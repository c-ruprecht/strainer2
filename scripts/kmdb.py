#!/usr/bin/env python3
"""Read .kmdb metagenome k-mer databases written by kmer_metagenome_db.

Library use:
    from scripts.kmdb import read_meta, read_kmdb
    meta = read_meta("SRR9224053.kmdb")          # dict of header + metadata
    df = read_kmdb("SRR9224053.kmdb")            # polars: kmer (str), count
    df = read_kmdb("x.kmdb", decode=False)       # kmer as packed uint64

CLI:
    python kmdb.py info   x.kmdb
    python kmdb.py export x.kmdb out.parquet [--packed]
    python kmdb.py export x.kmdb out.tsv.zst

Format (see src/kmdb.h): 64-byte header, zstd blocks of delta-varint keys
followed by varint counts, "key=value" metadata text, 32-byte index entries.
Canonical k-mers are max(fwd, revcomp) packed 2 bits/base (A0 C1 G2 T3),
first base most significant, so decoded strings equal the scrub's #kmer keys.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import polars as pl
import zstandard

HEADER = struct.Struct("<4sIIIQQQQQQ")   # 64 bytes
INDEX = struct.Struct("<QQQII")          # 32 bytes
ACGT = np.frombuffer(b"ACGT", dtype=np.uint8)


def _header(fh):
    raw = fh.read(HEADER.size)
    if len(raw) < HEADER.size:
        raise ValueError("file too short to be a .kmdb")
    (magic, version, k, block_n, n_kmers, n_blocks,
     index_offset, meta_offset, meta_len, flags) = HEADER.unpack(raw)
    if magic != b"KMDB":
        raise ValueError("not a .kmdb file (bad magic)")
    if version != 1:
        raise ValueError(f"unsupported .kmdb version {version}")
    return dict(version=version, k=k, block_n=block_n, n_kmers=n_kmers,
                n_blocks=n_blocks, index_offset=index_offset,
                meta_offset=meta_offset, meta_len=meta_len, flags=flags)


def _index(fh, h):
    fh.seek(h["index_offset"])
    raw = fh.read(INDEX.size * h["n_blocks"])
    return [INDEX.unpack_from(raw, i * INDEX.size) for i in range(h["n_blocks"])]


def read_meta(path: str | Path) -> dict:
    """Header fields plus metadata. Repeated keys (read_file) become lists."""
    with open(path, "rb") as fh:
        h = _header(fh)
        fh.seek(h["meta_offset"])
        text = fh.read(h["meta_len"]).decode()
    meta: dict = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key in meta:
            meta[key] = (meta[key] if isinstance(meta[key], list) else [meta[key]]) + [val]
        else:
            meta[key] = val
    out = {k: v for k, v in h.items() if not k.endswith("_offset") and k != "meta_len"}
    out.update(meta)
    return out


def _varints(buf: np.ndarray, n_values: int) -> np.ndarray:
    """Decode LEB128 varints (vectorised)."""
    is_last = buf < 0x80
    ends = np.flatnonzero(is_last)
    if len(ends) != n_values:
        raise ValueError(f"block payload holds {len(ends)} varints, expected {n_values}")
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    group = np.repeat(np.arange(n_values), ends - starts + 1)
    shift = (np.arange(len(buf)) - starts[group]).astype(np.uint64) * np.uint64(7)
    parts = (buf & 0x7F).astype(np.uint64) << shift
    return np.add.reduceat(parts, starts)


def _decode_block(comp: bytes, first: int, n: int, dctx) -> tuple[np.ndarray, np.ndarray]:
    raw = np.frombuffer(dctx.decompress(comp), dtype=np.uint8)
    vals = _varints(raw, 2 * n)
    keys = np.cumsum(vals[:n], dtype=np.uint64) + np.uint64(first)
    counts = vals[n:].astype(np.uint32)
    return keys, counts


def decode_kmers(packed: np.ndarray, k: int) -> np.ndarray:
    """uint64 packed k-mers -> numpy array of k-character bytes."""
    shifts = np.arange(2 * (k - 1), -1, -2, dtype=np.uint64)
    codes = ((packed[:, None] >> shifts[None, :]) & np.uint64(3)).astype(np.uint8)
    return ACGT[codes].view(f"S{k}").ravel()


def read_kmdb(path: str | Path, decode: bool = True) -> pl.DataFrame:
    """All k-mers of a .kmdb as a polars DataFrame (kmer, count)."""
    with open(path, "rb") as fh:
        h = _header(fh)
        index = _index(fh, h)
        dctx = zstandard.ZstdDecompressor()
        frames = []
        for first, last, offset, comp_size, n in index:
            fh.seek(offset)
            kb, cb = _decode_block(fh.read(comp_size), first, n, dctx)
            if n and int(kb[-1]) != last:
                raise ValueError(f"block at offset {offset} is corrupt")
            # decode per block: the string expansion is ~250 bytes per k-mer
            # transiently, so doing it for the whole file at once could need
            # tens of GB
            col = (pl.Series("kmer", decode_kmers(kb, h["k"])).cast(pl.Binary).cast(pl.Utf8)
                   if decode else pl.Series("kmer", kb))
            frames.append(pl.DataFrame([col, pl.Series("count", cb)]))
    if not frames:
        return pl.DataFrame({"kmer": pl.Series([], dtype=pl.Utf8 if decode else pl.UInt64),
                             "count": pl.Series([], dtype=pl.UInt32)})
    df = pl.concat(frames, rechunk=True)
    if df.height != h["n_kmers"]:
        raise ValueError(f"decoded {df.height} k-mers, header says {h['n_kmers']}")
    return df


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_info = sub.add_parser("info", help="print header and metadata")
    p_info.add_argument("kmdb")
    p_exp = sub.add_parser("export", help="write kmer/count as parquet or TSV")
    p_exp.add_argument("kmdb")
    p_exp.add_argument("out", help=".parquet, .tsv, .tsv.gz or .tsv.zst")
    p_exp.add_argument("--packed", action="store_true",
                       help="keep kmer as packed uint64 instead of a string")
    a = ap.parse_args(argv)

    if a.cmd == "info":
        for k, v in read_meta(a.kmdb).items():
            for item in (v if isinstance(v, list) else [v]):
                print(f"{k}\t{item}")
        return 0

    out = a.out
    if not out.endswith((".parquet", ".tsv", ".tsv.gz", ".tsv.zst")):
        print(f"unknown output type: {out} (use .parquet, .tsv, .tsv.gz, .tsv.zst)",
              file=sys.stderr)
        return 2
    df = read_kmdb(a.kmdb, decode=not a.packed)
    if out.endswith(".parquet"):
        df.write_parquet(out)
        print(f"wrote {df.height} k-mers to {out}", file=sys.stderr)
        return 0
    df = df.rename({"kmer": "#kmer"})       # match the scrub's TSV headers
    if out.endswith(".tsv.zst"):
        with open(out, "wb") as fh, zstandard.ZstdCompressor(level=9).stream_writer(fh) as zw:
            zw.write(df.write_csv(separator="\t").encode())
    else:
        import gzip
        opener = gzip.open if out.endswith(".gz") else open
        with opener(out, "wt") as fh:
            fh.write(df.write_csv(separator="\t"))
    print(f"wrote {df.height} k-mers to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
