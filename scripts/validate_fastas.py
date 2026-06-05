#!/usr/bin/env python3
"""Validate gzipped FASTA genomes before sourmash DNA sketching.

For each *.fasta.gz (also *.fa.gz / *.fna.gz) in a directory, checks:
  - the gzip head decompresses
  - there is a FASTA header (line starting with '>')
  - there is sequence after the header
  - the sequence is DNA, not protein  (catches gffread -y / CDS-translation mistakes)

Writes a manysketch manifest of ONLY the valid genomes, plus a rejects TSV.
Designed for hundreds of thousands of files: parallel, reads only the head of
each file. Run `gzip -t` separately if you also want full CRC integrity.
"""
import argparse
import glob
import gzip
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

# Only the canonical nucleotide letters (+ lowercase soft-masking). IUPAC
# ambiguity codes are deliberately excluded from the *fraction* because they
# overlap amino-acid letters and would weaken the DNA-vs-protein test. They are
# rare enough (<1%) in real assemblies not to push genomes below the threshold.
_DNA_DELETE = {ord(c): None for c in "ACGTNacgtn"}

READ_BYTES = 131072        # decompressed chars to read from the head of each file
SAMPLE_SEQ_CHARS = 8000    # sequence chars to sample for the alphabet test
MIN_DNA_FRACTION = 0.90

# extensions stripped to build the genome 'name'
_EXTS = (".fasta.gz", ".fa.gz", ".fna.gz", ".fasta", ".fa", ".fna")


def _name_of(path):
    base = os.path.basename(path)
    for ext in _EXTS:
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def classify(path):
    """Return (path, status). status == 'OK' means usable for DNA sketching."""
    try:
        with gzip.open(path, "rt", errors="replace") as fh:
            chunk = fh.read(READ_BYTES)
    except (OSError, EOFError, gzip.BadGzipFile):
        return path, "GZIP_UNREADABLE"

    if not chunk:
        return path, "EMPTY"

    lines = chunk.splitlines()
    if not any(l.startswith(">") for l in lines):
        return path, "NO_HEADER"

    # collect sequence: non-header lines after the first header
    seq_parts = []
    seen_header = False
    total = 0
    for l in lines:
        if l.startswith(">"):
            seen_header = True
            continue
        if seen_header:
            s = l.strip()
            seq_parts.append(s)
            total += len(s)
            if total >= SAMPLE_SEQ_CHARS:
                break

    seq = "".join(seq_parts)[:SAMPLE_SEQ_CHARS]
    if not seq:
        return path, "NO_SEQUENCE"

    non_dna = len(seq.translate(_DNA_DELETE))
    dna_fraction = 1.0 - (non_dna / len(seq))
    if dna_fraction < MIN_DNA_FRACTION:
        return path, "NOT_DNA"

    return path, "OK"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--manifest", required=True,
                    help="output manysketch CSV of VALID genomes")
    ap.add_argument("--rejects", required=True,
                    help="output TSV of rejected files (path<TAB>reason)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--pattern", default="*.fasta.gz")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.fasta_dir, "**", args.pattern),
                             recursive=True))
    if not files:
        print(f"ERROR: no files matching {args.pattern!r} under {args.fasta_dir}",
              file=sys.stderr)
        sys.exit(1)

    n = len(files)
    print(f"  scanning {n} files with {args.jobs} workers...", flush=True)

    counts = Counter()
    valid_rows = []
    reject_rows = []

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, (path, status) in enumerate(
                ex.map(classify, files, chunksize=64), 1):
            counts[status] += 1
            if status == "OK":
                valid_rows.append(f"{_name_of(path)},{path},")
            else:
                reject_rows.append(f"{path}\t{status}")
            if i % 10000 == 0:
                print(f"    {i}/{n}", flush=True)

    with open(args.manifest, "w") as fh:
        fh.write("name,genome_filename,protein_filename\n")
        fh.write("\n".join(valid_rows))
        if valid_rows:
            fh.write("\n")

    with open(args.rejects, "w") as fh:
        fh.write("path\treason\n")
        fh.write("\n".join(reject_rows))
        if reject_rows:
            fh.write("\n")

    print("\n  validation summary:")
    for status, c in counts.most_common():
        print(f"    {status:16s} {c}")

    valid = counts.get("OK", 0)
    print(f"\n  VALID={valid}  REJECTED={n - valid}  (manifest: {args.manifest})")

    # exit non-zero only on hard failure (nothing usable)
    sys.exit(0 if valid > 0 else 3)


if __name__ == "__main__":
    main()
