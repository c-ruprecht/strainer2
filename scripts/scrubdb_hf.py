#!/usr/bin/env python3
"""
scrubdb_hf.py - package, upload and restore the scrub database via a Hugging Face dataset repo.

Subcommands
  pack     Stage the DB into sharded archives + manifests (this folder IS the repo layout)
  upload   Create the (private) dataset repo if needed and upload the staging folder (resumable)
  pull     Download from HF, then run `restore`
  restore  Verify checksums, unpack, rebuild t96/t94 symlinks, rewrite genome paths, verify

Repo / staging layout
  MANIFEST.json                    machine-readable description used by `restore`
  README.md                        dataset card (written once, edit freely)
  manifest/genomes_t98.tsv         name, size, shard
  manifest/checksums.sha256        sha256sum-compatible, paths relative to repo root
  summaries/*.tsv, *.list          top-level summary files, unchanged (original paths)
  genomes_t98/genomes_t98.NNN.tar  (plain tar: genomes are already .fna.gz)
  rocksdb/<name>.rocksdb.tar.zst.part-NNN

Only genomes_t98 is uploaded. On restore:
  * genomes_t96/ and genomes_t94/ are rebuilt from representatives_t96.tsv / _t94.tsv as relative
    links (genomes_t96/X -> ../genomes_t98/X); rows are matched on basename(genome_path), then genome.
  * the genome_path column of representatives_t{98,96,94}.tsv is rewritten to the restored location,
    and representatives_t*.list are regenerated from it (one path per line).
`pack` checks that each representatives_<tier>.tsv reproduces genomes_<tier>/ exactly and lies within t98.
"""
import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FORMAT_VERSION = 2
TIERS = ("t98", "t96", "t94")
LINKED_TIERS = ("t96", "t94")
SUMMARY_SUFFIXES = (".tsv", ".list")
ROCKSDB_EXCLUDES = ("LOCK", "LOG", "LOG.old.*")
GENOME_EXTS = (".gz", ".bz2", ".xz", ".zst", ".fna", ".fa", ".fasta", ".fas", ".ffn", ".fsa")

CODECS = {
    "none": {"ext": ".tar", "comp": "cat", "decomp": "cat", "level": 0},
    "zstd": {"ext": ".tar.zst", "comp": "zstd -q -T{threads} -{level}", "decomp": "zstd -dc -q", "level": 3},
    "pigz": {"ext": ".tar.gz", "comp": "pigz -p {threads} -{level}", "decomp": "pigz -dc", "level": 6},
    "gzip": {"ext": ".tar.gz", "comp": "gzip -{level}", "decomp": "gzip -dc", "level": 6},
}

q = shlex.quote


# ----------------------------------------------------------------------------- helpers

def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def die(msg):
    log(f"ERROR: {msg}")
    sys.exit(1)


def run(cmd):
    """Run a shell pipeline; fail if any stage fails."""
    subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True)


def sha256(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_of(codec_name):
    return CODECS[codec_name]["comp"].split()[0]


def pick_codec(requested, fallback_order=("zstd", "pigz", "gzip")):
    order = [requested] if requested else list(fallback_order)
    for name in order:
        if shutil.which(tool_of(name)):
            return name
    die(f"no compressor found (tried {order}); `conda install -c conda-forge zstd`")


def scan(d):
    """name -> path for all non-directory entries (files and symlinks) in d."""
    d = Path(d)
    if not d.is_dir():
        die(f"missing directory: {d}")
    out = {}
    for e in os.scandir(d):
        if e.is_dir(follow_symlinks=True):
            log(f"  skipping subdirectory {e.path}")
            continue
        out[e.name] = Path(e.path)
    return out


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def read_table(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    return header, rows


def stem(name):
    changed = True
    while changed:
        changed = False
        for e in GENOME_EXTS:
            if name.endswith(e):
                name, changed = name[: -len(e)], True
    return name


def resolve_reps(tsv, t98_names, name_col, path_col):
    """Match each row of a representatives table to a genomes_t98 file name.
    Returns (header, rows, path_idx, hits, unresolved) where hits[i] is a t98 name or None."""
    header, rows = read_table(tsv)
    ip = header.index(path_col) if path_col in header else None
    iname = header.index(name_col) if name_col in header else None
    if ip is None and iname is None:
        die(f"{tsv}: neither '{path_col}' nor '{name_col}' column found")
    names = set(t98_names)
    by_stem = {}
    for n in t98_names:
        by_stem.setdefault(stem(n), []).append(n)
    hits, unresolved = [], []
    for r in rows:
        cands = []
        if ip is not None and ip < len(r) and r[ip]:
            cands.append(os.path.basename(r[ip]))
        if iname is not None and iname < len(r) and r[iname]:
            cands.append(r[iname])
        hit = None
        for c in cands:
            if c in names:
                hit = c
                break
            s = by_stem.get(stem(c), [])
            if len(s) == 1:
                hit = s[0]
                break
        hits.append(hit)
        if hit is None:
            unresolved.append(cands[0] if cands else "<empty row>")
    return header, rows, ip, hits, unresolved


def make_shards(names, sizes, target_bytes):
    shards, cur, cur_size = [], [], 0
    for n in sorted(names):
        if cur and cur_size + sizes[n] > target_bytes:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(n)
        cur_size += sizes[n]
    if cur:
        shards.append(cur)
    return shards


# ----------------------------------------------------------------------------- pack

def pack_shard(src_dir, names, out, staging, codec, threads, level):
    """tar (dereferencing symlinks) + optional compression of one shard. Resumable via a state file."""
    rel = out.relative_to(staging)
    state = staging / ".state" / (str(rel) + ".json")
    list_hash = hashlib.sha256("\0".join(names).encode()).hexdigest()
    if out.exists() and state.exists():
        st = json.loads(state.read_text())
        if st.get("list_hash") == list_hash:
            return str(rel), st["sha256"], "reused"

    state.parent.mkdir(parents=True, exist_ok=True)
    filelist = Path(str(out) + ".filelist")
    tmp = Path(str(out) + ".tmp")
    filelist.write_bytes(b"\0".join(n.encode() for n in names) + b"\0")
    comp = codec["comp"].format(threads=threads, level=level)
    run(f"tar -c -h -f - -C {q(str(src_dir))} --null -T {q(str(filelist))} | {comp} > {q(str(tmp))}")
    tmp.rename(out)
    filelist.unlink()
    digest = sha256(out)
    state.write_text(json.dumps({"list_hash": list_hash, "sha256": digest}))
    return str(rel), digest, "built"


def pack_rocksdb(rdb, staging, codec_name, threads, part_bytes):
    codec = CODECS[codec_name]
    outdir = staging / "rocksdb"
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / f"{rdb.name}{codec['ext']}.part-"
    state = staging / ".state" / "rocksdb.json"
    if state.exists():
        st = json.loads(state.read_text())
        if (st.get("source") == str(rdb) and st.get("codec") == codec_name
                and all((staging / p).exists() for p in st["parts"])):
            log("  rocksdb: reusing existing parts")
            return st
    for old in outdir.glob(f"{rdb.name}*.part-*"):
        old.unlink()
    excl = " ".join(f"--exclude={q(x)}" for x in ROCKSDB_EXCLUDES)
    comp = codec["comp"].format(threads=threads, level=codec["level"])
    run(f"tar -c -f - -C {q(str(rdb.parent))} {excl} {q(rdb.name)} | {comp} "
        f"| split -a 3 -d -b {part_bytes} - {q(str(prefix))}")
    parts = sorted(outdir.glob(f"{rdb.name}{codec['ext']}.part-*"))
    st = {
        "source": str(rdb),
        "name": rdb.name,
        "codec": codec_name,
        "parts": [str(p.relative_to(staging)) for p in parts],
        "sha256": {str(p.relative_to(staging)): sha256(p) for p in parts},
    }
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(st))
    return st


def cmd_pack(a):
    db = Path(a.db_dir).resolve()
    reps = db / "representatives"
    staging = Path(a.staging).resolve()
    shard_bytes = int(a.shard_gb * 1024 ** 3)
    for src in filter(None, [db, Path(a.rocksdb).resolve() if a.rocksdb else None]):
        if staging == src or src in staging.parents or staging in src.parents:
            die(f"--staging {staging} overlaps the source {src}; use a separate folder (e.g. scratch)")

    # --- inventory t98 (symlinks inside t98 are dereferenced)
    log(f"Scanning {reps / 'genomes_t98'}")
    t98 = scan(reps / "genomes_t98")
    broken = [n for n, p in t98.items() if not p.exists()]
    if broken:
        die(f"{len(broken)} broken entries in genomes_t98, e.g. {broken[:3]}")
    sizes = {n: p.stat().st_size for n, p in t98.items()}
    log(f"  genomes_t98: {len(t98)} files, {sum(sizes.values()) / 1024 ** 3:.1f} GiB")

    # genomes are normally already gzipped -> plain tar, no recompression
    if a.codec:
        codec_name = pick_codec(a.codec)
    else:
        n_comp = sum(n.endswith((".gz", ".bz2", ".xz", ".zst")) for n in t98)
        codec_name = "none" if n_comp > len(t98) / 2 else pick_codec(None)
    codec = CODECS[codec_name]
    level = a.level if a.level is not None else codec["level"]

    summaries = sorted(p for p in db.iterdir() if p.is_file() and p.suffix in SUMMARY_SUFFIXES)

    # --- every tier must be reproducible from its representatives_<tier>.tsv
    problems = 0
    for tier in TIERS:
        tsv = db / f"representatives_{tier}.tsv"
        if not tsv.exists():
            die(f"{tsv} not found")
        _, rows, _, hits, unresolved = resolve_reps(tsv, list(t98), a.name_col, a.path_col)
        resolved = {h for h in hits if h}
        on_disk = set(scan(reps / f"genomes_{tier}"))
        not_listed = on_disk - resolved
        dups = len([h for h in hits if h]) - len(resolved)
        log(f"  {tsv.name}: {len(rows)} rows, {len(resolved)} resolve into t98, {len(unresolved)} do not, "
            f"{dups} duplicate rows; {len(not_listed)} files in genomes_{tier}/ not in table")
        for tok in unresolved[:5]:
            log(f"    not in t98: {tok}")
        for n in sorted(not_listed)[:5]:
            log(f"    in genomes_{tier}/ but not in table: {n}")
        problems += len(unresolved) + len(not_listed)
    if problems and not a.allow_link_mismatch:
        die("genome dirs cannot be rebuilt exactly from the representatives tables (see above); "
            "fix them or rerun with --allow-link-mismatch")

    t98_shards = make_shards(t98, sizes, shard_bytes)
    log(f"  plan: {len(t98_shards)} shard(s) of ~{a.shard_gb} GiB, genome codec={codec_name}")

    done_markers = sorted(p.name for p in reps.glob("*.done"))
    rdb = Path(a.rocksdb).resolve() if a.rocksdb else None
    if rdb and not (rdb / "CURRENT").exists():
        die(f"{rdb} does not look like a RocksDB directory (no CURRENT file)")
    rdb_codec = pick_codec(a.rocksdb_codec) if rdb else None

    if a.dry_run:
        log(f"  summaries: {[p.name for p in summaries]}")
        log(f"  done markers: {done_markers}")
        log(f"  rocksdb: {rdb} (codec {rdb_codec})")
        log("Dry run - nothing written.")
        return

    for sub in ("manifest", "summaries", "genomes_t98"):
        (staging / sub).mkdir(parents=True, exist_ok=True)

    def shard_path(i):
        return staging / "genomes_t98" / f"genomes_t98.{i:03d}{codec['ext']}"

    rows = [(n, sizes[n], shard_path(i).name) for i, names in enumerate(t98_shards) for n in names]
    write_tsv(staging / "manifest" / "genomes_t98.tsv", ["name", "size", "shard"], rows)
    for p in summaries:
        shutil.copy2(p, staging / "summaries" / p.name)
    log(f"  copied {len(summaries)} summary files")

    threads_per_job = max(1, a.threads // a.jobs)
    checksums = {}
    jobs = [(names, shard_path(i)) for i, names in enumerate(t98_shards)]
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(pack_shard, reps / "genomes_t98", n, o, staging, codec, threads_per_job, level)
                for n, o in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            rel, digest, status = f.result()
            checksums[rel] = digest
            log(f"  [{k}/{len(jobs)}] {rel} ({status})")

    rdb_info = None
    if rdb:
        log(f"Packing RocksDB {rdb} with {rdb_codec} (make sure nothing is writing to it)")
        st = pack_rocksdb(rdb, staging, rdb_codec, a.threads, int(a.part_gb * 1024 ** 3))
        checksums.update(st["sha256"])
        rdb_info = {"name": st["name"], "codec": st["codec"], "parts": st["parts"], "source": str(rdb)}
        log(f"  rocksdb: {len(st['parts'])} part(s)")

    with open(staging / "manifest" / "checksums.sha256", "w") as f:
        for rel in sorted(checksums):
            f.write(f"{checksums[rel]}  {rel}\n")

    manifest = {
        "format_version": FORMAT_VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "db_name": db.name,
        "source_db_dir": str(db),
        "codec": codec_name,
        "counts": {"t98": len(t98)},
        "shards": {"genomes_t98": [str(shard_path(i).relative_to(staging)) for i in range(len(t98_shards))]},
        "reps_tables": {t: f"summaries/representatives_{t}.tsv" for t in TIERS},
        "linked_tiers": list(LINKED_TIERS),
        "name_col": a.name_col,
        "path_col": a.path_col,
        "rocksdb": rdb_info,
        "summaries": [p.name for p in summaries],
        "done_markers": done_markers,
    }
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    readme = staging / "README.md"
    if not readme.exists():
        readme.write_text(README_TEMPLATE.format(db_name=db.name, t98=len(t98)))
    log(f"Staging complete: {staging}")


README_TEMPLATE = """---
license: other
pretty_name: {db_name}
tags: [metagenomics, bacterial-genomes, strain-detection]
---

# {db_name}

Scrub / reference genome database (dereplicated high-quality bacterial genomes).

- genomes_t98: {t98} genomes (sharded tar archives of .fna.gz files)
- genomes_t96 / genomes_t94 are recreated on restore as symlinks into genomes_t98,
  from `summaries/representatives_t96.tsv` and `representatives_t94.tsv`
- sourmash RocksDB index (split archive)
- summary tables in `summaries/` (genome_path columns hold the original build paths;
  `restore` rewrites them for the representatives tables)

Restore with `python scrubdb_hf.py pull --repo <user>/<repo> --dest <dir>`.
See `MANIFEST.json` for the machine-readable layout.
"""


# ----------------------------------------------------------------------------- upload

def cmd_upload(a):
    from huggingface_hub import HfApi

    staging = Path(a.staging).resolve()
    if not (staging / "MANIFEST.json").exists():
        die(f"{staging}/MANIFEST.json not found - run `pack` first")
    api = HfApi()
    log(f"Logged in as: {api.whoami()['name']}")
    url = api.create_repo(a.repo, repo_type="dataset", private=not a.public, exist_ok=True)
    log(f"Repo: {url}")
    api.upload_large_folder(
        repo_id=a.repo,
        folder_path=staging,
        repo_type="dataset",
        ignore_patterns=[".state/*", "*.tmp", "*.filelist"],
        num_workers=a.workers,
    )
    if a.tag:
        api.create_tag(a.repo, tag=a.tag, repo_type="dataset", exist_ok=True)
        log(f"Tagged revision as {a.tag}")
    log("Upload finished.")


# ----------------------------------------------------------------------------- pull / restore

def cmd_pull(a):
    from huggingface_hub import snapshot_download

    allow = ["MANIFEST.json", "README.md", "manifest/*", "summaries/*"]
    if not a.no_genomes:
        allow += ["genomes_t98/*"]
    if not a.no_rocksdb:
        allow += ["rocksdb/*"]
    log(f"Downloading {a.repo}@{a.revision or 'main'} -> {a.download_dir}")
    snapshot_download(
        repo_id=a.repo,
        repo_type="dataset",
        revision=a.revision,
        local_dir=a.download_dir,
        allow_patterns=allow,
        max_workers=a.workers,
    )
    cmd_restore(a)


def rewrite_reps_tables(man, dl, dest, reps, t98_names):
    """Write representatives_<tier>.tsv with genome_path pointing at the restored files,
    and regenerate representatives_<tier>.list from it. Returns {tier: [resolved names]}."""
    resolved_by_tier = {}
    for tier, rel in man["reps_tables"].items():
        header, rows, ip, hits, unresolved = resolve_reps(dl / rel, t98_names, man["name_col"], man["path_col"])
        new_paths = []
        for r, h in zip(rows, hits):
            if h and ip is not None:
                r[ip] = str(reps / f"genomes_{tier}" / h)
            if ip is not None and ip < len(r):
                new_paths.append(r[ip])
        write_tsv(dest / f"representatives_{tier}.tsv", header, rows)
        if f"representatives_{tier}.list" in man["summaries"]:
            (dest / f"representatives_{tier}.list").write_text("".join(p + "\n" for p in new_paths))
        resolved_by_tier[tier] = [h for h in hits if h]
        log(f"  representatives_{tier}: {len(resolved_by_tier[tier])} paths rewritten"
            + (f", {len(unresolved)} rows not in t98 (left as is)" if unresolved else ""))
    return resolved_by_tier


def cmd_restore(a):
    dl = Path(a.download_dir).resolve()
    dest = Path(a.dest).resolve()
    man = json.loads((dl / "MANIFEST.json").read_text())
    if man["format_version"] != FORMAT_VERSION:
        die(f"unsupported format_version {man['format_version']} (script expects {FORMAT_VERSION})")
    reps = dest / "representatives"
    rdb_parent = Path(a.rocksdb_dest).resolve() if a.rocksdb_dest else dest
    src_db = Path(man["source_db_dir"])
    src_rdb = Path(man["rocksdb"]["source"]) if man["rocksdb"] else None
    clash = [p for p in (src_db, src_rdb) if p and (p == dest or p in dest.parents or dest in p.parents)]
    if src_rdb and man["rocksdb"] and (rdb_parent / man["rocksdb"]["name"]) == src_rdb:
        clash.append(src_rdb)
    if clash and not a.allow_overwrite_source:
        die(f"restore target overlaps the original database {clash[0]}; choose another --dest/--rocksdb-dest "
            "(or --allow-overwrite-source if you really mean it)")
    marks = dl / ".unpacked"
    marks.mkdir(exist_ok=True)

    want_genomes = not a.no_genomes
    want_rocksdb = not a.no_rocksdb and man["rocksdb"]
    for cname in {man["codec"]} | ({man["rocksdb"]["codec"]} if want_rocksdb else set()):
        tool = CODECS[cname]["decomp"].split()[0]
        if not shutil.which(tool):
            die(f"`{tool}` is required to unpack this repo")

    present = list(man["shards"]["genomes_t98"]) if want_genomes else []
    if want_rocksdb:
        present += man["rocksdb"]["parts"]
    missing = [p for p in present if not (dl / p).exists()]
    if missing:
        die(f"{len(missing)} archive(s) missing from {dl}, e.g. {missing[:3]}")

    def mark_of(rel):
        return marks / (rel.replace("/", "__") + ".ok")

    # --- checksums (skipping archives already unpacked in a previous run)
    if not a.skip_verify and present:
        sums = {}
        for line in (dl / "manifest" / "checksums.sha256").read_text().splitlines():
            h, rel = line.split(None, 1)
            sums[rel.strip()] = h
        todo = [p for p in man["shards"]["genomes_t98"] if want_genomes and not mark_of(p).exists()]
        if want_rocksdb and not (marks / "rocksdb.ok").exists():
            todo += man["rocksdb"]["parts"]
        log(f"Verifying sha256 of {len(todo)} archive(s)")
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for rel, h in zip(todo, ex.map(lambda p: sha256(dl / p), todo)):
                if h != sums.get(rel):
                    die(f"checksum mismatch: {rel}")
        log("  all checksums OK")

    # --- genome shards
    if want_genomes:
        decomp = CODECS[man["codec"]]["decomp"]
        target = reps / "genomes_t98"
        target.mkdir(parents=True, exist_ok=True)

        def unpack(rel):
            if mark_of(rel).exists():
                return rel, "already unpacked"
            run(f"{decomp} {q(str(dl / rel))} | tar -x --no-same-owner -f - -C {q(str(target))}")
            mark_of(rel).touch()
            return rel, "unpacked"

        shards = man["shards"]["genomes_t98"]
        log(f"Unpacking {len(shards)} genome shard(s) into {target}")
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(unpack, r) for r in shards]
            for k, f in enumerate(as_completed(futs), 1):
                rel, status = f.result()
                log(f"  [{k}/{len(shards)}] {rel} ({status})")

    # --- rocksdb
    if want_rocksdb:
        info = man["rocksdb"]
        rtarget = rdb_parent
        mark = marks / "rocksdb.ok"
        if mark.exists():
            log("RocksDB already unpacked")
        else:
            if (rtarget / info["name"]).exists():
                if not a.force:
                    die(f"{rtarget / info['name']} exists; use --force to overwrite")
                shutil.rmtree(rtarget / info["name"])
            rtarget.mkdir(parents=True, exist_ok=True)
            parts = " ".join(q(str(dl / p)) for p in info["parts"])
            log(f"Unpacking RocksDB -> {rtarget / info['name']}")
            run(f"cat {parts} | {CODECS[info['codec']]['decomp']} | tar -x --no-same-owner -f - -C {q(str(rtarget))}")
            mark.touch()

    # --- summaries back to the DB top level (original content)
    dest.mkdir(parents=True, exist_ok=True)
    for name in man["summaries"]:
        shutil.copy2(dl / "summaries" / name, dest / name)
    log(f"Restored {len(man['summaries'])} summary files into {dest}")

    # --- representatives tables: rewrite genome_path to the restored location
    t98_names = [r[0] for r in read_table(dl / "manifest" / "genomes_t98.tsv")[1]]
    if a.keep_original_paths:
        resolved_by_tier = {t: [h for h in resolve_reps(dl / rel, t98_names, man["name_col"], man["path_col"])[3] if h]
                            for t, rel in man["reps_tables"].items()}
    else:
        resolved_by_tier = rewrite_reps_tables(man, dl, dest, reps, t98_names)

    if not want_genomes:
        log("Genomes skipped (--no-genomes); not creating links.")
        return

    # --- t96 / t94 symlinks (relative, so the tree is relocatable)
    for tier in man["linked_tiers"]:
        d = reps / f"genomes_{tier}"
        d.mkdir(parents=True, exist_ok=True)
        for n in resolved_by_tier[tier]:
            link = d / n
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path("..") / "genomes_t98" / n)
        log(f"  genomes_{tier}: {len(set(resolved_by_tier[tier]))} symlinks")

    for m in man["done_markers"]:
        (reps / m).touch()

    # --- verify
    errors = 0
    for name, size, _ in read_table(dl / "manifest" / "genomes_t98.tsv")[1]:
        p = reps / "genomes_t98" / name
        if not p.is_file() or p.stat().st_size != int(size):
            errors += 1
            if errors <= 5:
                log(f"  BAD: {p}")
    for tier in man["linked_tiers"]:
        for link in (reps / f"genomes_{tier}").iterdir():
            if not link.exists():
                errors += 1
                if errors <= 5:
                    log(f"  BROKEN LINK: {link}")
    if errors:
        die(f"verification failed: {errors} problem(s)")
    log(f"Verified: t98={len(t98_names)}, " +
        ", ".join(f"{t}={len(set(resolved_by_tier[t]))}" for t in man["linked_tiers"]))

    if a.cleanup:
        for p in present:
            (dl / p).unlink(missing_ok=True)
        log("Removed downloaded archives (--cleanup)")
    log(f"Done: {dest}")


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="stage DB into archives + manifests")
    p.add_argument("--db-dir", required=True, help="DB directory containing representatives/ and summary TSVs")
    p.add_argument("--rocksdb", help="sourmash RocksDB directory to include")
    p.add_argument("--staging", required=True, help="output folder (= repo layout)")
    p.add_argument("--shard-gb", type=float, default=10, help="GiB per genome shard (default 10)")
    p.add_argument("--part-gb", type=float, default=10, help="GiB per RocksDB archive part (default 10)")
    p.add_argument("--codec", choices=list(CODECS),
                   help="genome shard codec (default: none if genomes are already compressed)")
    p.add_argument("--rocksdb-codec", choices=list(CODECS), help="default: zstd, else pigz, else gzip")
    p.add_argument("--level", type=int, help="compression level for genome shards")
    p.add_argument("--name-col", default="genome", help="genome id column in representatives_*.tsv")
    p.add_argument("--path-col", default="genome_path", help="genome path column in representatives_*.tsv")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--jobs", type=int, default=4, help="shards built in parallel")
    p.add_argument("--allow-link-mismatch", action="store_true",
                   help="pack even if representatives tables don't exactly reproduce the genome dirs")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("upload", help="upload staging folder to a HF dataset repo")
    p.add_argument("--staging", required=True)
    p.add_argument("--repo", required=True, help="e.g. yourname/scrubdb-derep-hq")
    p.add_argument("--public", action="store_true", help="create repo as public (default private)")
    p.add_argument("--tag", help="tag this upload, e.g. v2026.09")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_upload)

    def add_restore_args(p):
        p.add_argument("--download-dir", required=True, help="where archives are / will be downloaded")
        p.add_argument("--dest", required=True, help="DB directory to recreate")
        p.add_argument("--rocksdb-dest", help="parent dir for the RocksDB (default: --dest)")
        p.add_argument("--jobs", type=int, default=4)
        p.add_argument("--skip-verify", action="store_true", help="skip sha256 of archives")
        p.add_argument("--keep-original-paths", action="store_true",
                       help="don't rewrite genome_path in representatives_*.tsv / regenerate .list files")
        p.add_argument("--cleanup", action="store_true", help="delete archives after successful restore")
        p.add_argument("--force", action="store_true", help="overwrite existing RocksDB")
        p.add_argument("--no-genomes", action="store_true")
        p.add_argument("--no-rocksdb", action="store_true")
        p.add_argument("--allow-overwrite-source", action="store_true",
                       help="permit restoring onto the paths the DB was packed from (normally refused)")

    p = sub.add_parser("pull", help="download from HF and restore")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", help="branch, tag or commit (default main)")
    p.add_argument("--workers", type=int, default=8)
    add_restore_args(p)
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("restore", help="restore from an already-downloaded folder")
    add_restore_args(p)
    p.set_defaults(func=cmd_restore)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
