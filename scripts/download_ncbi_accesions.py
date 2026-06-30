import pandas as pd
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
import gzip
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# ============================================================
# Input parsing
# ============================================================

def read_sourmash_gather(path):
    """Read a single sourmash gather CSV and return unique match accessions."""
    df = pd.read_csv(path)
    col = "match_name" if "match_name" in df.columns else "name"
    return df[col].str.split(' ').str[0].unique()


# ============================================================
# Batch download by accession (flat .fna.gz output, parallel)
# ============================================================

BATCH_SIZE = 500
COMPLETED_FILE = "completed_accessions.txt"

_completed_lock = threading.Lock()


def load_completed(out_dir):
    path = os.path.join(out_dir, COMPLETED_FILE)
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def mark_completed(out_dir, accessions):
    path = os.path.join(out_dir, COMPLETED_FILE)
    with open(path, "a") as fh:
        fh.write("\n".join(accessions) + "\n")


def _gzip_into(src_fna, dest_gz):
    """Compress src_fna -> dest_gz via the gzip CLI (fast), Python fallback."""
    try:
        with open(dest_gz, "wb") as out:
            subprocess.run(["gzip", "-1", "-c", str(src_fna)], stdout=out, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        with open(src_fna, "rb") as s, gzip.open(dest_gz, "wb", compresslevel=1) as d:
            shutil.copyfileobj(s, d, length=16 * 1024 * 1024)


def download_batch(batch, batch_idx, out_dir):
    """Download one batch, extract, and gzip each .fna into out_dir.

    Returns the number of .fna.gz files written. Self-contained so it can run
    in a thread pool: every batch uses a globally-unique index for its zip,
    accession list, and temp extract dir, so nothing collides across workers.
    """
    zip_path = os.path.join(out_dir, f"batch_{batch_idx:04d}.zip")
    acc_file = os.path.join(out_dir, f"batch_{batch_idx:04d}_accessions.txt")
    with open(acc_file, "w") as fh:
        fh.write("\n".join(batch) + "\n")

    cmd = ["datasets", "download", "genome", "accession",
           "--inputfile", acc_file,
           "--include", "genome",
           "--filename", zip_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"datasets download failed for batch {batch_idx} (exit {result.returncode})")
    if not os.path.exists(zip_path):
        raise RuntimeError(f"Expected zip not found: {zip_path}")

    tmp_extract = os.path.join(out_dir, f"_tmp_batch_{batch_idx:04d}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_extract)

    n = 0
    for fna in Path(tmp_extract).rglob("*.fna"):
        dest = os.path.join(out_dir, fna.name + ".gz")
        if not os.path.exists(dest):
            _gzip_into(fna, dest)
            n += 1
        else:
            print(f"[warn] skipping duplicate: {fna.name}.gz", file=sys.stderr)

    shutil.rmtree(tmp_extract)
    os.remove(zip_path)
    os.remove(acc_file)
    return n


def download_by_accession(accessions, out_dir, download_workers=3):
    """Download a flat list of accessions into out_dir as .fna.gz files.

    Batches download (and gzip) in parallel via a thread pool. Both the
    `datasets` subprocess and the `gzip` CLI release the GIL, so threads
    overlap network and compression without pickling overhead. Resume is
    supported via the completed-accessions marker file.
    """
    completed = load_completed(out_dir)
    remaining = [a for a in accessions if a not in completed]
    if not remaining:
        print("[download] All accessions already downloaded, nothing to do.", file=sys.stderr)
        return
    if completed:
        print(f"[download] Resuming: {len(completed)} already done, {len(remaining)} remaining.",
              file=sys.stderr)

    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    print(f"[download] {len(batches)} batches, {download_workers} parallel workers",
          file=sys.stderr)

    def worker(idx, batch):
        got = download_batch(batch, idx, out_dir)
        with _completed_lock:
            mark_completed(out_dir, batch)
        return idx, len(batch), got

    done = 0
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = [pool.submit(worker, idx, b) for idx, b in enumerate(batches)]
        for fut in as_completed(futures):
            idx, requested, got = fut.result()
            done += 1
            print(f"[download] {done}/{len(batches)} (batch {idx}: {got}/{requested} genomes)",
                  file=sys.stderr)

    print(f"[download] All {len(batches)} batches complete.", file=sys.stderr)


# ============================================================
# Cascading taxonomy expansion per seed
# ============================================================

def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


def summary_accessions(taxid, limit=None):
    cmd = ["datasets", "summary", "genome", "taxon", str(taxid), "--as-json-lines"]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    result = _run(cmd)
    accessions = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        acc = rec.get("accession")
        if acc:
            accessions.append(acc)
    return accessions


def download_accessions(accessions, path):
    """Download a specific list of accessions to `path` (zip) via --inputfile."""
    if not accessions:
        return
    inputfile = Path(path).with_suffix(".accessions.txt")
    inputfile.write_text("\n".join(accessions))
    _run([
        "datasets", "download", "genome", "accession",
        "--inputfile", str(inputfile),
        "--filename", str(path),
    ])


def get_accession_lineage(accession, ranks=("species", "genus", "family", "order")):
    genome = _run(["datasets", "summary", "genome", "accession", accession])
    tax_id = json.loads(genome.stdout)["reports"][0]["organism"]["tax_id"]

    taxonomy = _run(["datasets", "summary", "taxonomy", "taxon", str(tax_id)])
    classification = json.loads(taxonomy.stdout)["reports"][0]["taxonomy"]["classification"]
    return {r: classification[r]["id"] for r in ranks if r in classification}


def accessions_in_zip(zip_path):
    """Extract the GC{A,F}_XXXXXXXXX.Y accession strings present in a datasets zip."""
    accs = set()
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) >= 3 and parts[0] == "ncbi_dataset" and parts[1] == "data":
                token = parts[2]
                if token.startswith("GCA_") or token.startswith("GCF_"):
                    accs.add(token)
    return accs


def build_genome_set(seed_accession, output_dir,
                     ranks=("species", "genus", "family", "order"),
                     limit=(1000, 500, 500, 500)):
    """Cascade species -> genus -> family -> order for one seed.

    Per-rank resume: if `{rank}_{taxid}.zip` already exists AND is valid, its
    accessions are loaded into `collected` so downstream ranks exclude them,
    and that rank is skipped. Corrupt zips are deleted and re-downloaded.

    Full-cascade skip: if the species zip already exists for a prior seed in
    the same species, skip the whole cascade (the lineage is identical).
    """
    assert len(ranks) == len(limit), "ranks and limit must be the same length"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lineage = get_accession_lineage(seed_accession, ranks=ranks)
    print(f"[genome_set] {seed_accession} lineage: {lineage}", file=sys.stderr)

    # Full-cascade skip if this species was already built by a previous seed
    species_taxid = lineage.get("species")
    if species_taxid is not None:
        species_zip = output_dir / f"species_{species_taxid}.zip"
        if species_zip.exists():
            try:
                zipfile.ZipFile(species_zip).close()  # verify it opens
            except (zipfile.BadZipFile, OSError):
                print(f"[genome_set] species {species_taxid} zip corrupt, "
                      f"re-running cascade", file=sys.stderr)
                species_zip.unlink()
            else:
                print(f"[genome_set] SKIP {seed_accession}: species {species_taxid} "
                      f"already built by a previous seed", file=sys.stderr)
                zips = []
                for rank in ranks:
                    taxid = lineage.get(rank)
                    if taxid is None:
                        continue
                    zp = output_dir / f"{rank}_{taxid}.zip"
                    if zp.exists():
                        zips.append(zp)
                return {"zips": zips, "accessions": set(), "lineage": lineage}

    collected = set()
    zip_paths = []

    for rank, rank_limit in zip(ranks, limit):
        taxid = lineage.get(rank)
        if taxid is None:
            print(f"[genome_set]   [{rank}] not in lineage, skipping", file=sys.stderr)
            continue
        if rank_limit <= 0:
            print(f"[genome_set]   [{rank}] limit is 0, skipping", file=sys.stderr)
            continue

        zip_path = output_dir / f"{rank}_{taxid}.zip"

        if zip_path.exists():
            try:
                got = accessions_in_zip(zip_path)
            except (zipfile.BadZipFile, OSError) as e:
                print(f"[genome_set]   [{rank} taxid={taxid}] corrupt zip ({e}), "
                      f"re-downloading", file=sys.stderr)
                zip_path.unlink()
            else:
                collected |= got
                zip_paths.append(zip_path)
                print(f"[genome_set]   [{rank} taxid={taxid}] cached ({len(got)} accessions)",
                      file=sys.stderr)
                continue

        available = summary_accessions(taxid, limit=rank_limit + len(collected))
        novel = [a for a in available if a not in collected]
        take = novel[:rank_limit]
        print(f"[genome_set]   [{rank} taxid={taxid}] {len(available)} avail, "
              f"{len(novel)} novel, taking {len(take)} (limit={rank_limit})",
              file=sys.stderr)

        if not take:
            continue

        download_accessions(take, zip_path)
        zip_paths.append(zip_path)
        got = accessions_in_zip(zip_path)
        collected |= got
        print(f"[genome_set]   [{rank}] downloaded {len(got)} | total collected {len(collected)}",
              file=sys.stderr)

    return {"zips": zip_paths, "accessions": collected, "lineage": lineage}


def extract_zip_to_taxon_folder(zip_path, genome_lists_dir):
    """Extract .fna files from one rank-keyed zip into genome_lists/{rank}_{taxid}/.
    Corrupt zips are logged, deleted, and skipped (returns 0)."""
    zip_path = Path(zip_path)
    stem = zip_path.stem
    target = Path(genome_lists_dir) / stem
    target.mkdir(parents=True, exist_ok=True)

    tmp_extract = target.parent / f"_tmp_{stem}"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_extract)
    except (zipfile.BadZipFile, OSError) as e:
        print(f"[extract] skipping corrupt zip {zip_path.name}: {e}", file=sys.stderr)
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract, ignore_errors=True)
        try:
            zip_path.unlink()
        except OSError:
            pass
        return 0

    n_extracted = 0
    for fna in Path(tmp_extract).rglob("*.fna"):
        dest = target / fna.name
        if not dest.exists():
            shutil.move(str(fna), dest)
            n_extracted += 1
    shutil.rmtree(tmp_extract)
    return n_extracted


def build_scrub_genome_lists(seed_accessions, genome_lists_dir,
                             ranks=("species", "genus", "family", "order"),
                             limit=(1000, 500, 100, 100)):
    """For each seed, cascade ranks and extract zips into genome_lists/{rank}_{taxid}/.

    Returns a dict mapping seed accession -> lineage (for the query->taxa mapping).
    """
    zip_cache = Path(genome_lists_dir) / "_zips"
    zip_cache.mkdir(parents=True, exist_ok=True)

    seed_lineages = {}
    all_zips = set()
    for i, seed in enumerate(seed_accessions, 1):
        print(f"[genome_lists] ({i}/{len(seed_accessions)}) seed={seed}", file=sys.stderr)
        try:
            result = build_genome_set(
                seed_accession=seed,
                output_dir=zip_cache,
                ranks=ranks,
                limit=limit,
            )
            seed_lineages[seed] = result["lineage"]
            all_zips.update(str(p) for p in result["zips"])
        except Exception as e:
            print(f"[genome_lists] seed {seed} failed: {e}", file=sys.stderr)
            continue

    # Extract each unique rank-zip into its taxonomy subfolder
    total = 0
    for zp in all_zips:
        total += extract_zip_to_taxon_folder(zp, genome_lists_dir)
    print(f"[genome_lists] Extracted {total} new .fna files into taxonomy subfolders",
          file=sys.stderr)

    return seed_lineages


# ============================================================
# Parallel gzip (for the cascade genome_lists/ dirs)
# ============================================================

def _gzip_one(fna_path):
    """Worker: gzip a single file in place, remove the original. Returns bytes saved."""
    fna_path = Path(fna_path)
    gz = fna_path.with_suffix(fna_path.suffix + ".gz")
    if gz.exists():
        fna_path.unlink()
        return 0
    orig_size = fna_path.stat().st_size
    # Use `gzip` CLI if available (faster than Python's gzip module) else fall back
    try:
        subprocess.run(["gzip", "-1", str(fna_path)], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        with open(fna_path, "rb") as src, gzip.open(gz, "wb", compresslevel=1) as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        fna_path.unlink()
    return orig_size


def gzip_all_genomes(scrub_db_path, n_workers=8):
    """Gzip every .fna in target_samples/ and genome_lists/*/ in parallel.

    Only needed for the cascade path; the accession-download path now gzips
    inline per batch.
    """
    scrub_db_path = Path(scrub_db_path)
    dirs = [scrub_db_path / "target_samples"]
    gl = scrub_db_path / "genome_lists"
    if gl.exists():
        dirs.extend(d for d in gl.iterdir() if d.is_dir() and not d.name.startswith("_"))

    all_files = []
    for d in dirs:
        if d.exists():
            all_files.extend(d.glob("*.fna"))

    if not all_files:
        print("[gzip] Nothing to compress", file=sys.stderr)
        return

    print(f"[gzip] Compressing {len(all_files)} files with {n_workers} workers",
          file=sys.stderr)

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_gzip_one, f) for f in all_files]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 500 == 0:
                print(f"[gzip] {done}/{len(all_files)}", file=sys.stderr)

    print(f"[gzip] Done: {len(all_files)} files", file=sys.stderr)


# ============================================================
# Lineage fetching
# ============================================================

def _fetch_lineage(acc):
    """Worker for parallel lineage fetching. Returns (accession, lineage_or_None)."""
    try:
        return acc, get_accession_lineage(acc)
    except Exception:
        return acc, None


def fetch_lineages_parallel(accessions, n_workers=8):
    """Fetch lineages for many accessions concurrently.

    NCBI rate limits: 5 req/s without an API key, 10 req/s with one. Cap
    n_workers accordingly — 8 is safe for short bursts, drop to 5 if you see
    429s.
    """
    lineages = {}
    failed = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fetch_lineage, acc): acc for acc in accessions}
        for i, fut in enumerate(as_completed(futures), 1):
            acc, lineage = fut.result()
            if lineage is None:
                failed.append(acc)
            else:
                lineages[acc] = lineage
            if i % 200 == 0:
                print(f"[lineage] {i}/{len(accessions)} fetched "
                      f"({len(failed)} failed so far)", file=sys.stderr)

    if failed:
        print(f"[lineage] {len(failed)} accessions failed lineage lookup", file=sys.stderr)
    return lineages


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Download genomes by accession from a sourmash CSV.')
    parser.add_argument('--sourmash', required=True,
                        help='A sourmash gather/search CSV; match_name/name column holds accessions.')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--download_workers', type=int, default=6,
                        help='Parallel datasets-download batches. Keep modest (4-8): '
                             'the bottleneck is bandwidth and NCBI rate limits, not CPU.')
    args = parser.parse_args()

    # ---------- DOWNLOAD PHASE (gzips inline per batch) ----------
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.sourmash)
    col = "match_name" if "match_name" in df.columns else "name"
    accessions = df[col].str.split(' ').str[0].unique()

    print(f"[download] {len(accessions)} unique accessions to download", file=sys.stderr)
    download_by_accession(accessions, args.outdir, download_workers=args.download_workers)


if __name__ == '__main__':
    main()