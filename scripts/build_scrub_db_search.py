import pandas as pd
import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import gzip
import time



# ============================================================
# Input parsing
# ============================================================

def read_sourmash_gather(path):
    """Read a single sourmash gather CSV and return unique match accessions."""
    df = pd.read_csv(path)
    col = "match_name" if "match_name" in df.columns else "name"
    return df[col].str.split(' ').str[0].unique()


# ============================================================
# Batch download by accession (flat .fna output)
# ============================================================

BATCH_SIZE = 200
COMPLETED_FILE = "completed_accessions.txt"

DOWNLOAD_RETRIES = 3      # default, overridable via --download_retries
RETRY_BACKOFF = 10        # seconds, multiplied by attempt number


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


import re

ACC_RE = re.compile(r"(GC[AF]_\d+\.\d+)")


def download_batch(batch, batch_idx, out_dir, retries=DOWNLOAD_RETRIES):
    """Download and extract one batch of accessions, flattening .fna into out_dir.

    The `datasets download` call is retried up to `retries` times on transient
    failure (non-zero exit or missing zip). Returns the set of accessions
    actually extracted (parsed from .fna filenames), which may be a subset of
    `batch` if NCBI silently skipped some.
    """
    zip_path = os.path.join(out_dir, f"batch_{batch_idx:04d}.zip")
    acc_file = os.path.join(out_dir, f"batch_{batch_idx:04d}_accessions.txt")
    with open(acc_file, "w") as fh:
        fh.write("\n".join(batch) + "\n")

    cmd = ["datasets", "download", "genome", "accession",
           "--inputfile", acc_file,
           "--include", "genome",
           "--filename", zip_path]

    last_err = None
    for attempt in range(1, retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode == 0 and os.path.exists(zip_path):
            break  # success
        # transient failure: clean up a partial zip and retry
        last_err = (f"exit {result.returncode}"
                    if result.returncode != 0 else "zip not found")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if attempt < retries:
            wait = RETRY_BACKOFF * attempt
            print(f"[download] Batch {batch_idx} attempt {attempt}/{retries} failed "
                  f"({last_err}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    else:
        # all attempts exhausted
        os.remove(acc_file)
        raise RuntimeError(
            f"datasets download failed for batch {batch_idx} after {retries} "
            f"attempts ({last_err})")

    tmp_extract = os.path.join(out_dir, f"_tmp_batch_{batch_idx:04d}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_extract)

    arrived = set()
    for fna in Path(tmp_extract).rglob("*.fna"):
        dest = os.path.join(out_dir, fna.name)
        m = ACC_RE.search(fna.name)
        if m:
            arrived.add(m.group(1))
        if not os.path.exists(dest):
            shutil.move(str(fna), dest)
        else:
            print(f"[warn] skipping duplicate: {fna.name}", file=sys.stderr)

    shutil.rmtree(tmp_extract)
    os.remove(zip_path)
    os.remove(acc_file)
    return arrived



def download_by_accession(accessions, out_dir, retries=DOWNLOAD_RETRIES):
    """Download a flat list of accessions into out_dir as .fna files, with resume.

    Only accessions that actually produced a .fna are marked completed, so a
    silent partial download won't permanently skip the missing ones. Missing
    accessions are appended to missing_accessions.txt per-batch so the list
    survives an interrupted run.
    """
    completed = load_completed(out_dir)
    remaining = [a for a in accessions if a not in completed]
    if not remaining:
        print("[download] All accessions already downloaded, nothing to do.", file=sys.stderr)
        return
    if completed:
        print(f"[download] Resuming: {len(completed)} already done, {len(remaining)} remaining.",
              file=sys.stderr)

    miss_path = os.path.join(out_dir, "missing_accessions.txt")
    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    grand_missing = 0
    for idx, batch in enumerate(batches):
        print(f"[download] Batch {idx + 1}/{len(batches)}: {len(batch)} accessions", file=sys.stderr)
        arrived = download_batch(batch, idx, out_dir, retries = retries)

        arrived_base = {a.rsplit(".", 1)[0] for a in arrived}
        got = [a for a in batch
               if a in arrived or a.rsplit(".", 1)[0] in arrived_base]
        missing = [a for a in batch if a not in got]

        if got:
            mark_completed(out_dir, got)
        if missing:
            grand_missing += len(missing)
            with open(miss_path, "a") as fh:   # append per-batch, durable across crashes
                fh.write("\n".join(missing) + "\n")
            print(f"[download] Batch {idx + 1}/{len(batches)}: "
                  f"{len(got)} arrived, {len(missing)} MISSING", file=sys.stderr)
        else:
            print(f"[download] Batch {idx + 1}/{len(batches)} done ({len(got)}/{len(batch)}).",
                  file=sys.stderr)

    if grand_missing:
        print(f"[download] {grand_missing} accessions could not be downloaded; "
              f"see {miss_path}", file=sys.stderr)
    else:
        print("[download] All accessions downloaded successfully.", file=sys.stderr)


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
# Sourmash prefilter: sketch + pairwise
# ============================================================

def sourmash_sketch(genome_dir, sketches_path, ksize=31, scaled=1000, threads=8):
    """Build manysketch CSV from all .fna in genome_dir and run sourmash manysketch."""
    fna_files = sorted(Path(genome_dir).glob("*.fna"))
    if not fna_files:
        raise RuntimeError(f"No .fna files found in {genome_dir}")

    csv_path = os.path.join(genome_dir, "manysketch.csv")
    with open(csv_path, "w") as fh:
        fh.write("name,genome_filename,protein_filename\n")
        for fna in fna_files:
            fh.write(f"{fna.name},{fna},\n")

    print(f"[sketch] Sketching {len(fna_files)} genomes (k={ksize}, scaled={scaled}) -> {sketches_path}",
          file=sys.stderr)

    cmd = [
        "sourmash", "scripts", "manysketch",
        csv_path,
        "--param-str", f"dna,k={ksize},scaled={scaled}",
        "-o", sketches_path,
        "-c", str(threads),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, file=sys.stderr, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"sourmash manysketch failed (exit {result.returncode})")

    print(f"[sketch] Done: {sketches_path}", file=sys.stderr)
    return sketches_path


def sourmash_pairwise(sketches_path, pairwise_path, thresh = 0.8, threads=8):#, write_all=True):
    """Run sourmash pairwise on a sketches zip. Returns the resulting DataFrame."""
    print(f"[pairwise] Computing pairwise similarities -> {pairwise_path}", file=sys.stderr)

    cmd = [
        "sourmash", "scripts", "pairwise",
        sketches_path,
        "-o", pairwise_path,
        "-c", str(threads),
        #"-t", str(thresh), 
        "--ani",
        "--write-all"
    ]
  

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, file=sys.stderr, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"sourmash pairwise failed (exit {result.returncode})")

    df = pd.read_csv(pairwise_path)
    print(f"[pairwise] Done: {len(df)} comparisons", file=sys.stderr)
    return df


# ============================================================
# Exact k-mer containment via genome_compare
# ============================================================

def _compare_pairs(args):
    """Hash a_file once, compare only against its partner files."""
    genome_compare_bin, a_file, b_files, strain_mode = args

    with tempfile.NamedTemporaryFile(mode="w", suffix=".list", delete=False) as fh:
        fh.write("\n".join(b_files) + "\n")
        list_path = fh.name

    try:
        cmd = [genome_compare_bin, "-a", a_file, "-B", list_path]
        if strain_mode:
            cmd.append("-S")
        result = subprocess.run(cmd, capture_output=True, text=True)

        fracs = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("a_file") or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 5:
                try:
                    fracs[parts[1]] = float(parts[4])
                except ValueError:
                    pass
        return a_file, fracs
    finally:
        os.unlink(list_path)


def compare_pairs_from_df(df, genome_compare_bin, genome_dir,
                          strain_mode=True, n_workers=4):
    """Run genome_compare on specific (query, match) pairs, grouped by query."""
    grouped = df.groupby("query_name")["match_name"].apply(list).to_dict()

    tasks = [
        (genome_compare_bin,
         os.path.join(genome_dir, query),
         [os.path.join(genome_dir, m) for m in matches],
         strain_mode)
        for query, matches in grouped.items()
    ]

    print(f"[kmer_compare] Running genome_compare on {len(tasks)} query genomes "
          f"({len(df)} total pairs, {n_workers} workers)", file=sys.stderr)
    
    results = {}
    total_queries = len(tasks)
    with multiprocessing.Pool(n_workers) as pool:
        for i, (a_file, fracs) in enumerate(pool.imap_unordered(_compare_pairs, tasks), 1):
            results[a_file] = fracs
            if i % 100 == 0:
                print(f"[kmer_compare] {i}/{total_queries} query genomes done",
                      file=sys.stderr)

    rows = []
    for a_file, fracs in results.items():
        for b_file, frac in fracs.items():
            rows.append({
                "query_name": os.path.basename(a_file),
                "match_name": os.path.basename(b_file),
                "kmer_coverage": frac,
            })

    df_result = pd.DataFrame(rows)
    print(f"[kmer_compare] Done: {len(df_result)} pair results", file=sys.stderr)
    return df_result


def kmer_compare(df_pairwise, genome_compare_bin, genome_dir,
                 min_jaccard=0.7, strain_mode=True, n_workers=4):
    """Filter sourmash pairs by jaccard, then run exact k-mer containment."""
    if "average_containment" in df_pairwise.columns:
        jac_col = "average_containment"
    else:
        raise ValueError(
            f"Cannot find jaccard/similarity column in pairwise CSV. "
            f"Columns: {list(df_pairwise.columns)}"
        )

    df_filtered = df_pairwise[df_pairwise["query_name"] != df_pairwise["match_name"]].copy()
    df_filtered = df_filtered[df_filtered[jac_col] >= min_jaccard].copy()

    print(f"[kmer_compare] {len(df_filtered)} pairs above Jaccard >= {min_jaccard} "
          f"(from {len(df_pairwise)} total)", file=sys.stderr)

    if df_filtered.empty:
        return df_filtered

    df_kmer = compare_pairs_from_df(
        df_filtered[["query_name", "match_name"]],
        genome_compare_bin,
        genome_dir,
        strain_mode=strain_mode,
        n_workers=n_workers,
    )

    df_merged = df_filtered.merge(df_kmer, on=["query_name", "match_name"], how="left")
    return df_merged


# ============================================================
# Greedy dereplication
# ============================================================

def greedy_choice(df_kmer_compare, percentage):
    df = df_kmer_compare
    threshhold = percentage
    li_choice = []
    dropset = set()
    for entry in df['query_name'].unique():
        if entry in dropset:
            continue
        li_choice.append(entry)
        li_drop = df.loc[(df['query_name'] == entry) &
                         (df['kmer_coverage'] > threshhold)]['match_name'].to_list()
        dropset.update(li_drop)

    # Retroactive: drop representatives that later got dropped by another representative
    li_choice = [r for r in li_choice if r not in dropset]

    df_drop = pd.DataFrame([
        {"genome_representative": rep,
         "genome_drop_list": df.loc[
             (df['query_name'] == rep) & (df['kmer_coverage'] > threshhold),
             'match_name'].tolist()}
        for rep in li_choice
    ])

    all_dropped = df_drop['genome_drop_list'].explode()
    overlap = all_dropped[all_dropped.isin(df_drop['genome_representative'])].unique()
    print(f"{len(overlap)} dropped genomes are also representatives:")
    print(overlap)
    drop_list = all_dropped.dropna().unique().tolist()
    return df_drop, drop_list

#map scrub database
def build_query_to_taxa_map(scrub_db_path, drug_csv, target_samples_csv, seed_lineages):
    """
    For each (query_name, match_accession) in the gather CSVs, record the
    taxonomic lineage of the match. The representative genomes for that query
    are then everything in the corresponding taxonomy subfolders.
    """
    import re
    scrub_db_path = Path(scrub_db_path)
    ACC_RE = re.compile(r"(GC[AF]_\d+\.\d+)")

    def extract_accession(s):
        if not isinstance(s, str):
            return None
        m = ACC_RE.search(s)
        return m.group(1) if m else None

    rows = []
    for source, csv_path in [("drug", drug_csv), ("target_samples", target_samples_csv)]:
        if csv_path is None or not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            query = r["query_name"]
            match_acc = extract_accession(r.get("match_name")) or \
                        extract_accession(r.get("match_filename"))
            if match_acc is None:
                continue

            # The lineage was recorded when we processed this accession as a seed.
            # If this query's match wasn't a seed (shouldn't happen, but defensive),
            # we skip — there's no taxonomy folder for it.
            lineage = seed_lineages.get(match_acc, {})
            rows.append({
                "query_name": query,
                "source": source,
                "match_accession": match_acc,
                "species": lineage.get("species"),
                "genus": lineage.get("genus"),
                "family": lineage.get("family"),
                "order": lineage.get("order"),
            })

    df_map = pd.DataFrame(rows)
    out_path = scrub_db_path / "query_to_taxa.tsv"
    df_map.to_csv(out_path, sep="\t", index=False)
    n_unmapped = df_map["species"].isna().sum()
    print(f"[mapping] Wrote {len(df_map)} query entries to {out_path} "
          f"({n_unmapped} without lineage)", file=sys.stderr)
    return df_map

def gzip_all_genomes(scrub_db_path):
    """Gzip every .fna in target_samples/ and genome_lists/*/ subfolders."""
    scrub_db_path = Path(scrub_db_path)
    dirs = [scrub_db_path / "target_samples"]
    gl = scrub_db_path / "genome_lists"
    if gl.exists():
        dirs.extend(d for d in gl.iterdir() if d.is_dir() and not d.name.startswith("_"))

    for d in dirs:
        if not d.exists():
            continue
        files = list(d.glob("*.fna"))
        if not files:
            continue
        print(f"[gzip] Compressing {len(files)} files in {d}", file=sys.stderr)
        for f in files:
            gz = f.with_suffix(f.suffix + ".gz")
            if gz.exists():
                f.unlink()
                continue
            with open(f, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
            f.unlink()

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
        subprocess.run(["gzip", "-1", str(fna_path)], check=True,
                       capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        with open(fna_path, "rb") as src, gzip.open(gz, "wb", compresslevel=1) as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        fna_path.unlink()
    return orig_size


def gzip_all_genomes(scrub_db_path, n_workers=8):
    """Gzip every .fna in target_samples/ and genome_lists/*/ in parallel."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

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




def main():
    parser = argparse.ArgumentParser(description='Build a scrub k-mer database.')
    parser.add_argument('--drug',
                        help='A sourmash search or gather CSV for all strains in the drug.')
    parser.add_argument('--kmer_ident', type=float, default = 0.96,
                        help='k-mer coverage threshold for dereplication.')
    parser.add_argument('--min_jaccard', type=float, default = 0.80,
                        help='Min jaccard from sourmash pairwise to trigger exact kmer compare.')
    parser.add_argument('--scrub_db_path', required=True,
                        help='Output directory for the scrub database.')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--genome_compare', required=True,
                        help='Path to the strainer genome_compare binary.')
    parser.add_argument('--download_retries', type=int, default=DOWNLOAD_RETRIES,
                        help='Retries per batch on transient NCBI download failure.')
    parser.add_argument('--force', action='store_true')
    
    args = parser.parse_args()

    os.makedirs(args.scrub_db_path, exist_ok=True)

    # ---------- DOWNLOAD PHASE ----------

    target_dir = os.path.join(args.scrub_db_path, "genomes")
    os.makedirs(target_dir, exist_ok=True)
    df_drug = pd.read_csv(args.drug)
    col = "match_name" if "match_name" in df_drug.columns else "name"
    target_accessions =  df_drug[col].str.split(' ').str[0].unique()
    print(f"[target_samples] {len(target_accessions)} accessions to download",
            file=sys.stderr)
    
    download_by_accession(target_accessions, target_dir, retries=args.download_retries)
    
    sketches_path = os.path.join(args.scrub_db_path, "sketches.zip")
    if args.force or not os.path.exists(sketches_path):
        sourmash_sketch(genome_dir=target_dir, sketches_path=sketches_path,
                        ksize=31, scaled=1000, threads=args.threads)
    
    
    pairwise_path =  os.path.join(args.scrub_db_path, "pairwise.csv")
    if args.force or not os.path.exists(pairwise_path):
        df_pairwise = sourmash_pairwise(sketches_path=sketches_path,
                                        pairwise_path=pairwise_path,
                                        thresh=args.min_jaccard,
                                        threads=args.threads)
    else:
        print(f"[pairwise] Reusing existing {pairwise_path} (use --force to recompute)",
              file=sys.stderr)
        df_pairwise = pd.read_csv(pairwise_path)

    kmer_results_path = os.path.join(args.scrub_db_path, "kmer_compare.csv")
    if args.force or not os.path.exists(kmer_results_path):
        df_kmer = kmer_compare(df_pairwise=df_pairwise,
                            genome_compare_bin=args.genome_compare,
                            genome_dir=target_dir,
                            min_jaccard=args.min_jaccard,
                            strain_mode=True,
                            n_workers=args.threads)
    else:
        print(f"[pairwise] Reusing existing {pairwise_path} (use --force to recompute)",
              file=sys.stderr)
        df_kmer = pd.read_csv(kmer_results_path)

    if df_kmer.empty:
        print("[dereplicate] No close pairs found.", file=sys.stderr)
    
    else:
        df_kmer.to_csv(kmer_results_path, index=False)
        dups = df_kmer[df_kmer["kmer_coverage"] >= args.kmer_ident]
        print(f"[dereplicate] {len(dups)} pairs above kmer_ident >= {args.kmer_ident}",
              file=sys.stderr)

        df_drop, li_droplist = greedy_choice(df_kmer, percentage=args.kmer_ident)
        df_drop.to_csv(os.path.join(args.scrub_db_path, "representative_genomes.tsv"),
                       sep="\t")
        df_drop, li_droplist = greedy_choice(df_kmer, percentage=args.kmer_ident)
        df_drop.to_csv(os.path.join(args.scrub_db_path, "representative_genomes.tsv"),
                       sep="\t")

        # Write the drop list (genomes flagged as redundant)
        drop_path = os.path.join(args.scrub_db_path, "drop_list.txt")
        with open(drop_path, "w") as fh:
            fh.write("\n".join(li_droplist) + "\n")
        print(f"[dereplicate] Wrote {len(li_droplist)} dropped genomes to {drop_path}",
              file=sys.stderr)

        # Compute the keep set: every downloaded .fna minus the dropped ones
        all_fna = {p.name for p in Path(target_dir).glob("*.fna")}
        drop_set = set(li_droplist)
        keep = sorted(all_fna - drop_set)

        keep_path = os.path.join(args.scrub_db_path, "keep_list.txt")
        with open(keep_path, "w") as fh:
            fh.write("\n".join(keep) + "\n")
        print(f"[dereplicate] {len(all_fna)} total genomes, {len(drop_set)} dropped, "
              f"{len(keep)} kept -> {keep_path}", file=sys.stderr)

        print("[dereplicate] Removing redundant assemblies from taxonomy subfolders",
              file=sys.stderr)
        
        # adding in representatives that are not in any
        # read back all downloaded files

        #n_deleted = 0
        #for name in li_droplist:
        #    for src in origin.get(name, []):
        #        fpath = src / name
        #        if fpath.exists():
        #            fpath.unlink()
        #            n_deleted += 1
        #print(f"[dereplicate] Deleted {n_deleted} file instance(s) across all taxonomy folders.",
        #      file=sys.stderr)


    # ---------- GZIP PHASE ----------
    gzip_all_genomes(args.scrub_db_path, n_workers=min(args.threads, 16))


if __name__ == '__main__':
    main()
