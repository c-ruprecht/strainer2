#!/usr/bin/env python3
"""
build_scrub_db.py

Download and dereplicate genomes from NCBI GenBank for a sourmash lineage s__ entry.

Workflow:
  1. Parse species name from a sourmash lineage string (s__ field).
  2. Check if the target database already exists (skip if so).
  3. Fetch assembly metadata via `datasets summary` and sort by quality
     (assembly level > contig N50 > number of contigs).  Optionally limit
     how many assemblies to download with --max_download.
  4. Download the selected assemblies by accession.
  5. All-vs-all kmer similarity matrix computed in parallel with genome_compare.
  6. Greedy dereplication on the matrix; stops early at --max_reps.
  7. Compress representative genomes to .fna.gz in the target directory.

Usage:
  build_scrub_db.py --lineage "s__Faecalibacterium prausnitzii" --target_dir /path/to/db
  build_scrub_db.py -s  -o /path/to/db \\
                    --max_download 500 --max_reps 100 --threads 8
"""

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LEVEL_PRIORITY = {
    "Complete Genome": 0,
    "Chromosome": 1,
    "Scaffold": 2,
    "Contig": 3,
}

# Maps --assembly-level flag values to their NCBI display names (for priority ordering)
NCBI_LEVEL_ORDER = ["complete", "chromosome", "scaffold", "contig"]


def parse_species_from_lineage(lineage):
    """Extract species name from a sourmash lineage string (s__ field)."""
    m = re.search(r's__([^;]+)', lineage)
    if not m:
        raise ValueError(f"No s__ entry found in lineage string: {lineage!r}")
    return m.group(1).strip()


def safe_name(species):
    """Return a filesystem-safe version of the species name."""
    return re.sub(r'[^\w.-]', '_', species)


def ncbi_species_name(species):
    """
    Strip GTDB clade suffixes (e.g. '_E', '_A') from a species name before
    querying NCBI, which uses NCBI taxonomy rather than GTDB designations.
    Handles suffixes on both genus and species epithet positions:
      'Faecalibacterium prausnitzii_E' -> 'Faecalibacterium prausnitzii'
      'Bacillus_A pacificus'           -> 'Bacillus pacificus'
    """
    parts = species.split(None, 1)
    cleaned = [re.sub(r'_[A-Z]+$', '', p).rstrip('_') for p in parts]
    return ' '.join(cleaned)


def genus_from_species(species):
    """Return the genus (first word) from a species name."""
    return species.split()[0]


# ---------------------------------------------------------------------------
# Assembly metadata & quality-ranked download
# ---------------------------------------------------------------------------

def count_assemblies(species, level):
    """
    Return the total number of assemblies in NCBI GenBank for this species + level.
    Queries without --limit so total_count reflects the true total.
    Returns None on failure.
    """
    cmd = [
        "datasets", "summary", "genome", "taxon", species,
        "--assembly-source", "genbank",
        "--assembly-level", level,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout).get("total_count")
    except (json.JSONDecodeError, AttributeError):
        return None


def _fetch_level_records(species, level, limit):
    """
    Fetch assembly metadata for a single assembly level.
    limit=0 means no limit. Returns [] on failure.
    """
    cmd = [
        "datasets", "summary", "genome", "taxon", species,
        "--assembly-source", "genbank",
        "--assembly-level", level,
        "--as-json-lines",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    records = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _sort_key(r):
    level = r.get("assembly_info", {}).get("assembly_level", "Contig")
    stats = r.get("assembly_stats", {})
    n50 = stats.get("contig_n50", 0)
    n_contigs = stats.get("number_of_contigs", 999_999)
    return (LEVEL_PRIORITY.get(level, 4), -n50, n_contigs)


def _query_ncbi_family(taxon):
    """Return the family name for a taxon string, or None."""
    cmd = ["datasets", "summary", "taxonomy", "taxon", taxon]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        reports = data.get("reports", [])
        if not reports:
            return None
        classification = reports[0].get("taxonomy", {}).get("classification", {})
        return classification.get("family", {}).get("name") or None
    except (json.JSONDecodeError, AttributeError, KeyError, IndexError):
        return None


def get_ncbi_family(ncbi_species):
    """
    Query NCBI taxonomy to return the family name for a species.
    Falls back to querying by genus if the species name is not found.
    Returns the family name string, or None on failure / not found.
    """
    family = _query_ncbi_family(ncbi_species)
    if family:
        return family
    genus = genus_from_species(ncbi_species)
    return _query_ncbi_family(genus)


def fetch_family_metadata(family, exclude_genus, assembly_levels, limit=0):
    """
    Fetch assembly metadata for a family, excluding any assembly whose
    organism_name starts with exclude_genus (i.e. the entire target genus,
    which was already tried at the genus level).
    Returns records sorted best-first.
    """
    requested = [l.strip().lower() for l in assembly_levels.split(",")]
    ordered_levels = [l for l in NCBI_LEVEL_ORDER if l in requested]

    all_records = []
    remaining = limit
    excl_lower = exclude_genus.lower()

    for level in ordered_levels:
        if limit and remaining == 0:
            break
        fetch_n = (remaining * 3) if limit else 0
        records = _fetch_level_records(family, level, fetch_n)
        filtered = [
            r for r in records
            if not r.get("organism", {}).get("organism_name", "").lower().startswith(excl_lower)
        ]
        filtered.sort(key=_sort_key)
        all_records.extend(filtered)
        if limit:
            remaining = max(0, limit - len(all_records))

    result = all_records[:limit] if limit else all_records
    print(
        f"[family] {len(result)} assemblies fetched for family {family!r} "
        f"(excluding genus {exclude_genus!r}).",
        file=sys.stderr,
    )
    return result


def fetch_genus_metadata(genus, exclude_species, assembly_levels, limit=0):
    """
    Fetch assembly metadata for all species in `genus`, excluding `exclude_species`.
    Records whose organism_name starts with exclude_species are dropped.
    Returns records sorted best-first.
    """
    requested = [l.strip().lower() for l in assembly_levels.split(",")]
    ordered_levels = [l for l in NCBI_LEVEL_ORDER if l in requested]

    all_records = []
    remaining = limit
    excl_lower = exclude_species.lower()

    for level in ordered_levels:
        if limit and remaining == 0:
            break
        # Fetch a buffer so filtering doesn't leave us short
        fetch_n = (remaining * 3) if limit else 0
        records = _fetch_level_records(genus, level, fetch_n)
        filtered = [
            r for r in records
            if not r.get("organism", {}).get("organism_name", "").lower().startswith(excl_lower)
        ]
        filtered.sort(key=_sort_key)
        all_records.extend(filtered)
        if limit:
            remaining = max(0, limit - len(all_records))

    result = all_records[:limit] if limit else all_records
    print(
        f"[genus] {len(result)} assemblies fetched for genus {genus!r} "
        f"(excluding {exclude_species!r}).",
        file=sys.stderr,
    )
    return result


def fetch_assembly_metadata(species, assembly_levels, limit=0):
    """
    Fetch assembly metadata sorted best-first (Complete > Chromosome > Scaffold > Contig,
    then highest N50, fewest contigs).

    Fetches complete genomes first, then fills remaining quota with lower-quality levels.
    Reports total assembly counts per level before fetching.
    """
    requested = [l.strip().lower() for l in assembly_levels.split(",")]
    ordered_levels = [l for l in NCBI_LEVEL_ORDER if l in requested]

    # --- count how many exist per level; stop early if quota already met ---
    print("[summary] counting available assemblies …", file=sys.stderr)
    total_counted = 0
    for level in ordered_levels:
        n = count_assemblies(species, level)
        label = f"{n:,}" if n is not None else "unknown"
        print(f"[summary]   {level:12s}: {label}", file=sys.stderr)
        if n:
            total_counted += n
        if limit and total_counted >= limit:
            break

    # --- fetch level-by-level, filling the quota ---
    all_records = []
    remaining = limit  # 0 = no limit

    for level in ordered_levels:
        if limit and remaining == 0:
            break
        level_limit = remaining if limit else 0
        print(
            f"[summary] fetching {level} metadata"
            f"{f' (limit={level_limit})' if level_limit else ''} …",
            file=sys.stderr,
        )
        records = _fetch_level_records(species, level, level_limit)
        records.sort(key=_sort_key)
        all_records.extend(records)
        if limit:
            remaining = max(0, limit - len(all_records))

    print(f"[summary] {len(all_records)} assemblies fetched total.", file=sys.stderr)
    return all_records


def download_by_accession(accessions, out_dir):
    """Download genomes for a specific list of accessions; return zip path."""
    zip_path = os.path.join(out_dir, "ncbi_dataset.zip")
    acc_file = os.path.join(out_dir, "accessions.txt")
    with open(acc_file, "w") as fh:
        fh.write("\n".join(accessions) + "\n")

    cmd = [
        "datasets", "download", "genome", "accession",
        "--inputfile", acc_file,
        "--assembly-source", "genbank",
        "--include", "genome",
        "--filename", zip_path,
    ]
    print(f"[download] {len(accessions)} accessions …", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"datasets download failed (exit {result.returncode})")
    if not os.path.exists(zip_path):
        raise RuntimeError(f"Expected zip not found: {zip_path}")
    return zip_path


def download_by_taxon(species, out_dir, assembly_levels):
    """Fallback: download all assemblies by taxon name."""
    zip_path = os.path.join(out_dir, "ncbi_dataset.zip")
    cmd = [
        "datasets", "download", "genome", "taxon", species,
        "--assembly-source", "genbank",
        "--assembly-level", assembly_levels,
        "--include", "genome",
        "--filename", zip_path,
    ]
    print(f"[download] {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"datasets download failed (exit {result.returncode})")
    if not os.path.exists(zip_path):
        raise RuntimeError(f"Expected zip not found: {zip_path}")
    return zip_path


def extract_genome_files(zip_path, extract_dir):
    """Unzip NCBI datasets archive and return list of genome FASTA paths."""
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # NCBI datasets layout: ncbi_dataset/data/{accession}/*.fna
    files = []
    for ext in ("*.fna", "*.fasta", "*.fa"):
        files.extend(
            str(p) for p in Path(extract_dir).rglob(ext)
            if p.stat().st_size > 0
        )
    return sorted(set(files))


def order_by_metadata(genome_files, metadata_records):
    """
    Re-order genome_files to match the quality ranking from metadata_records.
    Files not matched by accession keep their original relative order at the end.
    """
    # Build accession -> file path map from the extracted files
    # NCBI path: .../ncbi_dataset/data/{accession}/...
    acc_to_file = {}
    for f in genome_files:
        parts = Path(f).parts
        try:
            idx = parts.index("data")
            acc = parts[idx + 1]
            acc_to_file[acc] = f
        except (ValueError, IndexError):
            pass

    ordered = []
    seen = set()
    for rec in metadata_records:
        acc = rec.get("accession", "")
        if acc in acc_to_file:
            ordered.append(acc_to_file[acc])
            seen.add(acc_to_file[acc])

    # Append any files that didn't match a metadata record
    for f in genome_files:
        if f not in seen:
            ordered.append(f)

    return ordered


# ---------------------------------------------------------------------------
# Reference-genome identity filter
# ---------------------------------------------------------------------------

def load_reference_genomes(sourmash_csv, lineage):
    """
    Return genome file paths from a sourmash classification summary CSV
    whose lineage matches the s__ species in `lineage`.
    """
    target_species = parse_species_from_lineage(lineage)
    refs = []
    with open(sourmash_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                row_species = parse_species_from_lineage(row.get("lineage", ""))
            except ValueError:
                continue
            if row_species == target_species:
                fname = row.get("query_filename", "").strip()
                if fname and os.path.exists(fname):
                    refs.append(fname)
                elif fname:
                    print(f"[ref] warning: reference file not found: {fname}", file=sys.stderr)
    print(f"[ref] {len(refs)} reference genome(s) matched for {target_species!r}", file=sys.stderr)
    return refs


def _compare_ref_vs_all(args):
    """Worker: run genome_compare for one reference against the genome list file."""
    genome_compare_bin, ref, list_path, strain_mode = args
    cmd = [genome_compare_bin, "-a", ref, "-B", list_path]
    if strain_mode:
        cmd.append("-S")
    result = subprocess.run(cmd, capture_output=True, text=True)
    fracs = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 5:
            try:
                fracs[parts[1]] = float(parts[4])
            except ValueError:
                pass
    return fracs


def compute_ref_identities(genome_files, reference_files, genome_compare_bin, strain_mode, threads):
    """
    For each genome in genome_files compute max kmer identity to any reference genome.
    All reference comparisons run in parallel. Returns dict {genome_path: max_identity}.
    """
    max_identity = {gf: 0.0 for gf in genome_files}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".list", delete=False) as fh:
        fh.write("\n".join(genome_files) + "\n")
        list_path = fh.name

    try:
        args_list = [
            (genome_compare_bin, ref, list_path, strain_mode)
            for ref in reference_files
        ]
        with ProcessPoolExecutor(max_workers=min(threads, len(reference_files))) as ex:
            for fracs in ex.map(_compare_ref_vs_all, args_list):
                for gf, frac in fracs.items():
                    if gf in max_identity:
                        max_identity[gf] = max(max_identity[gf], frac)
    finally:
        os.unlink(list_path)

    return max_identity


def filter_by_ref_identity(genome_files, ref_identities, min_identity, max_identity):
    """
    Keep only genomes with min_identity <= ref_max_identity < max_identity.
    min_identity removes unrelated genomes; max_identity (the dereplication threshold)
    removes genomes that are essentially the same strain as the reference.
    """
    kept = []
    for gf in genome_files:
        ident = ref_identities.get(gf, 0.0)
        bname = os.path.basename(gf)
        if ident < min_identity:
            print(f"[filter] drop  {bname:60s}  ref_identity={ident:.3f}  (below min={min_identity})", file=sys.stderr)
        elif ident >= max_identity:
            print(f"[filter] drop  {bname:60s}  ref_identity={ident:.3f}  (same strain, >= threshold={max_identity})", file=sys.stderr)
        else:
            kept.append(gf)
            print(f"[filter] keep  {bname:60s}  ref_identity={ident:.3f}", file=sys.stderr)
    print(
        f"[filter] {len(kept)}/{len(genome_files)} genomes passed reference identity filter "
        f"({min_identity} <= identity < {max_identity}).",
        file=sys.stderr,
    )
    return kept


# ---------------------------------------------------------------------------
# Parallel all-vs-all dereplication
# ---------------------------------------------------------------------------

def _compare_one_vs_all(args):
    """
    Worker function (must be module-level for multiprocessing).
    Hashes `a_file` and computes containment of every other genome against it.
    Returns (a_file, {b_file: frac}).
    """
    genome_compare_bin, a_file, all_files, strain_mode = args

    others = [f for f in all_files if f != a_file]
    if not others:
        return a_file, {}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".list", delete=False) as fh:
        fh.write("\n".join(others) + "\n")
        list_path = fh.name

    try:
        cmd = [genome_compare_bin, "-a", a_file, "-B", list_path]
        if strain_mode:
            cmd.append("-S")
        result = subprocess.run(cmd, capture_output=True, text=True)

        fracs = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("a_file"):
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


def compute_similarity_matrix(genome_files, genome_compare_bin, strain_mode, threads):
    """
    Compute all-vs-all kmer containment in parallel.

    sim[a][b] = fraction of b's kmers found in a's hash (containment of b in a).
    Each of the N workers runs one genome_compare call (-a genome_i -B all_others),
    so the total number of subprocess invocations equals len(genome_files).
    """
    n = len(genome_files)
    print(
        f"[derep] computing {n}x{n} similarity matrix with {threads} thread(s) …",
        file=sys.stderr,
    )
    sim = {f: {} for f in genome_files}

    args_list = [
        (genome_compare_bin, f, genome_files, strain_mode)
        for f in genome_files
    ]

    with ProcessPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(_compare_one_vs_all, a): a[1] for a in args_list}
        done = 0
        for fut in as_completed(futures):
            a_file, fracs = fut.result()
            sim[a_file] = fracs
            done += 1
            print(f"[derep] {done}/{n} done", file=sys.stderr, end="\r")

    print(file=sys.stderr)  # newline after \r progress
    return sim


def greedy_cluster(genome_files, sim_matrix, threshold):
    """
    Greedy dereplication on a pre-computed similarity matrix.

    genome_files should be ordered best-first (quality ranking) so that higher-
    quality assemblies become representatives when strains are similar.
    Always runs to completion — use select_reps_by_ref_similarity() afterwards
    to apply max_reps.

    Returns:
        reps       : list of representative genome paths
        assignments: dict mapping every genome to (cluster_rep_path, similarity_to_rep)
                     Representatives map to (themselves, 1.0).
    """
    reps = []
    assignments = {}
    for genome in genome_files:
        if not reps:
            reps.append(genome)
            assignments[genome] = (genome, 1.0)
            print(f"[derep] kept  {os.path.basename(genome):60s}  max_sim=0.000", file=sys.stderr)
            continue

        sims = {r: sim_matrix[genome].get(r, 0.0) for r in reps}
        best_rep = max(sims, key=sims.get)
        max_sim = sims[best_rep]
        bname = os.path.basename(genome)
        if max_sim < threshold:
            reps.append(genome)
            assignments[genome] = (genome, 1.0)
            print(f"[derep] kept  {bname:60s}  max_sim={max_sim:.3f}", file=sys.stderr)
        else:
            assignments[genome] = (best_rep, max_sim)
            print(f"[derep] skip  {bname:60s}  max_sim={max_sim:.3f}", file=sys.stderr)
    return reps, assignments


def select_reps_by_ref_similarity(reps, ref_identities, max_reps):
    """
    Rank representative genomes by their max identity to the reference genome(s),
    descending, and return the top max_reps. If ref_identities is empty or max_reps
    is 0, returns reps unchanged.
    """
    if not max_reps:
        return reps
    if ref_identities:
        ranked = sorted(reps, key=lambda g: ref_identities.get(g, 0.0), reverse=True)
        selected = ranked[:max_reps]
        mean_sim = sum(ref_identities.get(g, 0.0) for g in selected) / len(selected)
        print(
            f"[select] top {len(selected)}/{len(reps)} representatives by ref similarity "
            f"(mean={mean_sim:.3f})",
            file=sys.stderr,
        )
        for g in selected:
            print(
                f"[select]   {os.path.basename(g):60s}  ref_identity={ref_identities.get(g, 0.0):.3f}",
                file=sys.stderr,
            )
        return selected
    return reps[:max_reps]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def compress_to_gz(src, dst):
    """Copy src to dst as gzip."""
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def accession_from_path(path):
    """Extract NCBI accession from the datasets directory layout."""
    parts = Path(path).parts
    try:
        idx = parts.index("data")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return Path(path).stem


# ---------------------------------------------------------------------------
# Genus / family fallback helpers
# ---------------------------------------------------------------------------

def _download_and_extract(metadata, tmp_subdir):
    """
    Download accessions from metadata records into tmp_subdir and return
    extracted genome file paths ordered by metadata quality. Returns [] on failure.
    """
    accessions = [r["accession"] for r in metadata if "accession" in r]
    if not accessions:
        return []
    os.makedirs(tmp_subdir, exist_ok=True)
    try:
        zip_path = download_by_accession(accessions, tmp_subdir)
    except RuntimeError as e:
        print(f"[download] failed: {e}", file=sys.stderr)
        return []
    files = extract_genome_files(zip_path, os.path.join(tmp_subdir, "extracted"))
    return order_by_metadata(files, metadata)


def supplement_with_genus(genome_files, ncbi_species, args, tmp_dir):
    """
    If fewer than max_download species genomes were found, download genus-level
    genomes (and, if the genus is monotypic, family-level genomes) to fill up
    to max_download before dereplication.

    Returns:
        genome_files      : original list + supplement genomes appended
        supplement_source : dict {path: "genus_supplement" | "family_supplement"}
                            (used to bypass ref-id min filter and label TSV rows)
    """
    if not args.max_download or len(genome_files) >= args.max_download:
        return genome_files, {}

    needed = args.max_download - len(genome_files)
    genus = genus_from_species(ncbi_species)
    supplement_source = {}

    print(
        f"\n[genus] only {len(genome_files)}/{args.max_download} species genomes available; "
        f"fetching up to {needed} from genus {genus!r} to fill download pool",
        file=sys.stderr,
    )
    genus_meta = fetch_genus_metadata(genus, ncbi_species, args.assembly_levels, limit=needed)
    if genus_meta:
        files = _download_and_extract(genus_meta, os.path.join(tmp_dir, "genus_supplement"))
        if files:
            print(f"[genus] {len(files)} genus genome(s) added to pool.", file=sys.stderr)
            supplement_source.update({f: "genus_supplement" for f in files})
            needed -= len(files)

    if needed > 0:
        # Genus exhausted or monotypic — escalate to family
        family = get_ncbi_family(ncbi_species)
        if family:
            print(
                f"[family] genus insufficient; fetching up to {needed} from family {family!r}",
                file=sys.stderr,
            )
            family_meta = fetch_family_metadata(family, genus, args.assembly_levels, limit=needed)
            if family_meta:
                files = _download_and_extract(family_meta, os.path.join(tmp_dir, "family_supplement"))
                if files:
                    print(f"[family] {len(files)} family genome(s) added to pool.", file=sys.stderr)
                    supplement_source.update({f: "family_supplement" for f in files})
        else:
            print(f"[family] could not determine family for {ncbi_species!r}.", file=sys.stderr)

    if not supplement_source:
        print(f"[genus] no higher-taxon genomes found to supplement pool.", file=sys.stderr)
        return genome_files, {}

    return genome_files + list(supplement_source), supplement_source


def genus_fallback(existing_reps, species, ncbi_species, args, tmp_dir, sname, species_dir, strain_mode):
    """
    Download and dereplicate genus-level genomes to fill up to args.max_reps.
    Genus genomes are deduplicated both against each other and against existing_reps.

    Returns:
        all_reps       : existing_reps + newly selected genus reps
        genus_rows     : list of dicts with cluster info for every genus genome considered,
                         keyed by accession. Each dict has keys:
                         accession, representative, similarity_to_cluster_rep, source="genus"
    """
    needed = args.max_reps - len(existing_reps)
    genus = genus_from_species(ncbi_species)
    fetch_limit = needed * 5  # buffer for dedup loss

    # --- collect candidates: genus first, then family if genus is insufficient ---
    candidates = []

    print(
        f"\n[genus] only {len(existing_reps)}/{args.max_reps} representatives found; "
        f"fetching up to {fetch_limit} from genus {genus!r}",
        file=sys.stderr,
    )
    genus_meta = fetch_genus_metadata(genus, ncbi_species, args.assembly_levels, limit=fetch_limit)
    if genus_meta:
        files = _download_and_extract(genus_meta, os.path.join(tmp_dir, "genus_fallback"))
        if files:
            print(f"[genus] {len(files)} genus genome(s) fetched.", file=sys.stderr)
            candidates.extend(files)

    if len(candidates) < fetch_limit:
        family = get_ncbi_family(ncbi_species)
        if family:
            still_needed = fetch_limit - len(candidates)
            print(
                f"[family] genus insufficient; fetching up to {still_needed} from family {family!r}",
                file=sys.stderr,
            )
            family_meta = fetch_family_metadata(family, genus, args.assembly_levels, limit=still_needed)
            if family_meta:
                files = _download_and_extract(family_meta, os.path.join(tmp_dir, "family_fallback"))
                if files:
                    print(f"[family] {len(files)} family genome(s) fetched.", file=sys.stderr)
                    candidates.extend(files)
        else:
            print(f"[family] could not determine family for {ncbi_species!r}.", file=sys.stderr)

    if not candidates:
        print(f"[genus] no higher-taxon candidates found.", file=sys.stderr)
        return existing_reps, []

    # --- all-vs-all similarity among candidates ---
    cand_sim = compute_similarity_matrix(candidates, args.genome_compare, strain_mode, args.threads)

    # --- max identity of each candidate vs existing species reps ---
    vs_reps = (
        compute_ref_identities(candidates, existing_reps, args.genome_compare, strain_mode, args.threads)
        if existing_reps else {c: 0.0 for c in candidates}
    )

    # --- greedy selection ---
    new_reps = []
    cluster_map = {}  # genome -> (cluster_rep_path, sim_to_rep)
    for genome in candidates:
        max_sim_to_reps = vs_reps.get(genome, 0.0)
        sims_to_new = {r: cand_sim[genome].get(r, 0.0) for r in new_reps}
        best_new = max(sims_to_new, key=sims_to_new.get) if sims_to_new else None
        max_sim_to_new = sims_to_new[best_new] if best_new else 0.0

        if max_sim_to_reps >= max_sim_to_new:
            max_sim, cluster_anchor = max_sim_to_reps, (existing_reps[0] if existing_reps else None)
        else:
            max_sim, cluster_anchor = max_sim_to_new, best_new

        bname = os.path.basename(genome)
        if max_sim < args.threshold:
            new_reps.append(genome)
            cluster_map[genome] = (genome, 1.0)
            print(f"[genus] kept  {bname:60s}  max_sim={max_sim:.3f}", file=sys.stderr)
            if len(new_reps) >= needed:
                break
        else:
            cluster_map[genome] = (cluster_anchor, max_sim)
            print(f"[genus] skip  {bname:60s}  max_sim={max_sim:.3f}", file=sys.stderr)

    print(
        f"[genus] added {len(new_reps)}/{needed} representative(s) from higher taxon.",
        file=sys.stderr,
    )

    # --- store ---
    for rep in new_reps:
        acc = accession_from_path(rep)
        out_path = os.path.join(species_dir, f"{sname}_genus_{acc}.fna.gz")
        compress_to_gz(rep, out_path)
        print(f"[store] {out_path}", file=sys.stderr)

    # --- build TSV rows ---
    rep_cluster_ids = {r: f"genus_{i+1}" for i, r in enumerate(new_reps)}
    rows = []
    for genome in candidates:
        if genome not in cluster_map:
            continue
        cluster_rep, sim_to_rep = cluster_map[genome]
        if genome in rep_cluster_ids:
            cluster_id = rep_cluster_ids[genome]
        elif cluster_rep in rep_cluster_ids:
            cluster_id = rep_cluster_ids[cluster_rep]
        else:
            cluster_id = "genus_species_rep"
        rows.append({
            "accession": accession_from_path(genome),
            "representative": genome in new_reps,
            "similarity_to_cluster_rep": sim_to_rep,
            "cluster_id": cluster_id,
            "source": "genus",
        })

    return existing_reps + new_reps, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def lineages_from_sourmash(sourmash_csv):
    """Return list of unique non-empty lineage strings found in a sourmash summary CSV."""
    seen = {}  # species_string -> first full lineage encountered
    with open(sourmash_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lineage = row.get("lineage", "").strip()
            if not lineage:
                continue
            try:
                species = parse_species_from_lineage(lineage)
            except ValueError:
                continue
            if species not in seen:
                seen[species] = lineage
    return list(seen.values())


def process_lineage(lineage, args):
    """Run the full download-filter-dereplicate pipeline for a single lineage."""
    species = parse_species_from_lineage(lineage)
    ncbi_species = ncbi_species_name(species)  # strips GTDB suffixes like _E
    sname = safe_name(species)
    species_dir = os.path.join(args.target_dir, sname)
    done_marker = os.path.join(species_dir, ".done")
    strain_mode = not args.no_strain_mode

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[info] species      : {species}", file=sys.stderr)
    print(f"[info] output dir   : {species_dir}", file=sys.stderr)
    print(f"[info] threshold    : {args.threshold}", file=sys.stderr)
    print(f"[info] max_download : {args.max_download or 'unlimited'}", file=sys.stderr)
    print(f"[info] max_reps     : {args.max_reps or 'unlimited'}", file=sys.stderr)
    print(f"[info] threads      : {args.threads}", file=sys.stderr)
    print(f"[info] strain mode  : {strain_mode}", file=sys.stderr)

    if os.path.exists(done_marker) and not args.force:
        print(
            f"[skip] Database already built for {species!r} at {species_dir}. "
            "Use --force to rebuild.",
            file=sys.stderr,
        )
        return

    os.makedirs(species_dir, exist_ok=True)

    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f"build_scrub_{sname}_", dir=args.tmp_dir)
    print(f"[info] tmp dir      : {tmp_dir}", file=sys.stderr)

    try:
        if args.genome_dir:
            # Use pre-downloaded genomes; still fetch metadata for quality ordering
            genome_files = sorted(set(
                str(p) for ext in ("*.fna", "*.fasta", "*.fa")
                for p in Path(args.genome_dir).rglob(ext)
                if p.stat().st_size > 0
            ))
            print(f"[info] {len(genome_files)} genome file(s) found in {args.genome_dir}.", file=sys.stderr)
            metadata = fetch_assembly_metadata(ncbi_species, args.assembly_levels, limit=args.max_download)
            if metadata and genome_files:
                genome_files = order_by_metadata(genome_files, metadata)
            genus_supplement = {}
        else:
            # --- fetch metadata & rank assemblies ---
            metadata = fetch_assembly_metadata(ncbi_species, args.assembly_levels, limit=args.max_download)

            if metadata:
                accessions = [r["accession"] for r in metadata if "accession" in r]
                zip_path = download_by_accession(accessions, tmp_dir)
                extract_dir = os.path.join(tmp_dir, "extracted")
                genome_files = extract_genome_files(zip_path, extract_dir)
            else:
                try:
                    zip_path = download_by_taxon(ncbi_species, tmp_dir, args.assembly_levels)
                    extract_dir = os.path.join(tmp_dir, "extracted")
                    genome_files = extract_genome_files(zip_path, extract_dir)
                except RuntimeError as exc:
                    print(
                        f"[warn] species-level download failed for {species!r}: {exc}\n"
                        f"[warn] no species genomes available; will attempt genus/family fallback.",
                        file=sys.stderr,
                    )
                    genome_files = []

            if genome_files:
                print(f"[info] {len(genome_files)} genome file(s) extracted.", file=sys.stderr)

            if metadata and genome_files:
                genome_files = order_by_metadata(genome_files, metadata)

            # --- pre-dereplication genus supplement (if species downloads < max_download) ---
            genome_files, genus_supplement = supplement_with_genus(
                genome_files, ncbi_species, args, tmp_dir
            )

            if args.download_only:
                manifest_path = os.path.join(species_dir, "genome_manifest.txt")
                with open(manifest_path, "w") as fh:
                    fh.write("\n".join(genome_files) + "\n")
                print(
                    f"[download_only] {len(genome_files)} genome(s) downloaded.\n"
                    f"[download_only] Manifest : {manifest_path}\n"
                    f"[download_only] Genome dir: {tmp_dir}",
                    file=sys.stderr,
                )
                return

        # --- reference-identity filter (optional, applied only to species-source genomes) ---
        ref_identities = {}
        if args.sourmash_summary:
            reference_files = load_reference_genomes(args.sourmash_summary, lineage)
            if reference_files:
                species_files = [gf for gf in genome_files if gf not in genus_supplement]
                ref_identities = compute_ref_identities(
                    genome_files, reference_files, args.genome_compare, strain_mode, args.threads
                )
                if args.min_ref_identity > 0.0:
                    # Filter species genomes by ref identity; always keep supplement genomes
                    filtered_species = filter_by_ref_identity(
                        species_files, ref_identities, args.min_ref_identity, args.threshold
                    )
                    genome_files = filtered_species + [gf for gf in genome_files if gf in genus_supplement]
                    if not filtered_species:
                        print(
                            "[warn] No species genomes passed the reference identity filter; "
                            "genus supplement genomes retained.",
                            file=sys.stderr,
                        )
                    if not genome_files:
                        print(
                            "[warn] No genomes remain after filtering; "
                            "will attempt post-derep genus fallback if --max_reps is set.",
                            file=sys.stderr,
                        )

        # --- build quality rows for all species-level genomes before dereplication ---
        reps = []
        quality_rows = []
        cluster_assignments = {}
        if genome_files:
            acc_to_meta = {r["accession"]: r for r in metadata if "accession" in r}
            tracked_genomes = list(ref_identities.keys()) if ref_identities else genome_files
            if metadata:
                tracked_genomes = order_by_metadata(tracked_genomes, metadata)

            for gf in tracked_genomes:
                acc = accession_from_path(gf)
                meta = acc_to_meta.get(acc, {})
                info = meta.get("assembly_info", {})
                stats = meta.get("assembly_stats", {})
                row = {
                    "path": gf,
                    "accession": acc,
                    "assembly_level": info.get("assembly_level", ""),
                    "contig_n50": stats.get("contig_n50", ""),
                    "number_of_contigs": stats.get("number_of_contigs", ""),
                }
                if ref_identities:
                    row["ref_max_identity"] = ref_identities.get(gf, "")
                quality_rows.append(row)

            # --- parallel all-vs-all similarity ---
            sim_matrix = compute_similarity_matrix(
                genome_files, args.genome_compare, strain_mode, args.threads
            )

            # --- save kmer identity matrix ---
            accessions = [accession_from_path(gf) for gf in genome_files]
            matrix_path = os.path.join(species_dir, "kmer_identity_matrix.csv")
            with open(matrix_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([""] + accessions)
                for gf, row_acc in zip(genome_files, accessions):
                    row = [row_acc] + [
                        sim_matrix[gf].get(other, 1.0 if gf == other else 0.0)
                        for other in genome_files
                    ]
                    writer.writerow(row)
            print(f"[info] kmer identity matrix written to {matrix_path}", file=sys.stderr)

            # --- greedy dereplication (always runs to completion) ---
            reps, cluster_assignments = greedy_cluster(genome_files, sim_matrix, args.threshold)
            print(
                f"[info] {len(reps)}/{len(genome_files)} genomes passed dereplication.",
                file=sys.stderr,
            )

            # --- select top max_reps by ref similarity (or by order if no refs) ---
            reps = select_reps_by_ref_similarity(reps, ref_identities, args.max_reps)
            print(
                f"[info] {len(reps)} representative(s) selected.",
                file=sys.stderr,
            )

        # --- genus fallback when species reps are insufficient ---
        genus_rows = []
        if args.max_reps and len(reps) < args.max_reps:
            reps, genus_rows = genus_fallback(
                reps, species, ncbi_species, args, tmp_dir, sname, species_dir, strain_mode
            )

        if not reps:
            print(f"[error] No representatives found for {species!r}.", file=sys.stderr)
            return

        # --- cluster TSV ---
        rep_set = set(reps)
        rep_to_cluster_id = {r: i + 1 for i, r in enumerate(
            r for r in genome_files if r in cluster_assignments and cluster_assignments[r][0] == r
        )} if cluster_assignments else {}

        tsv_path = os.path.join(species_dir, "dereplication_clusters.tsv")
        tsv_fieldnames = ["cluster_id", "accession", "source", "representative",
                          "similarity_to_ref", "similarity_to_cluster_rep"]
        with open(tsv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=tsv_fieldnames, delimiter="\t")
            writer.writeheader()
            # Species and pre-derep genus supplement genomes
            for genome in (genome_files if genome_files else []):
                if genome not in cluster_assignments:
                    continue
                cluster_rep, sim_to_rep = cluster_assignments[genome]
                cluster_id = rep_to_cluster_id.get(cluster_rep, "")
                source = genus_supplement.get(genome, "species")
                writer.writerow({
                    "cluster_id": cluster_id,
                    "accession": accession_from_path(genome),
                    "source": source,
                    "representative": genome in rep_set,
                    "similarity_to_ref": f"{ref_identities[genome]:.4f}" if genome in ref_identities else "",
                    "similarity_to_cluster_rep": f"{sim_to_rep:.4f}",
                })
            # Genus-fallback genomes
            for row in genus_rows:
                writer.writerow({
                    "cluster_id": row["cluster_id"],
                    "accession": row["accession"],
                    "source": "genus",
                    "representative": row["representative"],
                    "similarity_to_ref": "",
                    "similarity_to_cluster_rep": f"{row['similarity_to_cluster_rep']:.4f}",
                })
        print(f"[info] dereplication clusters written to {tsv_path}", file=sys.stderr)

        # --- quality summary CSV (species-level genomes only) ---
        if quality_rows:
            csv_fieldnames = ["accession", "assembly_level", "contig_n50", "number_of_contigs"]
            if ref_identities:
                csv_fieldnames.append("ref_max_identity")
            csv_fieldnames.append("representative")
            csv_path = os.path.join(species_dir, "quality_summary.csv")
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=csv_fieldnames)
                writer.writeheader()
                for row in quality_rows:
                    out = {
                        "accession": row["accession"],
                        "assembly_level": row["assembly_level"],
                        "contig_n50": row["contig_n50"],
                        "number_of_contigs": row["number_of_contigs"],
                        "representative": row["path"] in rep_set,
                    }
                    if ref_identities:
                        out["ref_max_identity"] = row.get("ref_max_identity", "")
                    writer.writerow(out)
            print(f"[info] quality summary written to {csv_path}", file=sys.stderr)

        # --- store reps (post-derep genus reps already stored in genus_fallback) ---
        for rep in reps:
            acc = accession_from_path(rep)
            src = genus_supplement.get(rep, "species")
            infix = "_family_" if src == "family_supplement" else "_genus_" if src == "genus_supplement" else "_"
            out_path = os.path.join(species_dir, f"{sname}{infix}{acc}.fna.gz")
            if not os.path.exists(out_path):
                compress_to_gz(rep, out_path)
                print(f"[store] {out_path}", file=sys.stderr)

        # --- done marker ---
        with open(done_marker, "w") as fh:
            fh.write(f"species={species}\n")
            fh.write(f"n_downloaded={len(genome_files)}\n")
            fh.write(f"n_representatives={len(reps)}\n")
            fh.write(f"threshold={args.threshold}\n")
            fh.write(f"assembly_levels={args.assembly_levels}\n")
            fh.write(f"max_download={args.max_download}\n")
            fh.write(f"max_reps={args.max_reps}\n")
            if ref_identities and reps:
                species_reps = [r for r in reps if r in ref_identities]
                if species_reps:
                    mean_ref_sim = sum(ref_identities.get(r, 0.0) for r in species_reps) / len(species_reps)
                    fh.write(f"mean_ref_identity={mean_ref_sim:.4f}\n")

        print(
            f"[done] {len(reps)} representative genome(s) stored in {species_dir}",
            file=sys.stderr,
        )

    finally:
        if args.keep_tmp or (getattr(args, 'download_only', False) and not args.genome_dir):
            print(f"[info] kept tmp dir: {tmp_dir}", file=sys.stderr)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def cmd_similarity_matrix(args):
    """Subcommand: compute all-vs-all similarity matrix for a list of local genome files."""
    genome_files = [line.strip() for line in open(args.genome_list) if line.strip()]
    missing = [f for f in genome_files if not os.path.isfile(f)]
    if missing:
        print(f"[error] {len(missing)} file(s) not found:", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)

    strain_mode = not args.no_strain_mode
    sim = compute_similarity_matrix(genome_files, args.genome_compare, strain_mode, args.threads)

    labels = [os.path.basename(f) for f in genome_files]
    with open(args.output, 'w') as fh:
        fh.write('\t' + '\t'.join(labels) + '\n')
        for f, label in zip(genome_files, labels):
            row = [f"{sim[f].get(g, 0.0):.4f}" for g in genome_files]
            fh.write(label + '\t' + '\t'.join(row) + '\n')

    print(f"[info] matrix written to {args.output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "build_scrub_db: download/dereplicate NCBI genomes, or compute a local similarity matrix."
        )
    )
    subparsers = parser.add_subparsers(dest='subcommand')

    # --- similarity subcommand ---
    sim_parser = subparsers.add_parser(
        'similarity',
        help='Compute all-vs-all kmer similarity matrix for a list of local genome files.'
    )
    sim_parser.add_argument(
        '--genome_list', '-i', required=True,
        help='Text file with one genome file path per line.'
    )
    sim_parser.add_argument(
        '--output', '-o', required=True,
        help='Output TSV file for the similarity matrix.'
    )
    sim_parser.add_argument(
        '--genome_compare', '-g', default='genome_compare',
        help='Path to the genome_compare binary (default: genome_compare in PATH).'
    )
    sim_parser.add_argument(
        '--threads', '-p', type=int, default=4,
        help='Parallel genome_compare processes (default: 4).'
    )
    sim_parser.add_argument(
        '--no_strain_mode', action='store_true',
        help='Disable strain mode (-S) in genome_compare.'
    )

    # --- build subcommand (existing behaviour) ---
    build_parser = subparsers.add_parser(
        'build',
        help='Download and dereplicate NCBI GenBank genomes for sourmash s__ lineage entries.'
    )
    build_parser.add_argument(
        "--lineage", "-l", default=None,
        help="Sourmash lineage string with an s__ entry "
             "(e.g. 's__Faecalibacterium prausnitzii'). "
             "Not required when --sourmash_summary is given.",
    )
    build_parser.add_argument(
        "--target_dir", "-o", required=True,
        help="Root output directory. Genomes are stored in <target_dir>/<species>/.",
    )
    parser.add_argument(
        "--lineage", "-l", default=None,
        help="Sourmash lineage string with an s__ entry "
             "(e.g. 's__Faecalibacterium prausnitzii'). "
             "Not required when --sourmash_summary is given.",
    )
    build_parser.add_argument(
        "--genome_compare", "-g", default="genome_compare",
        help="Path to the genome_compare binary (default: genome_compare in PATH).",
    )
    build_parser.add_argument(
        "--threshold", "-t", type=float, default=0.95,
        help="Kmer containment threshold for dereplication (default: 0.95).",
    )
    build_parser.add_argument(
        "--assembly_levels", "-a", default="complete,scaffold,contig",
        help="Comma-separated NCBI assembly levels to consider (default: complete,scaffold,contig).",
    )
    build_parser.add_argument(
        "--max_download", "-n", type=int, default=0,
        help="Maximum assemblies to download per species, ranked by quality. 0 = no limit.",
    )
    build_parser.add_argument(
        "--max_reps", "-r", type=int, default=0,
        help="Stop dereplication once this many representatives are collected. 0 = no limit.",
    )
    build_parser.add_argument(
        "--threads", "-p", type=int, default=4,
        help="Parallel genome_compare processes for the similarity matrix (default: 4).",
    )
    build_parser.add_argument(
        "--sourmash_summary", "-s", default=None,
        help="Sourmash classification summary CSV (columns: lineage, query_filename, …). "
             "All unique lineages are processed; matching genomes are used as references "
             "for --min_ref_identity filtering.",
    )
    build_parser.add_argument(
        "--min_ref_identity", type=float, default=0.0,
        help="Minimum kmer identity to any reference genome from --sourmash_summary "
             "required to keep a downloaded assembly (default: 0.0 = no filter).",
    )
    build_parser.add_argument(
        "--no_strain_mode", action="store_true",
        help="Disable strain mode (-S) in genome_compare.",
    )
    build_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Rebuild database even if it already exists.",
    )
    build_parser.add_argument(
        "--keep_tmp", action="store_true",
        help="Keep temporary download directory (useful for debugging).",
    )
    build_parser.add_argument(
        "--tmp_dir", default=None,
        help="Parent directory for the temporary download folder (default: system temp dir).",
    )
    build_parser.add_argument(
        "--download_only", action="store_true",
        help="Download and extract genomes then exit without running dereplication. "
             "Genome paths are written to <target_dir>/<species>/genome_manifest.txt and "
             "the tmp dir is preserved. Pass --tmp_dir to control its location so the cluster "
             "job can find it.",
    )
    build_parser.add_argument(
        "--genome_dir", default=None,
        help="Path to a directory of pre-downloaded genome FASTA files (.fna/.fasta/.fa). "
             "Skips the download step entirely and runs dereplication on the files found there. "
             "Intended for cluster jobs that receive genomes downloaded via --download_only.",
    )
    args = parser.parse_args()

    if args.subcommand == 'similarity':
        cmd_similarity_matrix(args)
        return

    if args.subcommand != 'build':
        parser.print_help()
        sys.exit(1)

    if not args.lineage and not args.sourmash_summary:
        build_parser.error("provide --lineage or --sourmash_summary (or both)")

    if args.sourmash_summary and not args.lineage:
        lineages = lineages_from_sourmash(args.sourmash_summary)
        print(
            f"[info] {len(lineages)} unique lineage(s) found in {args.sourmash_summary}",
            file=sys.stderr,
        )
    else:
        lineages = [args.lineage]

    for lineage in lineages:
        process_lineage(lineage, args)


if __name__ == "__main__":
    main()
