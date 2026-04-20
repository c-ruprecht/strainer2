import pandas as pd
import argparse
import csv
import gzip
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def read_sourmash_gather(paths):
    dfs = []
    for mash in paths:
        df = pd.read_csv(mash)
        dfs.append(df)
    
    df_mash = pd.concat(dfs)
    li_accession = df_mash['name'].str.split(' ').str[0].unique()
    return li_accession

BATCH_SIZE = 500
COMPLETED_FILE = "completed_accessions.txt"

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

def download_batch(batch, batch_idx, out_dir):
    """Download and extract one batch of accessions, flattening to out_dir."""
    zip_path = os.path.join(out_dir, f"batch_{batch_idx:04d}.zip")
    acc_file = os.path.join(out_dir, f"batch_{batch_idx:04d}_accessions.txt")
    with open(acc_file, "w") as fh:
        fh.write("\n".join(batch) + "\n")

    cmd = ["datasets", "download", "genome", "accession",
        "--inputfile", acc_file,
        "--assembly-source", "genbank",
        "--include", "genome",
        "--filename", zip_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"datasets download failed for batch {batch_idx} (exit {result.returncode})")
    if not os.path.exists(zip_path):
        raise RuntimeError(f"Expected zip not found: {zip_path}")

    # Extract to a temp location, then flatten .fna files into out_dir
    tmp_extract = os.path.join(out_dir, f"_tmp_batch_{batch_idx:04d}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_extract)

    # Move all .fna files to out_dir flat
    for fna in Path(tmp_extract).rglob("*.fna"):
        dest = os.path.join(out_dir, fna.name)
        if not os.path.exists(dest):
            shutil.move(str(fna), dest)
        else:
            print(f"[warn] skipping duplicate: {fna.name}", file=sys.stderr)

    shutil.rmtree(tmp_extract)
    os.remove(zip_path)
    os.remove(acc_file)

def download_by_accession(accessions, out_dir):
    completed = load_completed(out_dir)
    remaining = [a for a in accessions if a not in completed]
    if not remaining:
        print("[download] All accessions already downloaded, nothing to do.", file=sys.stderr)
        return
    if completed:
        print(f"[download] Resuming: {len(completed)} already done, {len(remaining)} remaining.", file=sys.stderr)

    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    for idx, batch in enumerate(batches):
        print(f"[download] Batch {idx + 1}/{len(batches)}: {len(batch)} accessions …", file=sys.stderr)
        download_batch(batch, idx, out_dir)
        mark_completed(out_dir, batch)
        print(f"[download] Batch {idx + 1}/{len(batches)} done.", file=sys.stderr)

#def dereplicate(dataset_zip, accession):
#    genome_files = [line.strip() for line in open(accession) if line.strip()]
#    missing = [f for f in genome_files if not os.path.isfile(f)]
#        if missing:
#            print(f"[error] {len(missing)} file(s) not found:", file=sys.stderr)
#            for f in missing:
#                print(f"  {f}", file=sys.stderr)
#            sys.exit(1)

#create input file for manysketch
#
## header: name,genome_filename,protein_filename
#echo "name,genome_filename,protein_filename" > manysketch.csv

#for f in /metrica/scratch/strainer_dev/scrub_db/ncbi_dataset/data/*/*.fna; do
#    name=$(basename "$f")
#    echo "${name},${f}," >> manysketch.csv
#done
# primary sourmash sketch filter 
# sourmash scripts manysketch manysketch.csv --param-str dna,k=31,scaled=1000 -o /metrica/scratch/strainer_dev/ske
#tches.zip     -c 28

#pairwaise comparison
#sourmash scripts pairwise \
#    /metrica/scratch/strainer_dev/sketches.zip \
#    -o /metrica/scratch/strainer_dev/pairwise.csv \
#    -c 30 \
#    --write-all
#clusterring

# run kmer similarity matrix
# lower threshhold maybe 0.8?
# remove all self kmer comparisons with 8
def sourmash_sketch(genome_dir, sketches_path, ksize=31, scaled=1000, threads=8):
    """
    Create a manysketch CSV from all .fna files in genome_dir,
    then run `sourmash scripts manysketch`.
    """
    fna_files = sorted(Path(genome_dir).glob("*.fna"))
    if not fna_files:
        raise RuntimeError(f"No .fna files found in {genome_dir}")
 
    csv_path = os.path.join(genome_dir, "manysketch.csv")
    with open(csv_path, "w") as fh:
        fh.write("name,genome_filename,protein_filename\n")
        for fna in fna_files:
            fh.write(f"{fna.name},{fna},\n")
 
    print(f"[sketch] Sketching {len(fna_files)} genomes "
          f"(k={ksize}, scaled={scaled}) → {sketches_path}",
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
 
 
def sourmash_pairwise(sketches_path, pairwise_path, threads=8, write_all=True):
    """
    Run `sourmash scripts pairwise` on a sketches zip file.
    Returns a DataFrame with columns: query_name, match_name, jaccard, ...
    """
    print(f"[pairwise] Computing pairwise similarities → {pairwise_path}",
          file=sys.stderr)
 
    cmd = [
        "sourmash", "scripts", "pairwise",
        sketches_path,
        "-o", pairwise_path,
        "-c", str(threads),
    ]
    if write_all:
        cmd.append("--write-all")
 
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

def _compare_pairs(args):
    """
    Hashes a_file once, compares only against its specific partner files.
    """
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
    """
    Run genome_compare on specific pairs from a DataFrame.
    df must have columns: query_name, match_name
 
    Groups by query_name so each genome is hashed only once.
    Returns a DataFrame with columns: query_name, match_name, kmer_coverage
    """
    # group by query so each query genome is hashed only once
    grouped = df.groupby("query_name")["match_name"].apply(list).to_dict()
 
    tasks = [
        (genome_compare_bin,
         os.path.join(genome_dir, query),
         [os.path.join(genome_dir, m) for m in matches],
         strain_mode)
        for query, matches in grouped.items()
    ]
 
    print(f"[kmer_compare] Running genome_compare on {len(tasks)} query genomes "
          f"({len(df)} total pairs, {n_workers} workers) …", file=sys.stderr)
 
    results = {}
    with multiprocessing.Pool(n_workers) as pool:
        for a_file, fracs in pool.imap_unordered(_compare_pairs, tasks):
            results[a_file] = fracs
 
    # flatten back to rows
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
    """
    Filter sourmash pairwise results by min_jaccard, then run genome_compare
    on the remaining pairs for exact k-mer containment.
 
    Returns a merged DataFrame with both jaccard and kmer_coverage.
    """
    # sourmash pairwise CSV has columns like:
    #   query_name, query_md5, match_name, match_md5, containment, ...
    # but the exact column for Jaccard may vary; check what's available
    if "jaccard" in df_pairwise.columns:
        jac_col = "jaccard"
    elif "similarity" in df_pairwise.columns:
        jac_col = "similarity"
    else:
        # branchwater pairwise uses 'jaccard'
        raise ValueError(
            f"Cannot find jaccard/similarity column in pairwise CSV. "
            f"Columns: {list(df_pairwise.columns)}"
        )
 
    # Remove self-comparisons
    df_filtered = df_pairwise[df_pairwise["query_name"] != df_pairwise["match_name"]].copy()
 
    # Filter by minimum Jaccard
    df_filtered = df_filtered[df_filtered[jac_col] >= min_jaccard].copy()
 
    print(f"[kmer_compare] {len(df_filtered)} pairs above Jaccard >= {min_jaccard} "
          f"(from {len(df_pairwise)} total)", file=sys.stderr)
 
    if df_filtered.empty:
        print("[kmer_compare] No pairs to compare.", file=sys.stderr)
        return df_filtered
 
    df_kmer = compare_pairs_from_df(
        df_filtered[["query_name", "match_name"]],
        genome_compare_bin,
        genome_dir,
        strain_mode=strain_mode,
        n_workers=n_workers,
    )
 
    # Merge back the jaccard values
    df_merged = df_filtered.merge(
        df_kmer, on=["query_name", "match_name"], how="left"
    )
 
    return df_merged

def greedy_choice(df_kmer_compare, percentage):
    df = df_kmer_compare
    display(df)
    threshhold = percentage
    li_choice = []
    dropset = set()
    for entry in df['query_name'].unique():
        if entry in dropset:
            continue
        else:
            li_choice.append(entry)
            li_drop = df.loc[(df['query_name']==entry) &
                (df['kmer_coverage'] > threshhold)]['match_name'].to_list()
            dropset.update(li_drop)  

    # retroactive: remove representatives that got dropped by a later representative
    li_choice = [r for r in li_choice if r not in dropset]

    df_drop = pd.DataFrame([{"genome_representative": rep, "genome_drop_list": df.loc[
                        (df['query_name'] == rep) & (df['kmer_coverage'] > threshhold),'match_name'].tolist()}
                        for rep in li_choice])
    # explode the lists, then check overlap
    all_dropped = df_drop['genome_drop_list'].explode()
    overlap = all_dropped[all_dropped.isin(df_drop['genome_representative'])].unique()
    print(f"{len(overlap)} dropped genomes are also representatives:")
    print(overlap)
    return df_drop

def main():
    parser = argparse.ArgumentParser(description='Map scrubbed kmers onto a genome.')
    
    parser.add_argument('--drug', help='A list of files pointing to sourmash of target strains to track')
    parser.add_argument('--target_samples', help='A list of files pointing to sourmash of target samples to track in')
    parser.add_argument('--kmer_ident' , default = 0.96, help = 'threshhold for dereplication')
    parser.add_argument('--scrub_db_path', help = 'path where to build the database or where to expand if it is in place')
    parser.add_argument('--threads' , type=int,default = 8, help = 'how many threads to use,e specially for kmer dereplication')
    parser.add_argument('--genome_compare', help = 'path to strainer genome compare of run from different path')
    parser.add_argument('--min_jaccard', default = 0.7, help = 'min jaccard distance to use for full kmer comparison')
    args = parser.parse_args()
    out_dir = args.scrub_db_path
    os.makedirs(args.scrub_db_path, exist_ok=True)
    #create a full file list
    with open(args.drug) as f:
        paths = [line.strip() for line in f if line.strip()]    
    
    
    li_accessions = read_sourmash_gather(paths)
    print(f'Found {len(li_accessions)} unique accession numbers to download')
    print('Downloading genomes to build scrub database')
    # download 
    assembly_path = os.path.join(args.scrub_db_path, 'assemblies')
    os.makedirs(assembly_path, exist_ok=True)
    download_by_accession(li_accessions, assembly_path)
    # create sourmash sketches 
    sketches_path = os.path.join(out_dir, "sketches.zip")
    sourmash_sketch(genome_dir=assembly_path,sketches_path=sketches_path, ksize=31,scaled=1000,threads=args.threads)
 
    # --- 4. Sourmash pairwise ---
    pairwise_path = os.path.join(out_dir, "pairwise.csv")
    df_pairwise = sourmash_pairwise(sketches_path=sketches_path,pairwise_path=pairwise_path,threads=args.threads)
 
    # --- 5. K-mer compare on close pairs ---
    kmer_results_path = os.path.join(out_dir, "kmer_compare.csv")
    df_kmer = kmer_compare(df_pairwise=df_pairwise,
                           genome_compare_bin=args.genome_compare,
                           genome_dir=assembly_path,
                           min_jaccard=args.min_jaccard, 
                           strain_mode=True,#not args.no_strain_mode,
                           n_workers=args.threads,)
 
    if not df_kmer.empty:
        df_kmer.to_csv(kmer_results_path, index=False)
        print(f"[build] K-mer comparison results  {kmer_results_path}", file=sys.stderr)
 
        # Show pairs above dereplication threshold
        if "kmer_coverage" in df_kmer.columns:
            dups = df_kmer[df_kmer["kmer_coverage"] >= args.kmer_ident]
            print(f"[build] {len(dups)} pairs above kmer_ident >= {args.kmer_ident} "
                  f"(candidates for dereplication)", file=sys.stderr)
        
        df_drop = greedy_choice(df_kmer, percentage = args.kmer_ident)
        df_drop.to_csv(os.path.join(out_dir,'representative_genomes.tsv', sep = '\t'))
        ### Build a csv that 
        print(' TO DO: REMOVE DEREPLICATED FILES, KEEP TRACK FOR LATER ADDITION OF MORE GENOMES')
    
    else:
        print("[build] No close pairs found.", file=sys.stderr)
    

    print("[build] Done.", file=sys.stderr)
    
if __name__ == '__main__':
    main()