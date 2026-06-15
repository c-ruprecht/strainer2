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
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix

def sourmash_sketch(genome_list, sketches_path,outdir, ksize=31, scaled=1000, threads=8):
    """Build manysketch CSV from all .fna in genome_dir and run sourmash manysketch."""
    
    df = pd.read_csv(genome_list, header= None)
    df.columns = ['genome_filename']
    df['name'] = df['genome_filename'].str.rsplit('/').str[-1]
    df['protein_filename']=""
    print(df)
    print(df["name"].value_counts().head(20))
    print(f"\nTotal: {len(df)}, Unique: {df['name'].nunique()}, Duplicates: {len(df) - df['name'].nunique()}")

    if len(df) - df['name'].nunique() > 0:
        print('Duplicate name entries in genome list')
    
    csv_path = os.path.join(outdir, "manysketch.csv")
    df[['name', 'genome_filename', 'protein_filename']].to_csv(csv_path, index = None)

    print(f"[sketch] Sketching {len(df)} genomes (k={ksize}, scaled={scaled}) -> {sketches_path}",
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
    """Run sourmash pairwise on a sketches zip. Returns the resulting DataFrame."""
    print(f"[pairwise] Computing pairwise similarities -> {pairwise_path}", file=sys.stderr)

    cmd = [
        "sourmash", "scripts", "pairwise",
        sketches_path,
        "-o", pairwise_path,
        "--threshold", "0",
        "--ani",
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



def cluster_genomes_by_ani(
    pairwise_csv: str,
    ani_threshold: float = 0.90,
    ani_column: str = "max_containment",
) -> pd.DataFrame:
    """
    Cluster genomes from a sourmash pairwise CSV using DBSCAN on ANI distance.

    Parameters
    ----------
    pairwise_csv : str
        Path to sourmash pairwise output CSV.
    ani_threshold : float
        Minimum ANI to consider two genomes in the same cluster (default 0.90).
    ani_column : str
        Column to use as ANI proxy — 'ani', 'max_containment', etc.

    Returns
    -------
    pd.DataFrame with columns: genome, cluster
        cluster == -1 means singleton (no neighbors above threshold).
    """
    df = pairwise_csv

    # All unique genomes
    genomes = pd.unique(df[["query_name", "match_name"]].values.ravel())
    n = len(genomes)
    genome_idx = {g: i for i, g in enumerate(genomes)}

    # Keep only pairs above ANI threshold
    df_filt = df[df[ani_column] >= ani_threshold].copy()
    df_filt["dist"] = 1.0 - df_filt[ani_column]

    # Build sparse distance matrix
    i_idx = df_filt["query_name"].map(genome_idx).values
    j_idx = df_filt["match_name"].map(genome_idx).values
    dists = df_filt["dist"].values

    # Symmetrize
    rows = np.concatenate([i_idx, j_idx])
    cols = np.concatenate([j_idx, i_idx])
    data = np.concatenate([dists, dists])

    sparse_dist = csr_matrix((data, (rows, cols)), shape=(n, n))

    # DBSCAN — min_samples=1 so every genome gets a cluster (no noise points)
    db = DBSCAN(eps=1.0 - ani_threshold, min_samples=1, metric="precomputed")
    labels = db.fit_predict(sparse_dist)

    cluster_df = pd.DataFrame({"genome": genomes, "cluster": labels})

    n_clusters = cluster_df["cluster"].nunique()
    sizes = cluster_df["cluster"].value_counts()
    print(f"Genomes   : {n}")
    print(f"Clusters  : {n_clusters}")
    print(f"Largest   : {sizes.iloc[0]} genomes")
    print(f"Singletons: {(sizes == 1).sum()}")

    return cluster_df

def main():
    parser = argparse.ArgumentParser(description='Build a scrub k-mer database.')
    parser.add_argument('--genome_list',
                        help='A list of genomes to dereplicate, one path per line')
    parser.add_argument('--min_sourmash', type=float, default=0.8,
                        help='Min jaccard from sourmash pairwise to trigger exact kmer compare.')
    parser.add_argument('--kmer_ident', type=float, default=0.96,
                        help='k-mer coverage threshold for dereplication.')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for the scrub database.')
    parser.add_argument('--genome_compare', required=False,
                        help='Path to the strainer genome_compare binary.')
    parser.add_argument('--threads', required=False, default = 12,
                        help='Path to the strainer genome_compare binary.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    #check if glist only appear once
    # Check for duplicate files in your genome list
    

    sketches_path = os.path.join(args.output_dir, "sketches.zip")
    sourmash_sketch(genome_list=args.genome_list, 
                    outdir = args.output_dir,
                    sketches_path=sketches_path,
                    ksize=31, scaled=1000, threads=args.threads)

    pairwise_path = os.path.join(args.output_dir, "sourmash-pairwise.csv")
    df_pairwise = sourmash_pairwise(sketches_path=sketches_path,
                                    pairwise_path=pairwise_path,
                                    threads=args.threads)
    

    # create primary clusteron max_containment ani
    cluster_df = cluster_genomes_by_ani(
        df_pairwise,
        ani_threshold=args.min_sourmash,
        ani_column="max_containment_ani",
    )
    # Inspect cluster sizes
    print(cluster_df["cluster"].value_counts().head(10))
    # --- Write per-cluster genome lists ---
    genome_paths = pd.read_csv(args.genome_list, header=None, names=["path"])
    genome_paths["genome"] = genome_paths["path"].str.rsplit("/").str[-1]
    name_to_path = genome_paths.set_index("genome")["path"].to_dict()

    cluster_lists_dir = os.path.join(args.output_dir, "cluster_lists")
    os.makedirs(cluster_lists_dir, exist_ok=True)

    for cluster_id, grp in cluster_df.groupby("cluster"):
        paths = []
        for genome_name in grp["genome"]:
            path = name_to_path.get(genome_name)
            if path is None:
                print(f"[warn] No path found for genome '{genome_name}' in cluster {cluster_id}",
                      file=sys.stderr)
                continue
            paths.append(path)

        list_path = os.path.join(cluster_lists_dir, f"pr_cluster_{cluster_id}.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(paths) + "\n")

    print(f"[cluster_lists] Written to {cluster_lists_dir}", file=sys.stderr)
    
    #write as last step for snakemake
    cluster_df.sort_values(['cluster'], ascending = True).to_csv(os.path.join(args.output_dir, 'primary_clusters.csv'))


if __name__ == '__main__':
    main()
