#!/usr/bin/env python3
"""
Cluster genomes from skani dist output and select representative per cluster
based on CheckM2 completeness and contamination. Copy or symlink to output dir.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN


def load_skani(skani_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(skani_tsv, sep="\t")
    # skani may produce empty files for singleton clusters
    if df.empty:
        return df
    df = df.rename(columns={
        "Ref_file": "query_name",
        "Query_file": "match_name",
        "ANI": "ani",
        "Align_fraction_ref": "af_ref",
        "Align_fraction_query": "af_query",
    })
    return df


def load_checkm2(checkm2_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(checkm2_tsv, sep="\t")
    df = df.rename(columns={
        "Name": "genome_name",
        "Completeness": "completeness",
        "Contamination": "contamination",
    })
    return df[["genome_name", "completeness", "contamination"]]


def path_to_name(p: str) -> str:
    name = Path(p).name
    for ext in [".fna.gz", ".fa.gz", ".fasta.gz", ".fna", ".fa", ".fasta"]:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(p).stem


def cluster_genomes(
    df: pd.DataFrame,
    ani_threshold: float = 99.0,
    af_threshold: float = 0.0,
    genome_list_path: str = None,
) -> pd.DataFrame:
    """
    DBSCAN clustering on skani ANI (0-100 scale).

    If skani TSV is empty (singleton cluster — skani produces no rows when
    only one genome is present), fall back to reading paths from genome_list_path
    so the genome still gets assigned to a cluster.
    """
    if df.empty:
        if genome_list_path is None:
            print("[cluster] Empty skani input and no genome_list — returning empty.",
                  file=sys.stderr)
            return pd.DataFrame(columns=["genome_path", "cluster"])
        genomes = []
        with open(genome_list_path) as fh:
            for line in fh:
                p = line.strip()
                if p:
                    genomes.append(p)
        print(f"[cluster] Singleton cluster: {len(genomes)} genome(s) from list",
              file=sys.stderr)
        return pd.DataFrame({"genome_path": genomes, "cluster": [0] * len(genomes)})

    # Drop self-hits
    df = df[df["query_name"] != df["match_name"]].copy()

    if af_threshold > 0:
        df = df[
            (df["af_ref"] >= af_threshold) &
            (df["af_query"] >= af_threshold)
        ].copy()

    genomes = pd.unique(df[["query_name", "match_name"]].values.ravel())
    n = len(genomes)
    genome_idx = {g: i for i, g in enumerate(genomes)}

    df_filt = df[df["ani"] >= ani_threshold].copy()
    df_filt["dist"] = (100.0 - df_filt["ani"]) / 100.0

    i_idx = df_filt["query_name"].map(genome_idx).values
    j_idx = df_filt["match_name"].map(genome_idx).values
    dists = df_filt["dist"].values

    rows = np.concatenate([i_idx, j_idx])
    cols = np.concatenate([j_idx, i_idx])
    data = np.concatenate([dists, dists])

    sparse_dist = csr_matrix((data, (rows, cols)), shape=(n, n))

    eps = (100.0 - ani_threshold) / 100.0
    db = DBSCAN(eps=eps, min_samples=1, metric="precomputed")
    labels = db.fit_predict(sparse_dist)

    cluster_df = pd.DataFrame({"genome_path": genomes, "cluster": labels})

    sizes = cluster_df["cluster"].value_counts()
    print(f"[cluster] Genomes   : {n}", file=sys.stderr)
    print(f"[cluster] Clusters  : {cluster_df['cluster'].nunique()}", file=sys.stderr)
    print(f"[cluster] Largest   : {sizes.iloc[0]} genomes", file=sys.stderr)
    print(f"[cluster] Singletons: {(sizes == 1).sum()}", file=sys.stderr)

    return cluster_df


def select_representatives(
    cluster_df: pd.DataFrame,
    checkm2_df: pd.DataFrame,
    completeness_weight: float = 1.0,
    contamination_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score = completeness * completeness_weight - contamination * contamination_weight
    Pick highest score per cluster; ties broken alphabetically by path.
    """
    cluster_df = cluster_df.copy()
    cluster_df["genome_name"] = cluster_df["genome_path"].apply(path_to_name)

    merged = cluster_df.merge(checkm2_df, on="genome_name", how="left")

    missing_comp = merged["completeness"].isna().sum()
    missing_cont = merged["contamination"].isna().sum()
    if missing_comp > 0:
        print(f"[warn] {missing_comp} genomes missing CheckM2 completeness — set to 0",
              file=sys.stderr)
    if missing_cont > 0:
        print(f"[warn] {missing_cont} genomes missing CheckM2 contamination — set to 100",
              file=sys.stderr)

    merged["completeness"] = merged["completeness"].fillna(0.0)
    merged["contamination"] = merged["contamination"].fillna(100.0)

    merged["score"] = (
        merged["completeness"] * completeness_weight
        - merged["contamination"] * contamination_weight
    )

    merged = merged.sort_values(
        ["cluster", "score", "genome_path"],
        ascending=[True, False, True],
    )

    reps = merged.groupby("cluster", sort=False).first().reset_index()
    reps = reps.rename(columns={"genome_path": "representative"})

    return merged, reps[["cluster", "representative", "completeness", "contamination", "score"]]


def export_representatives(
    reps_df: pd.DataFrame,
    dest_dir: str,
    mode: str = "symlink",
) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for _, row in reps_df.iterrows():
        src = Path(row["representative"])
        dst = Path(dest_dir) / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if mode == "symlink":
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        print(
            f"[export] cluster {row['cluster']:>6} | {mode} {src.name} "
            f"(completeness={row['completeness']:.1f}%, "
            f"contamination={row['contamination']:.1f}%, "
            f"score={row['score']:.2f})",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Cluster genomes via skani ANI and pick best by CheckM2 quality."
    )
    parser.add_argument("--skani", required=True,
                        help="skani dist output TSV")
    parser.add_argument("--checkm2", required=True,
                        help="CheckM2 quality_report.tsv")
    parser.add_argument("--genome_list", required=False, default=None,
                        help="Original genome list (paths, one per line) — used as "
                             "fallback for singleton clusters where skani produces no rows")
    parser.add_argument("--ani_threshold", type=float, default=99.0,
                        help="ANI threshold for clustering (0-100, default 99.0)")
    parser.add_argument("--af_threshold", type=float, default=0.0,
                        help="Min align fraction for both ref and query (0-100, default 0)")
    parser.add_argument("--completeness_weight", type=float, default=1.0,
                        help="Weight for completeness in scoring (default 1.0)")
    parser.add_argument("--contamination_weight", type=float, default=0.5,
                        help="Weight for contamination penalty in scoring (default 0.5)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write cluster assignments TSV")
    parser.add_argument("--reps_out", required=True,
                        help="Full path for representatives TSV output")
    parser.add_argument("--dest_dir", required=True,
                        help="Directory to place representative genomes")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                        help="symlink (default) or copy representatives to dest_dir")
    args = parser.parse_args()

    print(f"[main] Loading skani output: {args.skani}", file=sys.stderr)
    skani_df = load_skani(args.skani)

    print(f"[main] Loading CheckM2 report: {args.checkm2}", file=sys.stderr)
    checkm2_df = load_checkm2(args.checkm2)

    print(
        f"[main] Clustering at ANI >= {args.ani_threshold}%, AF >= {args.af_threshold}%",
        file=sys.stderr,
    )
    cluster_df = cluster_genomes(
        skani_df,
        ani_threshold=args.ani_threshold,
        af_threshold=args.af_threshold,
        genome_list_path=args.genome_list,
    )

    if cluster_df.empty:
        print("[main] No genomes to process — exiting.", file=sys.stderr)
        # Write empty outputs so Snakemake doesn't complain about missing files
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame().to_csv(os.path.join(args.output_dir, "clusters.tsv"), sep="\t", index=False)
        pd.DataFrame().to_csv(args.reps_out, sep="\t", index=False)
        return

    merged_df, reps_df = select_representatives(
        cluster_df,
        checkm2_df,
        completeness_weight=args.completeness_weight,
        contamination_weight=args.contamination_weight,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    clusters_out = os.path.join(args.output_dir, "clusters.tsv")
    merged_df.to_csv(clusters_out, sep="\t", index=False)
    reps_df.to_csv(args.reps_out, sep="\t", index=False)
    print(f"[main] Cluster assignments -> {clusters_out}", file=sys.stderr)
    print(f"[main] Representatives     -> {args.reps_out}", file=sys.stderr)

    export_representatives(reps_df, args.dest_dir, mode=args.mode)
    print(f"[main] Done. {len(reps_df)} representatives in {args.dest_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()