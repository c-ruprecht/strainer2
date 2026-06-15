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
from scipy.sparse.csgraph import connected_components


def path_to_name(p: str) -> str:
    """Normalize a genome path OR a CheckM2 'Name' to a bare identifier.

    Strips the directory, then a trailing `.gz`, then a single genome extension.
    Used on BOTH sides of the CheckM2 join so e.g.
        /.../MGYG000064733.fasta.gz   (glist / skani path)
        MGYG000064733.fasta           (CheckM2 Name, .gz already stripped)
    both reduce to `MGYG000064733`.
    """
    name = Path(str(p)).name
    if name.endswith(".gz"):
        name = name[:-3]
    for ext in (".fna", ".fasta", ".fa"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


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
    df = pd.read_csv(checkm2_tsv, sep="\t", low_memory=False)
    df = df.rename(columns={
        "Name": "genome_name",
        "Completeness": "completeness",
        "Contamination": "contamination",
    })
    # Normalize the CheckM2 Name with the SAME function used on genome paths,
    # so the merge key is comparable on both sides.
    df["key"] = df["genome_name"].apply(path_to_name)

    # Collapse duplicate keys (e.g. both X.fna and X.fasta were scored), keeping the
    # most-complete / least-contaminated row so a stale duplicate can't win selection.
    n_before = len(df)
    df = (
        df.sort_values(["completeness", "contamination"], ascending=[False, True])
          .drop_duplicates("key", keep="first")
    )
    if len(df) < n_before:
        print(f"[checkm2] Collapsed {n_before - len(df)} duplicate-key rows",
              file=sys.stderr)

    return df[["key", "genome_name", "completeness", "contamination"]]


def cluster_genomes(
    df: pd.DataFrame,
    ani_threshold: float = 99.0,
    af_threshold: float = 0.0,
    genome_list_path: str = None,
) -> pd.DataFrame:
    """
    Single-linkage (connected-components) clustering on skani ANI (0-100 scale).

    The node universe is the genome_list for this primary cluster (the genomes fed
    to skani), NOT the skani rows. skani only emits rows for pairs it finds similar,
    so genomes with no qualifying hit — and entire clusters where every pair fails
    the ANI/AF cut — would otherwise disappear from the output (or crash on a 0x0
    matrix). Seeding from the list makes those genomes fall out as their own
    singleton clusters, so every input genome is either a representative or assigned
    to one.
    """
    # --- node universe: every genome in this primary cluster ---
    all_genomes = []
    if genome_list_path is not None:
        with open(genome_list_path) as fh:
            all_genomes = [ln.strip() for ln in fh if ln.strip()]

    # Fall back to whatever skani mentions only if no list was provided.
    if not all_genomes:
        if df.empty:
            print("[cluster] Empty skani input and no genome_list — returning empty.",
                  file=sys.stderr)
            return pd.DataFrame(columns=["genome_path", "cluster"])
        all_genomes = list(pd.unique(df[["query_name", "match_name"]].values.ravel()))

    idx = {g: i for i, g in enumerate(all_genomes)}
    n = len(all_genomes)

    # --- edges: qualifying skani pairs (self-hits dropped, AF + ANI filtered) ---
    qi = np.empty(0, dtype=int)
    mi = np.empty(0, dtype=int)
    if not df.empty:
        e = df[df["query_name"] != df["match_name"]]
        if af_threshold > 0:
            e = e[(e["af_ref"] >= af_threshold) & (e["af_query"] >= af_threshold)]
        e = e[e["ani"] >= ani_threshold]

        qi = e["query_name"].map(idx).to_numpy()
        mi = e["match_name"].map(idx).to_numpy()
        keep = ~(pd.isna(qi) | pd.isna(mi))
        n_unmapped = int((~keep).sum())
        if n_unmapped:
            print(f"[cluster] {n_unmapped} skani edges referenced paths not in the "
                  f"genome_list — dropped (path mismatch between list and skani output?)",
                  file=sys.stderr)
        qi = qi[keep].astype(int)
        mi = mi[keep].astype(int)

    # One direction is enough; connected_components(directed=False) symmetrises.
    graph = csr_matrix((np.ones(len(qi), dtype=np.uint8), (qi, mi)), shape=(n, n))
    n_clusters, labels = connected_components(graph, directed=False, connection="weak")

    cluster_df = pd.DataFrame({"genome_path": all_genomes, "cluster": labels})

    sizes = cluster_df["cluster"].value_counts()
    print(f"[cluster] Genomes   : {n}", file=sys.stderr)
    print(f"[cluster] Clusters  : {n_clusters}", file=sys.stderr)
    print(f"[cluster] Largest   : {sizes.iloc[0]} genomes", file=sys.stderr)
    print(f"[cluster] Singletons: {int((sizes == 1).sum())}", file=sys.stderr)

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
    cluster_df["key"] = cluster_df["genome_path"].apply(path_to_name)

    merged = cluster_df.merge(checkm2_df, on="key", how="left")

    missing = merged["completeness"].isna()
    n_missing = int(missing.sum())
    if n_missing:
        examples = merged.loc[missing, "genome_path"].head(5).tolist()
        print(f"[warn] {n_missing}/{len(merged)} genomes had no CheckM2 match by key.",
              file=sys.stderr)
        for e in examples:
            print(f"          {e}  ->  key={path_to_name(e)}", file=sys.stderr)
        if n_missing == len(merged):
            print("        ALL genomes unmatched -> this is a naming mismatch, not "
                  "missing data. Compare the keys above against CheckM2 'Name'.",
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