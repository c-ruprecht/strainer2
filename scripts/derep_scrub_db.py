import argparse
import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def sourmash_sketch(genome_list, sketches_path, outdir, ksize=31, scaled=1000, threads=8):
    """Build a manysketch CSV from the genome list and run sourmash manysketch."""

    df = pd.read_csv(genome_list, header=None)
    df.columns = ['genome_filename']
    df['name'] = df['genome_filename'].str.rsplit('/').str[-1]
    df['protein_filename'] = ""
    print(df["name"].value_counts().head(20))
    print(f"\nTotal: {len(df)}, Unique: {df['name'].nunique()}, "
          f"Duplicates: {len(df) - df['name'].nunique()}")
    if len(df) - df['name'].nunique() > 0:
        print('Duplicate name entries in genome list')

    csv_path = os.path.join(outdir, "manysketch.csv")
    df[['name', 'genome_filename', 'protein_filename']].to_csv(csv_path, index=None)

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


def sourmash_pairwise(sketches_path, out_path, ani_floor=0.90,
                      ani_column="max_containment_ani", ksize=31, threads=8):
    """
    Run `sourmash scripts pairwise`, streaming the output through a FIFO so the
    full-width all-vs-all CSV never lands on disk. On the fly we:
      1. discard everything below `ani_floor` ANI   (awk, exact filter on the ANI col),
      2. keep only query_name, match_name, <ani_column>  (md5/jaccard/intersect dropped),
      3. zstd-compress the result to out_path (e.g. .../sourmash-pairwise.csv.zst).

    sourmash's own --threshold is on *containment*, not ANI, so we set it to
    ani_floor**ksize -- the containment that corresponds to ani_floor -- so pairwise
    itself drops the bulk of distant pairs before they ever reach the pipe, while
    still passing everything the awk ANI floor wants to keep.

    NOTE: this assumes the pairwise header uses query_name / match_name. Some plugin
    versions emit query / match -- check `head -1` on a small run and adjust if needed.
    """
    containment_threshold = ani_floor ** ksize

    fifo_dir = tempfile.mkdtemp(prefix="pw_fifo_")
    fifo = os.path.join(fifo_dir, "pw.csv")
    os.mkfifo(fifo)

    # Select the three columns by *header name* (robust to column reordering) and
    # apply the exact ANI floor. No single quotes inside the program, so it is safe
    # to wrap in single quotes for the shell.
    awk_prog = (
        'NR==1{{'
        'for(i=1;i<=NF;i++){{'
        'if($i=="query_name")q=i;'
        'if($i=="match_name")m=i;'
        'if($i=="{col}")a=i'
        '}}'
        'print "query_name,match_name,{col}";next'
        '}}'
        '($a+0)>={floor:.6f}{{print $q","$m","$a}}'
    ).format(col=ani_column, floor=ani_floor)

    consumer_cmd = (
        f"awk -F, '{awk_prog}' < {shlex.quote(fifo)} "
        f"| zstd -T0 -3 -f -o {shlex.quote(out_path)}"
    )
    print(f"[pairwise] ani_floor={ani_floor} -> containment --threshold "
          f"{containment_threshold:.3e}", file=sys.stderr)
    consumer = subprocess.Popen(consumer_cmd, shell=True, executable="/bin/bash")

    cmd = [
        "sourmash", "scripts", "pairwise",
        sketches_path,
        "-o", fifo,
        "--threshold", f"{containment_threshold:.8f}",
        "--ani",
        "-c", str(threads),
    ]
    producer = subprocess.run(cmd)

    # If pairwise died before opening the FIFO, the awk reader would block forever;
    # open the write end non-blocking and close it to hand the reader an EOF.
    if producer.returncode != 0:
        try:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass

    consumer.wait()
    shutil.rmtree(fifo_dir, ignore_errors=True)

    if producer.returncode != 0:
        raise RuntimeError(f"sourmash pairwise failed (exit {producer.returncode})")
    if consumer.returncode != 0:
        raise RuntimeError(f"projection/zstd consumer failed (exit {consumer.returncode})")

    print(f"[pairwise] Done: {out_path}", file=sys.stderr)
    return out_path


def _read_pairwise_csv(path):
    """Read the projected pairwise CSV (optionally .zst/.gz) into a Polars DataFrame.

    Post-projection the file is small (3 columns, ANI-filtered), so an eager read is
    fine. If you ever keep a very loose floor, switch to a streaming zstd -dc | scan.
    """
    path = str(path)
    if path.endswith(".zst"):
        proc = subprocess.run(["zstd", "-dcq", path], stdout=subprocess.PIPE, check=True)
        return pl.read_csv(io.BytesIO(proc.stdout))
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rb") as fh:
            return pl.read_csv(io.BytesIO(fh.read()))
    return pl.read_csv(path)


def cluster_genomes_by_ani(pairwise_path, all_genomes,
                           ani_threshold=0.95,
                           ani_column="max_containment_ani"):
    """
    Single-linkage (connected-components) clustering at `ani_threshold`.

    pairwise_path : projected pairwise file (query_name, match_name, <ani_column>),
                    optionally zstd/gzip compressed. Already ANI-prefiltered at the
                    source; the exact cut is re-applied here so the same artifact can
                    be reused at a stricter threshold without re-running sourmash.
    all_genomes   : full list of genome basenames. Seeding the node set from this
                    (not from the surviving edges) guarantees genomes with no
                    above-threshold neighbour still get their own singleton cluster.

    Returns a DataFrame[genome, cluster]; every genome gets a real cluster id
    (no -1 noise label), singletons included.
    """
    idx = {g: i for i, g in enumerate(all_genomes)}
    n = len(all_genomes)

    df = _read_pairwise_csv(pairwise_path)
    edges = df.filter(
        (pl.col(ani_column) >= ani_threshold)
        & (pl.col("query_name") != pl.col("match_name"))
    )

    qi = edges["query_name"].replace_strict(idx, default=-1).to_numpy()
    mi = edges["match_name"].replace_strict(idx, default=-1).to_numpy()
    keep = (qi >= 0) & (mi >= 0)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"[warn] {n_dropped} edges referenced genomes absent from genome_list",
              file=sys.stderr)
    qi, mi = qi[keep], mi[keep]

    # One direction is enough; connected_components(directed=False) symmetrises.
    graph = csr_matrix((np.ones(len(qi), dtype=np.uint8), (qi, mi)), shape=(n, n))
    n_clusters, labels = connected_components(graph, directed=False, connection="weak")

    cluster_df = pd.DataFrame({"genome": all_genomes, "cluster": labels})
    sizes = cluster_df["cluster"].value_counts()
    print(f"Genomes   : {n}", file=sys.stderr)
    print(f"Clusters  : {n_clusters}", file=sys.stderr)
    print(f"Largest   : {sizes.iloc[0]} genomes", file=sys.stderr)
    print(f"Singletons: {int((sizes == 1).sum())}", file=sys.stderr)
    return cluster_df


def main():
    parser = argparse.ArgumentParser(description='Build a scrub k-mer database.')
    parser.add_argument('--genome_list',
                        help='A list of genomes to dereplicate, one path per line')
    parser.add_argument('--min_sourmash', type=float, default=0.8,
                        help='Min max_containment_ani to put two genomes in the same primary cluster.')
    parser.add_argument('--kmer_ident', type=float, default=0.96,
                        help='k-mer coverage threshold for dereplication.')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for the scrub database.')
    parser.add_argument('--genome_compare', required=False,
                        help='Path to the strainer genome_compare binary.')
    parser.add_argument('--threads', required=False, type=int, default=12,
                        help='Threads for sourmash / compression.')
    parser.add_argument('--ksize', type=int, default=31)
    parser.add_argument('--scaled', type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- genome name <-> path map (basename keyed), built up front so we have the
    #     full node set for clustering (singletons no longer appear in the edge file) ---
    genome_paths = pd.read_csv(args.genome_list, header=None, names=["path"])
    genome_paths["genome"] = genome_paths["path"].str.rsplit("/").str[-1]
    name_to_path = genome_paths.set_index("genome")["path"].to_dict()
    all_genomes = genome_paths["genome"].tolist()

    # --- sketch ---
    sketches_path = os.path.join(args.output_dir, "sketches.zip")
    sourmash_sketch(genome_list=args.genome_list,
                    outdir=args.output_dir,
                    sketches_path=sketches_path,
                    ksize=args.ksize, scaled=args.scaled, threads=args.threads)

    # --- pairwise: projected + ANI-prefiltered + zstd, via FIFO (no big CSV lands) ---
    pairwise_path = os.path.join(args.output_dir, "sourmash-pairwise.csv.zst")
    ani_floor = max(0.0, args.min_sourmash - 0.05)   # keep margin below the cut
    sourmash_pairwise(sketches_path=sketches_path,
                      out_path=pairwise_path,
                      ani_floor=ani_floor,
                      ani_column="max_containment_ani",
                      ksize=args.ksize,
                      threads=args.threads)

    # --- primary clusters: connected components at the ANI cut ---
    cluster_df = cluster_genomes_by_ani(
        pairwise_path,
        all_genomes=all_genomes,
        ani_threshold=args.min_sourmash,
        ani_column="max_containment_ani",
    )
    print(cluster_df["cluster"].value_counts().head(10))

    # --- per-cluster genome lists ---
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

    # --- write last for snakemake checkpoint ---
    cluster_df.sort_values('cluster').to_csv(
        os.path.join(args.output_dir, 'primary_clusters.csv'), index=False)


if __name__ == '__main__':
    main()