import argparse
import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _verify_sketch_zip(zip_path, expected=None):
    """Sanity-check a sketch zip before trusting it downstream.

    branchwater's manysketch writer does not set ZIP64/large_file, so a single zip
    silently corrupts once it crosses ~4 GB or 65,535 entries: the file exists but its
    central directory is unusable and sourmash / pairwise then read garbage. Catch it
    here while it is cheap (reading only the central directory), rather than failing
    70% of the way into a multi-hour pairwise.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"{zip_path} is not a readable zip ({exc}); "
            "likely a ZIP64/large_file write overflow."
        )
    if "SOURMASH-MANIFEST.csv" not in names:
        raise RuntimeError(f"{zip_path} has no SOURMASH-MANIFEST.csv (corrupt write?).")
    n_sigs = sum(1 for x in names if x.endswith(".sig.gz"))
    if expected is not None and n_sigs != expected:
        # md5-identical sketches are deduplicated, so a small shortfall is normal;
        # a large one means a truncated / overflowed write -> worth a loud warning.
        print(
            f"[warn] {zip_path}: manifest lists {n_sigs} sketches but {expected} "
            "genomes were requested. A small gap is expected if any sketches share an "
            "md5; a large gap suggests a truncated or ZIP64-overflowed write.",
            file=sys.stderr,
        )
    return n_sigs


def sourmash_sketch(genome_list, sketches_path, outdir, ksize=31, scaled=1000,
                    threads=8, chunk_size=50000):
    """Sketch the genome list into a single, ZIP64-valid sketches.zip.

    We cannot let branchwater's manysketch write one big zip directly: its writer
    omits ZIP64/large_file, so any archive over ~4 GB (or 65,535 entries) is written
    corrupt -- exactly what bites a full UHGG run (~289k genomes). Instead we
    manysketch in <= chunk_size-genome shards (each comfortably under both limits),
    verify every shard, then concatenate them with `sourmash sig cat`, whose Python
    zipfile backend writes proper ZIP64. The result is one ordinary sketches.zip that
    pairwise consumes unchanged. (branchwater READS ZIP64 zips fine -- only its writer
    is the problem -- so the concatenated zip is safe downstream.)
    """
    df = pd.read_csv(genome_list, header=None)
    df.columns = ['genome_filename']
    df['name'] = df['genome_filename'].str.rsplit('/').str[-1]
    df['protein_filename'] = ""
    print(df["name"].value_counts().head(20))
    print(f"\nTotal: {len(df)}, Unique: {df['name'].nunique()}, "
          f"Duplicates: {len(df) - df['name'].nunique()}")
    if len(df) - df['name'].nunique() > 0:
        print('Duplicate name entries in genome list')

    df = df[['name', 'genome_filename', 'protein_filename']].reset_index(drop=True)

    chunks_dir = os.path.join(outdir, "sketch_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    n_total = len(df)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    shard_zips = []

    print(f"[sketch] {n_total} genomes -> {n_chunks} shard(s) of <= {chunk_size} "
          f"(k={ksize}, scaled={scaled})", file=sys.stderr)

    for i in range(n_chunks):
        sub = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        csv_path = os.path.join(chunks_dir, f"manysketch_{i:04d}.csv")
        zip_path = os.path.join(chunks_dir, f"sketches_{i:04d}.zip")
        sub.to_csv(csv_path, index=None)

        print(f"[sketch] shard {i + 1}/{n_chunks}: {len(sub)} genomes -> {zip_path}",
              file=sys.stderr)

        cmd = [
            "sourmash", "scripts", "manysketch",
            csv_path,
            "--param-str", f"dna,k={ksize},scaled={scaled}",
            "-o", zip_path,
            "-c", str(threads),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode != 0:
            raise RuntimeError(
                f"sourmash manysketch failed on shard {i} (exit {result.returncode})")

        _verify_sketch_zip(zip_path, expected=len(sub))
        shard_zips.append(zip_path)

    # Concatenate shards into one ZIP64-valid zip. `sig cat` streams signature-by-
    # signature (O(1) memory) and writes via Python's zipfile, which handles
    # >4 GB / >65,535 entries correctly -- unlike branchwater's writer.
    if len(shard_zips) == 1:
        # A single shard is already a complete, valid zip; just put it in place.
        if os.path.abspath(shard_zips[0]) != os.path.abspath(sketches_path):
            shutil.move(shard_zips[0], sketches_path)
    else:
        print(f"[sketch] concatenating {n_chunks} shards -> {sketches_path}",
              file=sys.stderr)
        cat_cmd = ["sourmash", "sig", "cat", *shard_zips, "-o", sketches_path]
        result = subprocess.run(cat_cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode != 0:
            raise RuntimeError(f"sourmash sig cat failed (exit {result.returncode})")

    _verify_sketch_zip(sketches_path, expected=n_total)
    print(f"[sketch] Done: {sketches_path} ({n_total} genomes, {n_chunks} shards)",
          file=sys.stderr)
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
    parser.add_argument('--sketch_chunk_size', type=int, default=50000,
                        help='Genomes per manysketch shard. Keeps each intermediate '
                             'zip well under the ~4 GB / 65,535-entry ZIP64 limit that '
                             "branchwater's manysketch writer mishandles, then shards "
                             'are concatenated into one valid sketches.zip.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- genome name <-> path map (basename keyed), built up front so we have the
    #     full node set for clustering (singletons no longer appear in the edge file) ---
    genome_paths = pd.read_csv(args.genome_list, header=None, names=["path"])
    genome_paths["genome"] = genome_paths["path"].str.rsplit("/").str[-1]
    name_to_path = genome_paths.set_index("genome")["path"].to_dict()
    all_genomes = genome_paths["genome"].tolist()

    # --- sketch (sharded -> single ZIP64-valid sketches.zip) ---
    sketches_path = os.path.join(args.output_dir, "sketches.zip")
    sourmash_sketch(genome_list=args.genome_list,
                    outdir=args.output_dir,
                    sketches_path=sketches_path,
                    ksize=args.ksize, scaled=args.scaled, threads=args.threads,
                    chunk_size=args.sketch_chunk_size)

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
