from collections import defaultdict
import polars as pl
from itertools import combinations
import polars as pl
import pandas as pd
import time
import gc
import argparse
import os
import gzip
import numpy as np
import numpy as np
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from collections import defaultdict

def get_test_dataset():
    ''' Creates a reabable test dataset to test rare kmer combination creation
    A: rare singleton all 0s
    B/C: rare pair, never in two at same time
    B/C/D/E: rare triplicate
    '''
    data = [
        ["A", "B", "C", "D", "E", "F"],
        [ 0,   1,   0,   1,   1,   0 ],
        [ 0,   0,   1,   1,   0,   1 ],
        [ 0,   1,   0,   0,   1,   1 ],
        [ 0 ,  0,   0,   0,   0,   0],
        [ 1,   1,   1,   1,   1,   1]
    ]

    df = pl.DataFrame(data, schema=["#kmer", "ST-1", "ST-2", 'ST-3', 'ST-4', 'ST-5'])
    print('Creating Test Data')
    print(df)

    # Creating test data set for checking pairs and trilicate counts etc:
    data = [
        ["A", "B", "C", "D", "E", "F"],
        [ 1,   1,   1,   1,   1,   0 ],
        [ 1,   1,   0,   1,   0,   1 ],
        [ 0,   0,   0,   1,   1,   1 ],
        [ 0 ,  0,   0,   0,   1,   0],
        [ 1,   1,   1,   1,   1,   1]
    ]
    df2 = pl.DataFrame(data, schema=["#kmer", "SAMPLE-1", "SAMPLE-2", 'SAMPLE-3', 'SAMPLE-4', 'SAMPLE-5'])  
    return df, df2

def strain_name_from_path(path):
    base = os.path.basename(path)
    return base.split('.')[0]


def create_all_kmer_pairs(df, kmer_column):
    ''' This creates all kmer pairs possible,
    legacy function from initial development 
    '''
    dict_kmer_pairs = defaultdict(int)
    # Get kmers present (non-zero) in this genome
    kmers_present = df.select(pl.col(kmer_column)).to_series().to_list()
    print(len(kmers_present))
    n_kmers = len(kmers_present)
    
    # Count pairs locally
    for kmerA, kmerB in combinations(kmers_present, 2):
        pair = (kmerA, kmerB) if kmerA < kmerB else (kmerB, kmerA)
        dict_kmer_pairs[pair] = 0
    
    # Cleanup
    del kmers_present
    gc.collect()
    
    return dict_kmer_pairs


def process_genome(genome_col, dict_kmer_pairs, df):
    """Process a single genome and return pair counts"""
    
    # Get kmers present (non-zero) in this genome
    kmers_present = df.filter(pl.col(genome_col) > 0).select(pl.col("#kmer")).to_series().to_list()
    
    for kmerA, kmerB in combinations(kmers_present, 2):
        pair = (kmerA, kmerB) if kmerA < kmerB else (kmerB, kmerA)
        dict_kmer_pairs[pair] += 1
    
    # Cleanup
    del kmers_present
    gc.collect()
    
    return dict_kmer_pairs

def process_sample(sample_col, dict_kmer_pairs, df):
    """Process a sample and return pair counts"""
        # Get kmers present (non-zero) in this genome
    kmers_present = df.filter(pl.col(genome_col) > 0).select(pl.col("#kmer")).to_series().to_list()
    
    for kmerA, kmerB in combinations(kmers_present, 2):
        pair = (kmerA, kmerB) if kmerA < kmerB else (kmerB, kmerA)
        dict_kmer_pairs[pair] = {'sample': sample_col,
                                'count': 1}
    
    # Cleanup
    del kmers_present
    gc.collect()
    
    return dict_kmer_pairs

import numpy as np
from itertools import combinations

def create_kmer_pairs(df, kmer_column="#kmer"):
    '''Create all kmer pairs and split by informativeness.
    
    Returns:
        inform:     dict {(kmerA, kmerB): 0}   — pairs that never co-occur
        non_inform: dict {(kmerA, kmerB): n}   — pairs co-occurring in n samples
    '''
    genome_cols = [c for c in df.columns if c != kmer_column]
    kmers = df[kmer_column].to_list()

    # N x G boolean matrix: is each kmer present in each sample?
    mat = (df.select(genome_cols).to_numpy() > 0)
    presence = [frozenset(np.nonzero(row)[0]) for row in mat]

    print(f"{len(kmers)} kmers, {len(genome_cols)} samples")

    inform = {}
    non_inform = {}
    for (i, a), (j, b) in combinations(enumerate(kmers), 2):
        pair = (a, b) if a < b else (b, a)
        c = len(presence[i] & presence[j])
        if c == 0:
            inform[pair] = 0
        else:
            non_inform[pair] = c

    print(f"Informative pairs: {len(inform):,}")
    print(f"Non-informative pairs: {len(non_inform):,}")
    return inform, non_inform

from multiprocessing import Pool
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os

# Module-level globals populated once per worker via initializer
_PRESENCE = None
_KMERS = None

def _init_worker(presence, kmers):
    global _PRESENCE, _KMERS
    _PRESENCE = presence
    _KMERS = kmers

def _process_anchor_chunk(args):
    '''Process a chunk of (i, j_list) anchors. Returns path to written parquet.'''
    chunk_id, anchor_items, output_dir, basename = args
    schema = pa.schema([
        ("kmerA", pa.string()), ("kmerB", pa.string()),
        ("kmerC", pa.string()), ("count", pa.int64()),
    ])
    inform_path = f"{output_dir}/{basename}.inform_triplets.part{chunk_id:04d}.parquet"
    non_inform_path = f"{output_dir}/{basename}.non_inform_triplets.part{chunk_id:04d}.parquet"
    inform_w = pq.ParquetWriter(inform_path, schema, compression="zstd")
    non_inform_w = pq.ParquetWriter(non_inform_path, schema, compression="zstd")

    i_a, i_b, i_c, i_n = [], [], [], []
    n_a, n_b, n_c, n_n = [], [], [], []
    n_inform = n_non_inform = 0
    n = len(_KMERS)
    BATCH = 500_000

    def flush(writer, cols):
        if cols[0]:
            writer.write_table(pa.table({
                "kmerA": cols[0], "kmerB": cols[1], "kmerC": cols[2], "count": cols[3]
            }, schema=schema))
            cols[0].clear(); cols[1].clear(); cols[2].clear(); cols[3].clear()

    inform_buf = [i_a, i_b, i_c, i_n]
    non_inform_buf = [n_a, n_b, n_c, n_n]

    for i, j_list in anchor_items:
        kA = _KMERS[i]
        for j in j_list:
            kB = _KMERS[j]
            ab = _PRESENCE[i] & _PRESENCE[j]
            # early termination: if ab is huge, unlikely to find informative triplets
            # (optional - depends on your data)
            for k in range(j + 1, n):
                c = len(ab & _PRESENCE[k])
                kC = _KMERS[k]
                if c == 0:
                    i_a.append(kA); i_b.append(kB); i_c.append(kC); i_n.append(0)
                    n_inform += 1
                    if len(i_a) >= BATCH:
                        flush(inform_w, inform_buf)
                else:
                    n_a.append(kA); n_b.append(kB); n_c.append(kC); n_n.append(c)
                    n_non_inform += 1
                    if len(n_a) >= BATCH:
                        flush(non_inform_w, non_inform_buf)

    flush(inform_w, inform_buf)
    flush(non_inform_w, non_inform_buf)
    inform_w.close()
    non_inform_w.close()
    return chunk_id, n_inform, n_non_inform

def create_kmer_triplets_parallel(non_inform_pairs, df_w_count, output_dir, basename,
                                   n_workers=None, kmer_column="#kmer"):
    from collections import defaultdict
    genome_cols = [c for c in df_w_count.columns if c != kmer_column]
    kmers = df_w_count[kmer_column].to_list()
    kmer_to_idx = {k: i for i, k in enumerate(kmers)}
    mat = (df_w_count.select(genome_cols).to_numpy() > 0)
    presence = [frozenset(np.nonzero(row)[0]) for row in mat]

    pairs_by_a = defaultdict(list)
    for (a, b) in non_inform_pairs.keys():
        pairs_by_a[kmer_to_idx[a]].append(kmer_to_idx[b])

    # split anchors into chunks for workers
    # balance by total work per chunk, not just number of anchors
    items = sorted(pairs_by_a.items())  # deterministic order
    n_workers = n_workers or max(1, os.cpu_count() - 1)

    # assign items to workers greedily by workload (j_list length × remaining-k-count)
    n_total = len(kmers)
    workloads = [(i, j_list, sum(n_total - j - 1 for j in j_list)) for i, j_list in items]
    workloads.sort(key=lambda x: -x[2])  # biggest first
    chunks = [[] for _ in range(n_workers)]
    chunk_loads = [0] * n_workers
    for i, j_list, load in workloads:
        w = chunk_loads.index(min(chunk_loads))
        chunks[w].append((i, j_list))
        chunk_loads[w] += load

    args_list = [(cid, chunk, output_dir, basename) for cid, chunk in enumerate(chunks) if chunk]

    with Pool(n_workers, initializer=_init_worker, initargs=(presence, kmers)) as pool:
        results = pool.map(_process_anchor_chunk, args_list)

    total_inform = sum(r[1] for r in results)
    total_non_inform = sum(r[2] for r in results)
    print(f"Done. Informative: {total_inform:,}  Non-informative: {total_non_inform:,}")
 


def create_kmer_triplets_streaming(
    non_inform_pairs,
    df_w_count,
    output_dir,
    basename,
    kmer_column="#kmer",
    batch_size=1_000_000,
):
    '''Extend non-informative pairs to triplets, streaming results to parquet.

    Writes two files:
      - {basename}.inform_kmer_triplets.parquet     (count == 0)
      - {basename}.non_inform_kmer_triplets.parquet (count > 0)

    Returns (n_inform, n_non_inform) counts.
    '''
    genome_cols = [c for c in df_w_count.columns if c != kmer_column]
    kmers = df_w_count[kmer_column].to_list()
    kmer_to_idx = {k: i for i, k in enumerate(kmers)}

    mat = (df_w_count.select(genome_cols).to_numpy() > 0)
    presence = [frozenset(np.nonzero(row)[0]) for row in mat]

    pairs_by_a = defaultdict(list)
    for (a, b) in non_inform_pairs.keys():
        pairs_by_a[kmer_to_idx[a]].append(kmer_to_idx[b])
    for i in pairs_by_a:
        pairs_by_a[i].sort()

    schema = pa.schema([
        ("kmerA", pa.string()),
        ("kmerB", pa.string()),
        ("kmerC", pa.string()),
        ("count", pa.int64()),
    ])
    inform_path = f"{output_dir}/{basename}.inform_kmer_triplets.parquet"
    non_inform_path = f"{output_dir}/{basename}.non_inform_kmer_triplets.parquet"
    inform_writer = pq.ParquetWriter(inform_path, schema, compression="zstd")
    non_inform_writer = pq.ParquetWriter(non_inform_path, schema, compression="zstd")

    # batch buffers
    i_a, i_b, i_c, i_n = [], [], [], []
    n_a, n_b, n_c, n_n = [], [], [], []
    n_inform = 0
    n_non_inform = 0
    n = len(kmers)

    def flush_inform():
        nonlocal i_a, i_b, i_c, i_n
        if not i_a:
            return
        table = pa.table({"kmerA": i_a, "kmerB": i_b, "kmerC": i_c, "count": i_n}, schema=schema)
        inform_writer.write_table(table)
        i_a, i_b, i_c, i_n = [], [], [], []

    def flush_non_inform():
        nonlocal n_a, n_b, n_c, n_n
        if not n_a:
            return
        table = pa.table({"kmerA": n_a, "kmerB": n_b, "kmerC": n_c, "count": n_n}, schema=schema)
        non_inform_writer.write_table(table)
        n_a, n_b, n_c, n_n = [], [], [], []

    total_pairs = sum(len(v) for v in pairs_by_a.values())
    processed = 0
    print(f"Extending {total_pairs:,} non-informative pairs to triplets", flush=True)

    for i, j_list in pairs_by_a.items():
        kA = kmers[i]
        for j in j_list:
            kB = kmers[j]
            ab = presence[i] & presence[j]
            for k in range(j + 1, n):
                c = len(ab & presence[k])
                kC = kmers[k]
                if c == 0:
                    i_a.append(kA); i_b.append(kB); i_c.append(kC); i_n.append(0)
                    n_inform += 1
                    if len(i_a) >= batch_size:
                        flush_inform()
                else:
                    n_a.append(kA); n_b.append(kB); n_c.append(kC); n_n.append(c)
                    n_non_inform += 1
                    if len(n_a) >= batch_size:
                        flush_non_inform()
            processed += 1
            if processed % 100_000 == 0:
                print(f"  [{processed:,}/{total_pairs:,}] inform={n_inform:,} non_inform={n_non_inform:,}", flush=True)

    flush_inform()
    flush_non_inform()
    inform_writer.close()
    non_inform_writer.close()

    print(f"Done. Informative triplets: {n_inform:,}  Non-informative triplets: {n_non_inform:,}", flush=True)
    return n_inform, n_non_inform

# convert pair-keyed dict to a DataFrame with columns
def pairs_dict_to_df(d):
    pairs = list(d.keys())
    return pl.DataFrame({
        "kmerA": [p[0] for p in pairs],
        "kmerB": [p[1] for p in pairs],
        "count": list(d.values()),
    })
def triplets_dict_to_df(d):
    pairs = list(d.keys())
    return pl.DataFrame({
        "kmerA": [p[0] for p in pairs],
        "kmerB": [p[1] for p in pairs],
        "kmerC": [p[2] for p in pairs],
        "count": list(d.values()),
    })
import polars as pl

def drop_reference_similar_strains(df, similarity_threshold=0.95, kmer_column="#kmer", verbose=True):
    '''Drop strain columns whose Jaccard similarity to an all-present reference
    exceeds the threshold.

    Jaccard(strain, all-ones) = |strain ∩ all| / |strain ∪ all|
                              = |strain| / N_kmers
    i.e. the fraction of the kmer panel present in that strain.
    '''
    strain_cols = [c for c in df.columns if c != kmer_column]
    n_kmers = df.shape[0]

    # fraction of kmers present (>0) in each strain column
    presence_frac = df.select([(pl.col(c) > 0).sum().truediv(n_kmers).alias(c) for c in strain_cols]).row(0, named=True)

    drop_cols = [c for c, frac in presence_frac.items() if frac >= similarity_threshold]
    keep_cols = [c for c in strain_cols if c not in set(drop_cols)]

    if verbose:
        print(f"Strains:   {len(keep_cols)}/ {len(strain_cols)}"
              f"(dropped {len(drop_cols)} with kmer fraction >= {similarity_threshold})")
        shown = sorted(drop_cols, key=lambda c: -presence_frac[c])[:10]
        for c in shown:
            print(f"  drop {c[:60]}... (fraction {presence_frac[c]:.3f})")
        if len(drop_cols) > 10:
            print(f"  ... and {len(drop_cols) - 10} more")

    return df.select([kmer_column] + keep_cols)

def get_singleton_hits(df_samples, df_informative, kmer_column="#kmer"):
    """Compute per-sample coverage for informative singletons.

    df_samples:    Polars DataFrame, wide format [#kmer, sample1, sample2, ...]
    df_informative: Polars DataFrame with #kmer column (the informative singleton panel)
    Returns a Polars DataFrame with one row per sample.
    """
    sample_cols = [c for c in df_samples.columns if c != kmer_column]
    n_total = len(df_informative)
    inform_set = df_informative[kmer_column]

    df_hits = df_samples.filter(pl.col(kmer_column).is_in(inform_set.implode()))

    # per-sample aggregates: observed count, mean hit count
    rows = []
    for s in sample_cols:
        col = df_hits[s]
        observed = (col > 0).sum()
        rows.append({
            "sample": s,
            "inform_singletons_observed": observed,
            "inform_singletons_count_mean": col.mean() if len(col) else 0.0,
            "inform_singletons_coverage": observed / n_total if n_total else 0.0,
        })
    return pl.DataFrame(rows)


def get_pair_hits(df_samples, df_informative, kmer_column="#kmer"):
    """Compute per-sample coverage for informative pairs.

    df_informative: Polars DataFrame with columns [kmerA, kmerB, ...]
    A pair is 'observed' in a sample when both kmers have count > 0 there.
    """
    sample_cols = [c for c in df_samples.columns if c != kmer_column]
    n_total = len(df_informative)

    # restrict the sample matrix to kmers that appear in any pair
    pair_kmers = pl.concat([df_informative["kmerA"], df_informative["kmerB"]]).unique()
    df_hits = df_samples.filter(pl.col(kmer_column).is_in(pair_kmers.implode()))

    # join pairs to their A and B counts per sample
    df_A = df_hits.rename({kmer_column: "kmerA", **{s: f"{s}__A" for s in sample_cols}})
    df_B = df_hits.rename({kmer_column: "kmerB", **{s: f"{s}__B" for s in sample_cols}})

    df_pair = (
        df_informative.select(["kmerA", "kmerB"])
        .join(df_A, on="kmerA", how="inner")
        .join(df_B, on="kmerB", how="inner")
    )

    # per-sample pair count = min(count_A, count_B); present when > 0
    rows = []
    for s in sample_cols:
        pair_count = pl.min_horizontal(pl.col(f"{s}__A"), pl.col(f"{s}__B"))
        stats = df_pair.select([
            (pair_count > 0).sum().alias("observed"),
            pair_count.mean().alias("mean_count"),
        ]).row(0, named=True)
        rows.append({
            "sample": s,
            "inform_pairs_observed": stats["observed"],
            "inform_pairs_count_mean": stats["mean_count"] or 0.0,
            "inform_pairs_coverage": stats["observed"] / n_total if n_total else 0.0,
        })
    return pl.DataFrame(rows)


import glob
import polars as pl

def get_triple_hits_streaming(df_samples, triplets_glob, kmer_column="#kmer"):
    """Stream informative triplets from multiple parquet part files and compute
    per-sample coverage without materializing the full triplet frame.

    triplets_glob: glob pattern like
        "/path/to/output/{basename}.inform_triplets.part*.parquet"
    """
    part_files = sorted(glob.glob(triplets_glob))
    if not part_files:
        raise FileNotFoundError(f"No triplet part files matched {triplets_glob}")

    sample_cols = [c for c in df_samples.columns if c != kmer_column]

    # per-sample running totals
    observed = {s: 0 for s in sample_cols}
    sum_count = {s: 0 for s in sample_cols}
    n_total = 0  # total triplets across all parts

    for path in part_files:
        df_part = pl.read_parquet(path)
        n_total += len(df_part)
        if len(df_part) == 0:
            continue

        # restrict the sample matrix to kmers touched by this part (small memory win)
        part_kmers = pl.concat([df_part["kmerA"], df_part["kmerB"], df_part["kmerC"]]).unique()
        df_hits = df_samples.filter(pl.col(kmer_column).is_in(part_kmers.implode()))

        df_A = df_hits.rename({kmer_column: "kmerA", **{s: f"{s}__A" for s in sample_cols}})
        df_B = df_hits.rename({kmer_column: "kmerB", **{s: f"{s}__B" for s in sample_cols}})
        df_C = df_hits.rename({kmer_column: "kmerC", **{s: f"{s}__C" for s in sample_cols}})

        df_trip = (
            df_part.select(["kmerA", "kmerB", "kmerC"])
            .join(df_A, on="kmerA", how="inner")
            .join(df_B, on="kmerB", how="inner")
            .join(df_C, on="kmerC", how="inner")
        )

        # accumulate per-sample stats from this part
        for s in sample_cols:
            trip_count = pl.min_horizontal(
                pl.col(f"{s}__A"), pl.col(f"{s}__B"), pl.col(f"{s}__C")
            )
            stats = df_trip.select([
                (trip_count > 0).sum().alias("observed"),
                trip_count.sum().alias("sum_count"),
            ]).row(0, named=True)
            observed[s] += stats["observed"] or 0
            sum_count[s] += stats["sum_count"] or 0

        del df_part, df_hits, df_A, df_B, df_C, df_trip

    rows = [{
        "sample": s,
        "inform_triples_observed": observed[s],
        "inform_triples_count_mean": sum_count[s] / n_total if n_total else 0.0,
        "inform_triples_coverage": observed[s] / n_total if n_total else 0.0,
    } for s in sample_cols]
    return pl.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description='Map scrubbed kmers onto a genome.')
    parser.add_argument('--csv_path', help='a set of kmer counts of pangenomes to create informative kmer pairs from')
    parser.add_argument('--output_dir')
    parser.add_argument('--testmode', action= 'store_true', help = 'Uses a test dataset instead of an input csv')
    parser.add_argument('--threads', type=int, default=None, help='Number of worker processes for triplet generation. Default: os.cpu_count() - 1')
    parser.add_argument('--ref_jaccard_threshold', type=float, default=0.95, help='Drop strains whose Jaccard similarity to an all-present reference exceeds this')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok = True)

    if args.testmode:
        df, df_samples = get_test_dataset()
        basename = 'testmode'
    else:
        basename = strain_name_from_path(args.csv_path)
        csv_path = args.csv_path
        # Polars for fast multithreaded reading of the csv file, needs more tha 16 gb ram
        df = pl.read_csv(csv_path, separator="\t")
        df = df.filter(pl.col("#kmer") != "total_evaluated")
   
    #Step 1 dropping all columns with no counts
    pair_cols = ["#kmer"] + [col for col in df.columns if col != "#kmer" and df[col].sum() > 0]
    df = df.select(pair_cols)
    print(df)
    
    # Step 2
    # Check for cols with only 1s, print warning and drop
    #ref_cols = [col for col, all_present in df.select(pl.exclude("#kmer").min().ge(1)).row(0, named=True).items() if all_present]
    #if ref_cols:
    #    print(f"Dropping {len(ref_cols)} reference columns (kmer present in all rows): {ref_cols[:3]}{'...' if len(ref_cols) > 3 else ''}")
    #    df = df.drop(ref_cols)
    print('Checking for too similar columns')
    df = drop_reference_similar_strains(df, similarity_threshold=args.ref_jaccard_threshold)

    # Step 3 Check for highly similar strains and remove


    # Get all informative kmer singletons:
    print('Getting all informative singleton kmers with count 0')
    df = df.with_columns(pl.sum_horizontal(pl.exclude('#kmer')).alias("count_kmer_singleton"))
    df_inform_singleton = df.filter(pl.col("count_kmer_singleton") == 0).select(["#kmer", "count_kmer_singleton"]) 
    print('Informative Singletons')   
    print(df_inform_singleton)
    df_inform_singleton.write_parquet(os.path.join(args.output_dir , f'{basename}.inform_kmer_singleton.parquet'), compression='zstd')
    
    # Gets all rows with kmer counts
    print('Creating pairs for all kmers with counts: ')
    df_w_count = df.filter(pl.col("count_kmer_singleton") > 0).drop("count_kmer_singleton")  
    n = df_w_count.shape[0]
    g = len([c for c in df_w_count.columns if c != "#kmer"])
    print(f"N kmers after zero-filter: {n:,}", flush=True)
    print(f"N genome columns: {g}", flush=True)


    dict_inform_pairs, dict_non_inform_pairs = create_kmer_pairs(df_w_count)
    df_inform_pairs = pairs_dict_to_df(dict_inform_pairs)
    print(df_inform_pairs)
    df_non_inform_pairs = pairs_dict_to_df(dict_non_inform_pairs)
    print(df_non_inform_pairs)
    df_inform_pairs.write_parquet(os.path.join(args.output_dir , f'{basename}.inform_kmer_pairs.parquet'), compression='zstd')
    if not args.testmode:
        del df_inform_pairs
        del df_non_inform_pairs
        del dict_inform_pairs
        gc.collect()
    
    # Get all informative triplets:
    # this is too much data for ram and needs to be streamed to files directly
    print('Creating triplicates')
    create_kmer_triplets_parallel(dict_non_inform_pairs,
                                df_w_count,
                                args.output_dir,
                                basename,
                                n_workers=args.threads,)
    if args.testmode:
        # reading the output
        print('Informative Triplets')
        print(pl.read_parquet(os.path.join(args.output_dir , f'{basename}.inform_kmer_triplets.parquet')))
        print('Non informative Triplets: ')
        print(pl.read_parquet(os.path.join(args.output_dir , f'{basename}.non_inform_kmer_triplets.parquet')))

        print('Creating coverage outputs')
        print(df_samples)
        df_cov_s = get_singleton_hits(df_samples, df_inform_singleton)
        df_cov_p = get_pair_hits(df_samples, df_inform_pairs)
        df_cov_t = get_triple_hits_streaming(df_samples,
                                             os.path.join(args.output_dir, f"{basename}.inform_triplets.part*.parquet"),)
        df_cov = df_cov_s.join(df_cov_p, on="sample", how="left").join(df_cov_t, on="sample", how="left")

        print(df_cov)


if __name__ == '__main__':
    main()
