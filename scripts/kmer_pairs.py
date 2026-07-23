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
import re
import random

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from collections import defaultdict
import glob
from multiprocessing import Pool


def get_test_dataset():
    '''Test dataset with multiple informative singletons, pairs
    '''
    strain_rows = [
        ["A1",    []],
        ["A2",    []],
        ["A3",    []],
        ["A4",    []],
        ["A5",    []],
        ["A6",    []],
        ["A7",    []],
        ["P1",    [1, 2, 3]],
        ["P2",    [4, 5, 6]],
        ["P3",    [4, 5]],
        ["P4",    [3]],
    ]
    df_indiv_counts = pl.DataFrame(
        strain_rows,
        schema={"#kmer": pl.String, "list_scrub_id": pl.List(pl.UInt32)},
        orient="row",
    )
    # global kmer counts
    rows_counts = [["A1",    1,0,0,0],
                    ["A2",    1,0,0,0],
                    ["A3",    1,0,0,0],
                    ["A4",    1,0,0,0],
                    ["A5",    1,0,0,0],
                    ["A6",    1,0,0,0],
                    ["A7",    1,0,0,0],
                    ["P1",    1,2,2,0],
                    ["P2",    1,2,2,0],
                    ["P3",    1,2,2,0],
                    ["P4",    1,2,0,0],
                    ]
    df_global_counts= pl.DataFrame(
        rows_counts,
        schema=['#kmer', 'reference_count', 'pangenome_count',	'metagenome_count', 'drug_count'],
        orient="row")
    

    summary_data = [[1,'ge', 'SCR1',12,0.95,True],
                    [2,'ge', 'SCR2',12,0.95,True],
                    [3,'ge', 'SCR3',12,0.95,True],
                    [4,'me', 'SCR4',12,0.95,True ],
                    [5,'me', 'SCR5',12,0.95,True ],
                    [6,'me', 'SCR5',12,0.95,True]
                     ]
    df_scrub_summary = pl.DataFrame(
        summary_data,
        schema=['scrub_id',	'sample_type','sample_id','n_unique_kmers', 'coverage_pct', 'is_in_global'],
        orient="row")
    

    # Sample panel — design counts to exercise coverage logic.
    # Make sure to give pair partners and triplet members co-occurring counts in some samples.
    sample_rows = [
        # kmer    S1   S2   S3   S4   S5
        ["A1",    1000,    10, 1000,   0,   0, 0],   # singleton observed in S1, S2
        ["A2",     1, 10,   1000,   0,   0, 0],
        ["A3",     1, 10,   1000,   0,   0, 0],
        ["A4",     1, 10,   1000,   0,   0, 0],
        ["A5",     1, 10,   1000,   0,   0, 0],
        ["A6",     1, 10,   1000,   0,   0, 0],
        ["A7",     1, 10,   1000,   0,   0, 0],
        ["P1",      1,   10,      0,   10,  10, 0],   # (P1,P2) pair observed in S2 (both>0)? need P2>0 too
        ["P2",      1,   10,      0,   10,  20, 0],
        ["P3",      1,   10,      0,   0,     0, 0],   # (P3,P4) pair observed in S4
    ]
    df_samples = pl.DataFrame(
        sample_rows,
        schema=["#kmer", "S1", "S2", "S3", "S4", "S5", "S6"],
        orient="row",
    )

    print("Creating Test Data")
    print(df_indiv_counts, df_global_counts, df_scrub_summary, df_samples)
    return df_indiv_counts, df_global_counts, df_scrub_summary, df_samples


_PAIR_SCHEMA = pa.schema([
    ("kmerA", pa.string()),
    ("kmerB", pa.string()),
    ("count", pa.int64()),
])

_PAIR_SCHEMA_PL = {"kmerA": pl.Utf8, "kmerB": pl.Utf8, "count": pl.Int64}

# fixes 0 pairs issue
def write_empty_pairs(path):
    """Valid zero-row parquet so downstream scans/globs still resolve."""
    pl.DataFrame(schema=_PAIR_SCHEMA_PL).write_parquet(path, compression="zstd")
    print(f"Wrote empty pairs parquet -> {path}", flush=True)
    return path


# ---------- worker globals (presence -> scrub_id sets) ----------

_SETS = None      # list[frozenset[int]] indexed by kmer position
_KMERS = None     # list[str] of kmer sequences


def _init_worker_disjoint(sets, kmers):
    global _SETS, _KMERS
    _SETS = sets
    _KMERS = kmers

def _process_disjoint_chunk(args):
    chunk_id, i_indices, output_dir, basename, write_non_inform, batch_size = args
    print(f"[worker {chunk_id}] starting, {len(i_indices)} anchors, writing to {output_dir}", flush=True)
    n = len(_KMERS)

    inform_path = os.path.join(
        output_dir, f"{basename}.inform_kmer_pairs.part{chunk_id:04d}.parquet"
    )
    inform_w = pq.ParquetWriter(inform_path, _PAIR_SCHEMA, compression="zstd")

    non_inform_w = None
    if write_non_inform:
        non_inform_path = os.path.join(
            output_dir, f"{basename}.non_inform_kmer_pairs.part{chunk_id:04d}.parquet"
        )
        non_inform_w = pq.ParquetWriter(non_inform_path, _PAIR_SCHEMA, compression="zstd")

    i_a, i_b, i_n = [], [], []
    n_a, n_b, n_n = [], [], []

    def flush(writer, cols):
        if cols[0]:
            writer.write_table(
                pa.table({"kmerA": cols[0], "kmerB": cols[1], "count": cols[2]},
                         schema=_PAIR_SCHEMA)
            )
            cols[0].clear(); cols[1].clear(); cols[2].clear()

    inform_buf = [i_a, i_b, i_n]
    non_inform_buf = [n_a, n_b, n_n]
    n_inform = 0
    n_non_inform = 0

    for i in i_indices:
        kA = _KMERS[i]
        sA = _SETS[i]
        for j in range(i + 1, n):
            kB = _KMERS[j]
            # disjoint => informative pair
            if sA.isdisjoint(_SETS[j]):
                a_, b_ = (kA, kB) if kA < kB else (kB, kA)
                i_a.append(a_); i_b.append(b_); i_n.append(0)
                n_inform += 1
                if len(i_a) >= batch_size:
                    flush(inform_w, inform_buf)
            else:
                n_non_inform += 1
                if write_non_inform:
                    c = len(sA & _SETS[j])
                    a_, b_ = (kA, kB) if kA < kB else (kB, kA)
                    n_a.append(a_); n_b.append(b_); n_n.append(c)
                    if len(n_a) >= batch_size:
                        flush(non_inform_w, non_inform_buf)

    flush(inform_w, inform_buf)
    inform_w.close()
    if write_non_inform:
        flush(non_inform_w, non_inform_buf)
        non_inform_w.close()

    return chunk_id, n_inform, n_non_inform


def create_disjoint_kmer_pairs_parallel(
    df_presence, output_dir, basename,
    n_workers=None,
    kmer_column="#kmer",
    list_column="list_scrub_id",
    batch_size=1_000_000,
    write_non_inform=False,
):
    """Parallel disjoint-pair generation from a presence-list dataframe.

    df_presence: Polars DataFrame with columns [kmer_column, list_column],
                 where list_column is List[UInt32] of scrub_ids per kmer.
                 Empty lists should already be filtered out upstream.
    Output: same parquet structure as create_kmer_pairs_parallel — informative
            pairs (disjoint scrub_id sets) get count=0; optional non-inform
            parquet stores |intersection|.

    Returns (n_inform, n_non_inform).
    """
    kmers = df_presence.get_column(kmer_column).to_list()
    n = len(kmers)
    if n < 2:
        print("Fewer than 2 kmers — nothing to pair.", flush=True)
        return 0, 0

    # build per-kmer scrub_id frozensets (workers share via fork)
    lists = df_presence.get_column(list_column).to_list()
    sets = [frozenset(v) for v in lists]

    n_workers = n_workers or max(1, os.cpu_count() - 1)
    n_workers = min(n_workers, n - 1)

    # balance anchors greedily by remaining work (same as your pair function)
    workloads = [(i, n - i - 1) for i in range(n - 1)]
    workloads.sort(key=lambda x: -x[1])
    chunks = [[] for _ in range(n_workers)]
    chunk_loads = [0] * n_workers
    for i, load in workloads:
        w = chunk_loads.index(min(chunk_loads))
        chunks[w].append(i)
        chunk_loads[w] += load

    args_list = [
        (cid, sorted(indices), output_dir, basename, write_non_inform, batch_size)
        for cid, indices in enumerate(chunks) if indices
    ]
    print(
        f"Disjoint pair generation: {n:,} kmers, "
        f"{n*(n-1)//2:,} candidate pairs, {len(args_list)} workers",
        flush=True,
    )

    with Pool(n_workers, initializer=_init_worker_disjoint, initargs=(sets, kmers)) as pool:
        results = pool.map(_process_disjoint_chunk, args_list)

    n_inform = sum(r[1] for r in results)
    n_non_inform = sum(r[2] for r in results)
    print(
        f"Done. Informative (disjoint) pairs: {n_inform:,}  "
        f"Non-informative pairs: {n_non_inform:,} (write_non_inform={write_non_inform})",
        flush=True,
    )
    return n_inform, n_non_inform

def kmer_pairs_from_presence(
    presence_tsv, summary_tsv, output_dir, basename, df_keep,
    presence_t=10, similarity_t=None,
    n_workers=None, write_non_inform=False,
    testmode = None, max_for_pairs = 20000
):
    # exclusion list from summary
    if testmode:
        df_presence = presence_tsv
        df_presence = df_presence.filter(pl.col('list_scrub_id').list.len()  <= presence_t - 1)
        print(df_presence)
    
    else:
        df = pl.read_csv(summary_tsv, separator='\t')
        if similarity_t is not None:
            df_t = df.filter(pl.col('coverage_pct') < similarity_t)
        else:
            df_t = df.filter(pl.col('is_in_global') == False)
        
        li_t = df_t.get_column('scrub_id').cast(pl.UInt32).to_list()

        # read & clean presence
        df_presence = (
            pl.scan_csv(presence_tsv, separator='\t')
            .filter(pl.col('list_scrub_id').str.count_matches(',') <= presence_t - 1) 
            .with_columns(
                pl.col('list_scrub_id')
                    .str.split(',')
                    .cast(pl.List(pl.UInt32))
                    .list.set_difference(li_t)
            )
            .filter(pl.col('list_scrub_id').list.len() > 0)
            .collect(engine='streaming')
        )
        print(df_presence)
        print(f"Presence rows after filtering: {df_presence.shape[0]:,}", flush=True)
        
    if df_keep is not None:
        print('removing all informative singletons from pair generation')
        df_presence = df_presence.join(df_keep, on='#kmer', how='semi')
        print(df_presence)
        print(f"After kmer subset filter: {df_presence.shape[0]:,}", flush=True)
        # if this is lenght = 0 need to stop here and return
        if len(df_presence) == 0:
            print('HELP')

    if len(df_presence) > max_for_pairs:
        print(f'too many potential pairs: >{max_for_pairs}')
        print('rarest subselection')
        df_presence = (
            df_presence
            .sort(pl.col('list_scrub_id').list.len())
            .head(max_for_pairs)
        )
        print(df_presence)
        #df_presence = df_presence.sample(n=max_for_pairs,  seed=42, shuffle=True)
        # maybe this should be sorted by presence list length


    os.makedirs(output_dir, exist_ok=True)
    return create_disjoint_kmer_pairs_parallel(
        df_presence, output_dir, basename,
        n_workers=n_workers,
        write_non_inform=write_non_inform,
    )

def create_all_pairs(
    kmer_set,
    output_dir, basename,
    batch_size=1_000_000,
    max_kmers=20000,
):
    """
    Change to create all pairs only for presenece 0 and presence 1
    These will all be informative pairs and the coverage_pairs = coverage**2 relationship should be true
    """
    kmers = sorted(kmer_set)

    # subsample if over cap (pair count grows ~n^2/2)
    if len(kmers) > max_kmers:
        kmers = sorted(random.sample(kmers, max_kmers))

    n = len(kmers)
    print(f"Combining {n} kmers among themselves "
          f"({n * (n - 1) // 2:,} pairs)", flush=True)

    path = os.path.join(output_dir, f"{basename}.inform_kmer_pairs.all.parquet")
    writer = pq.ParquetWriter(path, _PAIR_SCHEMA, compression="zstd")

    a_col, b_col, c_col = [], [], []
    n_pairs = 0

    def flush():
        if a_col:
            writer.write_table(
                pa.table({"kmerA": a_col, "kmerB": b_col, "count": c_col},
                         schema=_PAIR_SCHEMA)
            )
            a_col.clear(); b_col.clear(); c_col.clear()

    for i in range(n):
        kA = kmers[i]
        for j in range(i + 1, n):
            kB = kmers[j]
            a_, b_ = (kA, kB) if kA < kB else (kB, kA)
            a_col.append(a_); b_col.append(b_); c_col.append(0)
            n_pairs += 1
            if len(a_col) >= batch_size:
                flush()

    flush()
    writer.close()
    print(f"Wrote {n_pairs:,} pairs -> {path}", flush=True)
    return path, n_pairs

def create_pairs_with_singletons(
    singleton_kmer_set, pair_kmer_set,
    output_dir, basename,
    batch_size=1_000_000,
    self_singletons = False,
    max_singletons = 20000
):
    """Stream singleton-derived pairs to parquet (matches _PAIR_SCHEMA / count=0).

    Group 1: all combinations among informative singletons.
    Group 2: each singleton x each pair kmer.
    Returns (path, n_pairs).
    """
    singletons = sorted(singleton_kmer_set)
    pair_kmers = sorted(pair_kmer_set)
    n_s, n_p = len(singletons), len(pair_kmers)
    print(f"Combining {n_s} singletons among themselves and with {n_p} pair kmers",
          flush=True)

    path = os.path.join(output_dir, f"{basename}.inform_kmer_pairs.singletons.parquet")
    
    writer = pq.ParquetWriter(path, _PAIR_SCHEMA, compression="zstd")

    a_col, b_col, c_col = [], [], []
    n_pairs = 0

    def flush():
        if a_col:
            writer.write_table(
                pa.table({"kmerA": a_col, "kmerB": b_col, "count": c_col},
                         schema=_PAIR_SCHEMA)
            )
            a_col.clear(); b_col.clear(); c_col.clear()
    
    # match all singletons with all pair kmers
    pair_set = set(pair_kmers)
    for kA in singletons:
        skip = kA in pair_set
        for kB in pair_kmers:
            if skip and kA == kB:
                continue
            a_, b_ = (kA, kB) if kA < kB else (kB, kA)
            a_col.append(a_); b_col.append(b_); c_col.append(0)
            n_pairs += 1
            if len(a_col) >= batch_size:
                flush()
    
    # get self kmers
    if self_singletons == True:
        # get length singletons and randomly subselect if over x
        if n_s > max_singletons:
            singletons = random.sample(singletons, max_singletons)
        for i in range(len(singletons)):
            kA = singletons[i]
            for j in range(i + 1, len(singletons)):
                kB = singletons[j]
                a_, b_ = (kA, kB) if kA < kB else (kB, kA)
                a_col.append(a_); b_col.append(b_); c_col.append(0)
                n_pairs += 1
                if len(a_col) >= batch_size:
                    flush()
                    


    flush()
    writer.close()
    print(f"Wrote {n_pairs:,} singleton pairs -> {path}", flush=True)
    return path, n_pairs

def get_singletons_hits_streaming(df_samples, pairs_glob, kmer_column = "#kmer", batch_size = 250_000):
    part_files = sorted(glob.glob(pairs_glob))
    if not part_files:
        raise FileNotFoundError(f"No pair part files matched {pairs_glob}")

    strain_name = re.match(r"(.+)\.inform_kmer_pairs\..+\.parquet", os.path.basename(part_files[0])).group(1)
    sample_cols = [c for c in df_samples.columns if c != kmer_column]
    # get mean and standard deviation of counts per sample from hits
    global_stats = df_samples.select([expr
                                    for s in sample_cols
                                    for expr in [
                                        pl.col(s).mean().alias(f"mean__{s}"),
                                        pl.col(s).std().alias(f"std__{s}"),
                                        (pl.col(s)>0).sum().alias(f"sum__{s}"),
                                    ]
                                ]).row(0, named=True)

    dict_mean = {s: global_stats[f"mean__{s}"] for s in sample_cols}
    dict_std  = {s: global_stats[f"std__{s}"]  for s in sample_cols}
    dict_sum = {s: global_stats[f"sum__{s}"]  for s in sample_cols}
    total_unique_kmers = len(df_samples)



    observed = {s: 0 for s in sample_cols}
    sum_count_min = {s: 0 for s in sample_cols}
    sum_count_mean = {s: 0 for s in sample_cols}
    sum_count_max = {s: 0 for s in sample_cols}

    n_total = 0        # valid pairs: BOTH k-mers present in df_samples
    n_raw   = 0        # raw pairs in the parquet (kept for sanity-checking)
    for path in part_files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["kmerA", "kmerB"]):
            df_part = pl.from_arrow(batch)

            n_raw += len(df_part)
            if len(df_part) == 0:
                continue

            part_kmers = pl.concat([df_part["kmerA"], df_part["kmerB"]]).unique()
            df_hits = df_samples.filter(pl.col(kmer_column).is_in(part_kmers.implode()))

            df_A = df_hits.rename({kmer_column: "kmerA", **{s: f"{s}__A" for s in sample_cols}})
            df_B = df_hits.rename({kmer_column: "kmerB", **{s: f"{s}__B" for s in sample_cols}})

            df_pair = (
                df_part.select(["kmerA", "kmerB"])
                .join(df_A, on="kmerA", how="inner")
                .join(df_B, on="kmerB", how="inner")
            )

            # only pairs where both k-mers are in the (reduced) reference count as "total"
            n_total += len(df_pair)

            # single pass: all samples at once instead of n_samples selects
            exprs = []
            for s in sample_cols:
                a = pl.col(f"{s}__A")
                b = pl.col(f"{s}__B")
                both_nonzero = (a > 0) & (b > 0)

                pc_min  = pl.when(both_nonzero).then(pl.min_horizontal(a, b)).otherwise(0)
                pc_mean = pl.when(both_nonzero).then(pl.mean_horizontal(a, b)).otherwise(0)
                pc_max  = pl.when(both_nonzero).then(pl.max_horizontal(a, b)).otherwise(0)

                exprs.append((both_nonzero).sum().alias(f"o__{s}"))
                exprs.append(pc_min.sum().alias(f"cmin__{s}"))
                exprs.append(pc_mean.sum().alias(f"cmean__{s}"))
                exprs.append(pc_max.sum().alias(f"cmax__{s}"))


            stats = df_pair.select(exprs).row(0, named=True)
            
            for s in sample_cols:
                observed[s] += stats[f"o__{s}"] or 0
                sum_count_min[s] += stats[f"cmin__{s}"] or 0
                sum_count_mean[s] += stats[f"cmean__{s}"] or 0
                sum_count_max[s] += stats[f"cmax__{s}"] or 0

            del df_part, df_hits, df_A, df_B, df_pair


    # return dataframe
    rows = [{
        "strain": strain_name,
        "sample": s,
        "total_singleton_kmers": total_unique_kmers,
        "observed_singleton_kmers": dict_sum[s],
        "singleton_kmer_coverage": dict_sum[s]/total_unique_kmers,
        "singleton_kmer_count_mean": dict_mean[s],
        "singleton_kmer_count_std": dict_std[s],
        "singelton_pairs_total": n_total,
        "singelton_pairs_observed": observed[s],
        "singelton_pairs_coverage": observed[s] / n_total if n_total else 0.0,
        "singelton_pairs_count_mean-min": sum_count_min[s] / n_total if n_total else 0.0,
        "singelton_pairs_count_mean-mean": sum_count_mean[s] / n_total if n_total else 0.0,
        "singelton_pairs_count_mean-max": sum_count_max[s] / n_total if n_total else 0.0,
    } for s in sample_cols]
    return pl.DataFrame(rows)

def get_pair_hits_streaming(df_samples, pairs_glob, kmer_column="#kmer",
                            batch_size=250_000):
    # gets a kmer count by using the lower of the two kmers
    # mean is that count / total kmer pairs
    part_files = sorted(glob.glob(pairs_glob))
    if not part_files:
        raise FileNotFoundError(f"No pair part files matched {pairs_glob}")

    strain_name = re.match(r"(.+)\.inform_kmer_pairs\..+\.parquet", os.path.basename(part_files[0])).group(1)


    sample_cols = [c for c in df_samples.columns if c != kmer_column]
    # get mean and standard deviation of counts per sample from hits
    global_stats = df_samples.select([expr
                                    for s in sample_cols
                                    for expr in [
                                        pl.col(s).mean().alias(f"mean__{s}"),
                                        pl.col(s).std().alias(f"std__{s}"),
                                        (pl.col(s)>0).sum().alias(f"sum__{s}"),
                                    ]
                                ]).row(0, named=True)

    dict_mean = {s: global_stats[f"mean__{s}"] for s in sample_cols}
    dict_std  = {s: global_stats[f"std__{s}"]  for s in sample_cols}
    dict_sum = {s: global_stats[f"sum__{s}"]  for s in sample_cols}
    total_unique_kmers = len(df_samples)



    observed = {s: 0 for s in sample_cols}
    sum_count_min = {s: 0 for s in sample_cols}
    sum_count_mean = {s: 0 for s in sample_cols}
    sum_count_max = {s: 0 for s in sample_cols}

    n_total = 0        # valid pairs: both k-mers tracked in df_samples
    n_raw   = 0        # raw parquet pairs (sanity check)

    for path in part_files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size,
                                     columns=["kmerA", "kmerB"]):
            df_part = pl.from_arrow(batch)
            n_raw += len(df_part)
            if len(df_part) == 0:
                continue

            part_kmers = pl.concat([df_part["kmerA"], df_part["kmerB"]]).unique()
            df_hits = df_samples.filter(pl.col(kmer_column).is_in(part_kmers.implode()))

            df_A = df_hits.rename({kmer_column: "kmerA", **{s: f"{s}__A" for s in sample_cols}})
            df_B = df_hits.rename({kmer_column: "kmerB", **{s: f"{s}__B" for s in sample_cols}})

            df_pair = (
                df_part.select(["kmerA", "kmerB"])
                .join(df_A, on="kmerA", how="inner")
                .join(df_B, on="kmerB", how="inner")
            )

            # only pairs where BOTH k-mers are tracked count toward the denominator
            n_total += len(df_pair)

            # single pass: all samples at once instead of n_samples selects
            exprs = []
            for s in sample_cols:
                a = pl.col(f"{s}__A")
                b = pl.col(f"{s}__B")
                both_nonzero = (a > 0) & (b > 0)

                pc_min  = pl.when(both_nonzero).then(pl.min_horizontal(a, b)).otherwise(0)
                pc_mean = pl.when(both_nonzero).then(pl.mean_horizontal(a, b)).otherwise(0)
                pc_max  = pl.when(both_nonzero).then(pl.max_horizontal(a, b)).otherwise(0)

                exprs.append((both_nonzero).sum().alias(f"o__{s}"))
                exprs.append(pc_min.sum().alias(f"cmin__{s}"))
                exprs.append(pc_mean.sum().alias(f"cmean__{s}"))
                exprs.append(pc_max.sum().alias(f"cmax__{s}"))


            stats = df_pair.select(exprs).row(0, named=True)
            
            for s in sample_cols:
                observed[s] += stats[f"o__{s}"] or 0
                sum_count_min[s] += stats[f"cmin__{s}"] or 0
                sum_count_mean[s] += stats[f"cmean__{s}"] or 0
                sum_count_max[s] += stats[f"cmax__{s}"] or 0

            del df_part, df_hits, df_A, df_B, df_pair


    # return dataframe
    rows = [{
        "strain": strain_name,
        "sample": s,
        "total_individual_kmers": total_unique_kmers,
        "observed_individual_kmers": dict_sum[s],
        "individual_kmer_coverage": dict_sum[s]/total_unique_kmers,
        "individual_kmer_count_mean": dict_mean[s],
        "individual_kmer_count_std": dict_std[s],
        "pairs_total": n_total,
        "pairs_observed": observed[s],
        "pairs_coverage": observed[s] / n_total if n_total else 0.0,
        "pairs_count_mean-min": sum_count_min[s] / n_total if n_total else 0.0,
        "pairs_count_mean-mean": sum_count_mean[s] / n_total if n_total else 0.0,
        "pairs_count_mean-max": sum_count_max[s] / n_total if n_total else 0.0,
    } for s in sample_cols]
    return pl.DataFrame(rows)




def main():
    parser = argparse.ArgumentParser(description='Map scrubbed kmers onto a genome.')
    parser.add_argument('--csv_path', help='a set of kmer counts of pangenomes to create informative kmer pairs from')
    parser.add_argument('--output_dir')
    parser.add_argument('--testmode', action= 'store_true', help = 'Uses a test dataset instead of an input csv')
    parser.add_argument('--threads', type=int, default=None, help='Number of worker processes for triplet generation. Default: os.cpu_count() - 1')
    parser.add_argument('--presence_threshold', type=float, default=6, help='Drop strains who are present in more strains than this')
    parser.add_argument('--basename')
    parser.add_argument('--create_triples', action = 'store_true')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok = True)

    if args.testmode:
        df_indiv_counts, df_global_counts, df_scrub_summary, df_samples = get_test_dataset()
        basename = 'testmode'
        df_no_drugs = df_global_counts.filter((pl.col('drug_count')==0))
        df_inform_singletons = df_no_drugs.filter((pl.col('metagenome_count') == 0 ) & (pl.col('pangenome_count') == 0))        
        df_non_inform_singletons = df_no_drugs.filter(~(pl.col('metagenome_count') == 0 ) & ~(pl.col('pangenome_count') == 0))
        print(df_inform_singletons)

        # export parquet inform kmers
        print('Creating pairs from non informative singletons')
        kmer_pairs_from_presence(df_indiv_counts, df_scrub_summary, 
                                 args.output_dir , 
                                 basename = basename,
                                 df_keep=df_non_inform_singletons,
                                 presence_t = args.presence_threshold, 
                                 similarity_t=None, 
                                 n_workers=args.threads,
                                 testmode = True)
        pair_glob = os.path.join(args.output_dir, f"{basename}.inform_kmer_pairs.part*.parquet")

        unique_kmers = set()
        for path in sorted(glob.glob(pair_glob)):
            # read just one column at a time, dedupe per-part
            df_part_a = pl.read_parquet(path, columns=['kmerA'])
            unique_kmers.update(df_part_a.get_column('kmerA').unique().to_list())
            del df_part_a
            df_part_b = pl.read_parquet(path, columns=['kmerB'])
            unique_kmers.update(df_part_b.get_column('kmerB').unique().to_list())
            del df_part_b
            gc.collect()

        pair_kmers = unique_kmers
        print(f"Pair kmers: {len(pair_kmers):,}")

        # Add pairs from singletons
        singleton_kmers = set(df_inform_singletons["#kmer"].to_list())
        all_kmers = singleton_kmers | pair_kmers
        print(all_kmers)
        # write pairs file
        filtered_path = os.path.join(args.output_dir, f"{basename}.inform_kmer_pairs.pairs.parquet")
        #no selection in pair mode
        sel_pl = pl.Series(sorted(pair_kmers))
        (
            pl.scan_parquet(pair_glob, low_memory=True)
            .filter(pl.col("kmerA").is_in(sel_pl) & pl.col("kmerB").is_in(sel_pl))
            .sink_parquet(filtered_path, compression="zstd")
        )
        # only delete the parts once the selected file exists and is non-empty
        if os.path.exists(filtered_path) and os.path.getsize(filtered_path) > 0:
            parts = glob.glob(pair_glob)
            for p in parts:
                os.remove(p)
            print(f"Wrote {filtered_path} and removed {len(parts)} part files", flush=True)
        else:
            print("WARNING: final parquet missing or empty — keeping part files", flush=True)


        singleton_path, _ = create_pairs_with_singletons(
            singleton_kmers, pair_kmers,
            output_dir=args.output_dir, basename=basename,
            self_singletons=True,
            max_singletons=100
        )
    
    

        # get results
        inform_parts = sorted(glob.glob(os.path.join(args.output_dir, f"{basename}.*.parquet")))
        if inform_parts:
            print('All kmer pairs created')
            print(pl.read_parquet(inform_parts))

        print('Creating coverage outputs')
        df_cov_p = get_pair_hits_streaming(df_samples,os.path.join(args.output_dir, f"{basename}.inform_kmer_pairs.*.parquet"))

        # get pure singleton pairs sxs 
        print(df_inform_singletons)
        inform_parts = sorted(glob.glob(os.path.join(args.output_dir, f"{basename}.singletons.parquet")))
        if inform_parts:
            print('All kmer pairs created')
            print(pl.read_parquet(inform_parts))
        singleton_kmers = df_inform_singletons["#kmer"].to_list()
        df_singles = df_samples.filter(pl.col('#kmer').is_in(singleton_kmers))
        df_cov_s = get_singletons_hits_streaming(df_singles,os.path.join(args.output_dir, f"{basename}.inform_kmer_pairs.singletons.parquet"))
        #merge
        df_cov_sp = df_cov_p.join(df_cov_s, on=["strain", "sample"], how="inner")
        df_cov_sp.write_csv(os.path.join(args.output_dir, f'{basename}.coverage.csv'))
        print(df_cov_sp)


if __name__ == '__main__':
    main()
