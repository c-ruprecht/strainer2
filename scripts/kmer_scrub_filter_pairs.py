#!/usr/bin/env python3
"""Map scrubbed kmers onto a genome and export result dataframes.

Usage:
    python kmer_scrub_filter2.py <genome.fna.gz> <scrubbed_kmers.gz>
    python kmer_scrub_filter2.py <genome.fna.gz> <scrubbed_kmers.gz> --output-dir results/ --basename my_strain --figures
"""
import argparse
import gzip
import os
import sys
import time
import numpy as np
import ahocorasick
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Bio import SeqIO
from Bio.Seq import Seq
import polars as pl
import pandas as pd
import pyarrow as pa
import gc
import glob

import subprocess
import math

from collections import defaultdict
from itertools import combinations

import subprocess
import os
import polars as pl
import shutil
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from itertools import combinations
import os
from collections import defaultdict

from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
import os
import glob

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


_PAIR_SCHEMA = pa.schema([
    ("kmerA", pa.string()),
    ("kmerB", pa.string()),
    ("count", pa.int64()),
])


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
    presence_t=50, similarity_t=None,
    n_workers=None, write_non_inform=False,
):
    # exclusion list from summary
    df = pl.read_csv(summary_tsv, separator='\t')
    if similarity_t is not None:
        df_t = df.filter(pl.col('coverage_pct') < similarity_t)
    else:
        df_t = df.filter(pl.col('is_in_global') == False)
    li_t = df_t.get_column('scrub_id').cast(pl.UInt32).to_list()

    # read & clean presence
    df_presence = (
        pl.scan_csv(presence_tsv, separator='\t')
          .filter(pl.col('list_scrub_id').str.count_matches(',') < presence_t - 1)
          .with_columns(
              pl.col('list_scrub_id')
                .str.split(',')
                .cast(pl.List(pl.UInt32))
                .list.set_difference(li_t)
          )
          .filter(pl.col('list_scrub_id').list.len() > 0)
          #.filter(pl.col('#kmer').is_in(li_kmers))
          .collect(engine='streaming')
    )
    print(df_presence)
    print(f"Presence rows after filtering: {df_presence.shape[0]:,}", flush=True)
    if df_keep is not None:
        df_presence = df_presence.join(df_keep, on='#kmer', how='semi')
        print(f"After kmer subset filter: {df_presence.shape[0]:,}", flush=True)
    
    os.makedirs(output_dir, exist_ok=True)
    return create_disjoint_kmer_pairs_parallel(
        df_presence, output_dir, basename,
        n_workers=n_workers,
        write_non_inform=write_non_inform,
    )

def load_genome(genome_path):
    opener = gzip.open if genome_path.endswith('.gz') else open
    records = {}
    with opener(genome_path, 'rt') as fh:
        for record in SeqIO.parse(fh, 'fasta'):
            records[record.id] = record.seq
    return records

def strain_name_from_path(path):
    base = os.path.basename(path)
    for ext in ('.fna.gz', '.fasta.gz', '.fa.gz', '.fna', '.fasta', '.fa'):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base.split('.')[0]

def get_lowest_percentile(df, percentile=0.05, drug_scrub='percentile'):

    if df['drug_count'].isna().all():
        print('No drug counts found, continuing without drugscrub')
        df['drug_count'] = 0
        
    if drug_scrub == 'percentile':
        lowest = df[
            (df['reference_count'] <= df['reference_count'].quantile(percentile)) &
            (df['pangenome_count'] <= df['pangenome_count'].quantile(percentile)) &
            (df['metagenome_count'] <= df['metagenome_count'].quantile(percentile)) &
            (df['drug_count'] <= df['drug_count'].quantile(percentile))
        ].copy()
    
    if drug_scrub == 'count_hard':
        lowest = df[
            (df['reference_count'] <= df['reference_count'].quantile(percentile)) &
            (df['pangenome_count'] <= df['pangenome_count'].quantile(percentile)) &
            (df['metagenome_count'] <= df['metagenome_count'].quantile(percentile)) &
            (df['drug_count'] == 0)
        ].copy()
    return lowest

def build_mapped_kmers_ahocorasick(records, kmers, terminal_dist):
    # Build Aho-Corasick automaton with forward and reverse complement kmers
    # Important: only pass single-count kmers; only the first hit per kmer is kept
    A = ahocorasick.Automaton()
    for kmer in kmers:
        A.add_word(kmer, (kmer, False))
        A.add_word(str(Seq(kmer).reverse_complement()), (kmer, True))
    A.make_automaton()

    found = set()
    rows = []
    print(A)
    for record_id, seq in records.items():
        for pos, (kmer, is_rc) in A.iter(str(seq)):
            if kmer not in found:
                rows.append((record_id, kmer, pos - len(kmer) + 1, is_rc))
                found.add(kmer)

    df = pd.DataFrame(rows, columns=['contig_id', '#kmer', 'kmer_position', 'reverse_complement'])

    if len(df) < len(kmers):
        print('WARNING: not all kmers found in genome')
    elif len(df) > len(kmers):
        print('WARNING: kmers found more than once')
    else:
        print(f'  {len(df)} kmers mapped (all unique)')

    dict_len = {cid: len(seq) for cid, seq in records.items()}
    df['contig_length'] = df['contig_id'].map(dict_len)
    df['terminal_kmer'] = (
        (df['kmer_position'] < terminal_dist) |
        ((df['contig_length'] - df['kmer_position']) < terminal_dist)
    )
    df['label'] = df['terminal_kmer'].map({True: 'terminal', False: 'internal'})

    n_terminal = int(df['terminal_kmer'].sum())
    print(f'  Terminal kmers: {n_terminal} / {len(df)}')

    return df, dict_len


def assign_mapping_bin(df, bin_size):
    li_dfs = []
    for contig in df['contig_id'].unique():
        df_contig = df.loc[df['contig_id'] == contig].copy()
        df_contig['bin'] = (df_contig['kmer_position'] // bin_size) * bin_size
        li_dfs.append(df_contig)
    return pd.concat(li_dfs)


def smooth_downsample(df, total_target, bin_size, mode = None):
    """Downsample df so the total selected kmers equals total_target,
    with each contig's share proportional to its length.
    A global bin cap is computed as the bin_percentile quantile of bin counts across
    all contigs combined, then each contig's bins are smoothed down to that cap.
    The contig is further sampled down to its proportional share if still over.
    Removes terminal kmers to avoid bad assembly regions.
    """
    
    if mode == 'independent':
        
        kmer_gap = 31
        df = assign_mapping_bin(df.loc[df['terminal_kmer'] == False], bin_size)
        # drop all non ATCG in kmers
        df = df.loc[df['#kmer'].str.fullmatch(r'[ACGT]+')].copy()
        total_genome_length = df.groupby('contig_id')['contig_length'].first().sum()

        contig_results = []
        for contig_id, contig_df in df.groupby('contig_id'):

            contig_length = contig_df['contig_length'].iloc[0]
            contig_cap = max(1, int(total_target * contig_length / total_genome_length))
            print(f'Contig length: {contig_length}, max allowed kmers: {contig_cap}' )
            current_total = len(contig_df)
            excess = current_total - contig_cap
            print(f"Available Kmers on contig: {current_total}")
            print('Excess kmers on contig: ' + str(excess))
            
            #sort dataframe by position and counts
            sort_df = contig_df.sort_values(['bin','drug_count', 'pangenome_count', 'metagenome_count'])
            contig_result = sort_df.drop_duplicates('bin', keep = 'first')
            # need a check to see if kmers overlap anyway
            li_drop = []
            for pos_i, (_, row) in enumerate(contig_result.iterrows()):
                if row['reverse_complement'] == True:
                    if pos_i == 0:
                        continue
                    else:
                        neighbor = contig_result.iloc[pos_i-1]
                        distance = row['kmer_position'] - neighbor['kmer_position']
                        if neighbor['reverse_complement'] == True:
                            req_distance = 31
                        if neighbor['reverse_complement'] == False:
                            req_distance = 62
                        if distance < req_distance and distance > 0:
                            print('kmers are too close')
                            print('drop worse kmer: ')
                            pair =  contig_result.iloc[pos_i-1:pos_i+1]
                            drop_position = pair.sort_values(['drug_count', 'pangenome_count', 'metagenome_count'], ascending = False).iloc[0]['kmer_position']
                            print(drop_position)
                            li_drop.append(drop_position)
                if row['reverse_complement'] == False:
                    if pos_i == len(contig_result)-1:
                        continue
                    else:
                        neighbor = contig_result.iloc[pos_i+1]
                        distance = neighbor['kmer_position'] - row['kmer_position']
                        if neighbor['reverse_complement'] == True:
                            req_distance = 62
                        if neighbor['reverse_complement'] == False:
                            req_distance = 31
                        if distance < req_distance:
                            print('kmers are too close')
                            pair = contig_result.iloc[pos_i:pos_i+2]
                            drop_position = pair.sort_values(['drug_count', 'pangenome_count', 'metagenome_count'],ascending=False).iloc[0]['kmer_position']
                            print(f'drop worse kmer at position: {drop_position}')
                            li_drop.append(drop_position)

            print(f'Found too close kmers, dropped: {len(li_drop)}')
            contig_result = contig_result.loc[contig_result['kmer_position'].isin(li_drop) == False].copy()
            # if over contig cap, trim more common kmers until its hit
            if len(contig_result) > contig_cap:
                print('More rare kmers than contig cap allows')
                n_remove = len(contig_result) - contig_cap
                print(f'fremoving additional kmers: {n_remove}')
                
                contig_result = (contig_result.sort_values(['drug_count', 'pangenome_count', 'metagenome_count'], ascending = True)
                                 .iloc[:contig_cap])

            contig_results.append(contig_result)
        result = pd.concat(contig_results)
        print(f'  Total: {len(result)} after independent scrub')
        return result.sort_values(['contig_id', 'kmer_position'])


    else:
        df = assign_mapping_bin(df.loc[df['terminal_kmer'] == False], bin_size)

        bin_counts = df.groupby(['contig_id', 'bin']).size()
        #global_bin_cap = int(bin_counts.mean())
        #mean_bin_count_genome = bin_counts.groupby('contig_id').mean()

        bin_counts = bin_counts.reset_index()
        bin_counts = bin_counts.rename(columns = {0: 'size'})
        #bin_counts['to_scrub'] = bin_counts['size'] - global_bin_cap
        #print(bin_counts.sort_values(['to_scrub'],ascending = False))

        #print(bin_counts, global_bin_cap,mean_bin_count_genome)
        
        #print(f'  Global mean bin cap: {global_bin_cap}')

        total_genome_length = df.groupby('contig_id')['contig_length'].first().sum()

        contig_results = []
        for contig_id, contig_df in df.groupby('contig_id'):
            contig_length = contig_df['contig_length'].iloc[0]
            contig_cap = max(1, int(total_target * contig_length / total_genome_length))
            print(contig_cap)

            #df_scrub = bin_counts.loc[(bin_counts['contig_id']==contig_id) & 
            #                          (bin_counts['to_scrub']>0)]
            #print('counts for kmers not to be scrubbed')

            # HERE YOU NEED TO ACTUALLY ALSO GRAB NOT ONLY FROM THE OVERREPRESENTED ONES!
            #print(str(bin_counts.loc[(bin_counts['contig_id']==contig_id) & 
            #                          (bin_counts['to_scrub']<0)]['size'].sum()))
            
            current_total = len(contig_df)
            excess = current_total - contig_cap
            df_scrub = bin_counts.loc[bin_counts["contig_id"]==contig_id]
            
            #df_scrub['n_remove'] = (df_scrub['to_scrub'] / df_scrub['to_scrub'].sum() * excess).astype(int)
            
            #proportionally remove the excess from size
            df_scrub['n_remove'] = (df_scrub['size'] / df_scrub['size'].sum() * excess).astype(int)

            # cap so we never remove more than what's scrubable
            #df_scrub['n_remove'] = df_scrub['n_remove'].clip(upper=df_scrub['to_scrub'])

            # fix rounding remainder — assign to largest bins first
            remainder = excess - df_scrub['n_remove'].sum()
            if remainder > 0:
                largest = df_scrub.nlargest(remainder, 'size').index
                df_scrub.loc[largest, 'n_remove'] += 1

            # what each bin keeps after removal
            df_scrub['n_keep'] = df_scrub['size'] - df_scrub['n_remove']
            print(df_scrub.sort_values(['n_remove'], ascending = False))
            print('current contig kmers:' + str(current_total))
            print('Excess kmers in contig: ' + str(excess))
            #print('potential scrubs: ' + str(df_scrub['to_scrub'].sum()))
            print('total kmers that will be filtered: ' + str(df_scrub['n_remove'].sum()))
            # get all contig bins above global_bin_cap
            scrub_map = dict(zip(df_scrub['bin'], df_scrub['n_keep']))

            contig_keep = []
            for bin_name, group in contig_df.groupby('bin'):
                if bin_name in scrub_map:
                    n_in_bin = len(group)
                    n_keep = scrub_map[bin_name]
                    if n_keep < n_in_bin:
                        contig_keep.append(
                            group.sort_values('kmer_position').iloc[
                                np.linspace(0, n_in_bin - 1, n_keep, dtype=int)
                            ]
                        )
                    else:
                        contig_keep.append(group)
                else:
                    contig_keep.append(group)

            contig_result = pd.concat(contig_keep)

            if len(contig_result) > contig_cap:
                print('random sample')
                contig_result = contig_result.sample(n=contig_cap)

            contig_results.append(contig_result)
            print(f'  {contig_id}: {len(contig_df)} -> {len(contig_result)} kmers (contig cap: {contig_cap})')

        result = pd.concat(contig_results)
        print(f'  Total: {len(df)} -> {len(result)} kmers after smooth downsampling')
        return result.sort_values(['contig_id', 'kmer_position'])
    
def make_inform_kmers_independent(df, type = 'singleton'):
    
    if type == 'singleton':
        df = assign_mapping_bin(df.loc[df['terminal_kmer'] == False], 31)

        dict_drop = {}
        for contig_id, contig_df in df.groupby('contig_id'):
            contig_df = contig_df.sort_values(['bin','drug_count', 'pangenome_count', 'metagenome_count'])
            contig_df = contig_df.drop_duplicates('bin', keep = 'first')
            contig_df = contig_df.sort_values('kmer_position', ascending = True)
            li_drop = []
            print(contig_df[['#kmer', 'kmer_position', 'reverse_complement']])
            for pos_i, (_, row) in enumerate(contig_df.iterrows()):
                if row['reverse_complement'] == True:
                    if pos_i == 0:
                        continue
                    else:
                        neighbor = contig_df.iloc[pos_i-1]
                        distance = row['kmer_position'] - neighbor['kmer_position']
                        if neighbor['reverse_complement'] == True:
                            req_distance = 31
                        if neighbor['reverse_complement'] == False:
                            req_distance = 62
                        if distance < req_distance and distance > 0:
                            pair =  contig_df.iloc[pos_i-1:pos_i+1]
                            drop_position = pair.sort_values(['drug_count', 'pangenome_count', 'metagenome_count'], ascending = False).iloc[0]['kmer_position']
                            li_drop.append(drop_position)
                
                if row['reverse_complement'] == False:
                    if pos_i == len(contig_df)-1:
                        continue
                    else:
                        neighbor = contig_df.iloc[pos_i+1]
                        distance = neighbor['kmer_position'] - row['kmer_position']
                        if neighbor['reverse_complement'] == True:
                            req_distance = 62
                        if neighbor['reverse_complement'] == False:
                            req_distance = 31
                        if distance < req_distance:
                            pair = contig_df.iloc[pos_i:pos_i+2]
                            # for singletons this is always 0 so whats the point here? could do some other score like jaccard from rest?
                            drop_position = pair.sort_values(['drug_count', 'pangenome_count', 'metagenome_count'],ascending=False).iloc[0]['kmer_position']
                            li_drop.append(drop_position)
            dict_drop[contig_id] = li_drop
            print(f'Found too close kmers, on {contig_id}, drop: {len(li_drop)}')
        return dict_drop
    

def plot_genome_bins(df, df_smooth, basename, bin_size, output_dir, map_only = False):
    if map_only:
        df=df.copy()
        df.sort_values(['contig_length', 'contig_id', 'kmer_position'], inplace=True)
        df['kmer_count'] = 1
        df['bin'] = (df['kmer_position'] // bin_size) * bin_size

        contigs = df['contig_id'].unique()
        plot_dir = os.path.join(output_dir, 'contig_plots')
        os.makedirs(plot_dir, exist_ok=True)

        for contig in contigs:
            df_contig = df.loc[df['contig_id'] == contig]
            binned_all = df_contig.groupby('bin')['kmer_count'].sum().reset_index()
            binned_all = binned_all[binned_all['kmer_count'] > 0]

            y_max = binned_all['kmer_count'].max() if len(binned_all) else 1

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=binned_all['bin'], y=binned_all['kmer_count'],
                mode='markers', name='all rare kmers',
                marker=dict(color=px.colors.qualitative.D3[0], size=3),
            ))
            
            fig.update_xaxes(title_text='position (bp)')
            fig.update_yaxes(showline=True, showticklabels=True, range=[0, y_max * 1.05])
            fig.update_layout(
                title_text=f'{basename} — {contig}',
                height=400,
                width=800,
                template='simple_white',
            )
            safe_contig = contig.replace('/', '_').replace(' ', '_')
            fig.write_image(os.path.join(plot_dir, f'{basename}.{safe_contig}.svg'))
    else:
        df = df.copy()
        df.sort_values(['contig_length', 'contig_id', 'kmer_position'], inplace=True)
        df['kmer_count'] = 1
        df['bin'] = (df['kmer_position'] // bin_size) * bin_size

        df_smooth = df_smooth.copy()
        df_smooth['kmer_count'] = 1
        df_smooth['bin'] = (df_smooth['kmer_position'] // bin_size) * bin_size

        contigs = df['contig_id'].unique()
        plot_dir = os.path.join(output_dir, 'contig_plots')
        os.makedirs(plot_dir, exist_ok=True)

        for contig in contigs:
            df_contig = df.loc[df['contig_id'] == contig]
            binned_all = df_contig.groupby('bin')['kmer_count'].sum().reset_index()
            binned_all = binned_all[binned_all['kmer_count'] > 0]

            df_contig_smooth = df_smooth.loc[df_smooth['contig_id'] == contig]
            binned_smooth = df_contig_smooth.groupby('bin')['kmer_count'].sum().reset_index()
            binned_smooth = binned_smooth[binned_smooth['kmer_count'] > 0]

            y_max = binned_all['kmer_count'].max() if len(binned_all) else 1

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=binned_all['bin'], y=binned_all['kmer_count'],
                mode='markers', name='all rare kmers',
                marker=dict(color=px.colors.qualitative.D3[0], size=3),
            ))
            fig.add_trace(go.Scatter(
                x=binned_smooth['bin'], y=binned_smooth['kmer_count'],
                mode='markers', name='selected kmers',
                marker=dict(color=px.colors.qualitative.D3[1], size=3),
            ))
            fig.update_xaxes(title_text='position (bp)')
            fig.update_yaxes(showline=True, showticklabels=True, range=[0, y_max * 1.05])
            fig.update_layout(
                title_text=f'{basename} — {contig}',
                height=400,
                width=800,
                template='simple_white',
            )
            safe_contig = contig.replace('/', '_').replace(' ', '_')
            fig.write_image(os.path.join(plot_dir, f'{basename}.{safe_contig}.svg'))

def plot_kmer_counts(lowest_pct):
    df_plot = lowest_pct.sort_values(['pangenome_count', 'metagenome_count'], ascending=True).reset_index(drop=True).reset_index()
    df_plot_stack = df_plot.set_index(['index', '#kmer']).stack().reset_index()
    df_plot_stack = df_plot_stack.rename(columns={'level_2': 'scrub_type', 0: 'value'})
    df_plot_stack = df_plot_stack.loc[~df_plot_stack['scrub_type'].str.contains('freq')]
    fig = px.line(df_plot_stack,
                  x='index',
                  y='value',
                  #log_y=True,
                  template='simple_white',
                  color='scrub_type',
                  title='rare kmers by count')
    fig.update_yaxes(title_text='')
    return fig

def plot_box_coverage(df_lowest, df_smooth, basename, bin_size, map_only = False):
    if map_only:
        df = df_lowest
        df.sort_values(['contig_length', 'contig_id', 'kmer_position'], inplace=True)
        df['kmer_count'] = 1

        df['bin'] = (df['kmer_position'] // bin_size) * bin_size
        binned = df.groupby(['contig_id','bin'])['kmer_count'].sum().reset_index()

        fig = px.box(binned,
                    x = 'contig_id',
                    y = 'kmer_count',
                    #color = 'stage',
                    points = 'all',
                    template = 'simple_white',
                    title = basename,
                    width = 800,
                    height = 600)
    else:
        df_lowest['stage'] = 'pre_smooth'
        df_smooth['stage'] = 'post_smooth'
        print(df_lowest)
        print(df_smooth)

        df = pd.concat([df_lowest, df_smooth])


        df.sort_values(['contig_length', 'contig_id', 'kmer_position'], inplace=True)
        df['kmer_count'] = 1

        df['bin'] = (df['kmer_position'] // bin_size) * bin_size
        binned = df.groupby(['stage','contig_id','bin'])['kmer_count'].sum().reset_index()

        fig = px.box(binned,
                    x = 'contig_id',
                    y = 'kmer_count',
                    color = 'stage',
                    points = 'all',
                    template = 'simple_white',
                    title = basename,
                    width = 800,
                    height = 600)
    return fig

def main():
    parser = argparse.ArgumentParser(description='Map scrubbed kmers onto a genome.')
    parser.add_argument('--genome', help='Genome FASTA file (.fna or .fna.gz)')
    parser.add_argument('--counts_global', help='Either a kmer counts file or a scrubbed kmers file if map_scrubbed_kmers')
    parser.add_argument('--counts_individual', help='Either a kmer counts file or a scrubbed kmers file if map_scrubbed_kmers')
    parser.add_argument('--counts_summary')
    parser.add_argument('--output-dir', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--basename', default=None, help='Output basename (default: derived from genome filename)')
    parser.add_argument('--figures', action='store_true', default=False,
                        help='Save figures as SVG (default: False)')
    parser.add_argument('--threads', type = int)
    parser.add_argument('--percentage', type=float, default=0.01,
                        help='Percentile threshold for rare kmer selection (default: 0.05)')
    parser.add_argument('--percentile_union', type = float, default = 0.05, help = 'percentile passed for union of different kmer scrubs')
    parser.add_argument('--bin-size', type=int, default=1000, help='Bin size in bp for kmer density smoothing (default: 1000)')
    parser.add_argument('--terminal-dist', type=int, default=300,  help='Distance from contig ends to flag terminal kmers (default: 300)')
    parser.add_argument('--map_scrubbed_kmers_only', action='store_true', help = 'Takes a file of rare kmers as a list, one kmer per line that will be mapped to a target genome')
    parser.add_argument('--independent', action='store_true', help = 'reduces bin size to 31 to and only allows 1 kmer per bin')
    parser.add_argument('--force', action='store_true',
                        help='Recompute outputs even if they already exist')
    args = parser.parse_args()
    if args.map_scrubbed_kmers_only:
        strain = strain_name_from_path(args.genome)
        basename = args.basename if args.basename else strain
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'Loading genome: {args.genome}')
        records = load_genome(args.genome)
        print(f'  {len(records)} contigs')
        df_counts = pd.read_csv(args.kmer_counts, sep='\t', header = None)
        #print(df_counts)
        print(f'Total scrubbed kmers to map: {len(df_counts)}')
        kmers = df_counts[0].to_list()
        df, _ = build_mapped_kmers_ahocorasick(records, kmers, terminal_dist=args.terminal_dist)
        if args.figures:
            plot_genome_bins(df, df, basename, bin_size=args.bin_size, output_dir=args.output_dir, map_only = True )
            
            fig_bins2 = plot_box_coverage(df, df, basename, bin_size=args.bin_size, map_only=True)
            fig_bins2.write_image(os.path.join(args.output_dir, f'{basename}.box_genome_bins.svg'))

        df.to_csv(os.path.join(args.output_dir, f'{basename}.rare_kmers_mapped.tsv.gz'),
                        sep='\t', index=False, compression='gzip')
    else:
        
        strain = strain_name_from_path(args.genome)
        basename = args.basename if args.basename else strain
        os.makedirs(args.output_dir, exist_ok=True)
        

        # switch to scan csv for memory efficiency
        df_global_counts = pl.read_csv(args.counts_global, 
                                       separator= '\t', 
                                       schema_overrides={'reference_count': pl.UInt32,
                                                        'pangenome_count': pl.UInt32,
                                                        'metagenome_count': pl.UInt32,
                                                        'drug_count': pl.UInt32,}
                                        )
        print(df_global_counts)
        total_kmers = len(df_global_counts)

        

        
        print(f'Total kmers: {len(df_global_counts)}')
        print('Remove kmers with count >1 from ref genome:')
        df_global_counts = df_global_counts.filter(pl.col("reference_count") == 1)
        print(f'Remaining kmers: {len(df_global_counts)}')
        print('Removing all kmers present in drug scrub:')
        df_no_drugs = df_global_counts.filter(pl.col("drug_count") == pl.col("drug_count").min())
        print(f'Remaining kmers: {len(df_no_drugs)}')

        # you can then just grab the 0 counts here much faster
        print('Getting all kmers with counts')

        #mapping positions
        print('Mapping positions of remaining kmers')
        # Mapping kmers
        print(f'Loading genome: {args.genome}')
        records = load_genome(args.genome)
        

        # merge positions and counts
        n_targets = total_kmers * args.percentage 
        

        df_inform_singletons = df_no_drugs.filter((pl.col('metagenome_count') == 0 ) & (pl.col('pangenome_count') == 0))
        df_non_inform_singletons = df_no_drugs.filter(~(pl.col('metagenome_count') == 0 ) & ~(pl.col('pangenome_count') == 0))

        print(df_inform_singletons)

        # singletons should also be made independent!
        single_kmers = df_inform_singletons.get_column('#kmer').to_list()
        df_loc_singles, _ = build_mapped_kmers_ahocorasick(records, single_kmers, terminal_dist=args.terminal_dist)
        #df_loc_singles = pd.merge(df_inform_singletons.to_pandas(), df_locations, on = '#kmer', how = 'left')
        
        print(df_loc_singles)
        # for now just to test independence
        #dict_drop = make_inform_kmers_independent(df_loc_singles, type = 'singleton')
        #df_inform_singletons.filter()write_parquet(args.output_dir+f'{basename}.inform_kmer_singleton.parquet')

        # export parquet inform kmers
        li_kmers = df_non_inform_singletons.get_column('#kmer').to_list()
        print('Creating pairs from non informative singletons')
        print(df_non_inform_singletons)

        
        kmer_pairs_from_presence(args.counts_individual, args.counts_summary, args.output_dir , basename = basename,
                                 df_keep=df_non_inform_singletons,
                                 presence_t = 20, similarity_t=None,n_workers=args.threads)

        # Count how often each kmer is in a pair
        # make them independent

        # start finding pairs
        
        ### Build a greedy selection for n_target percentage
        #   maximal kmers to track 1% - singletons?
        #   keep better scored pairs, how to score: 
                #location: diff contig > same contig > distance from each other to maximize
        #   Start point just dataframe sorted by counts union?

       

        # singletons
        singleton_kmers = set(df_inform_singletons["#kmer"].to_list())

        # pair parts → unique kmers across both columns, computed lazily
        pair_glob = os.path.join(args.output_dir, f"{basename}.inform_kmer_pairs.part*.parquet")

        pair_kmers = set(
            pl.scan_parquet(pair_glob)
            .select(pl.concat([pl.col('kmerA'), pl.col('kmerB')]).unique().alias('#kmer'))
            .collect(engine='streaming')
            .get_column('#kmer')
            .to_list()
        )

        all_kmers = singleton_kmers | pair_kmers
        print(f"Singletons: {len(singleton_kmers):,}")
        print(f"Pair kmers: {len(pair_kmers):,}")
        print(f"Total kmers for strain_detect: {len(all_kmers):,}")
        df_locations ,_ = build_mapped_kmers_ahocorasick(records, all_kmers, terminal_dist=args.terminal_dist)

        #annotate kmer
        # Annotate source
        df_locations["from_singleton"] = df_locations["#kmer"].isin(singleton_kmers)
        df_locations["from_pair"] = df_locations["#kmer"].isin(pair_kmers)
        df_locations["source"] = df_locations.apply(
            lambda r: "both" if r["from_singleton"] and r["from_pair"]
                    else "singleton" if r["from_singleton"]
                    else "pair",
            axis=1,
        )      
        df_locations.to_csv(os.path.join(args.output_dir, f'{basename}.scrubbed_kmers.tsv.gz'),
                                         sep='\t', index=False)
        df_locations[['#kmer']].to_csv(os.path.join(args.output_dir, f'{basename}.scrubbed_kmers'),
                                sep='\t', index=False, header=None)

if __name__ == '__main__':
    main()
