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
import os
import pyarrow as pa
import pyarrow.parquet as pq
import re
import random
import subprocess
import math

from collections import defaultdict
from itertools import combinations, product

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
from kmer_pairs import *


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


def find_overlap_kmer_fast(df, max=0.8, chunk_size=10_000):
    dict_overlap = {}
    same_thresh = 31 * max
    cross_lo    = 31 * (1 - max)
    cross_hi    = 62 * max

    for contig_id, contig_df in df.groupby('contig_id'):
        kmers     = contig_df['#kmer'].values
        positions = contig_df['kmer_position'].values
        is_rc     = contig_df['reverse_complement'].values.astype(bool)
        n = len(kmers)
        
        overlap_lists = [[] for _ in range(n)]

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            
            # dist[i,j] = pos[j] - pos[i], i in chunk, j in all
            dist = positions[None, :] - positions[start:end, None]  # (chunk, n)
            rc_i = is_rc[start:end, None]
            rc_j = is_rc[None, :]

            same_strand = (rc_i == rc_j) & (np.abs(dist) < same_thresh)
            fwd_rc = (~rc_i) &  rc_j & (dist >  cross_lo) & (dist <  cross_hi)
            rc_fwd =   rc_i  & (~rc_j) & (dist > -cross_hi) & (dist < -cross_lo)

            overlap_matrix = same_strand | fwd_rc | rc_fwd
            # zero out self
            for local_i in range(end - start):
                overlap_matrix[local_i, start + local_i] = False

            for local_i in range(end - start):
                overlap_lists[start + local_i] = kmers[overlap_matrix[local_i]].tolist()

        for i in range(n):
            dict_overlap[kmers[i]] = overlap_lists[i]

    return dict_overlap

def find_overlap_kmer(df, max = 0.8):
    
    dict_overlap = {} 

    for contig_id, contig_df in df.groupby('contig_id'):

        for pos_i, (_, row) in enumerate(contig_df.iterrows()):
            if row['reverse_complement'] == True:
                overlap_df = contig_df.loc[((contig_df['reverse_complement'] == True ) & 
                                            (contig_df['kmer_position'] - row['kmer_position']  < 31 * max) & (contig_df['kmer_position'] - row['kmer_position']  > -31 * max))|
                                            ((contig_df['reverse_complement'] == False ) & 
                                             (row['kmer_position'] - contig_df['kmer_position'] < 62 * max) & (row['kmer_position'] - contig_df['kmer_position'] > 0 + 31*(1-max)))]
                #remove self kmer
                overlap_df = overlap_df.loc[overlap_df['#kmer'] != row['#kmer']]
                dict_overlap[row['#kmer']] = overlap_df ['#kmer'].to_list()
             
            if row['reverse_complement'] == False:
                overlap_df = contig_df.loc[((contig_df['reverse_complement'] == True ) & 
                                            ((contig_df['kmer_position'] - row['kmer_position'] < 62 * max) & (contig_df['kmer_position'] - row['kmer_position'] > 0 + 31*(1-max))))|
                                            ((contig_df['reverse_complement'] == False ) &
                                            ((contig_df['kmer_position'] -  row['kmer_position'] < 31 * max) & (contig_df['kmer_position'] -  row['kmer_position'] > -31 * max)))]
                
                #remove self kmer
                overlap_df = overlap_df.loc[overlap_df['#kmer'] != row['#kmer']]
                dict_overlap[row['#kmer']] = overlap_df ['#kmer'].to_list()

    return dict_overlap
def max_independent_kmers_greedy_heap(dict_overlap):
    import heapq
    
    # original one-directional degree BEFORE symmetry expansion
    original_degree = {k: len(v) for k, v in dict_overlap.items()}
    
    # expand to symmetric adj
    adj = {k: set(v) for k, v in dict_overlap.items()}
    for k, nbrs in list(adj.items()):
        for nb in nbrs:
            adj.setdefault(nb, set()).add(k)

    degree = {k: len(v) for k, v in adj.items()}  # live symmetric degree

    counter = 0
    heap = []
    for k in dict_overlap.keys():
        heapq.heappush(heap, (degree[k], original_degree[k], counter, k))
        counter += 1

    selected = []
    excluded = set()

    while heap:
        d, od, _, node = heapq.heappop(heap)

        if node in excluded:
            continue
        if d != degree[node]:
            heapq.heappush(heap, (degree[node], original_degree[node], counter, node))
            counter += 1
            continue

        selected.append(node)
        excluded.add(node)

        for nb in adj[node]:
            if nb in excluded:
                continue
            excluded.add(nb)
            for nb2 in adj[nb]:
                if nb2 not in excluded:
                    degree[nb2] -= 1
                    heapq.heappush(heap, (degree[nb2], original_degree[nb2], counter, nb2))
                    counter += 1

    return selected

def max_independent_kmers_greedy(dict_overlap):
    """Greedy minimum-degree independent set from an overlap dict.
    dict_overlap: {kmer: [overlapping_kmers]}. Returns a list of selected kmers."""

    adj = {k: set(v) for k, v in dict_overlap.items()}
    for k, nbrs in list(adj.items()):
        for nb in nbrs:
            adj.setdefault(nb, set()).add(k)

    selected = []
    remaining = set(adj)
    while remaining:
        # pick node with fewest remaining neighbors
        node = min(remaining, key=lambda k: len(adj[k] & remaining))
        selected.append(node)
        # remove node and all its neighbors from contention
        remaining.discard(node)
        remaining -= adj[node]
    return selected


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
    parser.add_argument('--figures', action='store_true', default=False, help='Save figures as SVG (default: False)')
    parser.add_argument('--threads', type = int,default = 32)
    parser.add_argument('--presence_t', type = int, help = 'maximal presence threshold for pair generation' ,default = 10)
    parser.add_argument('--pair_mode', type = str, default = "sxp", help = ' Either set to "sxs" for including singeltons x singletons pair generation or "sxp"')
    parser.add_argument('--percentage', type=float, default=0.01,
                        help='Percentile threshold for rare kmer selection (default: 0.05)')
    parser.add_argument('--percentile_union', type = float, default = 0.05, help = 'percentile passed for union of different kmer scrubs')
    parser.add_argument('--bin-size', type=int, default=1000, help='Bin size in bp for kmer density smoothing (default: 1000)')
    parser.add_argument('--terminal-dist', type=int, default=300,  help='Distance from contig ends to flag terminal kmers (default: 300)')
    parser.add_argument('--map_scrubbed_kmers_only', action='store_true', help = 'Takes a file of rare kmers as a list, one kmer per line that will be mapped to a target genome')
    parser.add_argument('--independent', action='store_true', help = 'reduces bin size to 31 to and only allows 1 kmer per bin')
    parser.add_argument('--force', action='store_true', help='Recompute outputs even if they already exist')
    #classic strainer to pairs arguments
    parser.add_argument('--create_classic_pairs', action='store_true', help = 'Takes a file of rare kmers as a list, one kmer per line that will be mapped to a target genome')
    parser.add_argument('--scrubbed_kmers' , help = 'Output of kmer_scrub_filter.py redirected to file.')
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
    
    # add entry point for classic strainer kmers
    if args.create_classic_pairs:
        
        if args.genome:
            strain = strain_name_from_path(args.genome)
        basename = args.basename if args.basename else strain
        os.makedirs(args.output_dir, exist_ok=True)

        # create histogram plot for scrub db
        df_hist = pd.read_csv(args.counts_summary, sep='\t')
        fig = px.histogram(
            df_hist,
            x='coverage_pct',
            log_y = True,
            color = 'sample_type',
            histfunc = 'count',
            template='simple_white',
            title=f'{basename} — coverage_pct distribution',
            range_x = [-0.01,1.1],
            )
        fig.add_vline(x=0.96, line_width=3, line_dash="dash", line_color="grey")
        fig.update_layout(width=800, height=500)
        fig.write_image(os.path.join(args.output_dir, f'{basename}.histogram_scrub_db.svg'))

        print('Creating pairs from classic kmer scrub count file')
        # read scrubbed kmers
        li_kmer_scrubs = [l.strip() for l in open(args.scrubbed_kmers) if l.strip() and not l.startswith("#")]
        print(f'Scrubbed kmer counts: {len(li_kmer_scrubs)}')

        #
        df_global_counts = pl.read_csv(args.counts_global, 
                                       separator= '\t', 
                                       schema_overrides={'reference_count': pl.UInt32,
                                                        'pangenome_count': pl.UInt32,
                                                        'metagenome_count': pl.UInt32,
                                                        'drug_count': pl.UInt32,}
                                        )
        # Create waterfall plot scrub
        df_gl = (df_global_counts.to_pandas().drop(columns=['reference_count']).set_index('#kmer'))
        row_sums = df_gl.sum(axis=1)
        count_hist = (row_sums.value_counts()
              .sort_index()
              .rename_axis('total_count')
              .reset_index(name='n_kmers'))
        
        print(count_hist)
        fig = px.scatter(count_hist,
                     y = 'n_kmers',
                     x = 'total_count',
                     log_y = True,
                     template = 'simple_white',
                     range_x  = [-0.1, 5000],
                     title = f'{basename}')
        fig.add_hline(y = count_hist.loc[count_hist['total_count']==0]['n_kmers'][0],line_width=3, line_dash="dash", line_color="grey")
        fig.write_image(os.path.join(args.output_dir, f'{basename}.scrub_counts.svg'))

        # create inform singletons from scrubbed files
        df_scrub_kmers = df_global_counts.filter(pl.col("#kmer").is_in(li_kmer_scrubs))
        print(df_scrub_kmers)

        
        df_inform_singletons = df_scrub_kmers.filter((pl.col('metagenome_count') == 0 ) & (pl.col('pangenome_count') == 0))        
        df_non_inform_singletons = df_scrub_kmers.filter(~((pl.col('metagenome_count') == 0) & (pl.col('pangenome_count') == 0)))

        print(df_inform_singletons)
        print(df_non_inform_singletons)

        # export parquet inform kmers
        print('Creating pairs from non informative singletons')
        kmer_pairs_from_presence(args.counts_individual, args.counts_summary, 
                                 args.output_dir , 
                                 basename = basename,
                                 df_keep=df_non_inform_singletons,
                                 presence_t = 1000, # set to just make pairs whereever possible
                                 similarity_t=None, 
                                 n_workers=args.threads,
                                 max_for_pairs = 100000)
        
        # pair parts → unique kmers across both columns, computed lazily
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
        singleton_kmers = set(df_inform_singletons["#kmer"].to_list())

        print(f"Pair kmers: {len(pair_kmers)}")
        # create pairs without regard for file sizes
        if args.pair_mode == "sxs":
            singleton_path, _ = create_pairs_with_singletons(singleton_kmers, pair_kmers,
                                                            output_dir=args.output_dir, basename=basename,
                                                            self_singletons=True,
                                                            max_singletons = 100000)
        if args.pair_mode == "sxp":
            singleton_path, _ = create_pairs_with_singletons(singleton_kmers, pair_kmers,
                                                            output_dir=args.output_dir, basename=basename,
                                                            self_singletons=False,
                                                            max_singletons = 100000)
        # Cleaning files
        pair_parts = sorted(glob.glob(pair_glob))
        filtered_path = os.path.join(args.output_dir, f"{basename}.inform_kmer_pairs.pairs.parquet")
        if pair_parts:
            (pl.scan_parquet(pair_parts)          # pass the expanded list, not the glob string
            .sink_parquet(filtered_path, compression="zstd"))
        else:
            #write an empty file so downstream readers still find it
            pl.DataFrame(schema={"kmerA": pl.Utf8,"kmerB": pl.Utf8,"count": pl.Int64}).write_parquet(filtered_path, compression="zstd")
        
        # only delete the parts once the selected file exists and is non-empty
        if os.path.exists(filtered_path) and os.path.getsize(filtered_path) > 0:
            parts = glob.glob(pair_glob)
            for p in parts:
                os.remove(p)
            print(f"Wrote {filtered_path} and removed {len(parts)} part files", flush=True)
        else:
            print("WARNING: final parquet missing or empty — keeping part files", flush=True)
        
        # map and export
        all_kmers = singleton_kmers | pair_kmers
        if args.genome:
            records = load_genome(args.genome)
            df_locations ,_ = build_mapped_kmers_ahocorasick(records, all_kmers, terminal_dist = args.terminal_dist)
            df_locations['origin'] = df_locations['#kmer'].apply(lambda x: 'singleton' if x in singleton_kmers else 'pair')
            df_locations.to_csv(os.path.join(args.output_dir, f'{basename}.rare_kmers_mapped.tsv.gz'), sep='\t', index=False, compression='gzip')
        else:
            pd.DataFrame(sorted(all_kmers), columns=['#kmer']).to_csv(os.path.join(args.output_dir, f'{basename}.rare_kmers_mapped.tsv.gz'), sep='\t', index=False, compression='gzip')
    
    if args.pair_mode == 'all_pairs':
        if args.genome:
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
        # Create waterfall plot scrub
        df_gl = (df_global_counts.to_pandas().drop(columns=['reference_count']).set_index('#kmer'))
        row_sums = df_gl.sum(axis=1)
        count_hist = (row_sums.value_counts()
              .sort_index()
              .rename_axis('total_count')
              .reset_index(name='n_kmers'))
        
        print(count_hist)
        fig = px.scatter(count_hist,
                     y = 'n_kmers',
                     x = 'total_count',
                     log_y = True,
                     template = 'simple_white',
                     range_x  = [-0.1, 5000],
                     title = f'{basename}')
        fig.add_hline(y = count_hist.loc[count_hist['total_count']==0]['n_kmers'][0],line_width=3, line_dash="dash", line_color="grey")
        fig.write_image(os.path.join(args.output_dir, f'{basename}.scrub_counts.svg'))

        # create histogram plot
        df_hist = pd.read_csv(args.counts_summary, sep='\t')
        fig = px.histogram(
            df_hist,
            x='coverage_pct',
            log_y = True,
            color = 'sample_type',
            histfunc = 'count',
            template='simple_white',
            title=f'{basename} — coverage_pct distribution',
            range_x = [0,1],
            )
        fig.add_vline(x=0.96, line_width=3, line_dash="dash", line_color="grey")
        fig.update_layout(width=800, height=500)
        fig.write_image(os.path.join(args.output_dir, f'{basename}.histogram_scrub_db.svg'))
        
        ## first get rarest kmers
        total_kmers = len(df_global_counts)

        print(f'Total kmers: {total_kmers}')
        print('Remove kmers with count >1 from ref genome:')
        df_global_counts = df_global_counts.filter(pl.col("reference_count") == 1)
        print(f'Remaining kmers: {len(df_global_counts)}')
        
        if "drug_count" in df_global_counts.columns:
            print('Removing all kmers present in drug scrub:')
            df_no_drugs = df_global_counts.filter(pl.col("drug_count") == 0)
        else:
            print("No drug scrub performed")
            df_no_drugs = df_global_counts
        print(f'Remaining kmers: {len(df_no_drugs)}')

        ## get position on genome
        print(f'Loading genome: {args.genome}')
        records = load_genome(args.genome)
        

        #df_rare = df_no_drugs.filter((pl.col('metagenome_count') == 0 ) & (pl.col('pangenome_count') == 0))        
        percentile = args.percentile_union
        lowest = df_no_drugs.filter(((pl.col('pangenome_count')  <= pl.col('pangenome_count').quantile(percentile, interpolation="higher"))  &
                                     (pl.col('metagenome_count') <= pl.col('metagenome_count').quantile(percentile, interpolation="higher"))
                            ))
        print(lowest)

        ## select kmers based on overlap
        all_kmers = set(lowest["#kmer"].to_list())
        df_locations ,_ = build_mapped_kmers_ahocorasick(records, all_kmers, terminal_dist=args.terminal_dist)
        print('dropping terminal kmers')
        df_locations = df_locations.loc[df_locations['terminal_kmer'] == False] 

        print('finding overlapping kmers for kmer selection')
        dict_overlap = find_overlap_kmer_fast(df_locations, max = 0.6)
        
        print('selecting kmers')
        selected = max_independent_kmers_greedy_heap(dict_overlap=dict_overlap)
        print(f"Selected pair kmers: {len(selected):,}")

        ## create pairs of all kmers
        create_all_pairs(selected, 
                        args.output_dir , 
                        basename = basename,
                        #n_workers=args.threads,
                        max_kmers = 50000)

        # export
        print(f"Total kmers for strain_detect: {len(selected):,}")
        df_locations.loc[df_locations['#kmer'].isin(selected)].to_csv(os.path.join(args.output_dir, f'{basename}.rare_kmers_mapped.tsv.gz'),
                                         sep='\t', index=False)
    # Standard pair generation 
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
        # Create waterfall plot scrub
        df_gl = (df_global_counts.to_pandas().drop(columns=['reference_count']).set_index('#kmer'))
        row_sums = df_gl.sum(axis=1)
        count_hist = (row_sums.value_counts()
              .sort_index()
              .rename_axis('total_count')
              .reset_index(name='n_kmers'))
        
        print(count_hist)
        fig = px.scatter(count_hist,
                     y = 'n_kmers',
                     x = 'total_count',
                     log_y = True,
                     template = 'simple_white',
                     range_x  = [-0.1, 5000],
                     title = f'{basename}')
        fig.add_hline(y = count_hist.loc[count_hist['total_count']==0]['n_kmers'][0],line_width=3, line_dash="dash", line_color="grey")
        fig.write_image(os.path.join(args.output_dir, f'{basename}.scrub_counts.svg'))

        # create histogram plot
        df_hist = pd.read_csv(args.counts_summary, sep='\t')
        fig = px.histogram(
            df_hist,
            x='coverage_pct',
            log_y = True,
            color = 'sample_type',
            histfunc = 'count',
            template='simple_white',
            title=f'{basename} — coverage_pct distribution',
            range_x = [0,1],
            )
        fig.add_vline(x=0.96, line_width=3, line_dash="dash", line_color="grey")
        fig.update_layout(width=800, height=500)
        fig.write_image(os.path.join(args.output_dir, f'{basename}.histogram_scrub_db.svg'))
        

        
        print(f'Total kmers: {len(df_global_counts)}')
        print('Remove kmers with count >1 from ref genome:')
        df_global_counts = df_global_counts.filter(pl.col("reference_count") == 1)
        print(f'Remaining kmers: {len(df_global_counts)}')
        
        if "drug_count" in df_global_counts.columns:
            print('Removing all kmers present in drug scrub:')
            df_no_drugs = df_global_counts.filter(pl.col("drug_count") == 0)
        else:
            print("No drug scrub performed")
            df_no_drugs = df_global_counts
        print(f'Remaining kmers: {len(df_no_drugs)}')

        # you can then just grab the 0 counts here much faster
        print('Getting all kmers with counts')


        print(f'Loading genome: {args.genome}')
        records = load_genome(args.genome)
        



        df_inform_singletons = df_no_drugs.filter((pl.col('metagenome_count') == 0 ) & (pl.col('pangenome_count') == 0))        
        # get all non unique singletons
        df_non_inform_singletons = df_no_drugs.filter(~((pl.col('metagenome_count') == 0) & (pl.col('pangenome_count') == 0)))

        print(df_inform_singletons)
        print(df_non_inform_singletons)

        # export parquet inform kmers
        print('Creating pairs from non informative singletons')
        kmer_pairs_from_presence(args.counts_individual, args.counts_summary, 
                                 args.output_dir , 
                                 basename = basename,
                                 df_keep=df_non_inform_singletons,
                                 presence_t = args.presence_t, 
                                 similarity_t=None, 
                                 n_workers=args.threads,
                                 max_for_pairs = 100000)
        
       

        # pair parts → unique kmers across both columns, computed lazily
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
        
        # map positions
        df_locations ,_ = build_mapped_kmers_ahocorasick(records, all_kmers, terminal_dist=args.terminal_dist)
        print('dropping terminal kmers')
        df_locations = df_locations.loc[df_locations['terminal_kmer'] == False] 

        print('finding overlapping kmers for kmer selection')
        dict_overlap = find_overlap_kmer_fast(df_locations, max = 0.6)
        
        print('selecting kmers')
        selected = max_independent_kmers_greedy_heap(dict_overlap=dict_overlap)

        # split selected kmers into two sets
        selected_pair_kmers = set(selected) & set(pair_kmers)
        selected_singletons = set(selected) & set(singleton_kmers)
        print(f"Selected pair kmers: {len(selected_pair_kmers):,}")
        print(f"Selected singletons: {len(selected_singletons):,}")

        # downselect to random subset if too many rare singletons found
        max_singletons = 100000
        if len(selected_singletons) > max_singletons:
            selected_singletons = set(random.Random(42).sample(sorted(selected_singletons), max_singletons)) 
            # ensure selection is correct for export
            selected = list(selected_pair_kmers | selected_singletons)

        #  Export locations
        df_locations['origin'] = df_locations['#kmer'].apply(lambda x: 'singleton' if x in selected_singletons else 'pair')
        print(f"Total kmers for strain_detect: {len(selected):,}")
        df_locations.loc[df_locations['#kmer'].isin(selected)].to_csv(os.path.join(args.output_dir, f'{basename}.rare_kmers_mapped.tsv.gz'),
                                         sep='\t', index=False)
        
        
        # write pairs file, create empty file with correct schema if no pairs exist
        filtered_path = os.path.join(args.output_dir, f"{basename}.inform_kmer_pairs.pairs.parquet")
        pair_parts = sorted(glob.glob(pair_glob))

        if pair_parts and selected_pair_kmers:
            sel_pl = pl.Series(sorted(selected_pair_kmers))
            (
                pl.scan_parquet(pair_parts, low_memory=True)   # expanded list, not the glob string
                .filter(pl.col("kmerA").is_in(sel_pl) & pl.col("kmerB").is_in(sel_pl))
                .sink_parquet(filtered_path, compression="zstd")
            )
        else:
            print("No informative pairs for this strain — writing empty pairs parquet", flush=True)
            write_empty_pairs(filtered_path)

        #old
        #filtered_path = os.path.join(args.output_dir, f"{basename}.inform_kmer_pairs.pairs.parquet")
        #sel_pl = pl.Series(sorted(selected_pair_kmers))
        #(
        #    pl.scan_parquet(pair_glob, low_memory=True)
        #    .filter(pl.col("kmerA").is_in(sel_pl) & pl.col("kmerB").is_in(sel_pl))
        #    .sink_parquet(filtered_path, compression="zstd")
        #)

        # only delete the parts once the selected file exists and is non-empty
        if os.path.exists(filtered_path) and os.path.getsize(filtered_path) > 0:
            parts = glob.glob(pair_glob)
            for p in parts:
                os.remove(p)
            print(f"Wrote {filtered_path} and removed {len(parts)} part files", flush=True)
        else:
            print("WARNING: final parquet missing or empty — keeping part files", flush=True)

        # Create pairs with singletons
        if args.pair_mode == "sxs":
            singleton_path, _ = create_pairs_with_singletons(
                selected_singletons, selected_pair_kmers,
                output_dir=args.output_dir, basename=basename,
                self_singletons=True,
                max_singletons = 100000,
            )
            
        if args.pair_mode == "sxp":
            singleton_path, _ = create_pairs_with_singletons(
                selected_singletons, selected_pair_kmers,
                output_dir=args.output_dir, basename=basename,
                self_singletons=False,
                max_singletons = 100000,
            )

if __name__ == '__main__':
    main()
