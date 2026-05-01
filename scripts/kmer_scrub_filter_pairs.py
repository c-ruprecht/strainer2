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
from kmer_pairs import drop_high_presence_strains, create_kmer_pairs_parallel 
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


def build_signatures_from_long(df_long, kmer_col="#kmer", strain_col="sample_id"):
    """From long-format presence data, produce signature equivalence classes.

    Returns:
      sig_df: Polars DataFrame with columns [sig_id, n_kmers, strain_set (list[int]), n_strains]
      kmer_to_sig: Polars DataFrame [#kmer, sig_id]
      strain_index: dict mapping strain name -> int (for strain_set encoding)
    """
    strains = df_long[strain_col].unique().sort().to_list()
    strain_index = {s: i for i, s in enumerate(strains)}

    # one row per kmer with its sorted list of strain indices = canonical signature
    kmer_sigs = (
        df_long
        .with_columns(pl.col(strain_col).replace_strict(strain_index).alias("_sidx"))
        .group_by(kmer_col)
        .agg(pl.col("_sidx").sort().alias("strain_set"))
        .with_columns(pl.col("strain_set").list.len().alias("n_strains"))
    )

    # group kmers by identical signatures
    sig_df = (
        kmer_sigs
        .group_by("strain_set")
        .agg([
            pl.col(kmer_col).alias("kmers"),
            pl.col(kmer_col).len().alias("n_kmers"),
            pl.col("n_strains").first(),
        ])
        .with_row_index("sig_id")
    )

    kmer_to_sig = (
        sig_df.explode("kmers")
              .select(pl.col("kmers").alias(kmer_col), "sig_id")
    )

    print(f"  {kmer_sigs.height:,} k-mers → {sig_df.height:,} unique signatures "
          f"({kmer_sigs.height / sig_df.height:.1f}x compression)")
    return sig_df, kmer_to_sig, strain_index

_PAIR_SCHEMA = pa.schema([
("kmerA", pa.string()),
("kmerB", pa.string()),
("sig_a", pa.uint32()),
("sig_b", pa.uint32()),
])


def find_informative_pairs(sig_df, output_dir, basename, batch_size=1_000_000):
    """Informative pair = two signatures with disjoint strain sets.
    Expand each informative signature pair into all kmer × kmer products.
    """
    sigs = sig_df.sort("sig_id")
    sig_sets = [frozenset(s) for s in sigs["strain_set"].to_list()]
    sig_kmers = sigs["kmers"].to_list()
    sig_ids = sigs["sig_id"].to_list()
    n_sig = len(sig_sets)

    inform_path = os.path.join(output_dir, f"{basename}.inform_kmer_pairs.parquet")
    writer = pq.ParquetWriter(inform_path, _PAIR_SCHEMA, compression="zstd")

    a_buf, b_buf, sa_buf, sb_buf = [], [], [], []
    n_inform = 0

    def flush():
        nonlocal a_buf, b_buf, sa_buf, sb_buf
        if not a_buf:
            return
        writer.write_table(pa.table(
            {"kmerA": a_buf, "kmerB": b_buf, "sig_a": sa_buf, "sig_b": sb_buf},
            schema=_PAIR_SCHEMA,
        ))
        a_buf, b_buf, sa_buf, sb_buf = [], [], [], []

    print(f"  Scanning {n_sig*(n_sig-1)//2:,} signature pairs", flush=True)
    for i, j in combinations(range(n_sig), 2):
        if sig_sets[i].isdisjoint(sig_sets[j]):
            # expand to all kmer pairs in this signature × signature block
            ka_list, kb_list = sig_kmers[i], sig_kmers[j]
            sid_a, sid_b = sig_ids[i], sig_ids[j]
            for ka in ka_list:
                for kb in kb_list:
                    a, b = (ka, kb) if ka < kb else (kb, ka)
                    a_buf.append(a); b_buf.append(b)
                    sa_buf.append(sid_a); sb_buf.append(sid_b)
                    n_inform += 1
                    if len(a_buf) >= batch_size:
                        flush()

    flush()
    writer.close()
    print(f"  Informative pairs: {n_inform:,}", flush=True)
    return n_inform
_TRIP_SCHEMA = pa.schema([
    ("kmerA", pa.string()), ("kmerB", pa.string()), ("kmerC", pa.string()),
    ("sig_a", pa.uint32()), ("sig_b", pa.uint32()), ("sig_c", pa.uint32()),
])



def drop_high_similarity_scrubs(input_path, total_counts, output_dir, threads=12, threshold=[0.01, 0.96]):
    print(f'Checking for highly similar strains')
    print(f'threshold: {threshold}')
    print(f'Total unique kmers of target strain: {total_counts}')
    print(f'threads: {threads}')

    df = pd.read_csv(input_path, sep = '\t')
    print(df)
    df_over = df.loc[(df['coverage_pct']>= threshold[1]) | (df['coverage_pct'] <= threshold[0])]
    fig = px.histogram(df,
                    x = 'coverage_pct',
                    log_y = True,
                    template = 'simple_white')
    fig.add_vrect(x0= threshold[0],x1 = threshold[1],
                  fillcolor="green", opacity=0.25, line_width=0,
                  annotation_text=f"threshold:  {threshold}")
    fig.write_image(os.path.join(output_dir, 'histogram_scrub_coverage.svg'))
    print('Dropping strains over threshold')
    print(df_over)
    return df_over['sample_id']


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
def make_kmers_independent(df, type = 'singleton'):
    
    if singleton:
        contig_results = []
        for contig_id, contig_df in df.groupby('contig_id'):
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
    return li_drop
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

        


        li_drop = drop_high_similarity_scrubs(input_path = args.counts_summary,
                                    total_counts= len(df_global_counts),
                                    threshold= [0.01,0.96],
                                    output_dir = os.path.join(args.output_dir))

        
        # Check for highly similar strains and drop
        # This should move to kmer scrub to not advance global counts when a strain is too close          

        print(f'Total kmers: {len(df_global_counts)}')
        print('Remove kmers with count >1 from ref genome:')
        df_global_counts = df_global_counts.filter(pl.col("reference_count") == 1)
        print(f'Remaining kmers: {len(df_global_counts)}')
         # remove all drug count entries to its minimum
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
        print(f'  {len(records)} contigs')
        all_kmers = df_no_drugs['#kmer']
        df_locations, _ = build_mapped_kmers_ahocorasick(records, all_kmers, terminal_dist=args.terminal_dist)
        print(df_locations)

        # merge positions and counts

        n_targets = total_kmers * args.percentage 
        

        df_inform_singletons = df_no_drugs.filter((pl.col('metagenome_count') == 0 ) & (pl.col('pangenome_count') == 0))
        df_non_inform_singletons = df_no_drugs.filter(~(pl.col('metagenome_count') == 0 ) & ~(pl.col('pangenome_count') == 0))

        print(df_inform_singletons)

        # singletons should also be made independent!
        df_loc_singles = pd.merge(df_inform_singletons.to_pandas(), df_locations, on = 'kmer', how = 'left')
        print(df_loc_singles)
        make_kmers_independent(df_loc_singles, type = 'singleton')

        # start finding pairs
        sys.exit()
        print(f'targetting {n_targets} informative kmer pairs')

        print(f'Non-informative singletons (need pair search): {len(df_non_inform_singletons):,}')

        ### Build a greedy selection for n_target percentage
        #   maximal kmers to track 1% - singletons?
        #   keep better scored pairs, how to score: 
                #location: diff contig > same contig > distance from each other to maximize
        #   Start point just dataframe sorted by counts union?

       
            
            
        

        singleton_kmers = set(df_inform_singletons["#kmer"].to_list())
        pair_kmers = set(df_inform_pairs["kmerA"].to_list()) | set(df_inform_pairs["kmerB"].to_list())
        all_kmers = singleton_kmers | pair_kmers
        print(f'Total kmers for strain_detect: {len(all_kmers)}')
        

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
