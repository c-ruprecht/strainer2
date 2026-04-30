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

def _process_kmer_partition(args):
    filtered_path, prefixes, keep_kmers_subset, strain_index = args

    sig_dict = {}
    plen = len(prefixes[0])
    
    # one scan per worker, prefix filter only — pushed down into parquet
    batch_iter = (
        pl.scan_parquet(filtered_path)
          .filter(pl.col('#kmer').str.slice(0, plen).is_in(prefixes))
          .collect_batches()
    )

    for batch in batch_iter:
        kmers = batch['#kmer'].to_list()
        samples = batch['sample_id'].to_list()
        for k, s in zip(kmers, samples):
            if k not in keep_kmers_subset:
                continue
            sidx = strain_index[s]
            st = sig_dict.get(k)
            if st is None:
                sig_dict[k] = {sidx}
            else:
                st.add(sidx)

    return {k: sorted(v) for k, v in sig_dict.items()}

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


def find_informative_triplets(
    sig_df, output_dir, basename,
    used_kmers=None,                 # set of k-mers already in informative pairs
    batch_size=1_000_000,
    max_kmers_per_sig=None,
):
    """Informative triplet = three signatures whose 3-way intersection is empty.
    
    used_kmers: optional set of k-mers to exclude (e.g. those already in pairs).
                Signatures left with zero remaining k-mers are dropped.
    """
    sigs = sig_df.sort("sig_id").to_dicts()
    
    # filter out used k-mers; drop signatures that empty out
    pruned = []
    for row in sigs:
        kept = row["kmers"] if used_kmers is None else [k for k in row["kmers"] if k not in used_kmers]
        if kept:
            if max_kmers_per_sig is not None:
                kept = kept[:max_kmers_per_sig]
            pruned.append({
                "sig_id": row["sig_id"],
                "strain_set": frozenset(row["strain_set"]),
                "kmers": kept,
            })
    
    n_sig = len(pruned)
    print(f"  Triplet search over {n_sig} signatures (after dropping pair-used k-mers)", flush=True)
    if n_sig < 3:
        print("  Fewer than 3 signatures remaining — no triplets possible.", flush=True)
        return 0

    inform_path = os.path.join(output_dir, f"{basename}.inform_kmer_triplets.parquet")
    writer = pq.ParquetWriter(inform_path, _TRIP_SCHEMA, compression="zstd")

    a_buf, b_buf, c_buf = [], [], []
    sa_buf, sb_buf, sc_buf = [], [], []
    n_inform = 0

    def flush():
        nonlocal a_buf, b_buf, c_buf, sa_buf, sb_buf, sc_buf
        if not a_buf:
            return
        writer.write_table(pa.table({
            "kmerA": a_buf, "kmerB": b_buf, "kmerC": c_buf,
            "sig_a": sa_buf, "sig_b": sb_buf, "sig_c": sc_buf,
        }, schema=_TRIP_SCHEMA))
        a_buf, b_buf, c_buf = [], [], []
        sa_buf, sb_buf, sc_buf = [], [], []

    for i, j in combinations(range(n_sig), 2):
        sij = pruned[i]["strain_set"] & pruned[j]["strain_set"]
        for k in range(j + 1, n_sig):
            if sij.isdisjoint(pruned[k]["strain_set"]):
                sid_i, sid_j, sid_k = pruned[i]["sig_id"], pruned[j]["sig_id"], pruned[k]["sig_id"]
                for ka in pruned[i]["kmers"]:
                    for kb in pruned[j]["kmers"]:
                        for kc in pruned[k]["kmers"]:
                            trip = sorted([(ka, sid_i), (kb, sid_j), (kc, sid_k)])
                            a_buf.append(trip[0][0]); b_buf.append(trip[1][0]); c_buf.append(trip[2][0])
                            sa_buf.append(trip[0][1]); sb_buf.append(trip[1][1]); sc_buf.append(trip[2][1])
                            n_inform += 1
                            if len(a_buf) >= batch_size:
                                flush()

    flush()
    writer.close()
    print(f"  Informative triplets: {n_inform:,}", flush=True)
    return n_inform

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
    parser.add_argument('--percentile', type=float, default=0.01,
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
        print(len(df_global_counts))

        


        li_drop = drop_high_similarity_scrubs(input_path = args.counts_summary,
                                    total_counts= len(df_global_counts),
                                    threshold= [0.01,0.96],
                                    output_dir = os.path.join(args.output_dir))

        # Drop individual counts with threshold larger than ...
        lf_counts_individual = (pl.scan_csv(args.counts_individual, separator='\t')
                                    .rename({'kmer': '#kmer'})
                                    .filter(~pl.col('sample_id').is_in(li_drop))
                                )
        
        print(lf_counts_individual)
        
        # Check for highly similar strains and drop
        #STOP here for now
        

        print(f'Total kmers: {len(df_global_counts)}')
        print('Remove kmers with count >1 from ref genome:')
        df_global_counts = df_global_counts.filter(pl.col("reference_count") == 1)
        print(f'Remaining kmers: {len(df_global_counts)}')
         # remove all drug count entries to its minimum
        print('Removing all kmers present in drug scrub:')
        df_no_drugs = df_global_counts.filter(pl.col("drug_count") == pl.col("drug_count").min())
        print(f'Remaining kmers: {len(df_no_drugs)}')

        print('Getting all kmers with counts')
        singles_path = os.path.join(args.output_dir , f'{basename}.inform_kmer_singleton.parquet')
        # ── Stage 1: Filter counts_individual once, sink to parquet ───────
        filtered_path = os.path.join(args.output_dir, f'{basename}.counts_filtered.parquet')
        drop_set = set(li_drop)

        if os.path.exists(filtered_path) and not args.force:
            print(f'Reusing filtered counts at {filtered_path}')
        else:
            print(f'Filtering {args.counts_individual} → {filtered_path}')
            print(f'  Dropping {len(drop_set):,} strains')
            t0 = time.time()
            (pl.scan_csv(args.counts_individual, separator='\t')
               .rename({'kmer': '#kmer'})
               .filter(~pl.col('sample_id').is_in(drop_set))
               .select(['#kmer', 'sample_id'])
               .sink_parquet(filtered_path, compression='zstd'))
            elapsed = time.time() - t0
            size_gb = os.path.getsize(filtered_path) / 1e9
            print(f'  Done in {elapsed:.0f}s, {size_gb:.1f} GB on disk')

        # ── Stage 2: Informative singletons ───────────────────────────────
        print('Getting all kmers with counts')
        singles_path = os.path.join(args.output_dir, f'{basename}.inform_kmer_singleton.parquet')

        if os.path.exists(singles_path) and not args.force:
            print(f'Skipping informative singletons — {singles_path} already exists. Use --force to regenerate.')
            df_inform_singletons = pl.read_parquet(singles_path)
        else:
            batch_iter = pl.scan_parquet(filtered_path).select('#kmer').collect_batches()

            chunks = []
            rows_seen = 0
            t0 = time.time()

            for i, batch in enumerate(batch_iter, 1):
                chunks.append(batch['#kmer'])
                rows_seen += batch.height

                if i % 100 == 0:
                    merged = pl.concat(chunks).unique()
                    chunks = [merged]
                    rate = rows_seen / (time.time() - t0)
                    print(f'  batch {i:>5}  {rows_seen:>13,} rows  |  '
                          f'{len(merged):>10,} unique  |  {rate/1e6:>5.2f} M rows/s')

            seen = pl.concat(chunks).unique() if chunks else pl.Series('#kmer', [], dtype=pl.Utf8)
            print(f'Final: {len(seen):,} unique k-mers')
            kmers_w_count = seen

            print('Get informative singletons')
            df_inform_singletons = df_no_drugs.filter(~pl.col('#kmer').is_in(kmers_w_count))
            df_inform_singletons.write_parquet(singles_path, compression='zstd')

        print(f'Informative singletons: {len(df_inform_singletons):,}')

        df_non_inform_singletons = df_no_drugs.filter(~pl.col('#kmer').is_in(df_inform_singletons['#kmer'].implode()))
        print(f'Non-informative singletons (need pair search): {len(df_non_inform_singletons):,}')

        # ── Stage 3: Build signatures from filtered parquet ───────────────
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        


        # in main():
        print('Building signatures via parallel batched scan')
        t0 = time.time()

        keep_kmers = set(df_non_inform_singletons['#kmer'].to_list())
        print(f'  k-mer set to track: {len(keep_kmers):,}')

        strains = (pl.scan_parquet(filtered_path)
                    .select('sample_id').unique()
                    .collect(engine='streaming')
                    ['sample_id'].sort().to_list())
        strain_index = {s: i for i, s in enumerate(strains)}
        print(f'  effective strains: {len(strains):,}')

        # Partition k-mers by 2-letter prefix → 16 disjoint groups, distribute across workers
        n_workers = min(args.threads or 8, 8)   # cap at 8
        all_prefixes = [a + b for a in 'ACGT' for b in 'ACGT']  # 16 prefixes total

        prefixes_per_worker = max(1, len(all_prefixes) // n_workers)
        worker_assignments = [
            all_prefixes[i:i + prefixes_per_worker]
            for i in range(0, len(all_prefixes), prefixes_per_worker)
        ]
        n_workers = len(worker_assignments)   # may have changed due to integer division
        plen = len(all_prefixes[0])
        print(f'  using {n_workers} workers, prefixes per worker: {prefixes_per_worker}')

        # map each prefix to its worker index
        prefix_to_worker = {}
        for w, prefs in enumerate(worker_assignments):
            for p in prefs:
                prefix_to_worker[p] = w

        # split keep_kmers into per-worker subsets
        keep_kmers_subsets = [set() for _ in range(n_workers)]
        for k in keep_kmers:
            w = prefix_to_worker.get(k[:plen])
            if w is not None:
                keep_kmers_subsets[w].add(k)
        del keep_kmers
        gc.collect()

        worker_args = [
            (filtered_path, prefs, subset, strain_index)
            for prefs, subset in zip(worker_assignments, keep_kmers_subsets)
            if subset
        ]
        del keep_kmers_subsets
        gc.collect()

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            partial_dicts = list(ex.map(_process_kmer_partition, worker_args))

        print(f'  workers done in {time.time()-t0:.0f}s')

        # Combine into single kmer_sigs frame — partitions are disjoint by construction
        print('Combining partitions')
        t0 = time.time()
        all_kmers, all_strain_lists = [], []
        for d in partial_dicts:
            all_kmers.extend(d.keys())
            all_strain_lists.extend(d.values())
        del partial_dicts
        gc.collect()

        kmer_sigs = pl.DataFrame({
            '#kmer': all_kmers,
            'strain_set': all_strain_lists,
        }, schema={'#kmer': pl.Utf8, 'strain_set': pl.List(pl.UInt32)})
        del all_kmers, all_strain_lists
        gc.collect()
        print(f'  combined into {kmer_sigs.height:,} k-mers in {time.time()-t0:.0f}s')

        # rest unchanged: group_by signature → sig_df
        sig_df = (
            kmer_sigs
            .with_columns(pl.col('strain_set').list.len().alias('n_strains'))
            .group_by('strain_set')
            .agg([
                pl.col('#kmer').alias('kmers'),
                pl.col('#kmer').len().alias('n_kmers'),
                pl.col('n_strains').first(),
            ])
            .with_row_index('sig_id')
        )
        print(f'  {kmer_sigs.height:,} k-mers → {sig_df.height:,} unique signatures '
            f'({kmer_sigs.height / max(sig_df.height,1):.1f}x compression)')

        del kmer_sigs
        gc.collect()
        # ── Convert dict to Polars DataFrame ──────────────────────────────
        print('Converting to signature table')
        t0 = time.time()
        kmer_list = list(sig_dict.keys())
        strain_lists = [sorted(sig_dict[k]) for k in kmer_list]
        del sig_dict
        gc.collect()

        kmer_sigs = pl.DataFrame({
            '#kmer': kmer_list,
            'strain_set': strain_lists,
        }, schema={'#kmer': pl.Utf8, 'strain_set': pl.List(pl.UInt32)})
        del kmer_list, strain_lists
        gc.collect()
        print(f'  built sig frame in {time.time()-t0:.0f}s')

        # ── Group k-mers by identical signatures → equivalence classes ────
        sig_df = (
            kmer_sigs
            .with_columns(pl.col('strain_set').list.len().alias('n_strains'))
            .group_by('strain_set')
            .agg([
                pl.col('#kmer').alias('kmers'),
                pl.col('#kmer').len().alias('n_kmers'),
                pl.col('n_strains').first(),
            ])
            .with_row_index('sig_id')
        )
        print(f'  {kmer_sigs.height:,} k-mers → {sig_df.height:,} unique signatures '
                f'({kmer_sigs.height / max(sig_df.height,1):.1f}x compression)')

        del kmer_sigs
        gc.collect()

        print('\nSignature class size distribution:')
        print(sig_df.select('n_kmers').describe())
        print('\nLargest signature classes:')
        print(sig_df.sort('n_kmers', descending=True)
                    .select(['sig_id', 'n_strains', 'n_kmers'])
                    .head(10))

        # ── Stage 4: Find informative pairs ───────────────────────────────
        print('\nFinding informative pairs')
        n_pairs = find_informative_pairs(sig_df, args.output_dir, basename)

        inform_pairs_path = os.path.join(args.output_dir, f'{basename}.inform_kmer_pairs.parquet')
        df_inform_pairs = pl.read_parquet(inform_pairs_path)
        print(f'Total informative pairs: {len(df_inform_pairs):,}')
        
        #clean up 
        gc.collect()
        del df_global_counts, df_indiv_counts, df_non_inform_singletons, df_no_drugs
            
            
        
        #mapping positions
        print('Mapping positions of remaining kmers')
        # Mapping kmers
        print(f'Loading genome: {args.genome}')
        records = load_genome(args.genome)
        print(f'  {len(records)} contigs')

        singleton_kmers = set(df_inform_singletons["#kmer"].to_list())
        pair_kmers = set(df_inform_pairs["kmerA"].to_list()) | set(df_inform_pairs["kmerB"].to_list())
        all_kmers = singleton_kmers | pair_kmers
        print(f'Total kmers for strain_detect: {len(all_kmers)}')
        
        df_locations, _ = build_mapped_kmers_ahocorasick(records, all_kmers, terminal_dist=args.terminal_dist)

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
