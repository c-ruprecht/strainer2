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


import subprocess
import os
import polars as pl
import shutil

def drop_high_similarity_scrubs(input_path, total_counts, output_dir, threads=12, threshold=0.96):
    print(f'Checking for highly similar strains')
    print(f'threshold: {threshold}')
    print(f'Total unique kmers of target strain: {total_counts}')
    print(f'threads: {threads}')

    path_counts = os.path.join(output_dir, 'similarity_counts.tsv')

    # Use pigz if available — parallel gzip decompression
    if input_path.endswith('.zst') or input_path.endswith('.zstd'):
        decompress = "zstd -dc -T0"
    elif input_path.endswith('.gz'):
        decompress = "pigz -dc -p 4" if shutil.which("pigz") else "zcat"
    else:
        raise ValueError(f"Unknown compression for {input_path}")    # Read header
    header_line = subprocess.check_output(
        f"{decompress} {input_path} | head -1", shell=True, text=True
    ).strip()
    header = header_line.split("\t")
    kmer_col = header.index("kmer") + 1
    sample_col = header.index("sample_id") + 1
    print(f'kmer col: {kmer_col}, sample_id col: {sample_col}')

    # Sort gets most of the threads; pigz gets a few; awk is single-threaded but fast
    sort_threads = max(1, threads - 4)
    sort_mem = f"{max(2, threads // 2)}G"  # scale memory budget with threads

    cmd = f"""
    set -euo pipefail
    {decompress} {input_path} \
      | tail -n +2 \
      | awk -F'\\t' -v s={sample_col} -v k={kmer_col} 'BEGIN{{OFS="\\t"}} {{print $s, $k}}' \
      | LC_ALL=C sort -u --parallel={sort_threads} -S {sort_mem} -T {output_dir} \
      | awk -F'\\t' 'BEGIN{{OFS="\\t"; print "sample_id","num_unique_kmers"}}
                     {{ if ($1 != prev) {{ if (prev != "") print prev, n; prev = $1; n = 0 }}
                        n++ }}
                     END{{ if (prev != "") print prev, n }}' \
      > {path_counts}
    """
    subprocess.run(["bash", "-c", cmd], check=True)

    df = pl.read_csv(path_counts, separator="\t")
    print(df)
    return df

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

        # Drop individual counts with threshold larger than ...
        lf_counts_individual = (pl.scan_csv(args.counts_individual, separator='\t')
                                    .rename({'kmer': '#kmer'})
                                    .unique() 
                                )
        
        print(lf_counts_individual)
        
        zst_path = os.path.join(args.output_dir, 'individual_counts.tsv.zst')
        if not os.path.exists(zst_path):
            print(f'Converting gzip to zstd: {zst_path}')
            subprocess.run(
                f"pigz -dc {args.counts_individual} | zstd -T0 -o {zst_path}",
                shell=True, check=True,
            )
        else:
            print(f'Reusing existing zstd file: {zst_path}')


        drop_high_similarity_scrubs(input_path = zst_path,
                                    total_counts= len(df_global_counts),
                                    threshold= 0.96,
                                    output_dir = args.output_dir,
                                    threads = args.threads)

        #STOP here for now
        sys.exit("Stopping before OOM step")

        print(f'Total kmers: {len(df_global_counts)}')
        print('Remove non unique kmers of ref genome:')
        df_global_counts = df_global_counts.filter(pl.col("reference_count") == 1)
        print(f'Remaining kmers: {len(df_global_counts)}')
         # remove all drug count entries to its minimum
        print('Removing all drug kmers > min (usually 0):')
        df_no_drugs = df_global_counts.filter(pl.col("drug_count") == pl.col("drug_count").min())
        print(f'Remaining kmers: {len(df_no_drugs)}')

        # get all informative 0s
        df_inform_singletons = df_no_drugs.filter((pl.col("pangenome_count") == 0 )&(pl.col('metagenome_count') == 0))
        df_inform_singletons.write_parquet(os.path.join(args.output_dir , f'{basename}.inform_kmer_singleton.parquet'), compression='zstd')

        # how do I reduce this to an ok number before trying to create pairs?
        print(f'Informative singletons: {len(df_inform_singletons)}')

        
        df_non_inform_singletons = df_no_drugs.filter(~(pl.col("pangenome_count") == 0 )&
                                                      ~(pl.col('metagenome_count') == 0))


        # Drop all kmers from individual that have been droppped
        # drop all non unique sample names
        keep_kmers = pl.LazyFrame({'#kmer': df_non_inform_singletons['#kmer']})
        print('Sinking individual counts to parquet (filtered to surviving kmers)')
        (pl.scan_csv(args.counts_individual, separator='\t')
            .rename({'kmer': '#kmer'})
            .join(keep_kmers, on='#kmer', how='semi')
            .sink_parquet(
                args.output_dir + '/counts_individual.parquet',
                compression='zstd',
            )
        )
        print('finished sink')

        df_indiv_counts = filter_long_counts_streaming(
            parquet_path=args.output_dir + '/counts_individual.parquet',
            output_dir=args.output_dir,
            reference_kmers=df_non_inform_singletons['#kmer'],  # match what's in the parquet
            presence_threshold=0.98,
        )
 
    
        
        
        
        # Creating pairs rom non informative singletons
        df_pairs = df_indiv_counts.filter(pl.col('#kmer').is_in(df_non_inform_singletons['#kmer'].implode())
                                          )
        #clean up 
        gc.collect()
        del df_global_counts, df_indiv_counts, df_non_inform_singletons, df_no_drugs
        
        print('Pivot dataframe for pair generation')
        # drop duplicate samples
        df_pairs=df_pairs.unique(subset=["#kmer", "sample_id"])
        df_pairs = df_pairs.pivot(on="sample_id",index="#kmer",values="count").fill_null(0)

        print('Creating pairs in parallel')
        n_inform_pairs, dict_non_inform_pairs = create_kmer_pairs_parallel(df_pairs, 
                                        args.output_dir,
                                        basename = args.basename,
                                        n_workers = args.threads)
        

        inform_pair_parts = sorted(glob.glob(os.path.join(args.output_dir, f"{basename}.inform_kmer_pairs.part*.parquet")))
        if inform_pair_parts:
            df_inform_pairs = pl.concat([pl.read_parquet(p) for p in inform_pair_parts])
        else:
            df_inform_pairs = pl.DataFrame()

        print(df_inform_pairs)
        print(f"Total informative pairs: {len(df_inform_pairs)}")
        
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
