#!/usr/bin/env python3
"""Map scrubbed kmers onto a genome and export result dataframes.

Usage:
    python kmer_scrub_filter2.py <genome.fna.gz> <scrubbed_kmers.gz>
    python kmer_scrub_filter2.py <genome.fna.gz> <scrubbed_kmers.gz> --output-dir results/ --basename my_strain --figures
"""
import argparse
import gzip
import os

import numpy as np
import ahocorasick
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Bio import SeqIO
from Bio.Seq import Seq


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


def smooth_downsample(df, target_count, bin_size):
    """Downsample df to target_count by capping high-coverage bins."""
    df = assign_mapping_bin(df.loc[df['terminal_kmer'] == False], bin_size)

    n_remove = len(df) - target_count
    if n_remove <= 0:
        return df

    bin_counts = df.groupby(['contig_id', 'bin'])['#kmer'].count()
    median_count = bin_counts.median()
    print(f'  Bin median: {median_count:.1f}, removing {n_remove} kmers to reach target {target_count}')

    bin_excess = (bin_counts - median_count).clip(lower=0)

    if bin_excess.sum() == 0:
        return df.sample(n=target_count)

    remove_per_bin = (bin_excess / bin_excess.sum() * n_remove).round().astype(int)
    remove_per_bin = remove_per_bin.clip(upper=bin_counts - 1)

    keep_indices = []
    for (contig_id, bin_val), group in df.groupby(['contig_id', 'bin']):
        n_drop = remove_per_bin.get((contig_id, bin_val), 0)
        if n_drop > 0:
            n_keep = len(group) - n_drop
            keep_indices.append(
                group.sort_values('kmer_position').iloc[
                    np.linspace(0, len(group) - 1, n_keep, dtype=int)
                ]
            )
        else:
            keep_indices.append(group)

    result = pd.concat(keep_indices)

    if len(result) > target_count:
        result = result.sample(n=target_count)

    return result.sort_values(['contig_id', 'kmer_position'])

def plot_genome_bins(df, df_smooth, basename, bin_size, output_dir):
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
                  log_y=True,
                  template='simple_white',
                  color='scrub_type',
                  title='rare kmers by count')
    fig.update_yaxes(title_text='')
    return fig

def plot_box_coverage(df_lowest, df_smooth, basename, bin_size):
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
    parser.add_argument('genome', help='Genome FASTA file (.fna or .fna.gz)')
    parser.add_argument('scrubbed_kmers', help='Scrubbed kmers file (.gz)')
    parser.add_argument('--output-dir', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--basename', default=None, help='Output basename (default: derived from genome filename)')
    parser.add_argument('--figures', action='store_true', default=False,
                        help='Save figures as SVG (default: False)')
    parser.add_argument('--percentile', type=float, default=0.01,
                        help='Percentile threshold for rare kmer selection (default: 0.05)')
    parser.add_argument('--percentile_union', type = float, default = 0.05, help = 'percentile passed for union of different kmer scrubs')
    parser.add_argument('--bin-size', type=int, default=1000,
                        help='Bin size in bp for kmer density smoothing (default: 1000)')
    parser.add_argument('--terminal-dist', type=int, default=300,
                        help='Distance from contig ends to flag terminal kmers (default: 300)')
    args = parser.parse_args()

    strain = strain_name_from_path(args.genome)
    basename = args.basename if args.basename else strain
    os.makedirs(args.output_dir, exist_ok=True)

    df_counts = pd.read_csv(args.scrubbed_kmers, sep='\t')
    lowest_pct = get_lowest_percentile(df_counts, percentile=argparse.percentile_union, drug_scrub='percentile')
    target = int(round(args.percentile * len(df_counts), 0))

    print(f'Total kmers: {len(df_counts)}')
    print(f'Rare kmers (5th pct): {len(lowest_pct)} ({len(lowest_pct)/len(df_counts)*100:.1f}%)')
    print(f'Target after smoothing: {target}')

    if args.figures:
        fig_counts = plot_kmer_counts(lowest_pct)
        fig_counts.write_image(os.path.join(args.output_dir, f'{basename}.kmer_counts.svg'))

    print(f'Loading genome: {args.genome}')
    records = load_genome(args.genome)
    print(f'  {len(records)} contigs')

    kmers = lowest_pct['#kmer'].to_list()
    if len(kmers) == 0:
        print('ERROR: 0 kmers after filter')
    else:
        print(f'Mapping {len(kmers)} kmers...')
    df, _ = build_mapped_kmers_ahocorasick(records, kmers, terminal_dist=args.terminal_dist)
    df_smooth = smooth_downsample(df, target_count=target, bin_size=args.bin_size)
    print(f'  {len(df_smooth)} kmers after smoothing')

    if args.figures:
        plot_genome_bins(df, df_smooth, basename, bin_size=args.bin_size, output_dir=args.output_dir)
        
        fig_bins2 = plot_box_coverage(df, df_smooth, basename, bin_size=args.bin_size)
        fig_bins2.write_image(os.path.join(args.output_dir, f'{basename}.box_genome_bins.svg'))
    df_smooth.to_csv(os.path.join(args.output_dir, f'{basename}.rare_kmers_mapped.tsv.gz'),
                     sep='\t', index=False, compression='gzip')
    df_smooth[['#kmer']].to_csv(os.path.join(args.output_dir, f'{basename}.scrubbed_kmers'),
                              sep='\t', index=False, header=None)

if __name__ == '__main__':
    main()
