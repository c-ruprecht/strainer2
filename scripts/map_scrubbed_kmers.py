#!/usr/bin/env python3
"""Map scrubbed kmers onto a genome and export result dataframes.

Usage:
    python map_scrubbed_kmers.py <genome.fna.gz> <scrubbed_kmers.gz> [output_prefix]
    python map_scrubbed_kmers.py <genome.fna.gz> <scrubbed_kmers.gz> --figures
"""
import argparse
import gzip
import os

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

TERMINAL_DIST = 300
WINDOW = 1000


def load_genome(genome_path):
    opener = gzip.open if genome_path.endswith('.gz') else open
    records = {}
    with opener(genome_path, 'rt') as fh:
        for record in SeqIO.parse(fh, 'fasta'):
            records[record.id] = record.seq
    return records


def load_kmers(kmer_path):
    try:
        with gzip.open(kmer_path, 'rt') as fh:
            content = fh.read()
    except gzip.BadGzipFile:
        with open(kmer_path, 'r') as fh:
            content = fh.read()
    return [line for line in content.splitlines() if '#' not in line and line]


def strain_name_from_path(path):
    base = os.path.basename(path)
    for ext in ('.fna.gz', '.fasta.gz', '.fa.gz', '.fna', '.fasta', '.fa'):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base.split('.')[0]


def build_mapped_kmers(records, kmers):
    rows = []
    for record_id, seq in records.items():
        for kmer in kmers:
            pos = seq.find(kmer)
            if pos != -1:
                rows.append((record_id, kmer, pos, False))
            else:
                kmer_rc = str(Seq(kmer).reverse_complement())
                pos_rc = seq.find(kmer_rc)
                if pos_rc != -1:
                    rows.append((record_id, kmer, pos_rc, True))

    df = pd.DataFrame(rows, columns=['contig_id', '#kmer', 'kmer_position', 'reverse_complement'])

    if len(df) < len(kmers):
        print('WARNING: NOT ALL KMERS ARE FOUND')
    elif len(df) > len(kmers):
        print('WARNING: KMERS ARE FOUND MORE THAN ONCE')
    else:
        print(f'All {len(df)} kmer hits found and unique')

    print(df.groupby('reverse_complement')['#kmer'].count()
            .rename({False: 'forward', True: 'reverse_complement'}).to_string())

    dict_len = {cid: len(seq) for cid, seq in records.items()}
    df['contig_length'] = df['contig_id'].map(dict_len)
    df['terminal_kmer'] = (
        (df['kmer_position'] < TERMINAL_DIST) |
        ((df['contig_length'] - df['kmer_position']) < TERMINAL_DIST)
    )
    df['label'] = df['terminal_kmer'].map({True: 'terminal', False: 'internal'})

    n_terminal = int(df['terminal_kmer'].sum())
    n_total = len(df)
    print(f'  Terminal kmers: {n_terminal} / {n_total} total')

    return df, dict_len


def build_density(df, dict_len):
    rolling_rows = []
    for contig_id, group in df.groupby('contig_id'):
        contig_len = dict_len[contig_id]
        counts = pd.Series(0, index=range(contig_len))
        counts.loc[group['kmer_position'].values] = 1
        rolling_mean = counts.rolling(window=WINDOW, center=True, min_periods=1).mean()
        sampled = rolling_mean.iloc[::WINDOW]
        rolling_rows.append(pd.DataFrame({
            'contig_id': contig_id,
            'position': sampled.index,
            'kmer_density': sampled.values,
        }))
    return pd.concat(rolling_rows, ignore_index=True)


def make_figures(df, df_rolling, dict_len, strain, output_path):
    import plotly.express as px
    import plotly.graph_objects as go

    n_terminal = int(df['terminal_kmer'].sum())
    n_total = len(df)

    contig_order = (
        df[['contig_id', 'contig_length']]
        .drop_duplicates()
        .sort_values('contig_length', ascending=False)['contig_id']
        .tolist()
    )

    fig = px.scatter(
        df,
        x='kmer_position',
        y='contig_id',
        color='label',
        color_discrete_map={'internal': 'steelblue', 'terminal': 'crimson'},
        category_orders={'contig_id': contig_order},
        template='simple_white',
        height=600,
        width=1000,
        labels={'kmer_position': 'position', 'contig_id': 'contig', 'label': ''},
        title=f'{strain}<br><sup>total kmers: {n_total} | terminal kmers: {n_terminal}</sup>',
    )
    fig.add_trace(go.Scatter(
        x=[dict_len[cid] for cid in contig_order],
        y=contig_order,
        mode='markers',
        marker=dict(symbol='line-ns', color='lightgray', size=10, line=dict(color='lightgray', width=2)),
        showlegend=False,
    ))
    fig.update_layout(title_x=0.5)
    fig.write_html(output_path)
    print(f'Figure saved to {output_path}')

    pivot = (df_rolling
             .pivot(index='contig_id', columns='position', values='kmer_density')
             .reindex(contig_order))
    fig2 = px.imshow(
        pivot,
        aspect='auto',
        color_continuous_scale='Blues',
        template='simple_white',
        height=600,
        width=1000,
        labels={'x': 'position (bp)', 'y': 'contig', 'color': f'kmers / {WINDOW} bp'},
        title=f'{strain} — rolling kmer density ({WINDOW} bp window)',
    )
    fig2.update_layout(title_x=0.5)
    density_path = output_path.replace('.kmer_map.html', '.kmer_density.html') if output_path.endswith('.kmer_map.html') else output_path + '.kmer_density.html'
    fig2.write_html(density_path)
    print(f'Density plot saved to {density_path}')


def main():
    parser = argparse.ArgumentParser(description='Map scrubbed kmers onto a genome.')
    parser.add_argument('genome', help='Genome FASTA file (.fna or .fna.gz)')
    parser.add_argument('scrubbed_kmers', help='Scrubbed kmers file (.gz)')
    parser.add_argument('output', nargs='?', help='Output basename (default: <strain>). Files will be written as <basename>.mapped_kmers.tsv etc.')
    parser.add_argument('--figures', action='store_true', default=False,
                        help='Also generate HTML figures (default: False)')
    args = parser.parse_args()

    strain = strain_name_from_path(args.genome)
    basename = args.output or strain

    os.makedirs(os.path.dirname(os.path.abspath(basename)), exist_ok=True)

    print(f'Strain: {strain}')
    print(f'Loading genome from {args.genome}...')
    records = load_genome(args.genome)
    print(f'  {len(records)} contigs loaded')

    print(f'Loading kmers from {args.scrubbed_kmers}...')
    kmers = load_kmers(args.scrubbed_kmers)
    print(f'  {len(kmers)} kmers loaded')

    print('Mapping kmers...')
    df, dict_len = build_mapped_kmers(records, kmers)

    tsv_path = f'{basename}.mapped_kmers.tsv'
    df.to_csv(tsv_path, sep='\t', index=False)
    print(f'Mapped kmers saved to {tsv_path}')

    df_rolling = build_density(df, dict_len)
    density_tsv_path = f'{basename}.kmer_density.tsv'
    df_rolling.to_csv(density_tsv_path, sep='\t', index=False)
    print(f'Density dataframe saved to {density_tsv_path}')

    if args.figures:
        make_figures(df, df_rolling, dict_len, strain, f'{basename}.kmer_map.html')


if __name__ == '__main__':
    main()
