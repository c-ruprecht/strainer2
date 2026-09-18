import argparse
import os
import gzip

import polars as pl
import pandas as pd
# read mapping
import ahocorasick
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

def add_block_ids(df, min_block_size=30, presence_col='list_scrub_id'):
    df = df.sort_values(['contig_id', 'kmer_position']).copy()
    presence = df[presence_col].fillna('')

    pos_break = df.groupby('contig_id')['kmer_position'].diff().ne(1)
    presence_break = presence.ne(presence.shift())
    df['block_id'] = (pos_break | presence_break).cumsum()

    df['block_size'] = df.groupby('block_id')['block_id'].transform('size')
    df = df[df['block_size'] >= min_block_size].copy()
    df['block_id'] = pd.factorize(df['block_id'])[0] + 1

    df_gpd = df[['block_id', 'block_size', 'presence_list']].drop_duplicates(subset=['block_id'])
    return df, df_gpd

def main():
    parser = argparse.ArgumentParser(
        description='Select strain-informative kmers and build kmer pairs.')
    parser.add_argument('--reference', required=True, 
                        help='Genome FASTA (.fna or .fna.gz)')
    parser.add_argument('--input_dir', required=True, 
                        help='directory of scrub output files, containing summary, global counts and presence file')
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--basename', default=None, 
                        help='Output basename (default: derived from --genome)')
    parser.add_argument('--terminal_dist', type=int, default=300,
                        help='Distance from contig ends to flag terminal kmers')
    parser.add_argument('--min_block_size', type =int, default = 20)
    parser.add_argument('--percentile', type = float, default = 0.01,
                        help = 'percentile union on pan genome and metagenome counts for rare kmer selection')
    args = parser.parse_args()

    basename = args.basename if args.basename else strain_name_from_path(args.genome)
    os.makedirs(args.output_dir, exist_ok=True)

    # read dataframes
    df_global = pl.read_csv(os.path.join(args.input_dir, '*.global_counts.tsv.zst'),
                            separator = '\t')
    df_summary = pl.read_csv(os.path.join(args.input_dir, '*.summary.tsv'),
                            separator = '\t')
    df_presence = pl.read_csv(os.path.join(args.input_dir, '*.presence.tsv.zst'),
                            separator = '\t')
    
    print(df_global.sort(pl.col('pangenome_count')))

    df_rare = get_lowest_percentile(df_global.to_pandas(), percentile=args.percentile, drug_scrub='count_hard')
    print(df_rare.sort_values('pangenome_count'))

    # map to reference
    genome = load_genome(args.reference)
    df_locations, _ = build_mapped_kmers_ahocorasick(genome, df_rare['#kmer'], terminal_dist= args.terminal_dist)
    if len(df_locations) == 0:
        print('No kmers mapped to reference!')


    df_merge = pd.merge(df_locations, df_global.to_pandas(), on = '#kmer', how = 'left')
    print(df_merge)
    df_merge2 = pd.merge(df_merge, df_presence.to_pandas(), on = '#kmer', how = 'left')

    # convert scrub ids to actual lists, empty list where pangenome and metagenome_count ==0 else drop na
    print(df_merge2)
    missing = df_merge2['list_scrub_id'].isna()
    has_hits = df_merge2[['pangenome_count', 'metagenome_count']].gt(0).any(axis=1)

    # capped presence list -> the empty entry is an artefact, not a real absence
    n_capped = int((missing & has_hits).sum())
    df_merge2 = df_merge2[~(missing & has_hits)].copy()
    # export for now
    # remaining NaNs are genuine absences -> empty list
    df_merge2['presence_list'] = df_merge2['list_scrub_id'].apply(
        lambda s: [int(x) for x in str(s).split(',')] if pd.notna(s) else [])

    print(f"dropped {n_capped} k-mers with capped presence lists")
    df_merge2  = df_merge2.loc[df_merge2['terminal_kmer'] == False].copy()

    df, df_gpd = add_block_ids(df_merge2, min_block_size=args.min_block_size)
    df.to_csv(os.path.join(args.output_dir, f"{basename}.rare_kmers.mapped.tsv"),
            sep = '\t', index = None)

    df_gpd.to_csv(os.path.join(args.output_dir, f"{basename}.kmer_blocks.tsv"), 
            sep = '\t', index = None)
    
    
if __name__ == '__main__':
    main()