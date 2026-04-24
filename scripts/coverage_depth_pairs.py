import pandas as pd
import plotly.express as px
import numpy as np
import os
import polars as pl
import gc
import argparse
from kmer_pairs import create_all_kmer_pairs, process_sample


def read_kmer_hits(path_to_kmer_hits, path_to_genome_map):
    #gets total reads 
    df = pd.read_csv(path_to_kmer_hits, sep = '\t')
    df_total_reads = df.loc[df['#kmer']=='total_evaluated'].copy().set_index('#kmer').T
    #df_total_reads = df_total_reads.reset_index()
    dict_total_reads = df_total_reads['total_evaluated'].to_dict()
    # get kmerhits and map locations
    df_kmer_hits = df.loc[df['#kmer']!= 'total_evaluated'].copy()
    df_locations = pd.read_csv(path_to_genome_map, sep = '\t')

    df_merge = pd.merge(df_kmer_hits, df_locations, on = ['#kmer'], how = 'left')
    df_merge = df_merge.set_index(df_locations.columns.to_list()).stack()
    df_merge = df_merge.reset_index()
    df_merge = df_merge.rename(columns={'level_13': 'sample', 0: 'count'})
    df_merge['strain'] = str(path_to_genome_map).split('/')[-1].split('.rare_kmers_mapped.')[0]
    return df_merge, dict_total_reads

def visualize_count_map(df_hits_stack, df_coverage, outdir, min_coverage = 0.02):
    #visualize coverage for strains with more coverage of unique kmers than threshhold
    for sample in df_coverage.loc[df_coverage['coverage_kmer_single']>min_coverage]['sample'].unique():
        os.makedirs(outdir+f'/plots/{sample}', exist_ok=True)
        print(sample)
        df_sample = df_hits_stack.loc[df_hits_stack['sample']==sample]
        fig = px.box(df_sample,
                     x= 'contig_id',
                     y = 'count_per10B_kmers',
                     template = 'simple_white',
                     color = 'sample',
                     title = df_sample['strain'].unique()[0]
                     )
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.write_image(outdir + f'/plots/{sample}/contig-box_{sample}.svg')
        fig = px.histogram(df_sample,
                x = 'count',
                #y = 'count',
                facet_col = 'sample',
                facet_col_wrap=4,
                template= 'simple_white',
                width = 1000)
        
        fig.write_image(outdir + f'/plots/{sample}/histogram-counts_{sample}.svg')
        
        os.makedirs(outdir+f'/plots/{sample}/contig_coverage_plots/', exist_ok=True)
        for contig, df_contig in df_sample.groupby('contig_id'):#speeds up for bad assemblies with a lot of contigs
            fig = px.scatter(df_contig,
                            x = 'kmer_position',
                            y = 'count',
                            color = 'contig_id',
                            facet_col = 'sample',
                            facet_col_wrap=4,
                            template= 'simple_white',
                            width = 1000)
            fig.write_image(outdir + f'/plots/{sample}/contig_coverage_plots/{contig}.svg')


    return

def main():
    parser = argparse.ArgumentParser(description='Calcualting coverage and depth with kmer locations')
    parser.add_argument('--location', help='a target strain .rare_kmers_mapped.tsv.gz file')
    parser.add_argument('--hits', help='a target_strain .kmer_hits.tsv.gz')
    parser.add_argument('--inform_kmer_singles')
    parser.add_argument('--inform_kmer_pairs', help = 'parquet file of kmer pair counts')
    parser.add_argument('--output_dir', help='directory where output is saved')
    parser.add_argument('--figures', help = 'if ommited no visualization will be produced')
    parser.add_argument('--min_coverage', help = 'minimal coverage for which figurs will be created. default 0.1 ')
    args = parser.parse_args()

    location = args.location
    hits = args.hits
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    df_hits_stack, dict_total_reads = read_kmer_hits(hits, location)
    print(df_hits_stack.columns, dict_total_reads)
    df_hits_stack['total_kmers_evaluated'] = df_hits_stack['sample'].map(dict_total_reads)
    df_hits_stack['count_per10B_kmers'] = round(df_hits_stack['count']/df_hits_stack['total_kmers_evaluated']*10**10,1)
    df_hits_stack


    df_cov_depth = df_hits_stack.groupby(['strain','sample']).agg(**{'total_unique_kmers': ('#kmer', 'nunique'),
                                                            'total_kmers_with_count': ('count_per10B_kmers', lambda x: sum(x>0)),
                                                            'count_mean': ('count', 'mean'),
                                                            'count_mean_per10B_kmers': ('count_per10B_kmers', 'mean'),
                                                            'count_mean_excl0_per10B_kmers': ('count_per10B_kmers', lambda x: x[x>0].mean())})

    df_cov_depth['coverage_kmer_single'] = df_cov_depth['total_kmers_with_count']/df_cov_depth['total_unique_kmers']
    df_cov_depth = df_cov_depth.reset_index()
    #df_cov_depth.sort_values(['coverage_kmer_single'], ascending= False).to_csv(output_dir+'/coverage_depth.tsv', index = False, sep = '\t')
    #print(df_cov_depth.loc[df_cov_depth['coverage']>0.02]['sample'].unique())
    visualize_count_map(df_hits_stack, df_cov_depth, outdir = output_dir, min_coverage=0.1)


    # Get informative kmer coverage
    df_samples = pl.read_csv(args.hits, separator="\t")
    df_samples = df_samples.filter(pl.col("#kmer") != "total_evaluated")
    pair_cols = ["#kmer"] + [col for col in df_samples.columns if col != "#kmer" and df_samples[col].sum() > 1]
    df_samples = df_samples.select(pair_cols)
    print('Samples without any kmer hits: ' + str(len(pair_cols)))
    print(df_samples)
    ### Singletons
    df_singletons = pl.read_parquet(args.inform_kmer_singles)
    

    ### Pairs
    print('creating all kmer pairs')
    df_kmer_pairs = pl.read_parquet(args.inform_kmer_pairs)
    print(df_kmer_pairs)
    sample_cols = [c for c in df_samples.columns if c != '#kmer']

    # Joins sample counts once on kmer A and once on kmer B
    result = df_kmer_pairs.join(
        df_samples.rename({'#kmer': 'kmerA'}),
        on='kmerA',
        how='left',
        suffix='_A'
    )

    result = result.join(
        df_samples.rename({'#kmer': 'kmerB'}),
        on='kmerB',
        how='left',
        suffix='_B'
    )
    ### HOW TO COMBINE COUNTS?!

    # minimal count of one of the two?
    # pl.min_horizontal(pl.col(s), pl.col(f'{s}_B')).alias(s) for s in sample_cols
    pair_cols = [pl.min_horizontal(pl.col(s), pl.col(f'{s}_B')).alias(s) for s in sample_cols]
    result = result.with_columns(pair_cols).drop([f'{s}_B' for s in sample_cols])
    print(result)
    total_pairs = len(result)
    df_cov_depth['total_kmer_pairs'] = len(result)
    # calculate full pair coverage
    coverage = (result.select([(pl.col(s) != 0).sum().alias(s) for s in sample_cols]).row(0, named=True))
    
    # divide by total to get coverage fraction
    dict_coverage = {sample: n_nonzero / total_pairs for sample, n_nonzero in coverage.items()}
    df_cov_depth['coverage_kmer_pair'] = df_cov_depth['sample'].map(dict_coverage)

    #
    depth_pairs = result.select([pl.col(s).mean().alias(s) for s in sample_cols]).row(0, named=True)
    df_cov_depth['mean_depth_kmer_pairs'] = df_cov_depth['sample'].map(depth_pairs)
    
    # Sum counts as weights  for all kmer pairs found
    weight_pairs = result.select([pl.col('count').filter(pl.col(s) > 0).sum().alias(s) for s in sample_cols]).row(0, named=True)
    df_cov_depth['weight_kmer_pairs'] = df_cov_depth['sample'].map(weight_pairs)
    ### NEED TO CHECK THIS
    # top 10% by reference count (>= 90th percentile)
    threshold = result.select(pl.col('count').quantile(0.01)).item()
    top_ref_pairs = result.filter(pl.col('count') <= threshold)
    total_top_pairs = top_ref_pairs.height

    coverage_top = top_ref_pairs.select(
        [(pl.col(s) != 0).sum().alias(s) for s in sample_cols]
    ).row(0, named=True)
    dict_coverage_top = {s: n / total_top_pairs for s, n in coverage_top.items()}

    depth_top = top_ref_pairs.select(
        [pl.col(s).mean().alias(s) for s in sample_cols]
    ).row(0, named=True)

    df_cov_depth['total_top1pct_ref_pairs'] = total_top_pairs
    df_cov_depth['coverage_top1pct_ref_pairs'] = df_cov_depth['sample'].map(dict_coverage_top)
    df_cov_depth['mean_depth_top1pct_ref_pairs'] = df_cov_depth['sample'].map(depth_top)
    # calcualted 0 count pait coverage
    # filter to pairs with reference count == 0
    zero_ref_pairs = result.filter(pl.col('count') == 0)
    total_zero_pairs = zero_ref_pairs.height
    coverage_zero = zero_ref_pairs.select([(pl.col(s) != 0).sum().alias(s) for s in sample_cols]).row(0, named=True)
    dict_coverage_zero = {s: n / total_zero_pairs for s, n in coverage_zero.items()}
    
    df_cov_depth['total_zero_ref_pairs'] = total_zero_pairs
    df_cov_depth['coverage_zero_ref_pairs'] = df_cov_depth['sample'].map(dict_coverage_zero)

    # row(0, named) is a dictioaniry style output that can be mapped
    depth_zero = zero_ref_pairs.select([pl.col(s).mean().alias(s) for s in sample_cols]).row(0, named=True)
    df_cov_depth['mean_depth_zero_ref_pairs'] = df_cov_depth['sample'].map(depth_zero)

    # coverage depth figure for zero pairs
    # coverage depth figure
    fig = px.scatter(df_cov_depth, x='count_mean', y='coverage_kmer_single',
                log_x=True, template='simple_white',
                hover_data=['sample'], width=600)

    fig1 = px.scatter(df_cov_depth, x='mean_depth_zero_ref_pairs', y='coverage_zero_ref_pairs',
                    log_x=True, template='simple_white',
                    hover_data=['sample'], range_y=[0,1], width=600)

    fig2 = px.scatter(df_cov_depth, x='mean_depth_kmer_pairs', y='coverage_kmer_pair',
                    log_x=True, template='simple_white',
                    hover_data=['sample'], range_y=[0,1], width=600)

    fig3 = px.scatter(df_cov_depth, x='mean_depth_top1pct_ref_pairs', y='coverage_top1pct_ref_pairs',
                    log_x=True, template='simple_white',
                    hover_data=['sample'], range_y=[0,1], width=600)

    # label + color each series
    fig.data[0].update(name='single kmers',    marker_color='#1f77b4', showlegend=True)
    fig1.data[0].update(name='zero-ref pairs', marker_color='#ff7f0e', showlegend=True)
    fig2.data[0].update(name='all pairs',      marker_color='#2ca02c', showlegend=True)
    fig3.data[0].update(name='rare-ref pairs [1 percentile]', marker_color='#d62728', showlegend=True)

    # add traces from fig1, fig2, fig3 onto fig
    for trace in fig1.data:
        fig.add_trace(trace)
    for trace in fig2.data:
        fig.add_trace(trace)
    for trace in fig3.data:
        fig.add_trace(trace)

    fig.update_layout(legend_title_text='metric', xaxis_title='depth', yaxis_title='coverage')
    fig.write_image(output_dir + '/coverage-depth-combined.svg')

    df_cov_depth.sort_values(['coverage_kmer_single'], ascending= False).to_csv(output_dir+'/coverage_depth.tsv', index = False, sep = '\t')


if __name__ == '__main__':
    main()
