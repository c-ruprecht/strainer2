import pandas as pd
import plotly.express as px
import numpy as np
import os
import argparse


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
    df_merge = df_merge.rename(columns={'level_9': 'sample', 0: 'count'})
    df_merge['strain'] = str(path_to_genome_map).split('/')[-1].split('.rare_kmers_mapped.')[0]
    return df_merge, dict_total_reads

def visualize_count_map(df_hits_stack, df_coverage, outdir, min_coverage = 0.02):
    #visualize coverage for strains with more coverage of unique kmers than threshhold

    for sample in df_coverage.loc[df_coverage['coverage']>min_coverage]['sample'].unique():
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
        fig.write_image(outdir + f'/contig-box_{sample}.svg')
        fig = px.histogram(df_sample,
                x = 'count',
                #y = 'count',
                facet_col = 'sample',
                facet_col_wrap=4,
                template= 'simple_white',
                width = 1000)
        
        fig.write_image(outdir + f'/histogram-counts_{sample}.svg')
        
        os.makedirs(outdir+f'/contig_coverage_plots/{sample}', exist_ok=True)
        for contig, df_contig in df_sample.groupby('contig_id'):#speeds up for bad assemblies with a lot of contigs
            fig = px.scatter(df_contig,
                            x = 'kmer_position',
                            y = 'count',
                            color = 'contig_id',
                            facet_col = 'sample',
                            facet_col_wrap=4,
                            template= 'simple_white',
                            width = 1000)
            fig.write_image(outdir + f'/contig_coverage_plots/{sample}/{contig}.svg')


    return

def main():
    parser = argparse.ArgumentParser(description='Calcualting coverage and depth with kmer locations')
    parser.add_argument('--location', help='a target strain .rare_kmers_mapped.tsv.gz file')
    parser.add_argument('--hits', help='a target_strain .kmer_hits.tsv.gz')
    parser.add_argument('--output_dir', help='directory where output is saved')
    parser.add_argument('--figures', help = 'if ommited no visualization will be produced')
    parser.add_argument('--min_coverage', help = 'minimal coverage for which figurs will be created. default 0.1 ')
    args = parser.parse_args()

    location = args.location
    hits = args.hits
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    df_hits_stack, dict_total_reads = read_kmer_hits(hits, location)

    df_hits_stack['total_kmers_evaluated'] = df_hits_stack['sample'].map(dict_total_reads)
    df_hits_stack['count_per10B_kmers'] = round(df_hits_stack['count']/df_hits_stack['total_kmers_evaluated']*10**10,1)
    df_hits_stack


    df_cov_depth = df_hits_stack.groupby(['strain','sample']).agg(**{'total_unique_kmers': ('#kmer', 'nunique'),
                                                            'total_kmers_with_count': ('count_per10B_kmers', lambda x: sum(x>0)),
                                                            'count_mean_per10B_kmers': ('count_per10B_kmers', 'mean'),
                                                            'count_mean_excl0_per10B_kmers': ('count_per10B_kmers', lambda x: x[x>0].mean())})

    df_cov_depth['coverage'] = df_cov_depth['total_kmers_with_count']/df_cov_depth['total_unique_kmers']
    df_cov_depth = df_cov_depth.reset_index()
    df_cov_depth.sort_values(['coverage'], ascending= False).to_csv(output_dir+'/coverage_depth.tsv', index = False, sep = '\t')
    #print(df_cov_depth.loc[df_cov_depth['coverage']>0.02]['sample'].unique())
    visualize_count_map(df_hits_stack, df_cov_depth, outdir = output_dir, min_coverage=0.1)


    fig = px.scatter(df_cov_depth,
                    x='count_mean_per10B_kmers',
                    y = 'coverage',
                    log_x= True,
                    template= 'simple_white',
                    hover_data = ['sample'],
                    width =600)
    #fig.show()
    fig.write_image(output_dir + f'/coverage-depth.svg')

if __name__ == '__main__':
    main()
