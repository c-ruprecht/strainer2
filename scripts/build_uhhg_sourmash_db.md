1. downloaded full gff set
2. convert gff to fasta
3. convert to sigs in batches

4. create batch zip files
find /metrica/databases/genomes/mgnify_uhgg_v2.0.2/batches -name "sigs.zip" | sort \
  > /metrica/databases/genomes/mgnify_uhgg_v2.0.2/all_batch_zips.txt

5. build rocks db

sourmash scripts index     /metrica/databases/genomes/mgnify_uhgg_v2.0.2/all_batch_zips.txt     -o /metrica/databases/genomes/mgnify_uhgg_v2.0.2/uhgg_v2.0.2_k31_s1000.rocksdb     -k 31     -c 32

6. run manysearch


sourmash scripts manysearch \
/metrica/scratch/strainer_dev/strains/sourmash-mtc01/signatures/all_samples.sig.zip \
/metrica/databases/genomes/mgnify_uhgg_v2.0.2/uhgg_v2.0.2_k31_s1000.rocksdb \
-o /metrica/codebase/strainer2-fork/scripts/dev/sourmash_uhhg_search.csv