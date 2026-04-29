#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include "kseq.h"
#include "BIO_sequence.h"
#include "BIO_hash.h"
#include "genome_compare.h"
#include "kmer_scrub_streaming.h"

KSEQ_INIT(gzFile, gzread)

/*
	kmer_scrub_count_individual

	Same CLI as before. Two output streams:

	  1) Global counts to stdout (TSV) — unchanged:
	       #kmer  reference_count  pangenome_count  metagenome_count  [drug_count]

	  2) Per-sample counts as a long-format gzipped TSV via -o:
	       kmer  sample_type  sample_id  count
	     where sample_type ∈ {ge, me, dr} and sample_id is the basename of
	     the input file with extensions stripped.

	Memory model
	------------
	Hash value vector layout per k-mer is (4 + num_threads) columns:
	  [0..3]                  global counts (ref, pan, meta, drug)
	  [4 .. 4+T-1]            per-thread scratch — exclusively owned by
	                          worker thread tid

	Per-sample output is streamed: each worker counts a sample into its
	own column, sweeps the hash to emit non-zero rows for that sample,
	zeros its column, and moves on to the next sample. A single dedicated
	writer thread owns the gzFile.

	Memory is therefore O(n_kmers × (4 + T)) regardless of the number of
	samples in -A / -B / -C.
*/

#define N_GLOBAL_COLS 4   /* ref, pan, meta, drug */

static void usage(void);
static void print_hash_counts(BIO_hash seqHash, const char *C_file);

int main(int argc, char *argv[])
{
	BIO_hash seqHash;
	char *A_file = NULL;       /* list of genome filenames */
	char *B_file = NULL;       /* list of metagenome filenames */
	char *C_file = NULL;       /* list of drug-strain genome filenames (optional) */
	char *r_file = NULL;       /* reference genome */
	char *p_file = NULL;       /* progress log (optional) */
	char *o_file = NULL;       /* per-sample long TSV.gz output (optional) */
	const int seed = 31;
	const int default_hash_val = 1;
	const int default_hash_increment = 1;
	int num_threads = 4;
	int c;
	FILE *progress = NULL;

	while ((c = getopt(argc, argv, "A:B:C:r:p:o:t:Hhud")) != EOF)
		switch (c) {
			case 'A': A_file = strdup(optarg); break;
			case 'B': B_file = strdup(optarg); break;
			case 'C': C_file = strdup(optarg); break;
			case 'r': r_file = strdup(optarg); break;
			case 'p': p_file = strdup(optarg); break;
			case 'o': o_file = strdup(optarg); break;
			case 't': num_threads = atoi(optarg); break;
			case 'u':
			case 'h':
			default: usage(); break;
		}

	if (!r_file || !A_file || !B_file) {
		usage();
		return 1;
	}
	if (num_threads < 1) num_threads = 1;

	if (p_file != NULL) {
		progress = fopen(p_file, "w");
		if (progress == NULL) {
			fprintf(stderr, "could not open progress file %s\n", p_file);
			exit(EXIT_FAILURE);
		}
		fprintf(progress, "adding kmer counts for:\n");
	}

	/*
		Hash size: 4 global columns + one scratch column per worker
		thread. This is the entire memory cost of per-sample tracking
		and is independent of the number of samples in -A/-B/-C.
	*/
	const int size_of_hash_vec = N_GLOBAL_COLS + num_threads;

	seqHash = BIO_initHash(DEFAULT_GENOME_HASH_SIZE);

	/* seed the hash with reference k-mers (column 0) */
	GEN_hash_sequences_set_count_vec(r_file, seed, seqHash,
	                                 default_hash_val, default_hash_increment,
	                                 0, size_of_hash_vec);

	/* --- Phase 1: global counts (columns 1, 2, 3) — unchanged API --- */
	GEN_all_kmer_counts(A_file, seed, seqHash, 1, progress, num_threads);
	GEN_all_kmer_counts(B_file, seed, seqHash, 2, progress, num_threads);
	if (C_file)
		GEN_all_kmer_counts_skip_file(C_file, r_file, seed, seqHash, 3,
		                              progress, num_threads);

	/* --- Phase 2: per-sample counts streamed to gzipped TSV --- */
	if (o_file) {
		streaming_writer *w = streaming_writer_open(o_file, 0);
		if (!w) {
			fprintf(stderr, "could not open per-sample output %s\n", o_file);
			exit(EXIT_FAILURE);
		}

		GEN_per_sample_kmer_counts_streaming(A_file, "ge", seed, seqHash,
		                                     num_threads, w, progress, NULL);
		GEN_per_sample_kmer_counts_streaming(B_file, "me", seed, seqHash,
		                                     num_threads, w, progress, NULL);
		if (C_file)
			GEN_per_sample_kmer_counts_streaming(C_file, "dr", seed, seqHash,
			                                     num_threads, w, progress, r_file);

		streaming_writer_close(w);
	}

	/* --- output 1: global counts to stdout (unchanged format) --- */
	print_hash_counts(seqHash, C_file);

	/* cleanup */
	BIO_destroyHashD(seqHash);
	free(A_file);
	free(B_file);
	free(C_file);
	free(r_file);
	free(p_file);
	free(o_file);
	if (progress != NULL) fclose(progress);
	return 0;
}

static void usage(void)
{
	fprintf(stderr,
	    "Usage: kmer_scrub_count_individual\n"
	    "                        -r <reference genome>\n"
	    "                        -A <file listing genome filenames>\n"
	    "                        -B <file listing metagenome filenames>\n"
	    "                       [-C <file listing drug-strain genome filenames>]\n"
	    "                       [-o <per-sample long-format output, .tsv.gz>]\n"
	    "                       [-p <progress log file>]\n"
	    "                       [-t <num threads, default 4>]\n"
	    "\n"
	    "  stdout: global counts (kmer, ref, pangenome, metagenome[, drug])\n"
	    "  -o:     long-format gzipped TSV (kmer, sample_type, sample_id, count)\n"
	    "          sample_type \xe2\x88\x88 {ge, me, dr}; load with polars.scan_csv.\n"
	    "          Memory bounded by O(n_kmers \xc3\x97 (4 + threads)).\n");
	exit(1);
}

static void print_hash_counts(BIO_hash seqHash, const char *C_file)
{
	char **allKeys = BIO_getHashKeys(seqHash);
	int hash_size = BIO_getHashSize(seqHash);
	unsigned int *counts = NULL;

	if (C_file)
		printf("#kmer\treference_count\tpangenome_count\tmetagenome_count\tdrug_count\n");
	else
		printf("#kmer\treference_count\tpangenome_count\tmetagenome_count\n");

	for (unsigned int i = 0; i < (unsigned int)hash_size; i++) {
		counts = (unsigned int*)BIO_searchHash(seqHash, allKeys[i]);
		if (C_file)
			printf("%s\t%u\t%u\t%u\t%u\n",
			       allKeys[i], counts[0], counts[1], counts[2], counts[3]);
		else
			printf("%s\t%u\t%u\t%u\n",
			       allKeys[i], counts[0], counts[1], counts[2]);
	}
	BIO_destroyHashKeys(allKeys);
}