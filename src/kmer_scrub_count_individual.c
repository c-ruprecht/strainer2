#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include "BIO_sequence.h"
#include "BIO_hash.h"
#include "genome_compare.h"
#include "kmer_scrub_streaming.h"

/*
	kmer_scrub_count_individual (single-pass variant)

	Same CLI as before. Same outputs:

	  1) Global counts to stdout (TSV) — unchanged format:
	       #kmer  reference_count  pangenome_count  metagenome_count  [drug_count]

	  2) Per-sample counts as a long-format gzipped TSV via -o:
	       kmer  sample_type  sample_id  count

	Architectural change vs the previous (two-pass) version
	-------------------------------------------------------
	Previously: Phase 1 read every input file to fill global cols 1/2/3,
	then Phase 2 read every input file again to fill per-sample cols.
	Each file was opened and decompressed twice.

	Now: a single pass per input file. Each worker's k-mer counter writes
	to BOTH the global column (1, 2, or 3) and its scratch column
	(4 + tid) on every k-mer match. After finishing a file, the worker
	sweeps the hash to emit non-zero rows for its scratch column and
	zero them, exactly as before.

	Wall-clock impact: roughly 2x speedup on I/O-bound workloads (which
	is everything at this scale, since metagenome decompression dominates).

	Memory model — unchanged:
	  Hash value vector has (4 + num_threads) columns.
	  Memory cost is O(n_kmers x (4 + T)), independent of n_samples.
*/

#define N_GLOBAL_COLS 4

static void usage(void);
static void print_hash_counts(BIO_hash seqHash, const char *C_file);

int main(int argc, char *argv[])
{
	BIO_hash seqHash;
	char *A_file = NULL;
	char *B_file = NULL;
	char *C_file = NULL;
	char *r_file = NULL;
	char *p_file = NULL;
	char *o_file = NULL;
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
		Hash size: 4 global columns + one scratch column per worker thread.
		Independent of n_samples — that's the whole point.
	*/
	const int size_of_hash_vec = N_GLOBAL_COLS + num_threads;

	seqHash = BIO_initHash(DEFAULT_GENOME_HASH_SIZE);

	/* seed the hash with reference k-mers (column 0) */
	GEN_hash_sequences_set_count_vec(r_file, seed, seqHash,
	                                 default_hash_val, default_hash_increment,
	                                 0, size_of_hash_vec);

	/*
		Single pass per list. Each call:
		  - reads each file in the list once
		  - increments counts[global_col] AND counts[4 + tid] per k-mer hit
		  - emits non-zero scratch rows + zeros the column per sample

		The -o flag is required for this binary; without per-sample
		output there's no reason to use _individual over the original
		kmer_scrub_count.
	*/
	if (!o_file) {
		fprintf(stderr, "error: -o is required for kmer_scrub_count_individual\n");
		fprintf(stderr, "       (use kmer_scrub_count if you only need globals)\n");
		exit(EXIT_FAILURE);
	}

	streaming_writer *w = streaming_writer_open(o_file, 0);
	if (!w) {
		fprintf(stderr, "could not open per-sample output %s\n", o_file);
		exit(EXIT_FAILURE);
	}

	/* -A → pangenome (col 1) + ge rows */
	GEN_per_sample_kmer_counts_dual(A_file, "ge", 1, seed, seqHash,
	                                num_threads, w, progress, NULL);

	/* -B → metagenome (col 2) + me rows */
	GEN_per_sample_kmer_counts_dual(B_file, "me", 2, seed, seqHash,
	                                num_threads, w, progress, NULL);

	/* -C → drug (col 3) + dr rows; reference-skip preserved */
	if (C_file)
		GEN_per_sample_kmer_counts_dual(C_file, "dr", 3, seed, seqHash,
		                                num_threads, w, progress, r_file);

	streaming_writer_close(w);

	/* global counts to stdout (unchanged format) */
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
	    "                        -o <per-sample long-format output, .tsv.gz>  REQUIRED\n"
	    "                       [-p <progress log file>]\n"
	    "                       [-t <num threads, default 4>]\n"
	    "\n"
	    "  stdout: global counts (kmer, ref, pangenome, metagenome[, drug])\n"
	    "  -o:     long-format gzipped TSV (kmer, sample_type, sample_id, count)\n"
	    "          sample_type in {ge, me, dr}; load with polars.scan_csv.\n"
	    "  Each input file is read exactly once. Memory bounded by\n"
	    "  O(n_kmers x (4 + threads)).\n");
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
