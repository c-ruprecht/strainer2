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
	kmer_scrub_count_individual (inverted-index variant)

	Outputs:
	  1) Global counts to stdout (TSV) — unchanged format:
	       #kmer  reference_count  pangenome_count  metagenome_count  [drug_count]

	  2) Inverted presence file via -o (optional), zstd-compressed TSV:
	       #kmer  list_scrub_id
	     where list_scrub_id is comma-separated u32 — the scrub_ids of
	     all samples that hit that k-mer.

	  3) Per-sample summary plain TSV via -S (optional):
	       scrub_id  sample_type  sample_id  n_unique_kmers  coverage_pct
	               is_in_global

	  scrub_id is sequential u32 (starting at 0), assigned in completion
	  order. The same id appears in the presence file's list_scrub_id
	  column.

	Coverage threshold (-T): if coverage_pct > threshold, the sample's
	counts are NOT folded into the global column (is_in_global=False).
	The sample STILL gets a scrub_id, summary row, and presence appends.
	Default 1.0 disables the gate.

	Architecture: single-pass per file. Workers do hot loop + scratch
	sweep in parallel; a single writer thread assigns scrub_ids and
	appends them to per-bucket id-lists (one queue record per sample).
	The presence file is materialized in one serial sweep at the end.

	Diagnostic counters (writer-bound vs. worker-bound) are printed at
	end. If the writer queue waits a lot for empty queue, workers are
	the bottleneck (good). If workers wait a lot for full queue, the
	writer is the bottleneck.
*/

#define N_GLOBAL_COLS 4
#define DEFAULT_COVERAGE_THRESHOLD 1.0

static void usage(void);
static void print_hash_counts(BIO_hash seqHash, const char *C_file);
static unsigned long count_seeded_kmers(BIO_hash seqHash);

int main(int argc, char *argv[])
{
	BIO_hash seqHash;
	char *A_file = NULL;
	char *B_file = NULL;
	char *C_file = NULL;
	char *r_file = NULL;
	char *p_file = NULL;
	char *o_file = NULL;
	char *s_file = NULL;
	const int seed = 31;
	const int default_hash_val = 1;
	const int default_hash_increment = 1;
	int num_threads = 4;
	double cov_threshold = DEFAULT_COVERAGE_THRESHOLD;
	int c;
	FILE *progress = NULL;

	while ((c = getopt(argc, argv, "A:B:C:r:p:o:S:t:T:Hhud")) != EOF)
		switch (c) {
			case 'A': A_file = strdup(optarg); break;
			case 'B': B_file = strdup(optarg); break;
			case 'C': C_file = strdup(optarg); break;
			case 'r': r_file = strdup(optarg); break;
			case 'p': p_file = strdup(optarg); break;
			case 'o': o_file = strdup(optarg); break;
			case 'S': s_file = strdup(optarg); break;
			case 't': num_threads = atoi(optarg); break;
			case 'T': cov_threshold = atof(optarg); break;
			case 'u':
			case 'h':
			default: usage(); break;
		}

	if (!r_file || !A_file || !B_file) {
		usage();
		return 1;
	}
	if (num_threads < 1) num_threads = 1;
	if (cov_threshold < 0.0) {
		fprintf(stderr, "error: -T must be >= 0 (got %g)\n", cov_threshold);
		exit(EXIT_FAILURE);
	}

	if (p_file != NULL) {
		progress = fopen(p_file, "w");
		if (progress == NULL) {
			fprintf(stderr, "could not open progress file %s\n", p_file);
			exit(EXIT_FAILURE);
		}
		fprintf(progress, "adding kmer counts for:\n");
	}

	const int size_of_hash_vec = N_GLOBAL_COLS + num_threads;

	seqHash = BIO_initHash(DEFAULT_GENOME_HASH_SIZE);

	GEN_hash_sequences_set_count_vec(r_file, seed, seqHash,
	                                 default_hash_val, default_hash_increment,
	                                 0, size_of_hash_vec);

	unsigned long total_ref_kmers = count_seeded_kmers(seqHash);
	fprintf(stderr, "total reference k-mers: %lu\n", total_ref_kmers);
	fprintf(stderr, "coverage threshold for global accumulation: %g%s\n",
	        cov_threshold,
	        cov_threshold >= 1.0 ? " (disabled — all samples included)" : "");

	presence_writer *w = NULL;
	if (o_file) {
		w = presence_writer_open(o_file, /*queue_capacity*/ 0);
		if (!w) {
			fprintf(stderr, "could not open presence writer %s\n", o_file);
			exit(EXIT_FAILURE);
		}
	} else {
		fprintf(stderr,
		        "no -o given: presence file will not be written "
		        "(summary still emitted if -S given)\n");
	}

	summary_writer *summary = NULL;
	if (s_file) {
		summary = summary_writer_open(s_file, total_ref_kmers);
		if (!summary) {
			fprintf(stderr, "could not open summary output %s\n", s_file);
			exit(EXIT_FAILURE);
		}
	}

	seen_registry *seen = seen_registry_new();
	if (!seen) {
		fprintf(stderr, "could not allocate seen_registry\n");
		exit(EXIT_FAILURE);
	}

	/* -A → pangenome (col 1) + ge rows */
	GEN_per_sample_kmer_counts_dual(A_file, "ge", 1, seed, seqHash,
	                                num_threads, w, summary, seen,
	                                cov_threshold, total_ref_kmers,
	                                progress, NULL);

	/* -B → metagenome (col 2) + me rows */
	GEN_per_sample_kmer_counts_dual(B_file, "me", 2, seed, seqHash,
	                                num_threads, w, summary, seen,
	                                cov_threshold, total_ref_kmers,
	                                progress, NULL);

	/* -C → drug (col 3) + dr rows; reference-skip preserved */
	if (C_file)
		GEN_per_sample_kmer_counts_dual(C_file, "dr", 3, seed, seqHash,
		                                num_threads, w, summary, seen,
		                                cov_threshold, total_ref_kmers,
		                                progress, r_file);

	/* Drain + join the writer thread, then materialize presence file. */
	if (w) {
		presence_writer_close(w);              /* joins writer thread */
		presence_writer_flush(w, seqHash);     /* serial zstd dump */
		presence_writer_print_diagnostics(w);
		presence_writer_destroy(w);
	}
	if (summary) summary_writer_close(summary);
	seen_registry_free(seen);

	/* Global counts to stdout (unchanged format). */
	print_hash_counts(seqHash, C_file);

	BIO_destroyHashD(seqHash);
	free(A_file);
	free(B_file);
	free(C_file);
	free(r_file);
	free(p_file);
	free(o_file);
	free(s_file);
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
	    "                       [-o <inverted presence file, .tsv.zst>]\n"
	    "                       [-S <per-sample summary output, .tsv>]\n"
	    "                       [-T <coverage threshold, default 1.0>]\n"
	    "                       [-p <progress log file>]\n"
	    "                       [-t <num threads, default 4>]\n"
	    "\n"
	    "  stdout: global counts (#kmer, ref, pangenome, metagenome[, drug])\n"
	    "  -o:     INVERTED presence index, zstd-compressed TSV.\n"
	    "          Format: #kmer<TAB>list_scrub_id (comma-separated u32 list).\n"
	    "          Only kmers with at least one sample hit are emitted.\n"
	    "          load with polars:\n"
	    "            pl.scan_csv('out.tsv.zst', separator='\\t')\n"
	    "              .with_columns(pl.col('list_scrub_id').str.split(','))\n"
	    "  -S:     plain TSV summary, columns:\n"
	    "          scrub_id, sample_type, sample_id, n_unique_kmers,\n"
	    "          coverage_pct, is_in_global\n"
	    "          scrub_id is sequential u32 starting at 0, assigned in\n"
	    "          sample-completion order. The same id appears in -o.\n"
	    "  -T:     coverage threshold. If coverage_pct > T, the sample's\n"
	    "          counts are NOT added to the global column (is_in_global\n"
	    "          becomes False), but scrub_id is still allocated and\n"
	    "          presence/summary rows still emitted. Default 1.0 disables.\n"
	    "  Each input file is read exactly once. Memory bounded by\n"
	    "  O(n_kmers x (4 + threads)) for the hash + O(total_kmer_appends x 4)\n"
	    "  for the inverted index. Duplicate (sample_type, sample_id) pairs\n"
	    "  are skipped with a warning.\n"
	    "\n"
	    "  Diagnostic counters (queue waits etc.) are printed at end of run\n"
	    "  to stderr. If 'worker queue waits' is large, the writer is the\n"
	    "  bottleneck. If 'writer queue waits' is large, the workers are.\n");
	exit(1);
}

static unsigned long count_seeded_kmers(BIO_hash seqHash)
{
	unsigned long n = 0;
	int hash_size = BIO_getHashSize(seqHash);
	char **allKeys = BIO_getHashKeys(seqHash);
	for (unsigned int i = 0; i < (unsigned int)hash_size; i++) {
		unsigned int *counts = (unsigned int *)BIO_searchHash(seqHash, allKeys[i]);
		if (counts && counts[0] > 0) n++;
	}
	BIO_destroyHashKeys(allKeys);
	return n;
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
