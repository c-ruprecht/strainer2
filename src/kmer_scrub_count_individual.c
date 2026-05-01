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

	Outputs:
	  1) Global counts to stdout (TSV) — unchanged format:
	       #kmer  reference_count  pangenome_count  metagenome_count  [drug_count]

	  2) Per-sample counts as a long-format zstd-compressed TSV via -o (optional):
	       kmer  sample_type  sample_id  count

	  3) Per-sample summary plain TSV via -S (optional):
	       sample_type  sample_id  n_unique_kmers  coverage_pct
	                              is_in_global  is_in_individual

	Coverage band gate (-T LOW,HIGH, default 0.0,1.0 = disabled):

	    coverage  <  LOW   → in_global=True,  in_individual=False
	      (sparse hit, contributes to background but not worth carrying
	       per-sample)

	    LOW <= coverage <= HIGH
	                       → in_global=True,  in_individual=True

	    coverage  >  HIGH  → in_global=False, in_individual=False
	      (looks like the reference itself; drop entirely so it doesn't
	       skew the global background)

	  -T accepts either "LOW,HIGH" (e.g. "0.1,0.96") or a single value
	  which is interpreted as HIGH with LOW=0.0 (back-compat with the
	  earlier single-threshold behavior).

	Architecture: single-pass per file, hash bounded by O(n_kmers x (4 + T)).

	Duplicate sample_id detection: if two input files yield the same
	(sample_type, sample_id) pair (e.g. same basename), the second one
	is skipped with a warning to stderr. Detection is process-global
	across -A / -B / -C lists.
*/

#define N_GLOBAL_COLS 4
#define DEFAULT_COV_LOW  0.0
#define DEFAULT_COV_HIGH 1.0

static void usage(void);
static void print_hash_counts(BIO_hash seqHash, const char *C_file);
static unsigned long count_seeded_kmers(BIO_hash seqHash);
static int parse_cov_range(const char *arg, double *lo, double *hi);

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
	double cov_low  = DEFAULT_COV_LOW;
	double cov_high = DEFAULT_COV_HIGH;
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
			case 'T':
				if (parse_cov_range(optarg, &cov_low, &cov_high) != 0) {
					fprintf(stderr,
					        "error: -T expects 'LOW,HIGH' (e.g. 0.1,0.96) "
					        "or a single HIGH value, got '%s'\n", optarg);
					exit(EXIT_FAILURE);
				}
				break;
			case 'u':
			case 'h':
			default: usage(); break;
		}

	if (!r_file || !A_file || !B_file) {
		usage();
		return 1;
	}
	if (num_threads < 1) num_threads = 1;
	if (cov_low < 0.0 || cov_high < 0.0) {
		fprintf(stderr, "error: -T values must be >= 0 (got %g,%g)\n",
		        cov_low, cov_high);
		exit(EXIT_FAILURE);
	}
	if (cov_low > cov_high) {
		fprintf(stderr,
		        "error: -T low (%g) must be <= high (%g)\n",
		        cov_low, cov_high);
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

	/* seed the hash with reference k-mers (column 0) */
	GEN_hash_sequences_set_count_vec(r_file, seed, seqHash,
	                                 default_hash_val, default_hash_increment,
	                                 0, size_of_hash_vec);

	/* count seeded reference k-mers — denominator for coverage_pct */
	unsigned long total_ref_kmers = count_seeded_kmers(seqHash);
	fprintf(stderr, "total reference k-mers: %lu\n", total_ref_kmers);
	fprintf(stderr,
	        "coverage band: low=%g high=%g  "
	        "(coverage<low → global only; coverage>high → drop)\n",
	        cov_low, cov_high);
	if (cov_low <= 0.0 && cov_high >= 1.0)
		fprintf(stderr, "  (band fully open — no samples will be filtered)\n");

	streaming_writer *w = NULL;
	if (o_file) {
		w = streaming_writer_open(o_file, 0);
		if (!w) {
			fprintf(stderr, "could not open per-sample output %s\n", o_file);
			exit(EXIT_FAILURE);
		}
	} else {
		fprintf(stderr,
		    "no -o given: per-sample counts will not be written "
		    "(in_individual will always be False)\n");
	}

	summary_writer *summary = NULL;
	if (s_file) {
		summary = summary_writer_open(s_file, total_ref_kmers);
		if (!summary) {
			fprintf(stderr, "could not open summary output %s\n", s_file);
			exit(EXIT_FAILURE);
		}
	}

	/* Shared dedup registry — process-global across -A/-B/-C. */
	seen_registry *seen = seen_registry_new();
	if (!seen) {
		fprintf(stderr, "could not allocate seen_registry\n");
		exit(EXIT_FAILURE);
	}

	/* -A → pangenome (col 1) + ge rows */
	GEN_per_sample_kmer_counts_dual(A_file, "ge", 1, seed, seqHash,
	                                num_threads, w, summary, seen,
	                                cov_low, cov_high, total_ref_kmers,
	                                progress, NULL);

	/* -B → metagenome (col 2) + me rows */
	GEN_per_sample_kmer_counts_dual(B_file, "me", 2, seed, seqHash,
	                                num_threads, w, summary, seen,
	                                cov_low, cov_high, total_ref_kmers,
	                                progress, NULL);

	/* -C → drug (col 3) + dr rows; reference-skip preserved */
	if (C_file)
		GEN_per_sample_kmer_counts_dual(C_file, "dr", 3, seed, seqHash,
		                                num_threads, w, summary, seen,
		                                cov_low, cov_high, total_ref_kmers,
		                                progress, r_file);

	if (w) streaming_writer_close(w);
	if (summary) summary_writer_close(summary);
	seen_registry_free(seen);

	/* global counts to stdout (unchanged format) */
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

/* Parse "LOW,HIGH" (e.g. "0.1,0.96") or a single value treated as HIGH
   with LOW defaulting to 0.0. Returns 0 on success, -1 on parse error.
   *lo and *hi are only modified on success. */
static int parse_cov_range(const char *arg, double *lo, double *hi)
{
	if (!arg || !*arg) return -1;

	const char *comma = strchr(arg, ',');
	char *end = NULL;

	if (comma == NULL) {
		/* single value → high only, low stays at default 0.0 */
		double v = strtod(arg, &end);
		if (end == arg || *end != '\0') return -1;
		*lo = 0.0;
		*hi = v;
		return 0;
	}

	/* split on the comma */
	size_t lo_len = (size_t)(comma - arg);
	char lo_buf[64];
	if (lo_len == 0 || lo_len >= sizeof lo_buf) return -1;
	memcpy(lo_buf, arg, lo_len);
	lo_buf[lo_len] = '\0';

	double lv = strtod(lo_buf, &end);
	if (end == lo_buf || *end != '\0') return -1;

	const char *hi_str = comma + 1;
	if (*hi_str == '\0') return -1;
	double hv = strtod(hi_str, &end);
	if (end == hi_str || *end != '\0') return -1;

	*lo = lv;
	*hi = hv;
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
	    "                       [-o <per-sample long-format output, .tsv.zst>]\n"
	    "                       [-S <per-sample summary output, .tsv>]\n"
	    "                       [-T <LOW,HIGH coverage band, default 0.0,1.0>]\n"
	    "                       [-p <progress log file>]\n"
	    "                       [-t <num threads, default 4>]\n"
	    "\n"
	    "  stdout: global counts (kmer, ref, pangenome, metagenome[, drug])\n"
	    "  -o:     long-format zstd-compressed TSV (one row per kmer x sample).\n"
	    "          If omitted, per-sample counts are not written; only stdout\n"
	    "          globals (and -S, if given) are produced. Even when -o is set,\n"
	    "          a sample is excluded from this file when its coverage falls\n"
	    "          outside the LOW..HIGH band.\n"
	    "          load with polars: pl.scan_csv('out.tsv.zst', separator='\\t')\n"
	    "  -S:     plain TSV summary, one row per sample, columns:\n"
	    "          sample_type, sample_id, n_unique_kmers, coverage_pct,\n"
	    "          is_in_global, is_in_individual\n"
	    "          coverage_pct = n_unique_kmers / total_reference_kmers\n"
	    "  -T:     coverage band as 'LOW,HIGH' (e.g. '0.1,0.96'):\n"
	    "            coverage <  LOW   → in global only (not individual)\n"
	    "            in [LOW, HIGH]    → in both\n"
	    "            coverage >  HIGH  → dropped from both\n"
	    "          A single value is interpreted as HIGH with LOW=0.0\n"
	    "          (back-compat with previous single-threshold behavior).\n"
	    "          Default 0.0,1.0 disables the band.\n"
	    "  Each input file is read exactly once. Memory bounded by\n"
	    "  O(n_kmers x (4 + threads)). Duplicate (sample_type, sample_id)\n"
	    "  pairs are skipped with a warning.\n");
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
