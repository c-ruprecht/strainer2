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

KSEQ_INIT(gzFile, gzread)

/*
	kmer_scrub_count

	Outputs:
	  1. Global counts to stdout (TSV) — unchanged from previous version:
	       #kmer  reference_count  pangenome_count  metagenome_count  [drug_count]

	  2. (NEW) Per-sample counts as a long-format gzipped TSV via -o:
	       kmer  sample_type  sample_id  count
	     where sample_type ∈ {ge, me, dr} and sample_id is the basename of
	     the input file with extensions stripped. Streamable into Polars:
	       pl.scan_csv("out.tsv.gz", separator="\t")

	Hash value layout (per kmer):
	  [0] reference count
	  [1] pangenome (global) count
	  [2] metagenome (global) count
	  [3] drug (global) count
	  [4 .. 4+nA-1]            per-genome sample counts        (ge_*)
	  [4+nA .. 4+nA+nB-1]      per-metagenome sample counts    (me_*)
	  [4+nA+nB .. ...]         per-drug-genome sample counts   (dr_*)

	Race conditions:
	  Per-sample columns are disjoint, so threads from the worker pool
	  processing different samples never contend on the same word. Within a
	  single sample the existing __sync_fetch_and_add in
	  GEN_calculate_kmer_count handles same-slot increments atomically.
*/

#define N_GLOBAL_COLS 4   /* ref, pan, meta, drug */

void usage();
void print_hash_counts(BIO_hash seqHash, const char *C_file);
void write_per_sample_long_tsv_gz(BIO_hash seqHash,
                                  char **ge_ids, int n_ge,
                                  char **me_ids, int n_me,
                                  char **dr_ids, int n_dr,
                                  const char *out_path);

/* helpers for sample list bookkeeping */
static char **read_file_list(const char *list_path, int *n_out);
static char  *basename_no_ext(const char *path);
static void   free_str_array(char **arr, int n);

/* declared in genome_compare.c (and should be added to genome_compare.h) */
extern void GEN_per_sample_kmer_counts(const char *list_file, const int seed,
                                       BIO_hash seqHash, unsigned int base_column,
                                       FILE *progress, int num_threads,
                                       const char *skip_file);

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

	if (p_file != NULL) {
		progress = fopen(p_file, "w");
		if (progress == NULL) {
			fprintf(stderr, "could not open progress file %s\n", p_file);
			exit(EXIT_FAILURE);
		}
		fprintf(progress, "adding kmer counts for:\n");
	}

	/* --- read sample lists so we know how many per-sample columns to allocate --- */
	int n_ge = 0, n_me = 0, n_dr = 0;
	char **ge_files = read_file_list(A_file, &n_ge);
	char **me_files = read_file_list(B_file, &n_me);
	char **dr_files = NULL;
	if (C_file) dr_files = read_file_list(C_file, &n_dr);

	if (n_ge == 0 || n_me == 0) {
		fprintf(stderr, "empty sample list in -A or -B\n");
		exit(EXIT_FAILURE);
	}

	/* derive sample IDs from basenames */
	char **ge_ids = malloc(sizeof(char*) * n_ge);
	char **me_ids = malloc(sizeof(char*) * n_me);
	char **dr_ids = (n_dr > 0) ? malloc(sizeof(char*) * n_dr) : NULL;
	for (int i = 0; i < n_ge; i++) ge_ids[i] = basename_no_ext(ge_files[i]);
	for (int i = 0; i < n_me; i++) me_ids[i] = basename_no_ext(me_files[i]);
	for (int i = 0; i < n_dr; i++) dr_ids[i] = basename_no_ext(dr_files[i]);

	/* total columns in the count vector */
	const int size_of_hash_vec = N_GLOBAL_COLS + n_ge + n_me + n_dr;

	const unsigned int GE_BASE = N_GLOBAL_COLS;
	const unsigned int ME_BASE = GE_BASE + n_ge;
	const unsigned int DR_BASE = ME_BASE + n_me;

	seqHash = BIO_initHash(DEFAULT_GENOME_HASH_SIZE);

	/* seed the hash with reference kmers (column 0) */
	GEN_hash_sequences_set_count_vec(r_file, seed, seqHash,
	                                 default_hash_val, default_hash_increment,
	                                 0, size_of_hash_vec);

	/* --- global counts: existing API, fills columns 1, 2, 3 --- */
	GEN_all_kmer_counts(A_file, seed, seqHash, 1, progress, num_threads);
	GEN_all_kmer_counts(B_file, seed, seqHash, 2, progress, num_threads);
	if (C_file)
		GEN_all_kmer_counts_skip_file(C_file, r_file, seed, seqHash, 3,
		                              progress, num_threads);

	/* --- per-sample counts: each file → its own dedicated column.
	   Pooled in parallel just like the global counts. Disjoint columns
	   eliminate inter-sample races; intra-sample races are handled by
	   the atomic increment in GEN_calculate_kmer_count. --- */
	if (o_file) {
		GEN_per_sample_kmer_counts(A_file, seed, seqHash, GE_BASE,
		                           progress, num_threads, NULL);
		GEN_per_sample_kmer_counts(B_file, seed, seqHash, ME_BASE,
		                           progress, num_threads, NULL);
		if (C_file)
			GEN_per_sample_kmer_counts(C_file, seed, seqHash, DR_BASE,
			                           progress, num_threads, r_file);
	}

	/* --- output 1: global counts to stdout (unchanged format) --- */
	print_hash_counts(seqHash, C_file);

	/* --- output 2: long-format gzipped TSV of per-sample counts --- */
	if (o_file) {
		write_per_sample_long_tsv_gz(seqHash,
		                             ge_ids, n_ge,
		                             me_ids, n_me,
		                             dr_ids, n_dr,
		                             o_file);
	}

	/* cleanup */
	BIO_destroyHashD(seqHash);
	free_str_array(ge_files, n_ge);
	free_str_array(me_files, n_me);
	free_str_array(dr_files, n_dr);
	free_str_array(ge_ids, n_ge);
	free_str_array(me_ids, n_me);
	free_str_array(dr_ids, n_dr);
	free(A_file);
	free(B_file);
	free(C_file);
	free(r_file);
	free(p_file);
	free(o_file);
	if (progress != NULL) fclose(progress);
	return 0;
}

void usage()
{
	fprintf(stderr,
	    "Usage: kmer_scrub_count -r <reference genome>\n"
	    "                        -A <file listing genome filenames>\n"
	    "                        -B <file listing metagenome filenames>\n"
	    "                       [-C <file listing drug-strain genome filenames>]\n"
	    "                       [-o <per-sample long-format output, .tsv.gz>]\n"
	    "                       [-p <progress log file>]\n"
	    "                       [-t <num threads, default 4>]\n"
	    "\n"
	    "  stdout: global counts (kmer, ref, pangenome, metagenome[, drug])\n"
	    "  -o:     long-format gzipped TSV (kmer, sample_type, sample_id, count)\n"
	    "          sample_type ∈ {ge, me, dr}; load with polars.scan_csv.\n");
	exit(1);
}

/* unchanged: writes the original global-counts TSV to stdout */
void print_hash_counts(BIO_hash seqHash, const char *C_file)
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

/*
	Long-format per-sample writer.

	One row per (kmer, sample) pair where count > 0. Skipping zeros keeps
	the file small — most kmers are absent from most samples — and Polars
	handles the implicit zeros via pivot/fill_null on the consumer side.
	If you want dense output, remove the `if (c == 0) continue;` lines.

	Writes via gzopen at level 9 (max compression) so the file streams
	directly into pl.scan_csv("...tsv.gz", separator="\t").
*/
void write_per_sample_long_tsv_gz(BIO_hash seqHash,
                                  char **ge_ids, int n_ge,
                                  char **me_ids, int n_me,
                                  char **dr_ids, int n_dr,
                                  const char *out_path)
{
	gzFile gz = gzopen(out_path, "wb9");
	if (!gz) {
		fprintf(stderr, "could not open %s for writing\n", out_path);
		exit(EXIT_FAILURE);
	}
	/* larger internal buffer = fewer syscalls on big writes */
	gzbuffer(gz, 1 << 20);

	/* header */
	gzprintf(gz, "kmer\tsample_type\tsample_id\tcount\n");

	const int GE_BASE = N_GLOBAL_COLS;
	const int ME_BASE = GE_BASE + n_ge;
	const int DR_BASE = ME_BASE + n_me;

	char **allKeys = BIO_getHashKeys(seqHash);
	int hash_size = BIO_getHashSize(seqHash);

	for (unsigned int i = 0; i < (unsigned int)hash_size; i++) {
		unsigned int *counts =
		    (unsigned int*)BIO_searchHash(seqHash, allKeys[i]);
		const char *kmer = allKeys[i];

		for (int j = 0; j < n_ge; j++) {
			unsigned int c = counts[GE_BASE + j];
			if (c == 0) continue;
			gzprintf(gz, "%s\tge\t%s\t%u\n", kmer, ge_ids[j], c);
		}
		for (int j = 0; j < n_me; j++) {
			unsigned int c = counts[ME_BASE + j];
			if (c == 0) continue;
			gzprintf(gz, "%s\tme\t%s\t%u\n", kmer, me_ids[j], c);
		}
		for (int j = 0; j < n_dr; j++) {
			unsigned int c = counts[DR_BASE + j];
			if (c == 0) continue;
			gzprintf(gz, "%s\tdr\t%s\t%u\n", kmer, dr_ids[j], c);
		}
	}

	BIO_destroyHashKeys(allKeys);
	gzclose(gz);
}

/* --- small helpers ----------------------------------------------------- */

static char **read_file_list(const char *list_path, int *n_out)
{
	FILE *f = fopen(list_path, "r");
	if (!f) {
		fprintf(stderr, "could not open list file %s\n", list_path);
		exit(EXIT_FAILURE);
	}
	int cap = 64, n = 0;
	char **arr = malloc(sizeof(char*) * cap);
	char buf[4096];
	while (fgets(buf, sizeof(buf), f)) {
		size_t len = strlen(buf);
		while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r' ||
		                   buf[len-1] == ' '  || buf[len-1] == '\t'))
			buf[--len] = '\0';
		if (len == 0) continue;
		if (n == cap) {
			cap *= 2;
			arr = realloc(arr, sizeof(char*) * cap);
		}
		arr[n++] = strdup(buf);
	}
	fclose(f);
	*n_out = n;
	return arr;
}

/* basename without directory and stripping common compression / fasta exts */
static char *basename_no_ext(const char *path)
{
	const char *base = strrchr(path, '/');
	base = base ? base + 1 : path;
	char *out = strdup(base);

	/* peel off up to two extensions: .fasta.gz, .fa.gz, .fastq.gz, .fna.gz, etc. */
	for (int pass = 0; pass < 2; pass++) {
		char *dot = strrchr(out, '.');
		if (!dot) break;
		const char *ext = dot + 1;
		if (strcmp(ext, "gz")    == 0 ||
		    strcmp(ext, "bz2")   == 0 ||
		    strcmp(ext, "xz")    == 0 ||
		    strcmp(ext, "fa")    == 0 ||
		    strcmp(ext, "fna")   == 0 ||
		    strcmp(ext, "ffn")   == 0 ||
		    strcmp(ext, "fasta") == 0 ||
		    strcmp(ext, "fastq") == 0 ||
		    strcmp(ext, "fq")    == 0)
			*dot = '\0';
		else
			break;
	}
	return out;
}

static void free_str_array(char **arr, int n)
{
	if (!arr) return;
	for (int i = 0; i < n; i++) free(arr[i]);
	free(arr);
}