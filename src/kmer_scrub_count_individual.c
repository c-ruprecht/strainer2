#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <errno.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "BIO_sequence.h"
#include "BIO_hash.h"
#include "genome_compare.h"
#include "kmer_scrub_streaming.h"

/*
	kmer_scrub_count_individual (inverted-index variant)

	All output goes to one directory (-O) under one basename (-n, or the
	reference filename with its directory and extension stripped):

	  1) <dir>/<base>.global_counts.tsv.zst
	       #kmer  reference_count  pangenome_count  metagenome_count
	              [drug_count]

	  2) <dir>/<base>.summary.tsv
	       scrub_id  sample_type  sample_id  n_unique_kmers
	                 coverage_pct  is_in_global

	  3) <dir>/<base>.presence.tsv.zst
	       #kmer  list_scrub_id
	     where list_scrub_id is comma-separated u32 — the scrub_ids of
	     all samples that hit that k-mer.

	scrub_id is sequential u32 (starting at 0), assigned in completion
	order. The same id ties the summary and presence files together.

	Coverage threshold (-T): if coverage_pct > threshold, the sample's
	counts are NOT folded into the global column (is_in_global=False).
	The sample STILL gets a scrub_id, summary row, and presence appends.
	Default 1.0 disables the gate.

	Presence cap (-P): k-mers hit by more than -P samples are dropped from
	the presence file (their id-list is freed and never written or grown).
	Bounds both output size and peak memory for ubiquitous k-mers. Default
	10; -P 0 disables the cap.

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
#define DEFAULT_PRESENCE_MAX 10
#define GLOBAL_COUNTS_ZSTD_LEVEL 9

static void usage(void);
static char *derive_basename(const char *path);
static char *build_path(const char *dir, const char *base, const char *suffix);
static int   ensure_dir(const char *dir);
static void  write_global_counts(BIO_hash seqHash, const char *path,
                                 int with_drug_col);
static unsigned long count_seeded_kmers(BIO_hash seqHash);

int main(int argc, char *argv[])
{
	BIO_hash seqHash;
	char *A_file = NULL;
	char *B_file = NULL;
	char *C_file = NULL;
	char *r_file = NULL;
	char *p_file = NULL;
	char *out_dir = NULL;
	char *basename_arg = NULL;
	const int seed = 31;
	const int default_hash_val = 1;
	const int default_hash_increment = 1;
	int num_threads = 4;
	double cov_threshold = DEFAULT_COVERAGE_THRESHOLD;
	long presence_max = DEFAULT_PRESENCE_MAX;
	int c;
	FILE *progress = NULL;

	while ((c = getopt(argc, argv, "A:B:C:r:p:O:n:t:T:P:Hhud")) != EOF)
		switch (c) {
			case 'A': A_file = strdup(optarg); break;
			case 'B': B_file = strdup(optarg); break;
			case 'C': C_file = strdup(optarg); break;
			case 'r': r_file = strdup(optarg); break;
			case 'p': p_file = strdup(optarg); break;
			case 'O': out_dir = strdup(optarg); break;
			case 'n': basename_arg = strdup(optarg); break;
			case 't': num_threads = atoi(optarg); break;
			case 'T': cov_threshold = atof(optarg); break;
			case 'P': presence_max = atol(optarg); break;
			case 'u':
			case 'h':
			default: usage(); break;
		}

	if (!r_file || !A_file) {
		usage();
		return 1;
	}
	if (!basename_arg) {
		basename_arg = derive_basename(r_file);
		if (!basename_arg) {
			fprintf(stderr, "error: could not derive an output basename "
			        "from -r %s; pass -n explicitly\n", r_file);
			exit(EXIT_FAILURE);
		}
		fprintf(stderr, "no -n given: using basename \"%s\" "
		        "(derived from -r %s)\n", basename_arg, r_file);
	}
	if (!out_dir) out_dir = strdup(".");
	if (num_threads < 1) num_threads = 1;
	if (cov_threshold < 0.0) {
		fprintf(stderr, "error: -T must be >= 0 (got %g)\n", cov_threshold);
		exit(EXIT_FAILURE);
	}
	if (presence_max < 0 || presence_max > (long)UINT32_MAX) {
		fprintf(stderr,
		        "error: -P must be between 0 and %u (got %ld)\n",
		        UINT32_MAX, presence_max);
		exit(EXIT_FAILURE);
	}
	if (strchr(basename_arg, '/') != NULL) {
		fprintf(stderr, "error: -n must be a bare basename, not a path "
		        "(got %s)\n", basename_arg);
		exit(EXIT_FAILURE);
	}
	if (ensure_dir(out_dir) != 0) {
		fprintf(stderr, "error: could not create output directory %s: %s\n",
		        out_dir, strerror(errno));
		exit(EXIT_FAILURE);
	}

	char *global_path   = build_path(out_dir, basename_arg,
	                                 ".global_counts.tsv.zst");
	char *summary_path  = build_path(out_dir, basename_arg, ".summary.tsv");
	char *presence_path = build_path(out_dir, basename_arg,
	                                 ".presence.tsv.zst");

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
	fprintf(stderr, "presence cap for inverted index: %ld%s\n",
	        presence_max,
	        presence_max == 0
	            ? " (disabled — all hit k-mers written)"
	            : " (k-mers seen in more samples are dropped)");
	fprintf(stderr, "output basename: %s/%s\n", out_dir, basename_arg);

	presence_writer *w = presence_writer_open(presence_path,
	                                          /*queue_capacity*/ 0,
	                                          (uint32_t)presence_max);
	if (!w) {
		fprintf(stderr, "could not open presence writer %s\n", presence_path);
		exit(EXIT_FAILURE);
	}

	summary_writer *summary = summary_writer_open(summary_path,
	                                              total_ref_kmers);
	if (!summary) {
		fprintf(stderr, "could not open summary output %s\n", summary_path);
		exit(EXIT_FAILURE);
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
	if (B_file)
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
	presence_writer_close(w);              /* joins writer thread */
	presence_writer_flush(w, seqHash);     /* serial zstd dump */
	presence_writer_print_diagnostics(w);
	presence_writer_destroy(w);

	summary_writer_close(summary);
	seen_registry_free(seen);

	write_global_counts(seqHash, global_path, C_file != NULL);

	BIO_destroyHashD(seqHash);
	free(A_file);
	free(B_file);
	free(C_file);
	free(r_file);
	free(p_file);
	free(out_dir);
	free(basename_arg);
	free(global_path);
	free(summary_path);
	free(presence_path);
	if (progress != NULL) fclose(progress);
	return 0;
}

static void usage(void)
{
	fprintf(stderr,
	    "Usage: kmer_scrub_count_individual\n"
	    "                        -r <reference genome>\n"
	    "                        -A <file listing genome filenames>\n"
	    "                       [-n <output basename, default: -r filename\n"
	    "                            with directory and extension stripped>]\n"
	    "                       [-O <output directory, default .>]\n"
	    "                       [-B <file listing metagenome filenames>]\n"
	    "                       [-C <file listing drug-strain genome filenames>]\n"
	    "                       [-T <coverage threshold, default 1.0>]\n"
	    "                       [-P <presence cap, default 10>]\n"
	    "                       [-p <progress log file>]\n"
	    "                       [-t <num threads, default 4>]\n"
	    "\n"
	    "  Writes three files into <-O>, all sharing the <-n> basename\n"
	    "  (the directory is created if it does not exist):\n"
	    "\n"
	    "    <base>.global_counts.tsv.zst\n"
	    "        zstd TSV. #kmer, reference_count, pangenome_count,\n"
	    "        metagenome_count[, drug_count when -C is given].\n"
	    "    <base>.summary.tsv\n"
	    "        plain TSV. scrub_id, sample_type, sample_id,\n"
	    "        n_unique_kmers, coverage_pct, is_in_global.\n"
	    "        scrub_id is sequential u32 starting at 0, assigned in\n"
	    "        sample-completion order, and keys into the presence file.\n"
	    "    <base>.presence.tsv.zst\n"
	    "        zstd TSV INVERTED presence index. #kmer<TAB>list_scrub_id\n"
	    "        (comma-separated u32). Only kmers with at least one\n"
	    "        sample hit are emitted. Load with polars:\n"
	    "          pl.scan_csv('...presence.tsv.zst', separator='\\t')\n"
	    "            .with_columns(pl.col('list_scrub_id').str.split(','))\n"
	    "\n"
	    "  -T:     coverage threshold. If coverage_pct > T, the sample's\n"
	    "          counts are NOT added to the global column (is_in_global\n"
	    "          becomes False), but scrub_id is still allocated and\n"
	    "          presence/summary rows still emitted. Default 1.0 disables.\n"
	    "  -P:     presence cap for the inverted index. A k-mer hit by\n"
	    "          more than P samples is dropped: its id-list is freed and\n"
	    "          it is never written nor allowed to grow further.\n"
	    "          Bounds both output size and peak memory for ubiquitous\n"
	    "          k-mers. Keeps k-mers with presence <= P. Default 10;\n"
	    "          P=0 disables the cap. Does NOT affect the global counts\n"
	    "          or the summary — only the presence file.\n"
	    "  Each input file is read exactly once. Memory bounded by\n"
	    "  O(n_kmers x (4 + threads)) for the hash + O(kept_kmer_appends x 4)\n"
	    "  for the inverted index. Duplicate (sample_type, sample_id) pairs\n"
	    "  are skipped with a warning.\n"
	    "\n"
	    "  Diagnostic counters (queue waits etc.) are printed at end of run\n"
	    "  to stderr. If 'worker queue waits' is large, the writer is the\n"
	    "  bottleneck. If 'writer queue waits' is large, the workers are.\n");
	exit(1);
}

/* Filename of `path` with directories and known sequence/compression
   extensions stripped: ref/GCF_000123.4_genomic.fna.gz -> GCF_000123.4_genomic.
   Extensions are whitelisted rather than "strip after the last dot" so that
   accession-style names keep their version suffix. Returns NULL if nothing
   usable is left. Caller frees. */
static char *derive_basename(const char *path)
{
	static const char *comp_ext[] = { ".gz", ".zst", ".bz2", ".xz", NULL };
	static const char *seq_ext[]  = { ".fa", ".fna", ".fasta", ".ffn", ".faa",
	                                  ".fas", ".fsa", ".seq", ".txt", ".list",
	                                  NULL };

	const char *slash = strrchr(path, '/');
	const char *name  = slash ? slash + 1 : path;
	if (*name == '\0') return NULL;

	char *base = strdup(name);
	if (!base) { perror("strdup basename"); exit(EXIT_FAILURE); }

	/* Strip at most one compression suffix, then one sequence suffix. */
	for (int round = 0; round < 2; round++) {
		const char **table = (round == 0) ? comp_ext : seq_ext;
		size_t len = strlen(base);
		for (const char **e = table; *e; e++) {
			size_t elen = strlen(*e);
			if (len > elen && strcasecmp(base + len - elen, *e) == 0) {
				base[len - elen] = '\0';
				break;
			}
		}
	}

	if (base[0] == '\0') { free(base); return NULL; }
	return base;
}

/* <dir>/<base><suffix>, with a trailing slash on dir collapsed. */
static char *build_path(const char *dir, const char *base, const char *suffix)
{
	size_t dir_len = strlen(dir);
	while (dir_len > 1 && dir[dir_len - 1] == '/') dir_len--;

	size_t need = dir_len + 1 + strlen(base) + strlen(suffix) + 1;
	char *path = malloc(need);
	if (!path) { perror("malloc path"); exit(EXIT_FAILURE); }

	snprintf(path, need, "%.*s/%s%s", (int)dir_len, dir, base, suffix);
	return path;
}

/* mkdir -p. Returns 0 on success (including "already a directory"). */
static int ensure_dir(const char *dir)
{
	struct stat st;
	if (stat(dir, &st) == 0)
		return S_ISDIR(st.st_mode) ? 0 : (errno = ENOTDIR, -1);

	char *tmp = strdup(dir);
	if (!tmp) return -1;

	size_t len = strlen(tmp);
	while (len > 1 && tmp[len - 1] == '/') tmp[--len] = '\0';

	for (char *p = tmp + 1; *p; p++) {
		if (*p != '/') continue;
		*p = '\0';
		if (mkdir(tmp, 0777) != 0 && errno != EEXIST) { free(tmp); return -1; }
		*p = '/';
	}
	if (mkdir(tmp, 0777) != 0 && errno != EEXIST) { free(tmp); return -1; }

	free(tmp);
	return 0;
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

static void write_global_counts(BIO_hash seqHash, const char *path,
                                int with_drug_col)
{
	zstd_out_t *z = zstd_out_open(path, GLOBAL_COUNTS_ZSTD_LEVEL);
	if (!z) {
		fprintf(stderr, "write_global_counts: cannot open %s: %s\n",
		        path, strerror(errno));
		exit(EXIT_FAILURE);
	}

	const char *header = with_drug_col
		? "#kmer\treference_count\tpangenome_count\tmetagenome_count\tdrug_count\n"
		: "#kmer\treference_count\tpangenome_count\tmetagenome_count\n";
	zstd_out_write(z, header, strlen(header));

	char **allKeys = BIO_getHashKeys(seqHash);
	int hash_size = BIO_getHashSize(seqHash);

	size_t row_cap = 4096;
	char  *rowbuf  = malloc(row_cap);
	if (!rowbuf) { perror("malloc rowbuf"); exit(EXIT_FAILURE); }

	uint64_t emitted = 0;
	for (unsigned int i = 0; i < (unsigned int)hash_size; i++) {
		unsigned int *counts =
			(unsigned int *)BIO_searchHash(seqHash, allKeys[i]);
		if (!counts) continue;

		size_t need = strlen(allKeys[i]) + 4 * 12 + 8;
		if (need > row_cap) {
			while (row_cap < need) row_cap *= 2;
			char *nbuf = realloc(rowbuf, row_cap);
			if (!nbuf) { perror("realloc rowbuf"); exit(EXIT_FAILURE); }
			rowbuf = nbuf;
		}

		int n;
		if (with_drug_col)
			n = snprintf(rowbuf, row_cap, "%s\t%u\t%u\t%u\t%u\n",
			             allKeys[i], counts[0], counts[1], counts[2],
			             counts[3]);
		else
			n = snprintf(rowbuf, row_cap, "%s\t%u\t%u\t%u\n",
			             allKeys[i], counts[0], counts[1], counts[2]);

		zstd_out_write(z, rowbuf, (size_t)n);
		emitted++;
	}

	free(rowbuf);
	BIO_destroyHashKeys(allKeys);
	if (zstd_out_close(z) != 0)
		fprintf(stderr, "write_global_counts: error closing %s\n", path);

	fprintf(stderr, "write_global_counts: emitted %" PRIu64 " kmer rows to %s\n",
	        emitted, path);
}
