/*
	kmer_metagenome_db — build a per-metagenome k-mer database (.kmdb).

	Two modes, chosen with -T:

	-T scrub  (default; unchanged behaviour)
	  A PRESENCE database that supplements the genome scrub database.
	  Keeps exactly the metagenome k-mers with

	    metagenome_count >= -c        (default 2: drops most error k-mers)
	    pangenome_count  == 0         (absent from every genome in -A)

	Only presence is stored: a k-mer seen 3x is identical to one seen 100x.
	Counts exist only transiently, as an 8-bit saturating counter used to
	apply -c (so -c must be <= 255). The database is a sorted k-mer list.

	Scrubbing a reference with `kmer_scrub_count_individual -B x.kmdb`
	then attributes each reference k-mer to the genomes OR to the
	metagenome, never both.

	Steps
	  1) count   all canonical k-mers of the reads (-i, repeatable; R1 and R2
	             of one sample go into one database). One reader thread feeds
	             read batches to -t workers; worker t owns hash partition t,
	             so counting needs no locks.
	  2) filter  drop k-mers with count < -c, rebuilding each partition at a
	             lower load factor so step 3's lookups are fast.
	  3) subtract stream every genome in -A (in parallel, one genome per
	             thread at a time) and delete every metagenome k-mer it
	             contains. Genome k-mers are never stored, so memory is set by
	             the metagenome alone.
	  4) write   sort partitions in parallel, k-way merge, write zstd blocks.

	-T sample
	  A COUNT database of one sample: every canonical k-mer of the reads
	  with its exact occurrence count (32-bit), no genome subtraction.
	  Step 1 counts as above, steps 2-3 are skipped, step 4 writes
	  key + count. -c (default 1 here) drops k-mers seen fewer times;
	  keep -c 1 if the database must reproduce read scanning exactly.
	  kmer_strain_detect accepts these files as targets in place of
	  FASTQs; it then looks k-mers up instead of re-reading the sample.
	  The metadata key kmer_positions equals kmer_strain_detect's
	  total_evaluated for the same reads.

	Low-memory mode (-M <passes> or -R <GB>)
	  The k-mer space is split into M hash slices by a second hash,
	  independent of the partition and slot hashes. Pass j reads the
	  reads again and runs steps 1-4 for slice j only, then writes its
	  sorted k-mers to a temporary run file (-W, default next to the
	  output). A final k-way merge of the M runs writes the database.
	  Slices are disjoint and every slice is processed exactly, so the
	  database is identical to a single-pass build.
	  In low-memory mode two more things change, neither affecting output:
	    - tables are pre-sized (from -R's estimate, or from pass 0 for
	      later passes), so the doubling spike (old + new table) is avoided;
	    - the -c filter marks dropped k-mers in place instead of rebuilding
	      the table, so step 2 needs no extra memory (step 3 lookups run
	      at load <= 0.7 instead of <= 0.5, slightly slower).
	  -R first runs one extra read pass with a HyperLogLog sketch to
	  estimate the number of distinct k-mers, then picks the smallest M
	  whose predicted peak fits. -E prints the estimate table and exits.

	Canonical form is max(fwd, revcomp) on 2-bit packed k-mers, identical
	to orient_string()'s choice on ASCII, so the database lines up with
	BIO_hash keys. Windows containing any non-ACGT base are skipped.
*/

#define _GNU_SOURCE
#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#ifdef __GLIBC__
#include <malloc.h>
#endif
#include "kseq.h"
#include "kmdb.h"

KSEQ_INIT(gzFile, gzread)

#define EMPTY_KEY        UINT64_MAX      /* never a valid k-mer for k <= 31 */
#define BATCH_BASES      (8u << 20)
#define N_SLOTS          4
#define INITIAL_TAB_CAP  (1u << 20)
#define COUNT_MAX        255u            /* scrub: saturating 8-bit counter */
#define COUNT32_MAX      UINT32_MAX      /* sample: saturating 32-bit counter */
#define MAX_PASSES       256
#define MAX_LOAD_NUM     7               /* tables grow at load 7/10 */
#define MAX_LOAD_DEN     10
#define RAM_SAFETY       0.90            /* -R: use at most 90% of the budget */
#define HLL_P            14              /* 16384 registers, ~0.8% std error */
#define HLL_M            (1u << HLL_P)
#define RUN_BUF          (1u << 20)      /* stdio buffer per temporary run */
#define FIXED_OVERHEAD   100e6           /* read buffers, writer, slack (bytes) */

enum { MODE_SCRUB = 0, MODE_SAMPLE = 1 };

/* ── small utils ───────────────────────────────────────────────────── */

static double now_s(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}

static void *xmalloc(size_t n)
{
	void *p = malloc(n ? n : 1);
	if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(EXIT_FAILURE); }
	return p;
}

static void *xrealloc(void *p, size_t n)
{
	p = realloc(p, n ? n : 1);
	if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(EXIT_FAILURE); }
	return p;
}

static inline uint64_t mix64(uint64_t x)
{
	x ^= x >> 33; x *= 0xff51afd7ed558ccdull;
	x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull;
	x ^= x >> 33;
	return x;
}

/* partition from the high 32 bits, slot from the low bits */
static inline uint32_t part_of(uint64_t h, uint32_t P)
{
	return (uint32_t)(((h >> 32) * (uint64_t)P) >> 32);
}

/* low-memory pass of a k-mer: a second, independent hash so slices are
   balanced and uncorrelated with partitions and slots */
static inline uint32_t pass_of(uint64_t h, uint32_t M)
{
	uint64_t x = mix64(h + 0x9e3779b97f4a7c15ull);
	return (uint32_t)(((x >> 32) * (uint64_t)M) >> 32);
}

static size_t pow2_at_least(size_t n)
{
	size_t c = 1024;
	while (c < n) c <<= 1;
	return c;
}

/* smallest power-of-two capacity that holds n keys below the growth load */
static size_t cap_for(double n)
{
	return pow2_at_least((size_t)(n * MAX_LOAD_DEN / MAX_LOAD_NUM) + 1);
}

static int has_suffix(const char *s, const char *suf)
{
	size_t n = strlen(s), m = strlen(suf);
	return n >= m && strcasecmp(s + n - m, suf) == 0;
}

static int is_seq_path(const char *p)
{
	static const char *ext[] = {
		".fa", ".fna", ".fasta", ".ffn", ".fas", ".fq", ".fastq", NULL };
	char buf[4096];
	snprintf(buf, sizeof buf, "%s", p);
	size_t n = strlen(buf);
	if (n > 3 && strcasecmp(buf + n - 3, ".gz") == 0) buf[n - 3] = '\0';
	for (int i = 0; ext[i]; i++) if (has_suffix(buf, ext[i])) return 1;
	return 0;
}

static char *sample_name_from(const char *path)
{
	const char *b = strrchr(path, '/');
	char *s = strdup(b ? b + 1 : path);
	static const char *ext[] = {
		".fastq.gz", ".fq.gz", ".fasta.gz", ".fa.gz", ".fna.gz",
		".fastq", ".fq", ".fasta", ".fa", ".fna", NULL };
	for (int i = 0; ext[i]; i++)
		if (has_suffix(s, ext[i])) { s[strlen(s) - strlen(ext[i])] = '\0'; break; }
	return s;
}

/* One path per non-blank line. A single sequence file is a list of one. */
static int read_path_list(const char *file, char ***out)
{
	int cap = 64, n = 0;
	char **v = xmalloc(sizeof(char *) * cap);
	if (is_seq_path(file)) {
		v[n++] = strdup(file);
		*out = v;
		return n;
	}
	FILE *fp = fopen(file, "r");
	if (!fp) {
		fprintf(stderr, "error: cannot open list %s: %s\n", file, strerror(errno));
		exit(EXIT_FAILURE);
	}
	char *line = NULL; size_t lcap = 0; ssize_t len;
	while ((len = getline(&line, &lcap, fp)) != -1) {
		while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r' ||
		                   line[len-1] == ' '  || line[len-1] == '\t'))
			line[--len] = '\0';
		char *q = line;
		while (*q == ' ' || *q == '\t') q++;
		if (*q == '\0' || *q == '#') continue;
		if (n == cap) { cap *= 2; v = xrealloc(v, sizeof(char *) * cap); }
		v[n++] = strdup(q);
	}
	free(line);
	fclose(fp);
	*out = v;
	return n;
}

/* ── temporary run files (low-memory mode) ────────────────────────── */
/* Removed at exit whatever happens (a job killed by a signal leaves
   them behind; they are named <out>.passNNN.tmp or <-W>/<name>.passNNN.tmp). */

static char **g_runs   = NULL;
static int    g_n_runs = 0;

static void cleanup_runs(void)
{
	for (int i = 0; i < g_n_runs; i++)
		if (g_runs[i]) { unlink(g_runs[i]); free(g_runs[i]); g_runs[i] = NULL; }
	g_n_runs = 0;
}

/* ── per-partition open-addressing table ──────────────────────────── */
/* scrub:  counts[] is a saturating 8-bit counter, only used to apply -c.
           After step 2 every survivor has count >= 1; step 3 sets
           0 = deleted (low-memory step 2 also marks dropped k-mers 0).
           counts32 is NULL.
   sample: counts32[] is a saturating 32-bit counter that is written to
           the database. counts is NULL. */

typedef struct {
	uint64_t *keys;
	uint8_t  *counts;
	uint32_t *counts32;
	size_t    cap;   /* power of two */
	size_t    n;
} tab_t;

static void tab_init(tab_t *t, size_t cap, int wide)
{
	t->cap      = cap;
	t->n        = 0;
	t->keys     = xmalloc(cap * sizeof(uint64_t));
	t->counts   = wide ? NULL : xmalloc(cap * sizeof(uint8_t));
	t->counts32 = wide ? xmalloc(cap * sizeof(uint32_t)) : NULL;
	memset(t->keys, 0xFF, cap * sizeof(uint64_t));
}

static void tab_free(tab_t *t)
{
	free(t->keys); free(t->counts); free(t->counts32);
	t->keys = NULL; t->counts = NULL; t->counts32 = NULL; t->cap = t->n = 0;
}

static void tab_put_new(tab_t *t, uint64_t key, uint8_t count)
{
	size_t mask = t->cap - 1, i = mix64(key) & mask;
	while (t->keys[i] != EMPTY_KEY) i = (i + 1) & mask;
	t->keys[i] = key;
	t->counts[i] = count;
	t->n++;
}

static void tab_put_new32(tab_t *t, uint64_t key, uint32_t count)
{
	size_t mask = t->cap - 1, i = mix64(key) & mask;
	while (t->keys[i] != EMPTY_KEY) i = (i + 1) & mask;
	t->keys[i] = key;
	t->counts32[i] = count;
	t->n++;
}

static void tab_grow(tab_t *t)
{
	tab_t nt;
	int wide = t->counts32 != NULL;
	tab_init(&nt, t->cap * 2, wide);
	for (size_t i = 0; i < t->cap; i++) {
		if (t->keys[i] == EMPTY_KEY) continue;
		if (wide) tab_put_new32(&nt, t->keys[i], t->counts32[i]);
		else      tab_put_new(&nt, t->keys[i], t->counts[i]);
	}
	tab_free(t);
	*t = nt;
}

static inline void tab_inc(tab_t *t, uint64_t key, uint64_t h)
{
	if (t->n * MAX_LOAD_DEN >= t->cap * MAX_LOAD_NUM) tab_grow(t);
	size_t mask = t->cap - 1, i = h & mask;
	for (;;) {
		uint64_t k = t->keys[i];
		if (k == key) {
			if (t->counts[i] != COUNT_MAX) t->counts[i]++;
			return;
		}
		if (k == EMPTY_KEY) {
			t->keys[i] = key;
			t->counts[i] = 1;
			t->n++;
			return;
		}
		i = (i + 1) & mask;
	}
}

static inline void tab_inc32(tab_t *t, uint64_t key, uint64_t h)
{
	if (t->n * MAX_LOAD_DEN >= t->cap * MAX_LOAD_NUM) tab_grow(t);
	size_t mask = t->cap - 1, i = h & mask;
	for (;;) {
		uint64_t k = t->keys[i];
		if (k == key) {
			if (t->counts32[i] != COUNT32_MAX) t->counts32[i]++;
			return;
		}
		if (k == EMPTY_KEY) {
			t->keys[i] = key;
			t->counts32[i] = 1;
			t->n++;
			return;
		}
		i = (i + 1) & mask;
	}
}

/* ── shared state ─────────────────────────────────────────────────── */

typedef struct {
	char     *seq;
	size_t    seq_len, seq_cap;
	uint32_t *lens;
	size_t    n, n_cap;
	uint64_t  seqno;
	int       ready;
	int       remaining;
} batch_t;

typedef struct {
	int        k;
	int        mode;           /* MODE_SCRUB / MODE_SAMPLE */
	uint32_t   P;              /* partitions == worker threads */
	uint32_t   min_count;      /* scrub 1..COUNT_MAX, sample 1..COUNT32_MAX */
	tab_t     *tabs;

	/* low-memory passes */
	uint32_t   M;              /* number of passes (1 = single pass) */
	uint32_t   pass;           /* current pass */
	uint8_t   *hll;            /* P * HLL_M registers, estimate pass only */

	/* reads pipeline */
	char     **read_files;
	int        n_read_files;
	batch_t    slots[N_SLOTS];
	uint64_t   n_published;
	int        eof;
	pthread_mutex_t mtx;
	pthread_cond_t  cv;
	uint64_t   n_reads, n_positions, n_valid;
	int        read_error;

	/* genome subtraction */
	char     **genomes;
	int        n_genomes;
	int        next_genome;    /* atomic */
	int        genomes_done;   /* atomic */
	int        genome_errors;  /* atomic */
	uint64_t   removed;        /* atomic */
	double     t_start;
} ctx_t;

typedef struct { ctx_t *c; uint32_t tid; } targ_t;

/* ── step 1: counting ─────────────────────────────────────────────── */

static void publish(ctx_t *c, batch_t *b)
{
	pthread_mutex_lock(&c->mtx);
	b->seqno     = c->n_published++;
	b->remaining = (int)c->P;
	b->ready     = 1;
	pthread_cond_broadcast(&c->cv);
	pthread_mutex_unlock(&c->mtx);
}

static batch_t *acquire_slot(ctx_t *c, uint64_t seqno)
{
	batch_t *b = &c->slots[seqno % N_SLOTS];
	pthread_mutex_lock(&c->mtx);
	while (b->ready) pthread_cond_wait(&c->cv, &c->mtx);
	pthread_mutex_unlock(&c->mtx);
	b->seq_len = 0;
	b->n = 0;
	return b;
}

static void *reader_main(void *arg)
{
	ctx_t *c = arg;
	uint64_t next = 0;
	batch_t *b = acquire_slot(c, next);

	for (int f = 0; f < c->n_read_files; f++) {
		gzFile fp = gzopen(c->read_files[f], "r");
		if (!fp) {
			fprintf(stderr, "error: cannot open reads %s: %s\n",
			        c->read_files[f], strerror(errno));
			c->read_error = 1;
			break;
		}
		gzbuffer(fp, 1 << 20);
		kseq_t *seq = kseq_init(fp);
		int l;
		while ((l = kseq_read(seq)) >= 0) {
			c->n_reads++;
			size_t len = seq->seq.l;
			if (len < (size_t)c->k) continue;
			c->n_positions += len - (size_t)c->k + 1;
			if (b->seq_len + len > b->seq_cap) {
				b->seq_cap = b->seq_len + len > BATCH_BASES
				           ? (b->seq_len + len) * 2 : BATCH_BASES * 2;
				b->seq = xrealloc(b->seq, b->seq_cap);
			}
			if (b->n == b->n_cap) {
				b->n_cap = b->n_cap ? b->n_cap * 2 : 65536;
				b->lens = xrealloc(b->lens, b->n_cap * sizeof(uint32_t));
			}
			memcpy(b->seq + b->seq_len, seq->seq.s, len);
			b->seq_len += len;
			b->lens[b->n++] = (uint32_t)len;
			if (b->seq_len >= BATCH_BASES) {
				publish(c, b);
				b = acquire_slot(c, ++next);
			}
		}
		if (l < -1) {
			fprintf(stderr, "error: %s is truncated or malformed (kseq %d)\n",
			        c->read_files[f], l);
			c->read_error = 1;
		}
		kseq_destroy(seq);
		gzclose(fp);
		fprintf(stderr, "  read %s (%.1fs)\n", c->read_files[f],
		        now_s() - c->t_start);
		if (c->read_error) break;
	}
	if (b->n > 0) publish(c, b);

	pthread_mutex_lock(&c->mtx);
	c->eof = 1;
	pthread_cond_broadcast(&c->cv);
	pthread_mutex_unlock(&c->mtx);
	return NULL;
}

/* Wait for batch s; NULL once the reader is done and s was never published. */
static batch_t *wait_batch(ctx_t *c, uint64_t s)
{
	batch_t *b = &c->slots[s % N_SLOTS];
	pthread_mutex_lock(&c->mtx);
	while (!(b->ready && b->seqno == s) && !(c->eof && s >= c->n_published))
		pthread_cond_wait(&c->cv, &c->mtx);
	int done = !(b->ready && b->seqno == s);
	pthread_mutex_unlock(&c->mtx);
	return done ? NULL : b;
}

static void release_batch(ctx_t *c, batch_t *b)
{
	pthread_mutex_lock(&c->mtx);
	if (--b->remaining == 0) {
		b->ready = 0;
		pthread_cond_broadcast(&c->cv);
	}
	pthread_mutex_unlock(&c->mtx);
}

static void *count_main(void *arg)
{
	targ_t *a = arg;
	ctx_t *c = a->c;
	const uint32_t tid = a->tid, P = c->P, M = c->M, pass = c->pass;
	const int k = c->k;
	const int shift = 2 * (k - 1);
	const uint64_t mask = (k == 32) ? UINT64_MAX : ((1ull << (2 * k)) - 1);
	tab_t *t = &c->tabs[tid];
	const int wide = c->mode == MODE_SAMPLE;   /* loop-invariant */
	uint64_t valid = 0;
	batch_t *b;

	for (uint64_t s = 0; (b = wait_batch(c, s)) != NULL; s++) {
		const char *p = b->seq;
		for (size_t r = 0; r < b->n; r++) {
			const char *end = p + b->lens[r];
			uint64_t fwd = 0, rev = 0;
			int l = 0;
			for (; p < end; p++) {
				uint8_t x = KMDB_NT2BIT[(unsigned char)*p];
				if (x > 3) { l = 0; fwd = rev = 0; continue; }
				fwd = ((fwd << 2) | x) & mask;
				rev = (rev >> 2) | ((uint64_t)(3 - x) << shift);
				if (++l >= k) {
					uint64_t can = fwd > rev ? fwd : rev;
					uint64_t h = mix64(can);
					if (part_of(h, P) == tid &&
					    (M == 1 || pass_of(h, M) == pass)) {
						if (wide) tab_inc32(t, can, h);
						else      tab_inc(t, can, h);
					}
					valid++;
				}
			}
		}
		release_batch(c, b);
	}
	if (tid == 0) c->n_valid = valid;   /* every worker sees every window */
	return NULL;
}

/* ── step 0 (-R / -E): HyperLogLog estimate of distinct k-mers ────── */

static void *hll_main(void *arg)
{
	targ_t *a = arg;
	ctx_t *c = a->c;
	const uint32_t tid = a->tid, P = c->P;
	const int k = c->k;
	const int shift = 2 * (k - 1);
	const uint64_t mask = (k == 32) ? UINT64_MAX : ((1ull << (2 * k)) - 1);
	uint8_t *reg = c->hll + (size_t)tid * HLL_M;
	uint64_t valid = 0;
	batch_t *b;

	for (uint64_t s = 0; (b = wait_batch(c, s)) != NULL; s++) {
		const char *p = b->seq;
		for (size_t r = 0; r < b->n; r++) {
			const char *end = p + b->lens[r];
			uint64_t fwd = 0, rev = 0;
			int l = 0;
			for (; p < end; p++) {
				uint8_t x = KMDB_NT2BIT[(unsigned char)*p];
				if (x > 3) { l = 0; fwd = rev = 0; continue; }
				fwd = ((fwd << 2) | x) & mask;
				rev = (rev >> 2) | ((uint64_t)(3 - x) << shift);
				if (++l >= k) {
					uint64_t can = fwd > rev ? fwd : rev;
					uint64_t h = mix64(can);
					if (part_of(h, P) == tid) {
						uint32_t idx = (uint32_t)(h & (HLL_M - 1));
						uint8_t rank = (uint8_t)(__builtin_ctzll(
							(h >> HLL_P) | (1ull << (64 - HLL_P))) + 1);
						if (rank > reg[idx]) reg[idx] = rank;
					}
					valid++;
				}
			}
		}
		release_batch(c, b);
	}
	if (tid == 0) c->n_valid = valid;
	return NULL;
}

static double hll_estimate(const uint8_t *reg)
{
	const double m = HLL_M;
	double sum = 0.0;
	uint32_t zeros = 0;
	for (uint32_t i = 0; i < HLL_M; i++) {
		sum += ldexp(1.0, -(int)reg[i]);
		if (reg[i] == 0) zeros++;
	}
	double e = (0.7213 / (1.0 + 1.079 / m)) * m * m / sum;
	if (e <= 2.5 * m && zeros) e = m * log(m / zeros);   /* small range */
	return e;
}

/* ── step 2: min-count filter ─────────────────────────────────────── */

static void *filter_main(void *arg)
{
	targ_t *a = arg;
	ctx_t *c = a->c;
	tab_t *t = &c->tabs[a->tid];

	/* Compact survivors to the front and shrink the arrays before building
	   the new table, so the old table and the new one are never both at
	   full size. */
	size_t keep = 0;
	for (size_t i = 0; i < t->cap; i++) {
		if (t->keys[i] != EMPTY_KEY && t->counts[i] >= c->min_count) {
			t->keys[keep]   = t->keys[i];
			t->counts[keep] = t->counts[i];
			keep++;
		}
	}
	uint64_t *ks = xrealloc(t->keys,   (keep ? keep : 1) * sizeof(uint64_t));
	uint8_t  *cs = xrealloc(t->counts, (keep ? keep : 1) * sizeof(uint8_t));

	tab_t nt;
	tab_init(&nt, pow2_at_least(keep * 2), 0);   /* load <= 0.5 */
	for (size_t i = 0; i < keep; i++) tab_put_new(&nt, ks[i], cs[i]);
	free(ks);
	free(cs);
	*t = nt;
	return NULL;
}

/* low-memory step 2: mark dropped k-mers deleted (count 0) in place.
   No allocation; the probe chains stay intact for step 3. t->n becomes
   the number of survivors (it is no longer used as a load measure). */
static void *filter_inplace_main(void *arg)
{
	targ_t *a = arg;
	ctx_t *c = a->c;
	tab_t *t = &c->tabs[a->tid];
	size_t keep = 0;
	for (size_t i = 0; i < t->cap; i++) {
		if (t->keys[i] == EMPTY_KEY) continue;
		if (t->counts[i] >= c->min_count) keep++;
		else t->counts[i] = 0;
	}
	t->n = keep;
	return NULL;
}

/* ── step 3: genome subtraction ───────────────────────────────────── */

static void *subtract_main(void *arg)
{
	targ_t *a = arg;
	ctx_t *c = a->c;
	const uint32_t P = c->P, M = c->M, pass = c->pass;
	const int k = c->k;
	const int shift = 2 * (k - 1);
	const uint64_t mask = (1ull << (2 * k)) - 1;
	uint64_t removed = 0;

	for (;;) {
		int g = __atomic_fetch_add(&c->next_genome, 1, __ATOMIC_RELAXED);
		if (g >= c->n_genomes) break;
		gzFile fp = gzopen(c->genomes[g], "r");
		if (!fp) {
			fprintf(stderr, "error: cannot open genome %s: %s\n",
			        c->genomes[g], strerror(errno));
			__atomic_fetch_add(&c->genome_errors, 1, __ATOMIC_RELAXED);
			continue;
		}
		kseq_t *seq = kseq_init(fp);
		int l;
		while ((l = kseq_read(seq)) >= 0) {
			const char *p = seq->seq.s, *end = p + seq->seq.l;
			uint64_t fwd = 0, rev = 0;
			int n = 0;
			for (; p < end; p++) {
				uint8_t x = KMDB_NT2BIT[(unsigned char)*p];
				if (x > 3) { n = 0; fwd = rev = 0; continue; }
				fwd = ((fwd << 2) | x) & mask;
				rev = (rev >> 2) | ((uint64_t)(3 - x) << shift);
				if (++n < k) continue;
				uint64_t can = fwd > rev ? fwd : rev;
				uint64_t h = mix64(can);
				if (M > 1 && pass_of(h, M) != pass) continue;
				tab_t *t = &c->tabs[part_of(h, P)];
				size_t tm = t->cap - 1, i = h & tm;
				uint64_t key;
				while ((key = t->keys[i]) != EMPTY_KEY) {
					if (key == can) {
						/* mark deleted (count 0); count removals once */
						if (__atomic_load_n(&t->counts[i], __ATOMIC_RELAXED) &&
						    __atomic_exchange_n(&t->counts[i], 0, __ATOMIC_RELAXED))
							removed++;
						break;
					}
					i = (i + 1) & tm;
				}
			}
		}
		if (l < -1) {
			fprintf(stderr, "error: genome %s is truncated or malformed\n",
			        c->genomes[g]);
			__atomic_fetch_add(&c->genome_errors, 1, __ATOMIC_RELAXED);
		}
		kseq_destroy(seq);
		gzclose(fp);

		int done = __atomic_add_fetch(&c->genomes_done, 1, __ATOMIC_RELAXED);
		if (done % 1000 == 0 || done == c->n_genomes)
			fprintf(stderr, "  subtracted %d / %d genomes (%.1fs)\n",
			        done, c->n_genomes, now_s() - c->t_start);
	}
	__atomic_fetch_add(&c->removed, removed, __ATOMIC_RELAXED);
	return NULL;
}

/* ── step 4: compact + sort (keys only) ───────────────────────────── */

static inline void swap_keys(uint64_t *k, size_t i, size_t j)
{
	uint64_t t = k[i]; k[i] = k[j]; k[j] = t;
}

/* In-place quicksort of unique keys. */
static void sort_keys(uint64_t *k, size_t n)
{
	while (n > 24) {
		size_t m = n / 2, hi = n - 1;
		/* median of three moved to k[0] -> Hoare partition is safe */
		if (k[m] < k[0])  swap_keys(k, m, 0);
		if (k[hi] < k[0]) swap_keys(k, hi, 0);
		if (k[hi] < k[m]) swap_keys(k, hi, m);
		swap_keys(k, 0, m);
		uint64_t p = k[0];
		ptrdiff_t i = -1, j = (ptrdiff_t)n;
		for (;;) {
			do i++; while (k[i] < p);
			do j--; while (k[j] > p);
			if (i >= j) break;
			swap_keys(k, (size_t)i, (size_t)j);
		}
		size_t left = (size_t)j + 1;           /* [0, j] and [j+1, n) */
		if (left < n - left) {
			sort_keys(k, left);
			k += left; n -= left;
		} else {
			sort_keys(k + left, n - left);
			n = left;
		}
	}
	for (size_t i = 1; i < n; i++) {
		uint64_t kk = k[i];
		size_t j = i;
		while (j > 0 && k[j - 1] > kk) { k[j] = k[j - 1]; j--; }
		k[j] = kk;
	}
}

static void *compact_sort_main(void *arg)
{
	targ_t *a = arg;
	tab_t *t = &a->c->tabs[a->tid];
	size_t j = 0;
	for (size_t i = 0; i < t->cap; i++) {
		if (t->keys[i] != EMPTY_KEY && t->counts[i] > 0)
			t->keys[j++] = t->keys[i];
	}
	t->n = j;
	/* counts are no longer needed: release them before the sort/merge */
	free(t->counts);
	t->counts = NULL;
	sort_keys(t->keys, j);
	return NULL;
}

/* sample mode: in-place quicksort of unique keys, carrying their counts */
static inline void swap_kv(uint64_t *k, uint32_t *v, size_t i, size_t j)
{
	uint64_t t = k[i]; k[i] = k[j]; k[j] = t;
	uint32_t u = v[i]; v[i] = v[j]; v[j] = u;
}

static void sort_kv(uint64_t *k, uint32_t *v, size_t n)
{
	while (n > 24) {
		size_t m = n / 2, hi = n - 1;
		if (k[m] < k[0])  swap_kv(k, v, m, 0);
		if (k[hi] < k[0]) swap_kv(k, v, hi, 0);
		if (k[hi] < k[m]) swap_kv(k, v, hi, m);
		swap_kv(k, v, 0, m);
		uint64_t p = k[0];
		ptrdiff_t i = -1, j = (ptrdiff_t)n;
		for (;;) {
			do i++; while (k[i] < p);
			do j--; while (k[j] > p);
			if (i >= j) break;
			swap_kv(k, v, (size_t)i, (size_t)j);
		}
		size_t left = (size_t)j + 1;
		if (left < n - left) {
			sort_kv(k, v, left);
			k += left; v += left; n -= left;
		} else {
			sort_kv(k + left, v + left, n - left);
			n = left;
		}
	}
	for (size_t i = 1; i < n; i++) {
		uint64_t kk = k[i];
		uint32_t vv = v[i];
		size_t j = i;
		while (j > 0 && k[j - 1] > kk) { k[j] = k[j - 1]; v[j] = v[j - 1]; j--; }
		k[j] = kk; v[j] = vv;
	}
}

/* sample mode: apply -c while compacting, then sort key+count pairs */
static void *compact_sort_counts_main(void *arg)
{
	targ_t *a = arg;
	ctx_t *c = a->c;
	tab_t *t = &c->tabs[a->tid];
	size_t j = 0;
	for (size_t i = 0; i < t->cap; i++) {
		if (t->keys[i] != EMPTY_KEY && t->counts32[i] >= c->min_count) {
			t->keys[j]     = t->keys[i];
			t->counts32[j] = t->counts32[i];
			j++;
		}
	}
	t->n = j;
	sort_kv(t->keys, t->counts32, j);
	return NULL;
}

static void run_parallel(ctx_t *c, void *(*fn)(void *), uint32_t n)
{
	pthread_t *th = xmalloc(sizeof(pthread_t) * n);
	targ_t *ta = xmalloc(sizeof(targ_t) * n);
	for (uint32_t i = 0; i < n; i++) {
		ta[i].c = c; ta[i].tid = i;
		if (pthread_create(&th[i], NULL, fn, &ta[i]) != 0) {
			fprintf(stderr, "pthread_create failed\n");
			exit(EXIT_FAILURE);
		}
	}
	for (uint32_t i = 0; i < n; i++) pthread_join(th[i], NULL);
	free(th); free(ta);
}

/* One pass over all read files, `worker` on every batch. 0 = ok. */
static int read_pass(ctx_t *c, void *(*worker)(void *))
{
	c->n_published = 0;
	c->eof = 0;
	c->read_error = 0;
	c->n_reads = c->n_positions = c->n_valid = 0;
	for (int i = 0; i < N_SLOTS; i++) c->slots[i].ready = 0;
	pthread_t rt;
	if (pthread_create(&rt, NULL, reader_main, c) != 0) {
		fprintf(stderr, "pthread_create failed\n");
		exit(EXIT_FAILURE);
	}
	run_parallel(c, worker, c->P);
	pthread_join(rt, NULL);
	return c->read_error ? -1 : 0;
}

static size_t total_n(ctx_t *c)
{
	size_t n = 0;
	for (uint32_t i = 0; i < c->P; i++) n += c->tabs[i].n;
	return n;
}

static double table_gb(ctx_t *c)
{
	size_t b = 0;
	size_t slot = c->mode == MODE_SAMPLE ? 12 : 9;   /* key + count */
	for (uint32_t i = 0; i < c->P; i++) b += c->tabs[i].cap * slot;
	return b / 1e9;
}

/* ── memory model ─────────────────────────────────────────────────── */
/* Peak of one pass in low-memory mode: every partition table at the
   capacity that holds its share below load 0.7. Pre-sized tables never
   grow; tables that grow from INITIAL_TAB_CAP briefly hold old + new
   (1.5x the final capacity). Steps 2-4 need no more than this in
   low-memory mode. 2% headroom covers estimate error and imbalance. */
static double est_peak_bytes(double distinct, uint32_t M, uint32_t P,
                             size_t slot, int presized)
{
	double per_part = distinct / ((double)M * P) * 1.02;
	size_t cap = cap_for(per_part);
	double spike = 1.0;
	if (!presized) {
		if (cap < INITIAL_TAB_CAP) cap = INITIAL_TAB_CAP;
		else if (cap > INITIAL_TAB_CAP) spike = 1.5;
	}
	return (double)P * cap * slot * spike + FIXED_OVERHEAD
	     + (M > 1 ? (double)M * RUN_BUF : 0.0);
}

static void print_estimate_table(FILE *fp, double distinct, uint32_t P,
                                 size_t slot, uint64_t valid, double ram_gb,
                                 uint32_t chosen)
{
	static const uint32_t ms[] = { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
	                               96, 128, 192, 256, 0 };
	fprintf(fp, "estimated distinct k-mers: %.3g  (HyperLogLog, ~1%% error; "
	        "%.3g valid k-mer windows, %.2f distinct per window)\n",
	        distinct, (double)valid, valid ? distinct / valid : 0.0);
	fprintf(fp, "predicted peak RAM with %u thread(s), %zu-byte slots:\n", P, slot);
	fprintf(fp, "  %6s  %13s  %16s  %16s  %10s\n", "passes", "k-mers/pass",
	        "-R (presized) GB", "-M only GB", "read passes");
	for (int i = 0; ms[i]; i++) {
		uint32_t M = ms[i];
		double pr = est_peak_bytes(distinct, M, P, slot, 1) / 1e9;
		double pm = est_peak_bytes(distinct, M, P, slot, 0) / 1e9;
		fprintf(fp, "  %6u  %13.3g  %16.2f  %16.2f  %10u%s\n", M,
		        distinct / M, pr, pm, M + 1,
		        M == chosen ? "   <- chosen for -R" : "");
	}
	fprintf(fp, "  (read passes: M counting passes + 1 estimate pass; -M only "
	        "skips the estimate pass\n   but pass 0 grows its tables, so it "
	        "needs the right-hand column.)\n");
	if (ram_gb > 0)
		fprintf(fp, "  budget -R %.2f GB, %.0f%% usable\n", ram_gb, RAM_SAFETY * 100);
}

/* ── output ───────────────────────────────────────────────────────── */

/* Single emission point. kmdb_writer_add() still takes a count argument in
   the current kmdb format, so we pass the constant 1. Once kmdb.[ch] has a
   key-only mode, replace this one call (and the open call in main). */
static inline int emit_kmer(kmdb_writer *w, uint64_t key)
{
	return kmdb_writer_add(w, key, 1);
}

/* sample mode: real counts */
static inline int emit_kmer_count(kmdb_writer *w, uint64_t key, uint32_t count)
{
	return kmdb_writer_add(w, key, count);
}

/* Where sorted k-mers go: the database (single pass) or a run file. */
typedef struct {
	kmdb_writer *w;
	FILE        *fp;
	int          counts;    /* sample mode: carry counts */
} sink_t;

static inline int sink_put(sink_t *s, uint64_t key, uint32_t count)
{
	if (s->w) return s->counts ? emit_kmer_count(s->w, key, count)
	                           : emit_kmer(s->w, key);
	if (fwrite(&key, sizeof key, 1, s->fp) != 1) return -1;
	if (s->counts && fwrite(&count, sizeof count, 1, s->fp) != 1) return -1;
	return 0;
}

/* k-way merge of the sorted partitions (min-heap on key) into s. */
static int merge_partitions(ctx_t *C, sink_t *s)
{
	uint32_t *heap = xmalloc(sizeof(uint32_t) * C->P);
	size_t *pos = calloc(C->P, sizeof(size_t));
	uint32_t hn = 0;
#define HKEY(p) (C->tabs[p].keys[pos[p]])
	for (uint32_t p = 0; p < C->P; p++) {
		if (C->tabs[p].n == 0) continue;
		uint32_t i = hn++;
		heap[i] = p;
		while (i > 0 && HKEY(heap[(i - 1) / 2]) > HKEY(heap[i])) {
			uint32_t t = heap[i]; heap[i] = heap[(i - 1) / 2]; heap[(i - 1) / 2] = t;
			i = (i - 1) / 2;
		}
	}
	int werr = 0;
	while (hn > 0 && !werr) {
		uint32_t p = heap[0];
		uint32_t cnt = C->mode == MODE_SAMPLE ? C->tabs[p].counts32[pos[p]] : 1;
		if (sink_put(s, C->tabs[p].keys[pos[p]], cnt) != 0) werr = 1;
		if (++pos[p] == C->tabs[p].n) heap[0] = heap[--hn];
		uint32_t i = 0;
		for (;;) {
			uint32_t l = 2 * i + 1, r = l + 1, m = i;
			if (l < hn && HKEY(heap[l]) < HKEY(heap[m])) m = l;
			if (r < hn && HKEY(heap[r]) < HKEY(heap[m])) m = r;
			if (m == i) break;
			uint32_t t = heap[i]; heap[i] = heap[m]; heap[m] = t;
			i = m;
		}
	}
#undef HKEY
	free(heap); free(pos);
	return werr ? -1 : 0;
}

/* final merge of the M disjoint, sorted run files into the database */
typedef struct { FILE *fp; uint64_t key; uint32_t cnt; } runrd_t;

/* 1 = got a record, 0 = clean EOF, -1 = error */
static int run_next(runrd_t *r, int counts)
{
	if (fread(&r->key, sizeof r->key, 1, r->fp) != 1)
		return ferror(r->fp) ? -1 : 0;
	r->cnt = 1;
	if (counts && fread(&r->cnt, sizeof r->cnt, 1, r->fp) != 1) return -1;
	return 1;
}

static int merge_runs(char **paths, uint32_t n, int counts, kmdb_writer *w)
{
	runrd_t *rd = calloc(n, sizeof *rd);
	uint32_t *heap = xmalloc(sizeof(uint32_t) * n);
	uint32_t hn = 0;
	int err = 0;
	for (uint32_t i = 0; i < n && !err; i++) {
		rd[i].fp = fopen(paths[i], "rb");
		if (!rd[i].fp) {
			fprintf(stderr, "error: cannot reopen %s: %s\n", paths[i], strerror(errno));
			err = 1;
			break;
		}
		setvbuf(rd[i].fp, NULL, _IOFBF, RUN_BUF);
		int rc = run_next(&rd[i], counts);
		if (rc < 0) { err = 1; break; }
		if (rc == 0) continue;
		uint32_t j = hn++;
		heap[j] = i;
		while (j > 0 && rd[heap[(j - 1) / 2]].key > rd[heap[j]].key) {
			uint32_t t = heap[j]; heap[j] = heap[(j - 1) / 2]; heap[(j - 1) / 2] = t;
			j = (j - 1) / 2;
		}
	}
	while (hn > 0 && !err) {
		runrd_t *r = &rd[heap[0]];
		int rc = counts ? emit_kmer_count(w, r->key, r->cnt) : emit_kmer(w, r->key);
		if (rc != 0) { err = 1; break; }
		rc = run_next(r, counts);
		if (rc < 0) { fprintf(stderr, "error: reading a run file failed\n"); err = 1; break; }
		if (rc == 0) heap[0] = heap[--hn];
		uint32_t j = 0;
		for (;;) {
			uint32_t l = 2 * j + 1, rr = l + 1, m = j;
			if (l < hn && rd[heap[l]].key < rd[heap[m]].key) m = l;
			if (rr < hn && rd[heap[rr]].key < rd[heap[m]].key) m = rr;
			if (m == j) break;
			uint32_t t = heap[j]; heap[j] = heap[m]; heap[m] = t;
			j = m;
		}
	}
	for (uint32_t i = 0; i < n; i++) if (rd[i].fp) fclose(rd[i].fp);
	free(rd); free(heap);
	return err ? -1 : 0;
}

/* ── main ─────────────────────────────────────────────────────────── */

static void usage(void)
{
	fprintf(stderr,
	"Usage: kmer_metagenome_db [-T scrub|sample] -i <reads.fastq.gz> [-i <reads_2.fastq.gz> ...]\n"
	"                          [-I <file listing read files>]\n"
	"                          [-A <genome list | single genome FASTA>]\n"
	"                          [-n <sample id>] [-O <output dir, default .>]\n"
	"                          [-o <output path, overrides -O/-n naming>]\n"
	"                          [-c <min count; scrub 1..255, default 2; sample default 1>]\n"
	"                          [-k <k, default 31>]\n"
	"                          [-t <threads, default 4>] [-b <k-mers per block>]\n"
	"                          [-z <zstd level, default 3>]\n"
	"                          [-M <passes> | -R <RAM GB>] [-E] [-W <temp dir>]\n"
	"\n"
	"  -T scrub (default)\n"
	"  Writes <-O>/<-n>.kmdb: a presence-only k-mer list holding the\n"
	"  metagenome k-mers with\n"
	"    count in the reads >= -c   and   absent from every -A genome.\n"
	"  Counts are NOT stored (3x and 100x are the same).\n"
	"  All -i/-I files are one sample (e.g. R1 + R2). -n defaults to the\n"
	"  first read file's name without its extension.\n"
	"  -A is the genome scrub DB list (the same list you pass to\n"
	"  kmer_scrub_count_individual -A). Without -A nothing is subtracted.\n"
	"  -c 1 keeps every k-mer seen at least once.\n"
	"  Memory: ~13-26 bytes per distinct k-mer in the reads (all k-mers,\n"
	"  including errors, before -c), briefly up to ~39 while a table grows,\n"
	"  plus ~70 MB of read buffers.\n"
	"\n"
	"  -T sample\n"
	"  Writes <-O>/<-n>.kmdb holding every canonical k-mer of the reads with\n"
	"  its exact count (32-bit). No genome subtraction (-A is rejected).\n"
	"  -c drops k-mers seen fewer than -c times (default 1 = keep all, which\n"
	"  reproduces read scanning exactly). Use as a kmer_strain_detect target.\n"
	"  Memory: ~17-34 bytes per distinct k-mer in the reads (12-byte slots),\n"
	"  briefly up to ~51 while a table grows; -c only shrinks the output file.\n"
	"\n"
	"  Low-memory mode (same database, more time)\n"
	"  -M <passes>  split the k-mers into M hash slices and build one slice per\n"
	"               pass (1..%d). Peak RAM drops roughly M-fold; the reads are\n"
	"               read M times and, in scrub mode, the -A genomes are streamed\n"
	"               M times. Sorted slices go to temporary run files (8 bytes per\n"
	"               output k-mer in scrub mode, 12 in sample mode) that are\n"
	"               merged at the end and then deleted.\n"
	"  -R <GB>      RAM budget: read the reads once to estimate the distinct\n"
	"               k-mers (HyperLogLog), print the estimate table, then pick the\n"
	"               smallest M whose predicted peak is <= %.0f%% of <GB>.\n"
	"               Tables are pre-sized, so -R needs less RAM than -M alone for\n"
	"               the same M.\n"
	"  -E           print the estimate table (stdout) and exit; with -R also show\n"
	"               the M it would choose. Nothing is written.\n"
	"  -W <dir>     directory for the run files (default: next to the output).\n"
	"               Use local scratch on a cluster.\n"
	"  -M 1 (or no -M/-R) is the original single-pass build.\n"
	"\n"
	"  The database is written to <out>.tmp and renamed when complete.\n",
	MAX_PASSES, RAM_SAFETY * 100);
	exit(1);
}

int main(int argc, char *argv[])
{
	char **reads = NULL; int n_reads = 0, cap_reads = 0;
	char *reads_list = NULL, *genome_list = NULL;
	char *name = NULL, *out_dir = NULL, *out_path = NULL, *tmp_dir = NULL;
	long long min_count = 0;
	int c_given = 0;           /* -c not given -> mode default */
	long k = 31, threads = 4, block_n = 0, level = 3, passes = 0;
	double ram_gb = 0.0;
	int estimate_only = 0;
	int mode = MODE_SCRUB;
	int c;

	while ((c = getopt(argc, argv, "T:i:I:A:n:O:o:c:k:t:b:z:M:R:EW:h")) != -1) {
		switch (c) {
		case 'i':
			if (n_reads == cap_reads) {
				cap_reads = cap_reads ? cap_reads * 2 : 4;
				reads = xrealloc(reads, sizeof(char *) * cap_reads);
			}
			reads[n_reads++] = strdup(optarg);
			break;
		case 'T':
			if      (!strcasecmp(optarg, "scrub"))  mode = MODE_SCRUB;
			else if (!strcasecmp(optarg, "sample")) mode = MODE_SAMPLE;
			else {
				fprintf(stderr, "error: -T must be scrub or sample (got %s)\n", optarg);
				return 1;
			}
			break;
		case 'I': reads_list  = strdup(optarg); break;
		case 'A': genome_list = strdup(optarg); break;
		case 'n': name        = strdup(optarg); break;
		case 'O': out_dir     = strdup(optarg); break;
		case 'o': out_path    = strdup(optarg); break;
		case 'c': min_count   = atoll(optarg); c_given = 1; break;
		case 'k': k           = atol(optarg); break;
		case 't': threads     = atol(optarg); break;
		case 'b': block_n     = atol(optarg); break;
		case 'z': level       = atol(optarg); break;
		case 'M': passes      = atol(optarg); break;
		case 'R': ram_gb      = atof(optarg); break;
		case 'E': estimate_only = 1; break;
		case 'W': tmp_dir     = strdup(optarg); break;
		default:  usage();
		}
	}
	if (reads_list) {
		char **more; int m = read_path_list(reads_list, &more);
		for (int i = 0; i < m; i++) {
			if (n_reads == cap_reads) {
				cap_reads = cap_reads ? cap_reads * 2 : 4;
				reads = xrealloc(reads, sizeof(char *) * cap_reads);
			}
			reads[n_reads++] = more[i];
		}
		free(more);
	}
	if (n_reads == 0) usage();
	if (k < 1 || k > KMDB_MAX_K) {
		fprintf(stderr, "error: -k must be 1..%d (got %ld)\n", KMDB_MAX_K, k);
		return 1;
	}
	if (!c_given) min_count = mode == MODE_SAMPLE ? 1 : 2;
	if (mode == MODE_SCRUB && (min_count < 1 || min_count > (long long)COUNT_MAX)) {
		fprintf(stderr, "error: -c must be 1..%u (got %lld); counts are an "
		        "8-bit saturating counter and are not stored\n",
		        COUNT_MAX, min_count);
		return 1;
	}
	if (mode == MODE_SAMPLE && (min_count < 1 || min_count > (long long)COUNT32_MAX)) {
		fprintf(stderr, "error: -c must be 1..%u (got %lld)\n",
		        COUNT32_MAX, min_count);
		return 1;
	}
	if (mode == MODE_SAMPLE && genome_list) {
		fprintf(stderr, "error: -A is not used with -T sample (no scrubbing is "
		        "performed); drop -A or use -T scrub\n");
		return 1;
	}
	if (passes && ram_gb > 0) {
		fprintf(stderr, "error: give -M or -R, not both\n");
		return 1;
	}
	if (passes < 0 || passes > MAX_PASSES) {
		fprintf(stderr, "error: -M must be 1..%d (got %ld)\n", MAX_PASSES, passes);
		return 1;
	}
	if (ram_gb < 0) {
		fprintf(stderr, "error: -R must be a positive number of GB\n");
		return 1;
	}
	if (threads < 1) threads = 1;
	if (threads > 256) threads = 256;
	if (block_n < 0 || block_n > (1L << 26)) {
		fprintf(stderr, "error: -b must be 1..%ld\n", 1L << 26);
		return 1;
	}
	if (level < 1 || level > 22) level = 3;
	for (int i = 0; i < n_reads; i++) {
		if (kmdb_is_kmdb_path(reads[i])) {
			fprintf(stderr, "error: -i %s is already a .kmdb\n", reads[i]);
			return 1;
		}
	}

	if (!name) name = sample_name_from(reads[0]);
	if (strchr(name, '/')) {
		fprintf(stderr, "error: -n must be a bare name (got %s)\n", name);
		return 1;
	}
	if (!out_path && !estimate_only) {
		if (!out_dir) out_dir = strdup(".");
		if (mkdir(out_dir, 0777) != 0 && errno != EEXIST) {
			fprintf(stderr, "error: cannot create %s: %s\n", out_dir, strerror(errno));
			return 1;
		}
		size_t need = strlen(out_dir) + strlen(name) + 8;
		out_path = xmalloc(need);
		snprintf(out_path, need, "%s/%s.kmdb", out_dir, name);
	}

#ifdef __GLIBC__
	/* Serve every table from its own mmap, so freeing a pass's tables
	   returns the memory to the OS instead of leaving a fragmented heap. */
	mallopt(M_MMAP_THRESHOLD, 4 << 20);
#endif

	ctx_t C;
	memset(&C, 0, sizeof C);
	C.k            = (int)k;
	C.mode         = mode;
	C.P            = (uint32_t)threads;
	C.min_count    = (uint32_t)min_count;
	C.M            = passes > 0 ? (uint32_t)passes : 1;
	C.read_files   = reads;
	C.n_read_files = n_reads;
	C.t_start      = now_s();
	pthread_mutex_init(&C.mtx, NULL);
	pthread_cond_init(&C.cv, NULL);
	C.tabs = calloc(C.P, sizeof(tab_t));
	const size_t slot = mode == MODE_SAMPLE ? 12 : 9;
	const int wide = mode == MODE_SAMPLE;

	if (genome_list && !estimate_only) {
		C.n_genomes = read_path_list(genome_list, &C.genomes);
		if (C.n_genomes == 0) {
			fprintf(stderr, "error: -A %s lists no genomes\n", genome_list);
			return 1;
		}
	}

	/* 0) -R / -E: estimate distinct k-mers, choose M */
	double est_distinct = 0.0;
	if (ram_gb > 0 || estimate_only) {
		fprintf(stderr, "[0] estimating distinct k-mers (HyperLogLog, one read pass)\n");
		C.hll = calloc((size_t)C.P * HLL_M, 1);
		if (!C.hll) { fprintf(stderr, "out of memory\n"); return 1; }
		if (read_pass(&C, hll_main) != 0) {
			fprintf(stderr, "error: reading failed\n");
			return 1;
		}
		for (uint32_t t = 1; t < C.P; t++)
			for (uint32_t i = 0; i < HLL_M; i++)
				if (C.hll[(size_t)t * HLL_M + i] > C.hll[i])
					C.hll[i] = C.hll[(size_t)t * HLL_M + i];
		est_distinct = hll_estimate(C.hll);
		uint64_t valid = C.n_valid;
		free(C.hll); C.hll = NULL;

		uint32_t chosen = 0;
		if (ram_gb > 0) {
			for (uint32_t m = 1; m <= MAX_PASSES; m++)
				if (est_peak_bytes(est_distinct, m, C.P, slot, 1)
				    <= ram_gb * 1e9 * RAM_SAFETY) { chosen = m; break; }
		}
		print_estimate_table(estimate_only ? stdout : stderr, est_distinct,
		                     C.P, slot, valid, ram_gb, chosen);
		if (ram_gb > 0 && !chosen) {
			double min_gb = est_peak_bytes(est_distinct, MAX_PASSES, C.P, slot, 1) / 1e9;
			fprintf(estimate_only ? stdout : stderr,
			        "error: even %d passes need ~%.2f GB > %.0f%% of -R %.2f GB\n",
			        MAX_PASSES, min_gb, RAM_SAFETY * 100, ram_gb);
			return 1;
		}
		if (estimate_only) return 0;
		C.M = chosen;
		fprintf(stderr, "-R %.2f GB: using %u pass(es), predicted peak %.2f GB\n",
		        ram_gb, C.M, est_peak_bytes(est_distinct, C.M, C.P, slot, 1) / 1e9);
	}
	const int lowmem = C.M > 1 || ram_gb > 0;

	if (mode == MODE_SCRUB)
		fprintf(stderr, "kmer_metagenome_db (presence-only): sample %s, k=%d, "
		        "min count %u, %u threads, %d read file(s), %d genome(s) to subtract",
		        name, C.k, C.min_count, C.P, n_reads, C.n_genomes);
	else
		fprintf(stderr, "kmer_metagenome_db (sample counts): sample %s, k=%d, "
		        "min count %u, %u threads, %d read file(s)",
		        name, C.k, C.min_count, C.P, n_reads);
	fprintf(stderr, lowmem ? ", low-memory: %u pass(es)\n" : "\n", C.M);

	/* run files for M > 1 */
	char **runs = NULL;
	if (C.M > 1) {
		runs = calloc(C.M, sizeof(char *));
		g_runs = runs;
		atexit(cleanup_runs);
		for (uint32_t j = 0; j < C.M; j++) {
			size_t need = (tmp_dir ? strlen(tmp_dir) + strlen(name)
			                       : strlen(out_path)) + 32;
			runs[j] = xmalloc(need);
			if (tmp_dir) snprintf(runs[j], need, "%s/%s.pass%03u.tmp", tmp_dir, name, j);
			else         snprintf(runs[j], need, "%s.pass%03u.tmp", out_path, j);
		}
		g_n_runs = (int)C.M;
		if (tmp_dir && mkdir(tmp_dir, 0777) != 0 && errno != EEXIST) {
			fprintf(stderr, "error: cannot create %s: %s\n", tmp_dir, strerror(errno));
			return 1;
		}
	}

	size_t tlen = strlen(out_path) + 5;
	char *tmp_path = xmalloc(tlen);
	snprintf(tmp_path, tlen, "%s.tmp", out_path);
	kmdb_writer *w = NULL;

	uint64_t distinct_all = 0, distinct_min = 0, final_n = 0, removed_total = 0;
	uint64_t count_sum = 0, count_max = 0, n_saturated = 0;
	uint64_t st_reads = 0, st_positions = 0, st_valid = 0;
	size_t prev_max_part = 0;   /* largest partition of the previous pass */
	int werr = 0;

	for (uint32_t pass = 0; pass < C.M; pass++) {
		C.pass = pass;
		char tag[48] = "";
		if (C.M > 1) snprintf(tag, sizeof tag, "pass %u/%u ", pass + 1, C.M);

		/* table sizes: pre-sized in low-memory mode when a size is known */
		size_t cap0 = INITIAL_TAB_CAP;
		if (lowmem && est_distinct > 0)
			cap0 = cap_for(est_distinct / ((double)C.M * C.P) * 1.02);
		else if (lowmem && pass > 0)
			cap0 = cap_for(prev_max_part * 1.03);
		for (uint32_t i = 0; i < C.P; i++) tab_init(&C.tabs[i], cap0, wide);

		/* 1) count */
		fprintf(stderr, "[1/4] %scounting k-mers\n", tag);
		if (read_pass(&C, count_main) != 0) {
			fprintf(stderr, "error: reading failed, no database written\n");
			return 1;
		}
		if (pass == 0) {
			st_reads = C.n_reads; st_positions = C.n_positions; st_valid = C.n_valid;
		}
		uint64_t pass_distinct = total_n(&C);
		prev_max_part = 0;
		for (uint32_t i = 0; i < C.P; i++)
			if (C.tabs[i].n > prev_max_part) prev_max_part = C.tabs[i].n;
		distinct_all += pass_distinct;
		fprintf(stderr, "      %" PRIu64 " reads, %" PRIu64 " valid k-mer windows, "
		        "%" PRIu64 " distinct k-mers (table %.2f GB, %.1fs)\n",
		        C.n_reads, C.n_valid, pass_distinct, table_gb(&C), now_s() - C.t_start);

		uint64_t pass_final;
		if (mode == MODE_SAMPLE) {
			/* sample: no rebuild, no subtraction; -c is applied while compacting */
			fprintf(stderr, "[2/4] skipped (sample mode: no rebuild for lookups)\n");
			fprintf(stderr, "[3/4] skipped (sample mode: no genome subtraction)\n");
			fprintf(stderr, "[4/4] %ssorting (k-mers with count >= %u)\n",
			        tag, C.min_count);
			run_parallel(&C, compact_sort_counts_main, C.P);
			pass_final = total_n(&C);
			distinct_min += pass_final;
			for (uint32_t p = 0; p < C.P; p++)
				for (size_t i = 0; i < C.tabs[p].n; i++) {
					uint32_t v = C.tabs[p].counts32[i];
					count_sum += v;
					if (v > count_max) count_max = v;
					if (v == COUNT32_MAX) n_saturated++;
				}
			fprintf(stderr, "      %" PRIu64 " k-mers kept (%.1f%%)\n", pass_final,
			        pass_distinct ? 100.0 * pass_final / pass_distinct : 0.0);
		} else {
			/* 2) filter */
			fprintf(stderr, "[2/4] %skeeping k-mers with count >= %u%s\n", tag,
			        C.min_count, lowmem ? " (in place)" : "");
			run_parallel(&C, lowmem ? filter_inplace_main : filter_main, C.P);
			uint64_t pass_min = total_n(&C);
			distinct_min += pass_min;
			fprintf(stderr, "      %" PRIu64 " k-mers kept (%.1f%%), table %.2f GB\n",
			        pass_min,
			        pass_distinct ? 100.0 * pass_min / pass_distinct : 0.0, table_gb(&C));

			/* 3) subtract genomes */
			if (C.n_genomes > 0) {
				fprintf(stderr, "[3/4] %sremoving k-mers present in %d genome(s)\n",
				        tag, C.n_genomes);
				C.next_genome = 0; C.genomes_done = 0; C.removed = 0;
				run_parallel(&C, subtract_main, C.P);
				if (C.genome_errors > 0) {
					fprintf(stderr, "error: %d genome(s) could not be read; their k-mers "
					        "would stay in the database, so none was written\n",
					        C.genome_errors);
					return 1;
				}
				removed_total += C.removed;
				fprintf(stderr, "      removed %" PRIu64 " k-mers shared with genomes, "
				        "%" PRIu64 " remain (%.1fs)\n", C.removed,
				        pass_min - C.removed, now_s() - C.t_start);
			} else {
				fprintf(stderr, "[3/4] no -A given, nothing subtracted\n");
			}

			/* 4) sort */
			fprintf(stderr, "[4/4] %ssorting\n", tag);
			run_parallel(&C, compact_sort_main, C.P);
			pass_final = total_n(&C);
		}
		final_n += pass_final;

		/* write: straight into the database, or into this pass's run */
		sink_t s = { NULL, NULL, mode == MODE_SAMPLE };
		if (C.M == 1) {
			fprintf(stderr, "      writing %s\n", out_path);
			w = kmdb_writer_open(tmp_path, C.k, (uint32_t)block_n, (int)level);
			if (!w) return 1;
			s.w = w;
			if (merge_partitions(&C, &s) != 0) werr = 1;
		} else {
			s.fp = fopen(runs[pass], "wb");
			if (!s.fp) {
				fprintf(stderr, "error: cannot create %s: %s\n", runs[pass], strerror(errno));
				return 1;
			}
			setvbuf(s.fp, NULL, _IOFBF, RUN_BUF);
			int rc = merge_partitions(&C, &s);
			if (fclose(s.fp) != 0 || rc != 0) {
				fprintf(stderr, "error: writing %s failed (disk full?)\n", runs[pass]);
				return 1;
			}
			fprintf(stderr, "      %" PRIu64 " k-mers -> %s (%.1fs)\n",
			        pass_final, runs[pass], now_s() - C.t_start);
		}
		for (uint32_t i = 0; i < C.P; i++) tab_free(&C.tabs[i]);
		if (werr) break;
	}

	if (C.M > 1 && !werr) {
		fprintf(stderr, "merging %u runs into %s\n", C.M, out_path);
		w = kmdb_writer_open(tmp_path, C.k, (uint32_t)block_n, (int)level);
		if (!w) return 1;
		if (merge_runs(runs, C.M, mode == MODE_SAMPLE, w) != 0) werr = 1;
		cleanup_runs();
	}
	if (count_sum && n_saturated)
		fprintf(stderr, "WARNING: %" PRIu64 " k-mer count(s) saturated at %u\n",
		        n_saturated, COUNT32_MAX);

	/* metadata */
	kmdb_writer_meta(w, "format", "kmdb");
	if (mode == MODE_SCRUB) {
		kmdb_writer_meta(w, "created_by", "kmer_metagenome_db 2 (presence-only)");
		kmdb_writer_meta(w, "presence_only", "1");
		kmdb_writer_meta(w, "db_type", "scrub");
	} else {
		kmdb_writer_meta(w, "created_by", "kmer_metagenome_db 2 (sample counts)");
		kmdb_writer_meta(w, "presence_only", "0");
		kmdb_writer_meta(w, "db_type", "sample");
	}
	kmdb_writer_meta(w, "sample_id", name);
	kmdb_writer_meta(w, "canonical", "max(fwd,revcomp) 2-bit A0C1G2T3 MSB-first");
	kmdb_writer_meta_u64(w, "k", (uint64_t)C.k);
	kmdb_writer_meta_u64(w, "min_count", C.min_count);
	for (int i = 0; i < n_reads; i++) kmdb_writer_meta(w, "read_file", reads[i]);
	kmdb_writer_meta(w, "genome_list", genome_list ? genome_list : "none");
	kmdb_writer_meta_u64(w, "n_genomes", (uint64_t)C.n_genomes);
	kmdb_writer_meta_u64(w, "n_reads", st_reads);
	kmdb_writer_meta_u64(w, "kmer_positions", st_positions);
	kmdb_writer_meta_u64(w, "valid_kmer_positions", st_valid);
	kmdb_writer_meta_u64(w, "distinct_kmers", distinct_all);
	kmdb_writer_meta_u64(w, "distinct_kmers_min_count", distinct_min);
	kmdb_writer_meta_u64(w, "removed_in_genomes", removed_total);
	kmdb_writer_meta_u64(w, "n_kmers", final_n);
	kmdb_writer_meta_u64(w, "passes", C.M);
	if (mode == MODE_SAMPLE) {
		kmdb_writer_meta_u64(w, "count_sum", count_sum);
		kmdb_writer_meta_u64(w, "count_max", count_max);
		kmdb_writer_meta_u64(w, "count_saturated", n_saturated);
	}
	{
		time_t now = time(NULL);
		char tbuf[64];
		strftime(tbuf, sizeof tbuf, "%Y-%m-%dT%H:%M:%S%z", localtime(&now));
		kmdb_writer_meta(w, "created", tbuf);
	}

	if (kmdb_writer_close(w) != 0 || werr) {
		unlink(tmp_path);
		fprintf(stderr, "error: writing failed, no database written\n");
		return 1;
	}
	if (rename(tmp_path, out_path) != 0) {
		fprintf(stderr, "error: rename %s -> %s: %s\n", tmp_path, out_path,
		        strerror(errno));
		return 1;
	}
	struct stat st;
	stat(out_path, &st);
	fprintf(stderr, "done: %" PRIu64 " k-mers, %.2f MB (%.2f bytes/k-mer), %.1fs\n",
	        final_n, st.st_size / 1e6,
	        final_n ? (double)st.st_size / final_n : 0.0, now_s() - C.t_start);

	free(C.tabs); free(tmp_path);
	for (int i = 0; i < n_reads; i++) free(reads[i]);
	for (int i = 0; i < C.n_genomes; i++) free(C.genomes[i]);
	free(reads); free(C.genomes); free(runs);
	free(reads_list); free(genome_list); free(name); free(out_dir); free(out_path);
	free(tmp_dir);
	for (int i = 0; i < N_SLOTS; i++) { free(C.slots[i].seq); free(C.slots[i].lens); }
	return 0;
}