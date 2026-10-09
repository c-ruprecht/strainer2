/*
	kmer_metagenome_db — build a per-metagenome k-mer database (.kmdb) that
	supplements the genome scrub database.

	Keeps exactly the metagenome k-mers with

	    metagenome_count >= -c        (default 2: drops most error k-mers)
	    pangenome_count  == 0         (absent from every genome in -A)

	so the result carries only signal the genome scrub DB does not already
	have. Scrubbing a reference with `kmer_scrub_count_individual -B x.kmdb`
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
#include <sys/stat.h>
#include "kseq.h"
#include "kmdb.h"

KSEQ_INIT(gzFile, gzread)

#define EMPTY_KEY        UINT64_MAX      /* never a valid k-mer for k <= 31 */
#define BATCH_BASES      (8u << 20)
#define N_SLOTS          4
#define INITIAL_TAB_CAP  (1u << 20)

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

static size_t pow2_at_least(size_t n)
{
	size_t c = 1024;
	while (c < n) c <<= 1;
	return c;
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

/* ── per-partition open-addressing table ──────────────────────────── */

typedef struct {
	uint64_t *keys;
	uint32_t *counts;
	size_t    cap;   /* power of two */
	size_t    n;
} tab_t;

static void tab_init(tab_t *t, size_t cap)
{
	t->cap    = cap;
	t->n      = 0;
	t->keys   = xmalloc(cap * sizeof(uint64_t));
	t->counts = xmalloc(cap * sizeof(uint32_t));
	memset(t->keys, 0xFF, cap * sizeof(uint64_t));
}

static void tab_free(tab_t *t)
{
	free(t->keys); free(t->counts);
	t->keys = NULL; t->counts = NULL; t->cap = t->n = 0;
}

static void tab_put_new(tab_t *t, uint64_t key, uint32_t count)
{
	size_t mask = t->cap - 1, i = mix64(key) & mask;
	while (t->keys[i] != EMPTY_KEY) i = (i + 1) & mask;
	t->keys[i] = key;
	t->counts[i] = count;
	t->n++;
}

static void tab_grow(tab_t *t)
{
	tab_t nt;
	tab_init(&nt, t->cap * 2);
	for (size_t i = 0; i < t->cap; i++)
		if (t->keys[i] != EMPTY_KEY) tab_put_new(&nt, t->keys[i], t->counts[i]);
	tab_free(t);
	*t = nt;
}

static inline void tab_inc(tab_t *t, uint64_t key, uint64_t h)
{
	if (t->n * 10 >= t->cap * 7) tab_grow(t);
	size_t mask = t->cap - 1, i = h & mask;
	for (;;) {
		uint64_t k = t->keys[i];
		if (k == key) {
			if (t->counts[i] != UINT32_MAX) t->counts[i]++;
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
	uint32_t   P;              /* partitions == worker threads */
	uint32_t   min_count;
	tab_t     *tabs;

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

static void *count_main(void *arg)
{
	targ_t *a = arg;
	ctx_t *c = a->c;
	const uint32_t tid = a->tid, P = c->P;
	const int k = c->k;
	const int shift = 2 * (k - 1);
	const uint64_t mask = (k == 32) ? UINT64_MAX : ((1ull << (2 * k)) - 1);
	tab_t *t = &c->tabs[tid];
	uint64_t valid = 0;

	for (uint64_t s = 0;; s++) {
		batch_t *b = &c->slots[s % N_SLOTS];
		pthread_mutex_lock(&c->mtx);
		while (!(b->ready && b->seqno == s) && !(c->eof && s >= c->n_published))
			pthread_cond_wait(&c->cv, &c->mtx);
		int done = !(b->ready && b->seqno == s);
		pthread_mutex_unlock(&c->mtx);
		if (done) break;

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
					if (part_of(h, P) == tid) tab_inc(t, can, h);
					valid++;
				}
			}
		}

		pthread_mutex_lock(&c->mtx);
		if (--b->remaining == 0) {
			b->ready = 0;
			pthread_cond_broadcast(&c->cv);
		}
		pthread_mutex_unlock(&c->mtx);
	}
	if (tid == 0) c->n_valid = valid;   /* every worker sees every window */
	return NULL;
}

/* ── step 2: min-count filter ─────────────────────────────────────── */

static void *filter_main(void *arg)
{
	targ_t *a = arg;
	ctx_t *c = a->c;
	tab_t *t = &c->tabs[a->tid];

	/* Compact survivors to the front and shrink the arrays before building
	   the new table, so the old table and the new one are never both at
	   full size: peak is max(old table, 36 bytes x survivors). */
	size_t keep = 0;
	for (size_t i = 0; i < t->cap; i++) {
		if (t->keys[i] != EMPTY_KEY && t->counts[i] >= c->min_count) {
			t->keys[keep]   = t->keys[i];
			t->counts[keep] = t->counts[i];
			keep++;
		}
	}
	uint64_t *ks = xrealloc(t->keys,   (keep ? keep : 1) * sizeof(uint64_t));
	uint32_t *cs = xrealloc(t->counts, (keep ? keep : 1) * sizeof(uint32_t));

	tab_t nt;
	tab_init(&nt, pow2_at_least(keep * 2));   /* load <= 0.5 */
	for (size_t i = 0; i < keep; i++) tab_put_new(&nt, ks[i], cs[i]);
	free(ks);
	free(cs);
	*t = nt;
	return NULL;
}

/* ── step 3: genome subtraction ───────────────────────────────────── */

static void *subtract_main(void *arg)
{
	targ_t *a = arg;
	ctx_t *c = a->c;
	const uint32_t P = c->P;
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
				tab_t *t = &c->tabs[part_of(h, P)];
				size_t tm = t->cap - 1, i = h & tm;
				uint64_t key;
				while ((key = t->keys[i]) != EMPTY_KEY) {
					if (key == can) {
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

/* ── step 4: compact + sort ───────────────────────────────────────── */

static inline void swap2(uint64_t *k, uint32_t *c, size_t i, size_t j)
{
	uint64_t tk = k[i]; k[i] = k[j]; k[j] = tk;
	uint32_t tc = c[i]; c[i] = c[j]; c[j] = tc;
}

/* In-place quicksort of keys (unique) carrying counts along. */
static void sort2(uint64_t *k, uint32_t *c, size_t n)
{
	while (n > 24) {
		size_t m = n / 2, hi = n - 1;
		/* median of three moved to k[0] -> Hoare partition is safe */
		if (k[m] < k[0])  swap2(k, c, m, 0);
		if (k[hi] < k[0]) swap2(k, c, hi, 0);
		if (k[hi] < k[m]) swap2(k, c, hi, m);
		swap2(k, c, 0, m);
		uint64_t p = k[0];
		ptrdiff_t i = -1, j = (ptrdiff_t)n;
		for (;;) {
			do i++; while (k[i] < p);
			do j--; while (k[j] > p);
			if (i >= j) break;
			swap2(k, c, (size_t)i, (size_t)j);
		}
		size_t left = (size_t)j + 1;           /* [0, j] and [j+1, n) */
		if (left < n - left) {
			sort2(k, c, left);
			k += left; c += left; n -= left;
		} else {
			sort2(k + left, c + left, n - left);
			n = left;
		}
	}
	for (size_t i = 1; i < n; i++) {
		uint64_t kk = k[i]; uint32_t cc = c[i];
		size_t j = i;
		while (j > 0 && k[j - 1] > kk) { k[j] = k[j - 1]; c[j] = c[j - 1]; j--; }
		k[j] = kk; c[j] = cc;
	}
}

static void *compact_sort_main(void *arg)
{
	targ_t *a = arg;
	tab_t *t = &a->c->tabs[a->tid];
	size_t j = 0;
	for (size_t i = 0; i < t->cap; i++) {
		if (t->keys[i] != EMPTY_KEY && t->counts[i] > 0) {
			t->keys[j]   = t->keys[i];
			t->counts[j] = t->counts[i];
			j++;
		}
	}
	t->n = j;
	sort2(t->keys, t->counts, j);
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

static size_t total_n(ctx_t *c)
{
	size_t n = 0;
	for (uint32_t i = 0; i < c->P; i++) n += c->tabs[i].n;
	return n;
}

static double table_gb(ctx_t *c)
{
	size_t b = 0;
	for (uint32_t i = 0; i < c->P; i++) b += c->tabs[i].cap * 12;
	return b / 1e9;
}

/* ── main ─────────────────────────────────────────────────────────── */

static void usage(void)
{
	fprintf(stderr,
	"Usage: kmer_metagenome_db -i <reads.fastq.gz> [-i <reads_2.fastq.gz> ...]\n"
	"                          [-I <file listing read files>]\n"
	"                          [-A <genome list | single genome FASTA>]\n"
	"                          [-n <sample id>] [-O <output dir, default .>]\n"
	"                          [-o <output path, overrides -O/-n naming>]\n"
	"                          [-c <min count, default 2>] [-k <k, default 31>]\n"
	"                          [-t <threads, default 4>] [-b <k-mers per block>]\n"
	"                          [-z <zstd level, default 3>]\n"
	"\n"
	"  Writes <-O>/<-n>.kmdb holding the metagenome k-mers with\n"
	"    count in the reads >= -c   and   absent from every -A genome.\n"
	"  All -i/-I files are one sample (e.g. R1 + R2). -n defaults to the\n"
	"  first read file's name without its extension.\n"
	"  -A is the genome scrub DB list (the same list you pass to\n"
	"  kmer_scrub_count_individual -A). Without -A nothing is subtracted.\n"
	"  -c 1 keeps every k-mer (exact parity with scrubbing the FASTQ).\n"
	"  Memory: peak ~25-30 bytes per distinct k-mer in the reads (all k-mers,\n"
	"  including errors, before -c), plus ~70 MB of read buffers.\n"
	"  The database is written to <out>.tmp and renamed when complete.\n");
	exit(1);
}

int main(int argc, char *argv[])
{
	char **reads = NULL; int n_reads = 0, cap_reads = 0;
	char *reads_list = NULL, *genome_list = NULL;
	char *name = NULL, *out_dir = NULL, *out_path = NULL;
	long min_count = 2, k = 31, threads = 4, block_n = 0, level = 3;
	int c;

	while ((c = getopt(argc, argv, "i:I:A:n:O:o:c:k:t:b:z:h")) != -1) {
		switch (c) {
		case 'i':
			if (n_reads == cap_reads) {
				cap_reads = cap_reads ? cap_reads * 2 : 4;
				reads = xrealloc(reads, sizeof(char *) * cap_reads);
			}
			reads[n_reads++] = strdup(optarg);
			break;
		case 'I': reads_list  = strdup(optarg); break;
		case 'A': genome_list = strdup(optarg); break;
		case 'n': name        = strdup(optarg); break;
		case 'O': out_dir     = strdup(optarg); break;
		case 'o': out_path    = strdup(optarg); break;
		case 'c': min_count   = atol(optarg); break;
		case 'k': k           = atol(optarg); break;
		case 't': threads     = atol(optarg); break;
		case 'b': block_n     = atol(optarg); break;
		case 'z': level       = atol(optarg); break;
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
	if (min_count < 1 || min_count > (long)UINT32_MAX) {
		fprintf(stderr, "error: -c must be >= 1 (got %ld)\n", min_count);
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
	if (!out_path) {
		if (!out_dir) out_dir = strdup(".");
		if (mkdir(out_dir, 0777) != 0 && errno != EEXIST) {
			fprintf(stderr, "error: cannot create %s: %s\n", out_dir, strerror(errno));
			return 1;
		}
		size_t need = strlen(out_dir) + strlen(name) + 8;
		out_path = xmalloc(need);
		snprintf(out_path, need, "%s/%s.kmdb", out_dir, name);
	}

	ctx_t C;
	memset(&C, 0, sizeof C);
	C.k            = (int)k;
	C.P            = (uint32_t)threads;
	C.min_count    = (uint32_t)min_count;
	C.read_files   = reads;
	C.n_read_files = n_reads;
	C.t_start      = now_s();
	pthread_mutex_init(&C.mtx, NULL);
	pthread_cond_init(&C.cv, NULL);
	C.tabs = xmalloc(sizeof(tab_t) * C.P);
	for (uint32_t i = 0; i < C.P; i++) tab_init(&C.tabs[i], INITIAL_TAB_CAP);

	if (genome_list) {
		C.n_genomes = read_path_list(genome_list, &C.genomes);
		if (C.n_genomes == 0) {
			fprintf(stderr, "error: -A %s lists no genomes\n", genome_list);
			return 1;
		}
	}

	fprintf(stderr, "kmer_metagenome_db: sample %s, k=%d, min count %u, "
	        "%u threads, %d read file(s), %d genome(s) to subtract\n",
	        name, C.k, C.min_count, C.P, n_reads, C.n_genomes);

	/* 1) count */
	fprintf(stderr, "[1/4] counting k-mers\n");
	pthread_t rt;
	pthread_create(&rt, NULL, reader_main, &C);
	run_parallel(&C, count_main, C.P);
	pthread_join(rt, NULL);
	if (C.read_error) {
		fprintf(stderr, "error: reading failed, no database written\n");
		return 1;
	}
	uint64_t distinct_all = total_n(&C);
	fprintf(stderr, "      %" PRIu64 " reads, %" PRIu64 " valid k-mer windows, "
	        "%" PRIu64 " distinct k-mers (table %.2f GB, %.1fs)\n",
	        C.n_reads, C.n_valid, distinct_all, table_gb(&C), now_s() - C.t_start);

	/* 2) filter */
	fprintf(stderr, "[2/4] keeping k-mers with count >= %u\n", C.min_count);
	run_parallel(&C, filter_main, C.P);
	uint64_t distinct_min = total_n(&C);
	fprintf(stderr, "      %" PRIu64 " k-mers kept (%.1f%%), table %.2f GB\n",
	        distinct_min,
	        distinct_all ? 100.0 * distinct_min / distinct_all : 0.0, table_gb(&C));

	/* 3) subtract genomes */
	if (C.n_genomes > 0) {
		fprintf(stderr, "[3/4] removing k-mers present in %d genome(s)\n",
		        C.n_genomes);
		run_parallel(&C, subtract_main, C.P);
		if (C.genome_errors > 0) {
			fprintf(stderr, "error: %d genome(s) could not be read; their k-mers "
			        "would stay in the database, so none was written\n",
			        C.genome_errors);
			return 1;
		}
		fprintf(stderr, "      removed %" PRIu64 " k-mers shared with genomes, "
		        "%" PRIu64 " remain (%.1fs)\n", C.removed,
		        distinct_min - C.removed, now_s() - C.t_start);
	} else {
		fprintf(stderr, "[3/4] no -A given, nothing subtracted\n");
	}

	/* 4) sort + write */
	fprintf(stderr, "[4/4] sorting and writing %s\n", out_path);
	run_parallel(&C, compact_sort_main, C.P);
	uint64_t final_n = total_n(&C);

	size_t tlen = strlen(out_path) + 5;
	char *tmp_path = xmalloc(tlen);
	snprintf(tmp_path, tlen, "%s.tmp", out_path);
	kmdb_writer *w = kmdb_writer_open(tmp_path, C.k, (uint32_t)block_n, (int)level);
	if (!w) return 1;

	/* k-way merge of the sorted partitions (min-heap on key) */
	uint32_t *heap = xmalloc(sizeof(uint32_t) * C.P);
	size_t *pos = calloc(C.P, sizeof(size_t));
	uint32_t hn = 0;
#define HKEY(p) (C.tabs[p].keys[pos[p]])
	for (uint32_t p = 0; p < C.P; p++) {
		if (C.tabs[p].n == 0) continue;
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
		if (kmdb_writer_add(w, C.tabs[p].keys[pos[p]], C.tabs[p].counts[pos[p]]) != 0)
			werr = 1;
		if (++pos[p] == C.tabs[p].n) heap[0] = heap[--hn];
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

	/* metadata */
	kmdb_writer_meta(w, "format", "kmdb");
	kmdb_writer_meta(w, "created_by", "kmer_metagenome_db 1");
	kmdb_writer_meta(w, "sample_id", name);
	kmdb_writer_meta(w, "canonical", "max(fwd,revcomp) 2-bit A0C1G2T3 MSB-first");
	kmdb_writer_meta_u64(w, "k", (uint64_t)C.k);
	kmdb_writer_meta_u64(w, "min_count", C.min_count);
	for (int i = 0; i < n_reads; i++) kmdb_writer_meta(w, "read_file", reads[i]);
	kmdb_writer_meta(w, "genome_list", genome_list ? genome_list : "none");
	kmdb_writer_meta_u64(w, "n_genomes", (uint64_t)C.n_genomes);
	kmdb_writer_meta_u64(w, "n_reads", C.n_reads);
	kmdb_writer_meta_u64(w, "kmer_positions", C.n_positions);
	kmdb_writer_meta_u64(w, "valid_kmer_positions", C.n_valid);
	kmdb_writer_meta_u64(w, "distinct_kmers", distinct_all);
	kmdb_writer_meta_u64(w, "distinct_kmers_min_count", distinct_min);
	kmdb_writer_meta_u64(w, "removed_in_genomes", C.removed);
	kmdb_writer_meta_u64(w, "n_kmers", final_n);
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

	for (uint32_t i = 0; i < C.P; i++) tab_free(&C.tabs[i]);
	free(C.tabs); free(heap); free(pos); free(tmp_path);
	for (int i = 0; i < n_reads; i++) free(reads[i]);
	for (int i = 0; i < C.n_genomes; i++) free(C.genomes[i]);
	free(reads); free(C.genomes);
	free(reads_list); free(genome_list); free(name); free(out_dir); free(out_path);
	for (int i = 0; i < N_SLOTS; i++) { free(C.slots[i].seq); free(C.slots[i].lens); }
	return 0;
}
