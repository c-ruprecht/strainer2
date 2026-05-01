#include "kmer_scrub_streaming.h"
#include "BIO_sequence.h"   /* must precede genome_compare.h: defines BIO_sequences */
#include "BIO_hash.h"
#include "genome_compare.h"
#include "kseq.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>
#include <zlib.h>
#include <zstd.h>

KSEQ_INIT(gzFile, gzread)

#define GLOBAL_COLS 4

/* ── zstd stream writer (replaces gzFile) ─────────────────────────────
   Wraps a stdio FILE* with a ZSTD_CStream. Produces a single zstd frame
   stream that decompresses cleanly with `zstd -dc` or polars' scan_csv.
   ──────────────────────────────────────────────────────────────────── */

typedef struct {
	FILE       *fp;
	ZSTD_CStream *cs;
	void       *out_buf;
	size_t      out_cap;
} zstd_out_t;

static zstd_out_t *zstd_out_open(const char *path, int level)
{
	zstd_out_t *z = calloc(1, sizeof(*z));
	if (!z) return NULL;
	z->fp = fopen(path, "wb");
	if (!z->fp) { free(z); return NULL; }
	z->cs = ZSTD_createCStream();
	if (!z->cs) { fclose(z->fp); free(z); return NULL; }
	size_t init_rc = ZSTD_initCStream(z->cs, level);
	if (ZSTD_isError(init_rc)) {
		ZSTD_freeCStream(z->cs); fclose(z->fp); free(z);
		return NULL;
	}
	z->out_cap = ZSTD_CStreamOutSize();
	z->out_buf = malloc(z->out_cap);
	if (!z->out_buf) {
		ZSTD_freeCStream(z->cs); fclose(z->fp); free(z);
		return NULL;
	}
	return z;
}

/* Compress `len` bytes from `data` and write to underlying file. */
static int zstd_out_write(zstd_out_t *z, const void *data, size_t len)
{
	ZSTD_inBuffer in = { data, len, 0 };
	while (in.pos < in.size) {
		ZSTD_outBuffer out = { z->out_buf, z->out_cap, 0 };
		size_t rc = ZSTD_compressStream(z->cs, &out, &in);
		if (ZSTD_isError(rc)) return -1;
		if (out.pos > 0 &&
		    fwrite(z->out_buf, 1, out.pos, z->fp) != out.pos)
			return -1;
	}
	return 0;
}

static int zstd_out_close(zstd_out_t *z)
{
	if (!z) return 0;
	int err = 0;
	for (;;) {
		ZSTD_outBuffer out = { z->out_buf, z->out_cap, 0 };
		size_t rem = ZSTD_endStream(z->cs, &out);
		if (ZSTD_isError(rem)) { err = -1; break; }
		if (out.pos > 0 &&
		    fwrite(z->out_buf, 1, out.pos, z->fp) != out.pos) {
			err = -1; break;
		}
		if (rem == 0) break;
	}
	ZSTD_freeCStream(z->cs);
	if (fclose(z->fp) != 0) err = -1;
	free(z->out_buf);
	free(z);
	return err;
}

/* ── Bounded row queue + writer thread ────────────────────────────────── */

typedef struct sample_id_s {
	char           *id;
	char            type[3];
	int             refcount;
	pthread_mutex_t mtx;
} sample_id_t;

typedef struct {
	const char  *kmer;
	sample_id_t *sid;
	uint32_t     count;
} row_t;

struct streaming_writer_s {
	zstd_out_t      *zout;
	row_t           *queue;
	size_t           cap;
	size_t           head, tail, size;
	int              shutdown;
	pthread_mutex_t  mtx;
	pthread_cond_t   not_empty;
	pthread_cond_t   not_full;
	pthread_t        writer_tid;
};

static void sample_id_unref(sample_id_t *sid);

/* Small helper: format and write one row through the zstd stream.
   We use a stack buffer big enough for any realistic kmer + ids. */
static void write_row_zstd(zstd_out_t *z, const row_t *r)
{
	char buf[1024];
	int n = snprintf(buf, sizeof buf, "%s\t%s\t%s\t%u\n",
	                 r->kmer, r->sid->type, r->sid->id, r->count);
	if (n < 0) return;
	if ((size_t)n >= sizeof buf) {
		/* extremely unlikely, but be safe */
		fprintf(stderr, "write_row_zstd: row too long, truncated\n");
		n = sizeof buf - 1;
	}
	zstd_out_write(z, buf, (size_t)n);
}

static void *writer_main(void *arg)
{
	streaming_writer *w = (streaming_writer *)arg;

	for (;;) {
		pthread_mutex_lock(&w->mtx);
		while (w->size == 0 && !w->shutdown)
			pthread_cond_wait(&w->not_empty, &w->mtx);
		if (w->size == 0 && w->shutdown) {
			pthread_mutex_unlock(&w->mtx);
			break;
		}
		row_t batch[256];
		size_t n = 0;
		while (n < (sizeof batch / sizeof batch[0]) && w->size > 0) {
			batch[n++] = w->queue[w->head];
			w->head = (w->head + 1) % w->cap;
			w->size--;
		}
		pthread_cond_broadcast(&w->not_full);
		pthread_mutex_unlock(&w->mtx);

		for (size_t i = 0; i < n; i++) {
			write_row_zstd(w->zout, &batch[i]);
			sample_id_unref(batch[i].sid);
		}
	}
	return NULL;
}

streaming_writer *streaming_writer_open(const char *path, size_t queue_capacity)
{
	if (queue_capacity == 0) queue_capacity = 1u << 20;

	streaming_writer *w = calloc(1, sizeof(*w));
	if (!w) return NULL;

	/* level 3 = zstd default. ~3-5x faster decompression than gzip-6,
	   compression ratio in the same neighborhood. Higher levels mostly
	   slow down compression for diminishing returns. */
	w->zout = zstd_out_open(path, 3);
	if (!w->zout) {
		fprintf(stderr, "streaming_writer_open: zstd_out_open %s failed: %s\n",
		        path, strerror(errno));
		free(w);
		return NULL;
	}

	w->queue = calloc(queue_capacity, sizeof(row_t));
	if (!w->queue) {
		zstd_out_close(w->zout);
		free(w);
		return NULL;
	}
	w->cap = queue_capacity;
	pthread_mutex_init(&w->mtx, NULL);
	pthread_cond_init(&w->not_empty, NULL);
	pthread_cond_init(&w->not_full, NULL);

	const char *header = "kmer\tsample_type\tsample_id\tcount\n";
	zstd_out_write(w->zout, header, strlen(header));

	if (pthread_create(&w->writer_tid, NULL, writer_main, w) != 0) {
		fprintf(stderr, "streaming_writer_open: pthread_create failed\n");
		zstd_out_close(w->zout);
		free(w->queue);
		free(w);
		return NULL;
	}
	return w;
}

void streaming_writer_close(streaming_writer *w)
{
	if (!w) return;

	pthread_mutex_lock(&w->mtx);
	w->shutdown = 1;
	pthread_cond_broadcast(&w->not_empty);
	pthread_mutex_unlock(&w->mtx);

	pthread_join(w->writer_tid, NULL);

	zstd_out_close(w->zout);
	pthread_mutex_destroy(&w->mtx);
	pthread_cond_destroy(&w->not_empty);
	pthread_cond_destroy(&w->not_full);
	free(w->queue);
	free(w);
}

static void writer_push(streaming_writer *w, row_t r)
{
	pthread_mutex_lock(&w->mtx);
	while (w->size == w->cap)
		pthread_cond_wait(&w->not_full, &w->mtx);
	w->queue[w->tail] = r;
	w->tail = (w->tail + 1) % w->cap;
	w->size++;
	pthread_cond_signal(&w->not_empty);
	pthread_mutex_unlock(&w->mtx);
}

/* ── Summary writer (plain TSV, low volume → simple mutex) ─────────── */

struct summary_writer_s {
	FILE           *fp;
	unsigned long   total_ref_kmers;
	pthread_mutex_t mtx;
};

summary_writer *summary_writer_open(const char *path,
                                    unsigned long total_reference_kmers)
{
	if (!path) return NULL;
	summary_writer *s = calloc(1, sizeof(*s));
	if (!s) return NULL;
	s->fp = fopen(path, "w");
	if (!s->fp) {
		fprintf(stderr, "summary_writer_open: fopen %s failed: %s\n",
		        path, strerror(errno));
		free(s);
		return NULL;
	}
	s->total_ref_kmers = total_reference_kmers;
	pthread_mutex_init(&s->mtx, NULL);
	fprintf(s->fp,
	        "sample_type\tsample_id\tn_unique_kmers\tcoverage_pct"
	        "\tis_in_global\tis_in_individual\n");
	return s;
}

void summary_writer_close(summary_writer *s)
{
	if (!s) return;
	fclose(s->fp);
	pthread_mutex_destroy(&s->mtx);
	free(s);
}

static void summary_write_row(summary_writer *s,
                              const char *sample_type,
                              const char *sample_id,
                              unsigned long n_unique_kmers,
                              int is_in_global,
                              int is_in_individual)
{
	if (!s) return;
	double coverage = s->total_ref_kmers > 0
	    ? (double)n_unique_kmers / (double)s->total_ref_kmers
	    : 0.0;
	pthread_mutex_lock(&s->mtx);
	fprintf(s->fp, "%s\t%s\t%lu\t%.6f\t%s\t%s\n",
	        sample_type, sample_id, n_unique_kmers, coverage,
	        is_in_global     ? "True" : "False",
	        is_in_individual ? "True" : "False");
	fflush(s->fp);  /* small file; flush so it's tail-able during long runs */
	pthread_mutex_unlock(&s->mtx);
}

/* ── Sample ID ref-counting ───────────────────────────────────────────── */

static sample_id_t *sample_id_new(const char *id, const char *type)
{
	sample_id_t *s = malloc(sizeof(*s));
	if (!s) { perror("malloc"); exit(EXIT_FAILURE); }
	s->id = strdup(id);
	if (!s->id) { perror("strdup"); exit(EXIT_FAILURE); }
	strncpy(s->type, type, sizeof(s->type) - 1);
	s->type[sizeof(s->type) - 1] = '\0';
	s->refcount = 1;
	pthread_mutex_init(&s->mtx, NULL);
	return s;
}

static void sample_id_ref(sample_id_t *s)
{
	pthread_mutex_lock(&s->mtx);
	s->refcount++;
	pthread_mutex_unlock(&s->mtx);
}

static void sample_id_unref(sample_id_t *s)
{
	pthread_mutex_lock(&s->mtx);
	int rc = --s->refcount;
	pthread_mutex_unlock(&s->mtx);
	if (rc == 0) {
		pthread_mutex_destroy(&s->mtx);
		free(s->id);
		free(s);
	}
}

/* ── Duplicate-sample registry ─────────────────────────────────────────
   Small linear-probing hash of "type:id" strings. Volumes are tiny
   (thousands of samples max), so a simple grow-on-load-factor design
   is plenty.
   ──────────────────────────────────────────────────────────────────── */

struct seen_registry_s {
	char           **slots;   /* NULL or owned heap string "type:id" */
	size_t           cap;
	size_t           size;
	pthread_mutex_t  mtx;
};

static unsigned long sr_hash(const char *s)
{
	unsigned long h = 1469598103934665603UL;  /* FNV-1a */
	for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211UL; }
	return h;
}

static void sr_insert_into(char **slots, size_t cap, char *key)
{
	size_t i = sr_hash(key) % cap;
	while (slots[i] != NULL) i = (i + 1) % cap;
	slots[i] = key;
}

static void sr_grow(seen_registry *r)
{
	size_t new_cap = r->cap * 2;
	char **new_slots = calloc(new_cap, sizeof(char *));
	if (!new_slots) { perror("calloc"); exit(EXIT_FAILURE); }
	for (size_t i = 0; i < r->cap; i++)
		if (r->slots[i]) sr_insert_into(new_slots, new_cap, r->slots[i]);
	free(r->slots);
	r->slots = new_slots;
	r->cap = new_cap;
}

seen_registry *seen_registry_new(void)
{
	seen_registry *r = calloc(1, sizeof(*r));
	if (!r) return NULL;
	r->cap = 256;
	r->slots = calloc(r->cap, sizeof(char *));
	if (!r->slots) { free(r); return NULL; }
	pthread_mutex_init(&r->mtx, NULL);
	return r;
}

void seen_registry_free(seen_registry *r)
{
	if (!r) return;
	for (size_t i = 0; i < r->cap; i++) free(r->slots[i]);
	free(r->slots);
	pthread_mutex_destroy(&r->mtx);
	free(r);
}

/* Returns 1 if (type, id) was already seen (and does NOT add it again),
   0 if newly registered. */
static int seen_registry_check_and_add(seen_registry *r,
                                       const char *type,
                                       const char *id)
{
	if (!r) return 0;

	char key[512];
	int n = snprintf(key, sizeof key, "%s:%s", type, id);
	if (n < 0 || (size_t)n >= sizeof key) {
		fprintf(stderr, "seen_registry: key too long for %s:%s\n", type, id);
		return 0;  /* don't block; just skip dedup for pathological case */
	}

	pthread_mutex_lock(&r->mtx);
	if (r->size * 2 >= r->cap) sr_grow(r);

	size_t i = sr_hash(key) % r->cap;
	while (r->slots[i] != NULL) {
		if (strcmp(r->slots[i], key) == 0) {
			pthread_mutex_unlock(&r->mtx);
			return 1;
		}
		i = (i + 1) % r->cap;
	}
	r->slots[i] = strdup(key);
	if (!r->slots[i]) { perror("strdup"); exit(EXIT_FAILURE); }
	r->size++;
	pthread_mutex_unlock(&r->mtx);
	return 0;
}

/* ── Helpers ──────────────────────────────────────────────────────────── */

static char *basename_no_ext(const char *path)
{
	const char *base = strrchr(path, '/');
	base = base ? base + 1 : path;
	char *out = strdup(base);
	if (!out) { perror("strdup"); exit(EXIT_FAILURE); }

	for (int pass = 0; pass < 2; pass++) {
		char *dot = strrchr(out, '.');
		if (!dot) break;
		const char *ext = dot + 1;
		if (strcmp(ext, "gz")    == 0 ||
		    strcmp(ext, "bz2")   == 0 ||
		    strcmp(ext, "xz")    == 0 ||
		    strcmp(ext, "zst")   == 0 ||
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

extern int   contains_N(char *str);
extern char *orient_string(char *seed_seq, char *seedStrRevComp, int seed);

/* Per-sample k-mer counting hot loop.
   Increments ONLY the per-thread scratch column. The decision to fold
   scratch into the global column is deferred until after the sample is
   fully counted, so we can gate on coverage. */
static void calculate_kmer_count_scratch(const char *file,
                                         const int seed,
                                         BIO_hash h,
                                         unsigned int scratch_col)
{
	gzFile fp = gzopen(file, "r");
	if (fp == NULL) {
		fprintf(stderr, "could not read file %s in calculate_kmer_count_scratch()\n",
		        file);
		return;
	}
	kseq_t *seq = kseq_init(fp);

	char *seedStrRevComp = (char *)malloc(sizeof(char) * (seed + 1));
	char *orientStr;
	unsigned int *count = NULL;
	char temp_nuc;
	char *seed_seq;
	int has_N;

	while (kseq_read(seq) >= 0) {
		if ((int)seq->seq.l < seed) continue;

		BIO_stringToUpper(seq->seq.s);
		seed_seq = seq->seq.s;
		has_N = contains_N(seed_seq);

		for (unsigned int i = 0; i < seq->seq.l - seed + 1; i++) {
			temp_nuc = seed_seq[seed];
			seed_seq[seed] = '\0';
			orientStr = orient_string(seed_seq, seedStrRevComp, seed);

			if (!has_N || !contains_N(orientStr)) {
				count = (unsigned int *)BIO_searchHash(h, orientStr);
				if (count != NULL) {
					/* Scratch is owned by this thread alone, but other threads
					   may be touching the same hash bucket for a different
					   sample's scratch column. Use atomic add to be safe under
					   shared bucket access. */
					__sync_fetch_and_add(&count[scratch_col], 1);
				}
			}

			seed_seq[seed] = temp_nuc;
			seed_seq++;
		}
	}

	kseq_destroy(seq);
	gzclose(fp);
	free(seedStrRevComp);
}

/* ── Worker pool ──────────────────────────────────────────────────────── */

typedef struct {
	char *filepath;
} sample_job_t;

typedef struct {
	sample_job_t   **jobs;
	int              jhead, jtail, jsize, jcap;
	int              shutdown;
	pthread_mutex_t  jmtx;
	pthread_cond_t   jhas_work;

	int              tid_counter;

	int              seed;
	BIO_hash         h;
	int              num_threads;
	streaming_writer *writer;
	summary_writer   *summary;
	seen_registry   *seen;
	const char      *sample_type;
	unsigned int     global_col;
	double           cov_low;
	double           cov_high;
	unsigned long    total_ref_kmers;
	const char      *skip_file;
	FILE            *progress;
	pthread_mutex_t  progress_mtx;
} worker_pool_t;

/* Two-phase sample finalize:

   Phase 1 (count_unique_in_column): sweep scratch, count distinct k-mers
   that hit (i.e. counts[scratch_col] > 0 AND counts[0] > 0 — only
   reference-seeded k-mers count toward coverage).

   Phase 2 (emit_zero_and_maybe_merge): sweep scratch a second time,
   conditionally emit per-sample rows to the streaming writer,
   conditionally fold into the global column atomically, and zero scratch.

   We can't fuse phase 1 into phase 2 because the gating decision needs
   the coverage value before any global merge or row emission happens.
   Phase 1 is cheap (one int compare per bucket, no atomic ops, no row
   pushes).
*/
static unsigned long count_unique_in_column(BIO_hash h,
                                            unsigned int scratch_col)
{
	unsigned long n = 0;
	for (unsigned int i = 0; i < h->M; i++) {
		unsigned int *counts = (unsigned int *)h->data[i].DATA;
		if (counts == NULL) continue;
		/* Only count reference-seeded buckets. counts[0] > 0 means the
		   k-mer was added during the reference seeding pass. */
		if (counts[0] > 0 && counts[scratch_col] > 0) n++;
	}
	return n;
}

/* Sweep scratch_col exactly once, doing whichever of {emit rows, merge
   into global} are enabled, and always zeroing scratch at the end. */
static void emit_zero_and_maybe_merge(BIO_hash h,
                                      unsigned int scratch_col,
                                      unsigned int global_col,
                                      int merge_into_global,
                                      int emit_individual,
                                      sample_id_t *sid,
                                      streaming_writer *writer)
{
	/* If the writer was not provided (e.g. driver run without -o), force
	   emit_individual off — there's nothing to emit to. */
	if (writer == NULL) emit_individual = 0;

	for (unsigned int i = 0; i < h->M; i++) {
		unsigned int *counts = (unsigned int *)h->data[i].DATA;
		if (counts == NULL) continue;
		unsigned int c = counts[scratch_col];
		if (c == 0) continue;

		if (emit_individual) {
			sample_id_ref(sid);
			row_t r;
			r.kmer  = h->data[i].key;
			r.sid   = sid;
			r.count = c;
			writer_push(writer, r);
		}

		if (merge_into_global) {
			/* Other workers may be merging their own samples into the
			   same global column concurrently — must be atomic. */
			__sync_fetch_and_add(&counts[global_col], c);
		}

		counts[scratch_col] = 0;
	}
}

static void *worker_main(void *arg)
{
	worker_pool_t *p = (worker_pool_t *)arg;

	int tid = __sync_fetch_and_add(&p->tid_counter, 1);
	unsigned int scratch_col = (unsigned int)(GLOBAL_COLS + tid);

	for (;;) {
		pthread_mutex_lock(&p->jmtx);
		while (p->jsize == 0 && !p->shutdown)
			pthread_cond_wait(&p->jhas_work, &p->jmtx);
		if (p->jsize == 0 && p->shutdown) {
			pthread_mutex_unlock(&p->jmtx);
			break;
		}
		sample_job_t *job = p->jobs[p->jhead];
		p->jhead = (p->jhead + 1) % p->jcap;
		p->jsize--;
		pthread_mutex_unlock(&p->jmtx);

		const char *path = job->filepath;

		if (p->skip_file != NULL && strcmp(path, p->skip_file) == 0) {
			fprintf(stderr, "skipping %s (identical match to reference)\n",
			        path);
			free(job->filepath);
			free(job);
			continue;
		}

		char *id = basename_no_ext(path);

		/* Duplicate check BEFORE doing any work. We claim the (type, id)
		   slot atomically; if already claimed, skip the file entirely. */
		if (seen_registry_check_and_add(p->seen, p->sample_type, id)) {
			fprintf(stderr,
			        "skipping %s: sample_id '%s' (type=%s) already processed\n",
			        path, id, p->sample_type);
			free(id);
			free(job->filepath);
			free(job);
			continue;
		}

		if (p->progress) {
			pthread_mutex_lock(&p->progress_mtx);
			time_t ltime = time(NULL);
			fprintf(p->progress, "%s\t%s", path, asctime(localtime(&ltime)));
			fflush(p->progress);
			pthread_mutex_unlock(&p->progress_mtx);
		}

		sample_id_t *sid = sample_id_new(id, p->sample_type);

		/* Hot loop: scratch-only counting. */
		calculate_kmer_count_scratch(path, p->seed, p->h, scratch_col);

		/* Phase 1: how many reference-seeded k-mers did this sample hit? */
		unsigned long n_unique = count_unique_in_column(p->h, scratch_col);

		double coverage = p->total_ref_kmers > 0
		    ? (double)n_unique / (double)p->total_ref_kmers
		    : 0.0;

		/* Coverage band gates. Two independent decisions:
		     coverage > cov_high  → drop from BOTH global and individual
		     coverage < cov_low   → drop from individual ONLY (still global)
		   Otherwise both flags True. */
		int is_in_global     = (coverage <= p->cov_high);
		int is_in_individual = is_in_global && (coverage >= p->cov_low);

		if (!is_in_global) {
			fprintf(stderr,
			        "excluding %s [%s] from global+individual: "
			        "coverage=%.6f > cov_high=%.6f\n",
			        id, p->sample_type, coverage, p->cov_high);
		} else if (!is_in_individual) {
			fprintf(stderr,
			        "excluding %s [%s] from individual only: "
			        "coverage=%.6f < cov_low=%.6f\n",
			        id, p->sample_type, coverage, p->cov_low);
		}

		/* Phase 2: do whichever passes are enabled, in a single sweep,
		   and always zero scratch for the next sample on this thread. */
		emit_zero_and_maybe_merge(p->h, scratch_col, p->global_col,
		                          is_in_global, is_in_individual,
		                          sid, p->writer);

		/* Reflect the actual outcome in the summary. If writer was NULL
		   we couldn't have emitted individual rows regardless. */
		int summary_in_individual =
		    is_in_individual && (p->writer != NULL);

		summary_write_row(p->summary, p->sample_type, id,
		                  n_unique, is_in_global, summary_in_individual);

		free(id);
		sample_id_unref(sid);

		free(job->filepath);
		free(job);
	}
	return NULL;
}

/* ── Public entry point ───────────────────────────────────────────────── */

void GEN_per_sample_kmer_counts_dual(const char *list_path,
                                     const char *sample_type,
                                     unsigned int global_col,
                                     int seed,
                                     BIO_hash h,
                                     int num_threads,
                                     streaming_writer *writer,
                                     summary_writer  *summary,
                                     seen_registry   *seen,
                                     double cov_low,
                                     double cov_high,
                                     unsigned long total_reference_kmers,
                                     FILE *progress,
                                     const char *skip_file)
{
	FILE *fp = fopen(list_path, "r");
	if (!fp) {
		fprintf(stderr,
		        "GEN_per_sample_kmer_counts_dual: cannot open %s: %s\n",
		        list_path, strerror(errno));
		exit(EXIT_FAILURE);
	}

	int nlines = 0;
	{
		char *line = NULL;
		size_t cap = 0;
		while (getline(&line, &cap, fp) != -1) {
			char *q = line;
			while (*q == ' ' || *q == '\t') q++;
			if (*q != '\n' && *q != '\0') nlines++;
		}
		free(line);
		rewind(fp);
	}

	if (nlines == 0) {
		fclose(fp);
		return;
	}

	worker_pool_t p;
	memset(&p, 0, sizeof p);
	p.jcap            = nlines + 1;
	p.jobs            = malloc(sizeof(sample_job_t *) * p.jcap);
	p.shutdown        = 0;
	p.tid_counter     = 0;
	p.seed            = seed;
	p.h               = h;
	p.num_threads     = num_threads;
	p.writer          = writer;
	p.summary         = summary;
	p.seen            = seen;
	p.sample_type     = sample_type;
	p.global_col      = global_col;
	p.cov_low         = cov_low;
	p.cov_high        = cov_high;
	p.total_ref_kmers = total_reference_kmers;
	p.skip_file       = skip_file;
	p.progress        = progress;
	pthread_mutex_init(&p.jmtx, NULL);
	pthread_cond_init(&p.jhas_work, NULL);
	pthread_mutex_init(&p.progress_mtx, NULL);

	{
		char *line = NULL;
		size_t cap = 0;
		ssize_t n;
		while ((n = getline(&line, &cap, fp)) != -1) {
			while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r' ||
			                 line[n-1] == ' '  || line[n-1] == '\t'))
				line[--n] = '\0';
			if (n == 0) continue;

			sample_job_t *job = malloc(sizeof(*job));
			job->filepath = strdup(line);
			p.jobs[p.jtail] = job;
			p.jtail = (p.jtail + 1) % p.jcap;
			p.jsize++;
		}
		free(line);
	}
	fclose(fp);

	pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
	for (int i = 0; i < num_threads; i++)
		pthread_create(&threads[i], NULL, worker_main, &p);

	pthread_mutex_lock(&p.jmtx);
	p.shutdown = 1;
	pthread_cond_broadcast(&p.jhas_work);
	pthread_mutex_unlock(&p.jmtx);

	for (int i = 0; i < num_threads; i++)
		pthread_join(threads[i], NULL);

	free(threads);
	free(p.jobs);
	pthread_mutex_destroy(&p.jmtx);
	pthread_cond_destroy(&p.jhas_work);
	pthread_mutex_destroy(&p.progress_mtx);
}
