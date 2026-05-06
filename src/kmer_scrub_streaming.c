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
#include <inttypes.h>
#include <errno.h>
#include <zlib.h>
#include <zstd.h>

KSEQ_INIT(gzFile, gzread)

#define GLOBAL_COLS 4

/* ── zstd stream writer (used at flush time) ────────────────────────── */

typedef struct {
	FILE         *fp;
	ZSTD_CStream *cs;
	void         *out_buf;
	size_t        out_cap;
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

/* ── Per-bucket dynamic id list ──────────────────────────────────────── */

typedef struct {
	uint32_t *ids;
	uint32_t  size;
	uint32_t  cap;
} id_list_t;

static void id_list_append(id_list_t *l, uint32_t id)
{
	if (l->size == l->cap) {
		uint32_t newcap = l->cap == 0 ? 4 : l->cap * 2;
		uint32_t *newids = realloc(l->ids, newcap * sizeof(uint32_t));
		if (!newids) { perror("realloc id_list"); exit(EXIT_FAILURE); }
		l->ids = newids;
		l->cap = newcap;
	}
	l->ids[l->size++] = id;
}

/* ── Per-sample queue record ─────────────────────────────────────────── */

typedef struct sample_record_s {
	char         *sample_id;     /* owned heap copy */
	char          sample_type[3];
	unsigned long n_unique_kmers;
	double        coverage_pct;
	int           is_in_global;
	uint32_t     *bucket_indices;  /* owned; bucket idx of each hit kmer */
	uint32_t      n_kmers;
} sample_record_t;

/* ── Summary writer ──────────────────────────────────────────────────── */

struct summary_writer_s {
	FILE           *fp;
	unsigned long   total_ref_kmers;
	pthread_mutex_t mtx;  /* used only when writer is NULL */
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
	        "scrub_id\tsample_type\tsample_id\tn_unique_kmers"
	        "\tcoverage_pct\tis_in_global\n");
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
                              uint32_t scrub_id,
                              const char *sample_type,
                              const char *sample_id,
                              unsigned long n_unique_kmers,
                              double coverage_pct,
                              int is_in_global,
                              int needs_lock)
{
	if (!s) return;
	if (needs_lock) pthread_mutex_lock(&s->mtx);
	fprintf(s->fp, "%u\t%s\t%s\t%lu\t%.6f\t%s\n",
	        scrub_id, sample_type, sample_id,
	        n_unique_kmers, coverage_pct,
	        is_in_global ? "True" : "False");
	fflush(s->fp);
	if (needs_lock) pthread_mutex_unlock(&s->mtx);
}

/* ── Presence writer ──────────────────────────────────────────────────── */

struct presence_writer_s {
	const char  *out_path;        /* not duped; caller must keep valid */

	sample_record_t **queue;
	size_t            cap;
	size_t            head, tail, size;
	int               shutdown;
	pthread_mutex_t   mtx;
	pthread_cond_t    not_empty;
	pthread_cond_t    not_full;
	pthread_t         writer_tid;
	int               thread_started;

	uint32_t          next_scrub_id;
	summary_writer   *summary;

	id_list_t        *id_lists;
	unsigned int      n_buckets;

	/* Diagnostic counters */
	uint64_t  diag_writer_waits;
	uint64_t  diag_worker_waits;
	uint64_t  diag_writer_wait_ns;
	uint64_t  diag_worker_wait_ns;
	size_t    diag_max_queue_depth;
	uint64_t  diag_samples_processed;
	uint64_t  diag_kmer_appends;
};

static inline uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void *presence_writer_main(void *arg);

presence_writer *presence_writer_open(const char *path,
                                      size_t queue_capacity)
{
	if (queue_capacity == 0) queue_capacity = 256;

	presence_writer *w = calloc(1, sizeof(*w));
	if (!w) return NULL;
	w->out_path = path;
	w->cap = queue_capacity;
	w->queue = calloc(queue_capacity, sizeof(sample_record_t *));
	if (!w->queue) { free(w); return NULL; }
	pthread_mutex_init(&w->mtx, NULL);
	pthread_cond_init(&w->not_empty, NULL);
	pthread_cond_init(&w->not_full, NULL);

	if (pthread_create(&w->writer_tid, NULL, presence_writer_main, w) != 0) {
		fprintf(stderr, "presence_writer_open: pthread_create failed\n");
		pthread_mutex_destroy(&w->mtx);
		pthread_cond_destroy(&w->not_empty);
		pthread_cond_destroy(&w->not_full);
		free(w->queue);
		free(w);
		return NULL;
	}
	w->thread_started = 1;
	return w;
}

/* Called from the public entry point on first use to wire up the writer
   to its companion summary file and to allocate the per-bucket side
   array. Idempotent. */
static void presence_writer_attach_summary(presence_writer *w,
                                           summary_writer *s)
{
	if (!w) return;
	pthread_mutex_lock(&w->mtx);
	if (w->summary == NULL) w->summary = s;
	pthread_mutex_unlock(&w->mtx);
}

static void presence_writer_attach_hash(presence_writer *w, BIO_hash h)
{
	if (!w) return;
	pthread_mutex_lock(&w->mtx);
	if (w->id_lists == NULL) {
		w->n_buckets = (unsigned int)h->M;
		w->id_lists  = calloc(w->n_buckets, sizeof(id_list_t));
		if (!w->id_lists) {
			fprintf(stderr,
			        "presence_writer_attach_hash: calloc(%u) failed\n",
			        w->n_buckets);
			exit(EXIT_FAILURE);
		}
	}
	pthread_mutex_unlock(&w->mtx);
}

static void writer_push_sample(presence_writer *w, sample_record_t *rec)
{
	uint64_t t0 = 0;
	int waited = 0;
	pthread_mutex_lock(&w->mtx);
	if (w->size == w->cap) { waited = 1; t0 = now_ns(); }
	while (w->size == w->cap)
		pthread_cond_wait(&w->not_full, &w->mtx);
	if (waited) {
		w->diag_worker_waits++;
		w->diag_worker_wait_ns += (now_ns() - t0);
	}
	w->queue[w->tail] = rec;
	w->tail = (w->tail + 1) % w->cap;
	w->size++;
	if (w->size > w->diag_max_queue_depth)
		w->diag_max_queue_depth = w->size;
	pthread_cond_signal(&w->not_empty);
	pthread_mutex_unlock(&w->mtx);
}

static void *presence_writer_main(void *arg)
{
	presence_writer *w = (presence_writer *)arg;

	for (;;) {
		uint64_t t0 = 0;
		int waited = 0;
		pthread_mutex_lock(&w->mtx);
		if (w->size == 0 && !w->shutdown) { waited = 1; t0 = now_ns(); }
		while (w->size == 0 && !w->shutdown)
			pthread_cond_wait(&w->not_empty, &w->mtx);
		if (waited) {
			w->diag_writer_waits++;
			w->diag_writer_wait_ns += (now_ns() - t0);
		}
		if (w->size == 0 && w->shutdown) {
			pthread_mutex_unlock(&w->mtx);
			break;
		}
		sample_record_t *rec = w->queue[w->head];
		w->head = (w->head + 1) % w->cap;
		w->size--;
		pthread_cond_broadcast(&w->not_full);
		pthread_mutex_unlock(&w->mtx);

		/* ── Process this sample serially ─────────────────────────── */
		uint32_t scrub_id = w->next_scrub_id++;

		summary_write_row(w->summary, scrub_id,
		                  rec->sample_type, rec->sample_id,
		                  rec->n_unique_kmers, rec->coverage_pct,
		                  rec->is_in_global, /*needs_lock=*/0);

		if (w->id_lists != NULL && rec->bucket_indices != NULL) {
			for (uint32_t k = 0; k < rec->n_kmers; k++) {
				uint32_t idx = rec->bucket_indices[k];
				if (idx < w->n_buckets) {
					id_list_append(&w->id_lists[idx], scrub_id);
					w->diag_kmer_appends++;
				}
			}
		}

		w->diag_samples_processed++;

		free(rec->sample_id);
		free(rec->bucket_indices);
		free(rec);
	}
	return NULL;
}

void presence_writer_close(presence_writer *w)
{
	if (!w) return;
	if (!w->thread_started) return;

	pthread_mutex_lock(&w->mtx);
	w->shutdown = 1;
	pthread_cond_broadcast(&w->not_empty);
	pthread_mutex_unlock(&w->mtx);

	pthread_join(w->writer_tid, NULL);
	w->thread_started = 0;
}

void presence_writer_flush(presence_writer *w, BIO_hash h)
{
	if (!w || !w->out_path) return;
	if (w->thread_started) {
		fprintf(stderr,
		        "presence_writer_flush: must call presence_writer_close first\n");
		return;
	}
	if (!w->id_lists) {
		fprintf(stderr, "presence_writer_flush: no id_lists (no samples?)\n");
		return;
	}

	zstd_out_t *z = zstd_out_open(w->out_path, /*level*/ 9);
	if (!z) {
		fprintf(stderr, "presence_writer_flush: cannot open %s: %s\n",
		        w->out_path, strerror(errno));
		return;
	}

	const char *header = "#kmer\tlist_scrub_id\n";
	zstd_out_write(z, header, strlen(header));

	/* Buffer for one row. Worst case: kmer string + tab + N_max ids x 11
	   bytes (uint32) + commas + \n. We grow on demand. */
	size_t row_cap = 256 * 1024;
	char  *rowbuf  = malloc(row_cap);
	if (!rowbuf) { perror("malloc rowbuf"); exit(EXIT_FAILURE); }

	uint64_t emitted = 0;
	for (unsigned int i = 0; i < w->n_buckets; i++) {
		id_list_t *l = &w->id_lists[i];
		if (l->size == 0) continue;
		const char *key = h->data[i].key;
		if (!key) continue;

		size_t need = strlen(key) + 2 + (size_t)l->size * 12 + 2;
		if (need > row_cap) {
			while (row_cap < need) row_cap *= 2;
			char *nbuf = realloc(rowbuf, row_cap);
			if (!nbuf) { perror("realloc rowbuf"); exit(EXIT_FAILURE); }
			rowbuf = nbuf;
		}

		int n = snprintf(rowbuf, row_cap, "%s\t", key);
		for (uint32_t j = 0; j < l->size; j++) {
			n += snprintf(rowbuf + n, row_cap - (size_t)n,
			              j == 0 ? "%u" : ",%u", l->ids[j]);
		}
		rowbuf[n++] = '\n';
		zstd_out_write(z, rowbuf, (size_t)n);
		emitted++;
	}

	free(rowbuf);
	zstd_out_close(z);
	fprintf(stderr,
	        "presence_writer_flush: emitted %" PRIu64 " kmer rows to %s\n",
	        emitted, w->out_path);
}

void presence_writer_print_diagnostics(presence_writer *w)
{
	if (!w) return;
	fprintf(stderr,
	        "─── presence writer diagnostics ─────────────────────────────\n"
	        "  samples processed:   %" PRIu64 "\n"
	        "  scrub_id appends:    %" PRIu64 "\n"
	        "  worker queue waits:  %" PRIu64
	        " (total %.3fs blocked on full queue)\n"
	        "                       ↑ if >>0: WRITER is the bottleneck\n"
	        "  writer queue waits:  %" PRIu64
	        " (total %.3fs blocked on empty queue)\n"
	        "                       ↑ if >>0: WORKERS are the bottleneck\n"
	        "  max queue depth:     %zu / %zu\n"
	        "─────────────────────────────────────────────────────────────\n",
	        w->diag_samples_processed,
	        w->diag_kmer_appends,
	        w->diag_worker_waits,
	        (double)w->diag_worker_wait_ns / 1e9,
	        w->diag_writer_waits,
	        (double)w->diag_writer_wait_ns / 1e9,
	        w->diag_max_queue_depth,
	        w->cap);
}

void presence_writer_destroy(presence_writer *w)
{
	if (!w) return;
	if (w->thread_started) presence_writer_close(w);
	if (w->id_lists) {
		for (unsigned int i = 0; i < w->n_buckets; i++)
			free(w->id_lists[i].ids);
		free(w->id_lists);
	}
	pthread_mutex_destroy(&w->mtx);
	pthread_cond_destroy(&w->not_empty);
	pthread_cond_destroy(&w->not_full);
	free(w->queue);
	free(w);
}

/* ── Duplicate-sample registry ───────────────────────────────────────── */

struct seen_registry_s {
	char           **slots;
	size_t           cap;
	size_t           size;
	pthread_mutex_t  mtx;
};

static unsigned long sr_hash(const char *s)
{
	unsigned long h = 1469598103934665603UL;
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

static int seen_registry_check_and_add(seen_registry *r,
                                       const char *type,
                                       const char *id)
{
	if (!r) return 0;
	char key[512];
	int n = snprintf(key, sizeof key, "%s:%s", type, id);
	if (n < 0 || (size_t)n >= sizeof key) {
		fprintf(stderr, "seen_registry: key too long for %s:%s\n", type, id);
		return 0;
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

/* ── Helpers ─────────────────────────────────────────────────────────── */

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

/* Hot loop: scratch-only counting. */
static void calculate_kmer_count_scratch(const char *file,
                                         const int seed,
                                         BIO_hash h,
                                         unsigned int scratch_col)
{
	gzFile fp = gzopen(file, "r");
	if (fp == NULL) {
		fprintf(stderr,
		        "could not read file %s in calculate_kmer_count_scratch()\n",
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
				if (count != NULL)
					__sync_fetch_and_add(&count[scratch_col], 1);
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

	/* Internal scrub_id allocator used only when writer == NULL.
	   When writer is non-NULL the writer thread allocates ids serially. */
	uint32_t         fallback_next_id;

	int              seed;
	BIO_hash         h;
	int              num_threads;
	presence_writer *writer;
	summary_writer  *summary;
	seen_registry   *seen;
	const char      *sample_type;
	unsigned int     global_col;
	double           cov_threshold;
	unsigned long    total_ref_kmers;
	const char      *skip_file;
	FILE            *progress;
	pthread_mutex_t  progress_mtx;
} worker_pool_t;

/* Single-pass: collect bucket indices of hit kmers, optionally fold
   into global, zero scratch. n_unique is the count from the cheap
   pre-sweep (count_unique_in_column) and is exact for reference-seeded
   buckets, which is what we collect here. Returns malloc'd uint32 array
   (size n_unique) or NULL if n_unique == 0. */
static unsigned long count_unique_in_column(BIO_hash h,
                                            unsigned int scratch_col)
{
	unsigned long n = 0;
	for (unsigned int i = 0; i < h->M; i++) {
		unsigned int *counts = (unsigned int *)h->data[i].DATA;
		if (counts == NULL) continue;
		if (counts[0] > 0 && counts[scratch_col] > 0) n++;
	}
	return n;
}

static uint32_t *collect_zero_and_maybe_merge(BIO_hash h,
                                              unsigned int scratch_col,
                                              unsigned int global_col,
                                              int merge_into_global,
                                              unsigned long n_unique)
{
	uint32_t *out = NULL;
	if (n_unique > 0) {
		out = malloc((size_t)n_unique * sizeof(uint32_t));
		if (!out) { perror("malloc kmer indices"); exit(EXIT_FAILURE); }
	}
	uint32_t k = 0;
	for (unsigned int i = 0; i < h->M; i++) {
		unsigned int *counts = (unsigned int *)h->data[i].DATA;
		if (counts == NULL) continue;
		unsigned int c = counts[scratch_col];
		if (c == 0) continue;

		if (counts[0] > 0 && k < (uint32_t)n_unique)
			out[k++] = (uint32_t)i;

		if (merge_into_global)
			__sync_fetch_and_add(&counts[global_col], c);
		counts[scratch_col] = 0;
	}
	/* k should equal n_unique. If hash mutated mid-flight (it can't here)
	   we'd notice. Defensive: pad with 0s if short, but realistically k==n_unique. */
	return out;
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
			free(job->filepath); free(job);
			continue;
		}

		char *id = basename_no_ext(path);
		if (seen_registry_check_and_add(p->seen, p->sample_type, id)) {
			fprintf(stderr,
			        "skipping %s: sample_id '%s' (type=%s) already processed\n",
			        path, id, p->sample_type);
			free(id); free(job->filepath); free(job);
			continue;
		}

		if (p->progress) {
			pthread_mutex_lock(&p->progress_mtx);
			time_t ltime = time(NULL);
			fprintf(p->progress, "%s\t%s", path, asctime(localtime(&ltime)));
			fflush(p->progress);
			pthread_mutex_unlock(&p->progress_mtx);
		}

		calculate_kmer_count_scratch(path, p->seed, p->h, scratch_col);

		unsigned long n_unique = count_unique_in_column(p->h, scratch_col);
		double coverage = p->total_ref_kmers > 0
		    ? (double)n_unique / (double)p->total_ref_kmers
		    : 0.0;
		int is_in_global = (coverage <= p->cov_threshold);
		if (!is_in_global) {
			fprintf(stderr,
			        "excluding %s from global %s column: coverage=%.6f > T=%.6f\n",
			        id, p->sample_type, coverage, p->cov_threshold);
		}

		uint32_t *idxs = collect_zero_and_maybe_merge(
		    p->h, scratch_col, p->global_col, is_in_global, n_unique);

		if (p->writer) {
			/* Hand off to writer; writer assigns scrub_id and writes summary. */
			sample_record_t *rec = malloc(sizeof(*rec));
			if (!rec) { perror("malloc rec"); exit(EXIT_FAILURE); }
			rec->sample_id = id;  /* ownership transferred */
			strncpy(rec->sample_type, p->sample_type,
			        sizeof(rec->sample_type) - 1);
			rec->sample_type[sizeof(rec->sample_type) - 1] = '\0';
			rec->n_unique_kmers = n_unique;
			rec->coverage_pct   = coverage;
			rec->is_in_global   = is_in_global;
			rec->bucket_indices = idxs;       /* ownership transferred */
			rec->n_kmers        = (uint32_t)n_unique;
			writer_push_sample(p->writer, rec);
			/* DO NOT free id or idxs — owned by writer now. */
		} else {
			/* No writer: emit summary inline with a fallback scrub_id.
			   The id will not be referenced from any presence file (which
			   is also absent in this mode), but we keep the column for
			   schema consistency. */
			uint32_t scrub_id = __sync_fetch_and_add(&p->fallback_next_id, 1);
			summary_write_row(p->summary, scrub_id,
			                  p->sample_type, id,
			                  n_unique, coverage, is_in_global,
			                  /*needs_lock=*/1);
			free(idxs);
			free(id);
		}

		free(job->filepath);
		free(job);
	}
	return NULL;
}

/* ── Public entry point ──────────────────────────────────────────────── */

void GEN_per_sample_kmer_counts_dual(const char *list_path,
                                     const char *sample_type,
                                     unsigned int global_col,
                                     int seed,
                                     BIO_hash h,
                                     int num_threads,
                                     presence_writer *writer,
                                     summary_writer  *summary,
                                     seen_registry   *seen,
                                     double cov_threshold,
                                     unsigned long total_reference_kmers,
                                     FILE *progress,
                                     const char *skip_file)
{
	if (writer) {
		presence_writer_attach_summary(writer, summary);
		presence_writer_attach_hash(writer, h);
	}

	FILE *fp = fopen(list_path, "r");
	if (!fp) {
		fprintf(stderr,
		        "GEN_per_sample_kmer_counts_dual: cannot open %s: %s\n",
		        list_path, strerror(errno));
		exit(EXIT_FAILURE);
	}

	int nlines = 0;
	{
		char *line = NULL; size_t cap = 0;
		while (getline(&line, &cap, fp) != -1) {
			char *q = line;
			while (*q == ' ' || *q == '\t') q++;
			if (*q != '\n' && *q != '\0') nlines++;
		}
		free(line);
		rewind(fp);
	}
	if (nlines == 0) { fclose(fp); return; }

	worker_pool_t p;
	memset(&p, 0, sizeof p);
	p.jcap            = nlines + 1;
	p.jobs            = malloc(sizeof(sample_job_t *) * p.jcap);
	p.seed            = seed;
	p.h               = h;
	p.num_threads     = num_threads;
	p.writer          = writer;
	p.summary         = summary;
	p.seen            = seen;
	p.sample_type     = sample_type;
	p.global_col      = global_col;
	p.cov_threshold   = cov_threshold;
	p.total_ref_kmers = total_reference_kmers;
	p.skip_file       = skip_file;
	p.progress        = progress;
	pthread_mutex_init(&p.jmtx, NULL);
	pthread_cond_init(&p.jhas_work, NULL);
	pthread_mutex_init(&p.progress_mtx, NULL);

	{
		char *line = NULL; size_t cap = 0; ssize_t n;
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
