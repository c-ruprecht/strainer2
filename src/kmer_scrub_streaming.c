#include "kmer_scrub_streaming.h"
#include "BIO_sequence.h"   /* must precede genome_compare.h: defines BIO_sequences */
#include "BIO_hash.h"
#include "genome_compare.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>

/*
	GEN_calculate_kmer_count is defined in genome_compare.c but not
	declared in genome_compare.h. Forward-declare it here so we can call
	it without modifying the existing header.
*/
extern void GEN_calculate_kmer_count(const char *file, const int seed,
                                     BIO_hash h, unsigned int vec_column);

/*
	Number of "global" columns occupied by the existing
	GEN_all_kmer_counts* machinery. Per-thread scratch columns start at
	GLOBAL_COLS and run through (GLOBAL_COLS + num_threads - 1).
*/
#define GLOBAL_COLS 4

/* ── Bounded row queue ────────────────────────────────────────────────── */

/*
	Rows pushed by workers and consumed by the writer thread. We store a
	pointer to the kmer string (stable for the hash's lifetime, no copy
	needed) and a heap-allocated copy of sample_id (one allocation per
	(sample, type) — we share a single sample_id pointer across all rows
	for a given sample using a small ref-counted holder).

	Reference counting matters here: hundreds of millions of rows can
	share one sample_id string, and we cannot free the string until the
	writer has drained the last row that references it.
*/
typedef struct sample_id_s {
	char           *id;         /* heap-allocated, freed when refcount hits 0 */
	char            type[3];    /* "ge", "me", "dr" — null-terminated */
	int             refcount;   /* incremented before push, decremented in writer */
	pthread_mutex_t mtx;        /* guards refcount */
} sample_id_t;

typedef struct {
	const char  *kmer;          /* pointer into BIO_hash key, do not free */
	sample_id_t *sid;           /* shared, ref-counted */
	uint32_t     count;
} row_t;

struct streaming_writer_s {
	gzFile           gz;
	row_t           *queue;
	size_t           cap;
	size_t           head, tail, size;
	int              shutdown;          /* set by streaming_writer_close */
	pthread_mutex_t  mtx;
	pthread_cond_t   not_empty;
	pthread_cond_t   not_full;
	pthread_t        writer_tid;
};

static void sample_id_unref(sample_id_t *sid);

/* Writer thread: drains rows and gzwrites them. */
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

		/* drain a chunk under the lock to amortize signaling cost */
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
			gzprintf(w->gz, "%s\t%s\t%s\t%u\n",
			         batch[i].kmer,
			         batch[i].sid->type,
			         batch[i].sid->id,
			         batch[i].count);
			sample_id_unref(batch[i].sid);
		}
	}
	return NULL;
}

streaming_writer *streaming_writer_open(const char *path, size_t queue_capacity)
{
	if (queue_capacity == 0) queue_capacity = 1u << 20; /* ~1M rows */

	streaming_writer *w = calloc(1, sizeof(*w));
	if (!w) return NULL;

	w->gz = gzopen(path, "wb9");
	if (!w->gz) {
		fprintf(stderr, "streaming_writer_open: gzopen %s failed: %s\n",
		        path, strerror(errno));
		free(w);
		return NULL;
	}
	gzbuffer(w->gz, 1u << 20); /* 1 MiB internal buffer */

	w->queue = calloc(queue_capacity, sizeof(row_t));
	if (!w->queue) {
		gzclose(w->gz);
		free(w);
		return NULL;
	}
	w->cap = queue_capacity;
	w->head = w->tail = w->size = 0;
	w->shutdown = 0;
	pthread_mutex_init(&w->mtx, NULL);
	pthread_cond_init(&w->not_empty, NULL);
	pthread_cond_init(&w->not_full, NULL);

	/* header */
	gzprintf(w->gz, "kmer\tsample_type\tsample_id\tcount\n");

	if (pthread_create(&w->writer_tid, NULL, writer_main, w) != 0) {
		fprintf(stderr, "streaming_writer_open: pthread_create failed\n");
		gzclose(w->gz);
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

	gzclose(w->gz);
	pthread_mutex_destroy(&w->mtx);
	pthread_cond_destroy(&w->not_empty);
	pthread_cond_destroy(&w->not_full);
	free(w->queue);
	free(w);
}

/* Push one row. Blocks if the queue is full (backpressure). */
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

/* ── Sample ID ref-counting ───────────────────────────────────────────── */

static sample_id_t *sample_id_new(const char *id, const char *type)
{
	sample_id_t *s = malloc(sizeof(*s));
	if (!s) { perror("malloc"); exit(EXIT_FAILURE); }
	s->id = strdup(id);
	if (!s->id) { perror("strdup"); exit(EXIT_FAILURE); }
	strncpy(s->type, type, sizeof(s->type) - 1);
	s->type[sizeof(s->type) - 1] = '\0';
	s->refcount = 1; /* held by producer until all rows pushed */
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

/* ── basename helpers (mirror the originals from kmer_scrub_count_individual) */

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

/* ── Worker thread pool ───────────────────────────────────────────────── */

typedef struct {
	char *filepath;     /* heap-allocated, owned by the job */
} sample_job_t;

typedef struct {
	/* job queue */
	sample_job_t   **jobs;
	int              jhead, jtail, jsize, jcap;
	int              shutdown;
	pthread_mutex_t  jmtx;
	pthread_cond_t   jhas_work;

	/* worker tid assignment — atomic counter consumed by workers at start */
	int              tid_counter;

	/* shared per-call state */
	int              seed;
	BIO_hash         h;
	int              num_threads;
	streaming_writer *writer;
	const char      *sample_type;   /* "ge" / "me" / "dr" */
	const char      *skip_file;     /* may be NULL */
	FILE            *progress;
	pthread_mutex_t  progress_mtx;
} worker_pool_t;

/*
	Sweep the hash: for every k-mer where this thread's column is non-zero,
	push a row to the writer and zero the column.

	Safety:
	 - Each worker reads/writes only its own column index. No two workers
	   touch the same uint32_t word.
	 - Other workers may be concurrently incrementing *their own* columns
	   in the same value vector. Different memory addresses, no race.
	 - The hash structure itself (h->data, h->M, h->data[i].DATA) is
	   read-only during this phase — no rule mutates the hash topology
	   after seeding, and BIO_searchHash is a pure read.
*/
static void emit_and_zero_column(BIO_hash h,
                                 unsigned int col,
                                 sample_id_t *sid,
                                 streaming_writer *writer)
{
	for (unsigned int i = 0; i < h->M; i++) {
		unsigned int *counts = (unsigned int *)h->data[i].DATA;
		if (counts == NULL) continue;
		unsigned int c = counts[col];
		if (c == 0) continue;

		sample_id_ref(sid);
		row_t r;
		r.kmer  = h->data[i].key;  /* stable pointer for hash lifetime */
		r.sid   = sid;
		r.count = c;
		writer_push(writer, r);

		counts[col] = 0;
	}
}

static void *worker_main(void *arg)
{
	worker_pool_t *p = (worker_pool_t *)arg;

	/* claim a stable worker tid in [0, num_threads) */
	int tid = __sync_fetch_and_add(&p->tid_counter, 1);
	unsigned int col = (unsigned int)(GLOBAL_COLS + tid);

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

		/* skip-file convention: matches the existing -C reference dedup */
		if (p->skip_file != NULL && strcmp(path, p->skip_file) == 0) {
			fprintf(stderr, "skipping %s (identical match)\n", path);
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

		/* derive sample_id from basename (strip .fasta.gz / .fna.gz / etc.) */
		char *id = basename_no_ext(path);
		sample_id_t *sid = sample_id_new(id, p->sample_type);
		free(id);

		/* count this sample into our exclusive column */
		GEN_calculate_kmer_count(path, p->seed, p->h, col);

		/* emit non-zero rows and zero the column */
		emit_and_zero_column(p->h, col, sid, p->writer);

		/* drop our own reference; writer holds the rest */
		sample_id_unref(sid);

		free(job->filepath);
		free(job);
	}
	return NULL;
}

/* ── Public entry point ───────────────────────────────────────────────── */

void GEN_per_sample_kmer_counts_streaming(const char *list_path,
                                          const char *sample_type,
                                          int seed,
                                          BIO_hash h,
                                          int num_threads,
                                          streaming_writer *writer,
                                          FILE *progress,
                                          const char *skip_file)
{
	FILE *fp = fopen(list_path, "r");
	if (!fp) {
		fprintf(stderr,
		        "GEN_per_sample_kmer_counts_streaming: cannot open %s: %s\n",
		        list_path, strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* count lines to size the job queue */
	int nlines = 0;
	{
		char *line = NULL;
		size_t cap = 0;
		while (getline(&line, &cap, fp) != -1) {
			/* skip blank lines */
			char *p = line;
			while (*p == ' ' || *p == '\t') p++;
			if (*p != '\n' && *p != '\0') nlines++;
		}
		free(line);
		rewind(fp);
	}

	if (nlines == 0) {
		fclose(fp);
		return; /* nothing to do; not an error */
	}

	worker_pool_t p;
	memset(&p, 0, sizeof p);
	p.jcap        = nlines + 1;
	p.jobs        = malloc(sizeof(sample_job_t *) * p.jcap);
	p.jhead = p.jtail = p.jsize = 0;
	p.shutdown    = 0;
	p.tid_counter = 0;
	p.seed        = seed;
	p.h           = h;
	p.num_threads = num_threads;
	p.writer      = writer;
	p.sample_type = sample_type;
	p.skip_file   = skip_file;
	p.progress    = progress;
	pthread_mutex_init(&p.jmtx, NULL);
	pthread_cond_init(&p.jhas_work, NULL);
	pthread_mutex_init(&p.progress_mtx, NULL);

	/* enqueue all samples up front */
	{
		char *line = NULL;
		size_t cap = 0;
		ssize_t n;
		while ((n = getline(&line, &cap, fp)) != -1) {
			/* trim trailing newline / whitespace */
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

	/* launch workers */
	pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
	for (int i = 0; i < num_threads; i++)
		pthread_create(&threads[i], NULL, worker_main, &p);

	/* signal shutdown — workers will exit once the queue drains */
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