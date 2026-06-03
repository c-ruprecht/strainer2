#include "metagenome_db.h"
#include "up2bit.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <zlib.h>
#include <zstd.h>
#include <pthread.h>
#include <inttypes.h>
#include <unistd.h>

/* ──────────────────────────────────────────────────────────── */
#define READ_LINE_BUF       (1 << 16)   /* 64 KiB; covers short-read SRA   */
#define REC_INIT_CAP        (1 << 20)   /* 1M records initial phase-1 array */
#define CURSOR_BUF_RECORDS  4096        /* per-input read buffer in merge   */

/* stderr is shared across worker threads; serialize whole lines so the
 * per-file stats and progress messages never interleave. */
static pthread_mutex_t g_stderr_lock = PTHREAD_MUTEX_INITIALIZER;

/* ════════════════════════════════════════════════════════════
 * 2-bit encoding helpers — reused verbatim from the in-memory
 * implementation. Mapping (via (char & 0x6) >> 1): A=0, C=1, T=2, G=3.
 * Complement is therefore (b ^ 2): A<->T, C<->G.
 * ════════════════════════════════════════════════════════════ */

static uint64_t reverse_complement_2bit(uint64_t kmer, int k)
{
    uint64_t rc = 0;
    for (int i = 0; i < k; i++) {
        rc <<= 2;
        rc |= (kmer & 3ULL) ^ 2ULL;
        kmer >>= 2;
    }
    return rc;
}

static inline uint64_t canonical_2bit(uint64_t kmer, int k)
{
    uint64_t rc = reverse_complement_2bit(kmer, k);
    return (kmer < rc) ? kmer : rc;
}

static int encode_kmer(const char *seq, int k, uint64_t *out)
{
    uint64_t enc = 0;
    for (int i = 0; i < k; i++) {
        char ch = seq[i] | 0x20;   /* fold to lowercase for validation */
        if (ch != 'a' && ch != 'c' && ch != 't' && ch != 'g')
            return -1;
        uint8_t b = ((uint8_t)seq[i] & 0x6) >> 1;
        enc = (enc << 2) | b;
    }
    *out = enc;
    return 0;
}

/* ════════════════════════════════════════════════════════════
 * Phase 1 — extract, sort, dedup, write one sample's temp file
 * ════════════════════════════════════════════════════════════ */

/* qsort order: kmer ascending, then sample_id ascending. */
static int rec_cmp(const void *pa, const void *pb)
{
    const kmer_record *a = pa;        /* const void* -> const T*: no qual loss */
    const kmer_record *b = pb;
    if (a->kmer < b->kmer) return -1;
    if (a->kmer > b->kmer) return  1;
    if (a->sample_id < b->sample_id) return -1;
    if (a->sample_id > b->sample_id) return  1;
    return 0;
}

int mdb_sort_sample(const char *filepath, uint32_t sample_id,
                    int kmer_len, const char *tmpdir,
                    char *out_path, size_t out_path_sz)
{
    if (out_path_sz) out_path[0] = '\0';

    gzFile fp = gzopen(filepath, "r");
    if (!fp) {
        pthread_mutex_lock(&g_stderr_lock);
        fprintf(stderr, "error: cannot open %s\n", filepath);
        pthread_mutex_unlock(&g_stderr_lock);
        return -1;
    }

    size_t       cap  = REC_INIT_CAP;
    size_t       n    = 0;
    kmer_record *recs = malloc(cap * sizeof *recs);
    char        *line = malloc(READ_LINE_BUF);
    if (!recs || !line) {
        perror("malloc phase-1 buffers");
        free(recs); free(line); gzclose(fp);
        return -1;
    }

    /* FASTA/FASTQ state machine.
     *   state 0: expect header ('@' fastq, '>' fasta)
     *   state 1: sequence line
     *   state 2: fastq '+' separator
     *   state 3: fastq quality line
     * Note: like the original, this treats a FASTA record as a single
     * sequence line. Inputs here are short-read SRA FASTQ, so that holds;
     * multi-line FASTA would need a continuation tweak. */
    int  is_fastq = 0;
    int  state    = 0;
    long n_positions = 0;   /* total kmer windows scanned        */
    long n_skipped   = 0;   /* windows with a non-ACGT base      */

    while (gzgets(fp, line, READ_LINE_BUF) != NULL) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        if (state == 0) {
            if      (line[0] == '@') { is_fastq = 1; state = 1; }
            else if (line[0] == '>') { is_fastq = 0; state = 1; }
            continue;
        }

        if (state == 1) {
            for (int i = 0; i + kmer_len <= len; i++) {
                n_positions++;
                uint64_t enc;
                if (encode_kmer(line + i, kmer_len, &enc) != 0) {
                    n_skipped++;
                    continue;
                }
                uint64_t canon = canonical_2bit(enc, kmer_len);

                if (n == cap) {
                    cap *= 2;
                    kmer_record *grown = realloc(recs, cap * sizeof *recs);
                    if (!grown) {
                        perror("realloc recs");
                        free(recs); free(line); gzclose(fp);
                        return -1;
                    }
                    recs = grown;
                }
                recs[n].kmer      = canon;
                recs[n].sample_id = sample_id;
                n++;
            }
            state = is_fastq ? 2 : 0;
            continue;
        }

        if (is_fastq && state == 2) { state = 3; continue; }  /* '+'     */
        if (is_fastq && state == 3) { state = 0; continue; }  /* quality */
    }

    free(line);
    gzclose(fp);

    long n_written = 0;
    if (n > 0) {
        qsort(recs, n, sizeof *recs, rec_cmp);

        /* Dedup consecutive identical (kmer, sample_id). Every record in
         * this file carries the same sample_id, so this collapses repeated
         * kmers within the sample down to a single occurrence — the temp
         * file ends up with each canonical kmer exactly once. */
        size_t w = 1;
        for (size_t r = 1; r < n; r++) {
            if (recs[r].kmer      != recs[w-1].kmer ||
                recs[r].sample_id != recs[w-1].sample_id)
                recs[w++] = recs[r];
        }
        n = w;

        snprintf(out_path, out_path_sz, "%s/%" PRIu32 ".tmp.bin",
                 tmpdir, sample_id);

        FILE *tf = fopen(out_path, "wb");
        if (!tf) {
            pthread_mutex_lock(&g_stderr_lock);
            fprintf(stderr, "error: cannot create temp file %s\n", out_path);
            pthread_mutex_unlock(&g_stderr_lock);
            free(recs);
            if (out_path_sz) out_path[0] = '\0';
            return -1;
        }
        if (fwrite(recs, sizeof *recs, n, tf) != n) {
            pthread_mutex_lock(&g_stderr_lock);
            fprintf(stderr, "error: short write to %s\n", out_path);
            pthread_mutex_unlock(&g_stderr_lock);
            fclose(tf); free(recs);
            if (out_path_sz) out_path[0] = '\0';
            return -1;
        }
        fclose(tf);
        n_written = (long)n;
    }

    free(recs);

    pthread_mutex_lock(&g_stderr_lock);
    fprintf(stderr,
            "[%" PRIu32 "] positions: %ld  new_pairs: %ld  skipped: %ld\n",
            sample_id, n_positions, n_written, n_skipped);
    pthread_mutex_unlock(&g_stderr_lock);

    return (n_written > 0) ? 0 : 1;
}

/* ════════════════════════════════════════════════════════════
 * Phase 2 — buffered input cursor over one sorted temp file
 * ════════════════════════════════════════════════════════════ */

typedef struct {
    FILE        *fp;
    char        *path;            /* owned; unlinked (if asked) + freed on close */
    kmer_record *buf;
    size_t       buf_len;         /* records currently in buf  */
    size_t       buf_pos;         /* next record to serve      */
    kmer_record  head;            /* current front record (valid when !done) */
    int          done;
    int          unlink_on_close;
} cursor_t;

static void cursor_close(cursor_t *c)
{
    if (c->fp) { fclose(c->fp); c->fp = NULL; }
    if (c->unlink_on_close && c->path) unlink(c->path);
    free(c->path); c->path = NULL;
    free(c->buf);  c->buf  = NULL;
    c->done = 1;
}

/* Pull the next record into head; close + mark done when exhausted. */
static void cursor_advance(cursor_t *c)
{
    if (c->buf_pos >= c->buf_len) {
        c->buf_len = fread(c->buf, sizeof *c->buf, CURSOR_BUF_RECORDS, c->fp);
        c->buf_pos = 0;
        if (c->buf_len == 0) { cursor_close(c); return; }
    }
    c->head = c->buf[c->buf_pos++];
}

static int cursor_open(cursor_t *c, const char *path, int unlink_on_close)
{
    memset(c, 0, sizeof *c);
    c->fp = fopen(path, "rb");
    if (!c->fp) {
        pthread_mutex_lock(&g_stderr_lock);
        fprintf(stderr, "error: cannot open temp file %s\n", path);
        pthread_mutex_unlock(&g_stderr_lock);
        return -1;
    }
    c->path = strdup(path);
    c->buf  = malloc(CURSOR_BUF_RECORDS * sizeof *c->buf);
    if (!c->path || !c->buf) {
        perror("cursor_open alloc");
        cursor_close(c);
        return -1;
    }
    c->unlink_on_close = unlink_on_close;
    cursor_advance(c);   /* prime head; immediately done if file is empty */
    return 0;
}

/* ════════════════════════════════════════════════════════════
 * Phase 2 — k-way merge iterator (min-heap over cursor heads)
 * ════════════════════════════════════════════════════════════ */

typedef struct {
    cursor_t   *cursors;
    size_t      n_cursors;
    int        *heap;        /* heap of cursor indices, ordered by head */
    size_t      heap_size;
    int         have_prev;
    kmer_record prev;        /* last emitted record, for dedup */
} merge_iter;

static int head_cmp(const cursor_t *a, const cursor_t *b)
{
    if (a->head.kmer < b->head.kmer) return -1;
    if (a->head.kmer > b->head.kmer) return  1;
    if (a->head.sample_id < b->head.sample_id) return -1;
    if (a->head.sample_id > b->head.sample_id) return  1;
    return 0;
}

static void heap_sift_up(merge_iter *it, size_t i)
{
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (head_cmp(&it->cursors[it->heap[i]],
                     &it->cursors[it->heap[parent]]) >= 0)
            break;
        int t = it->heap[i]; it->heap[i] = it->heap[parent]; it->heap[parent] = t;
        i = parent;
    }
}

static void heap_sift_down(merge_iter *it, size_t i)
{
    for (;;) {
        size_t l = 2*i + 1, r = 2*i + 2, m = i;
        if (l < it->heap_size &&
            head_cmp(&it->cursors[it->heap[l]], &it->cursors[it->heap[m]]) < 0)
            m = l;
        if (r < it->heap_size &&
            head_cmp(&it->cursors[it->heap[r]], &it->cursors[it->heap[m]]) < 0)
            m = r;
        if (m == i) break;
        int t = it->heap[i]; it->heap[i] = it->heap[m]; it->heap[m] = t;
        i = m;
    }
}

static int merge_iter_init(merge_iter *it, char *const *paths, size_t n,
                           int unlink_inputs)
{
    memset(it, 0, sizeof *it);
    it->cursors = calloc(n ? n : 1, sizeof *it->cursors);
    it->heap    = malloc((n ? n : 1) * sizeof *it->heap);
    if (!it->cursors || !it->heap) { perror("merge_iter_init"); return -1; }
    it->n_cursors = n;

    for (size_t i = 0; i < n; i++) {
        if (cursor_open(&it->cursors[i], paths[i], unlink_inputs) != 0)
            return -1;
        if (!it->cursors[i].done) {
            size_t pos = it->heap_size++;
            it->heap[pos] = (int)i;
            heap_sift_up(it, pos);
        }
    }
    return 0;
}

static void merge_iter_free(merge_iter *it)
{
    for (size_t i = 0; i < it->n_cursors; i++)
        if (!it->cursors[i].done)
            cursor_close(&it->cursors[i]);   /* cleans up on early abort */
    free(it->cursors);
    free(it->heap);
}

/* Emit the next (kmer, sample_id) in global sorted order.
 * Returns 1 and fills *out, or 0 when all inputs are drained. */
static int merge_iter_next(merge_iter *it, kmer_record *out)
{
    for (;;) {
        if (it->heap_size == 0) return 0;

        int         ci  = it->heap[0];
        kmer_record rec = it->cursors[ci].head;

        cursor_advance(&it->cursors[ci]);
        if (it->cursors[ci].done) {
            it->heap[0] = it->heap[--it->heap_size];
            if (it->heap_size > 0) heap_sift_down(it, 0);
        } else {
            heap_sift_down(it, 0);   /* root's head changed; restore order */
        }

        /* Dedup identical (kmer, sample_id). Defensive only: each sample
         * lives in exactly one input file, so duplicates cannot actually
         * arise — but the check keeps the merge correct if that ever
         * changes (e.g. re-merging overlapping shards). */
        if (it->have_prev &&
            rec.kmer == it->prev.kmer &&
            rec.sample_id == it->prev.sample_id)
            continue;

        it->prev = rec;
        it->have_prev = 1;
        *out = rec;
        return 1;
    }
}

/* ════════════════════════════════════════════════════════════
 * Phase 2 — sinks
 * ════════════════════════════════════════════════════════════ */

static void zstd_write(ZSTD_CStream *cs, FILE *fout,
                       void *obuf, size_t obuf_sz,
                       const void *data, size_t data_sz)
{
    ZSTD_inBuffer in = { data, data_sz, 0 };
    while (in.pos < in.size) {
        ZSTD_outBuffer out = { obuf, obuf_sz, 0 };
        size_t rc = ZSTD_compressStream(cs, &out, &in);
        if (ZSTD_isError(rc)) {
            fprintf(stderr, "zstd error: %s\n", ZSTD_getErrorName(rc));
            exit(1);
        }
        fwrite(obuf, 1, out.pos, fout);
    }
}

/* Format one TSV row "ACGT...\t1,5,23,...\n" into *prow (grown as needed).
 * Returns the row length in bytes. Writing is left to the caller so the
 * same formatted row can be fanned out to several outputs. */
static size_t format_tsv_row(char **prow, size_t *prow_cap,
                             int kmer_len, uint64_t kmer,
                             const uint32_t *ids, size_t n_ids)
{
    char decoded[64];
    decode_DNA_2_bit(kmer, kmer_len, decoded);
    decoded[kmer_len] = '\0';   /* be robust if decode does not NUL-terminate */

    size_t needed = (size_t)kmer_len + 2 + n_ids * 12 + 2;
    if (needed > *prow_cap) {
        *prow_cap = needed * 2;
        char *grown = realloc(*prow, *prow_cap);
        if (!grown) { perror("realloc row"); exit(1); }
        *prow = grown;
    }

    char *row = *prow;
    int   pos = snprintf(row, *prow_cap, "%s\t", decoded);
    for (size_t j = 0; j < n_ids; j++) {
        if (j) row[pos++] = ',';
        pos += snprintf(row + pos, *prow_cap - (size_t)pos, "%" PRIu32, ids[j]);
    }
    row[pos++] = '\n';
    return (size_t)pos;
}

/* Keep a kmer in this output iff its count is in [min_count, max_count). */
static inline int out_keep(const mdb_output *o, size_t count)
{
    if (count < o->min_count) return 0;
    if (o->max_count != 0 && count >= o->max_count) return 0;
    return 1;
}

/* Format the bare k-mer "ACGT...\n" into buf; returns its length. */
static size_t format_kmer_row(char *buf, size_t bufsz,
                              int kmer_len, uint64_t kmer)
{
    char decoded[64];
    decode_DNA_2_bit(kmer, kmer_len, decoded);
    decoded[kmer_len] = '\0';
    int pos = snprintf(buf, bufsz, "%s\n", decoded);
    return (size_t)pos;
}

/* Fan one completed kmer out to every passing sink. The id-list row and
 * the bare-kmer row are each formatted at most once, and only if some sink
 * of that schema actually keeps this kmer. */
static void emit_kmer_to_sinks(const mdb_output *outs, size_t n_outs,
                               FILE **fp, ZSTD_CStream **cs, uint64_t *nr,
                               void *obuf, size_t obuf_sz,
                               char **prow, size_t *prow_cap,
                               int kmer_len, uint64_t kmer,
                               const uint32_t *ids, size_t count)
{
    int need_id = 0, need_km = 0;
    for (size_t s = 0; s < n_outs; s++)
        if (out_keep(&outs[s], count)) {
            if (outs[s].kmer_only) need_km = 1; else need_id = 1;
        }
    if (!need_id && !need_km) return;

    size_t idlen = 0, kmlen = 0;
    char   krow[64];
    if (need_id) idlen = format_tsv_row(prow, prow_cap, kmer_len, kmer, ids, count);
    if (need_km) kmlen = format_kmer_row(krow, sizeof krow, kmer_len, kmer);

    for (size_t s = 0; s < n_outs; s++) {
        if (!out_keep(&outs[s], count)) continue;
        if (outs[s].kmer_only)
            zstd_write(cs[s], fp[s], obuf, obuf_sz, krow, kmlen);
        else
            zstd_write(cs[s], fp[s], obuf, obuf_sz, *prow, idlen);
        nr[s]++;
    }
}

/* Final pass: merge inputs and write each configured zstd TSV output.
 * Each distinct kmer's row is formatted once, then written to every
 * output whose prevalence threshold it passes (max_count==0 => keep all,
 * else keep iff id-count < max_count). */
static int merge_to_tsv_multi(char *const *paths, size_t n, int kmer_len,
                              const mdb_output *outs, size_t n_outs)
{
    merge_iter it;
    if (merge_iter_init(&it, paths, n, 1) != 0) { merge_iter_free(&it); return -1; }

    /* per-output runtime state (parallel arrays) */
    FILE         **fp = calloc(n_outs, sizeof *fp);
    ZSTD_CStream **cs = calloc(n_outs, sizeof *cs);
    uint64_t      *nr = calloc(n_outs, sizeof *nr);   /* rows kept per output */
    if (!fp || !cs || !nr) { perror("calloc tsv sinks"); exit(1); }

    size_t    obuf_sz = ZSTD_CStreamOutSize();
    void     *obuf    = malloc(obuf_sz);
    size_t    id_cap  = 256;
    uint32_t *ids     = malloc(id_cap * sizeof *ids);
    size_t    row_cap = 1 << 16;
    char     *row     = malloc(row_cap);
    if (!obuf || !ids || !row) { perror("malloc tsv sink"); exit(1); }

    for (size_t s = 0; s < n_outs; s++) {
        fp[s] = fopen(outs[s].path, "wb");
        if (!fp[s]) {
            pthread_mutex_lock(&g_stderr_lock);
            fprintf(stderr, "error: cannot open output %s: ", outs[s].path);
            perror(NULL);
            pthread_mutex_unlock(&g_stderr_lock);
            exit(1);
        }
        cs[s] = ZSTD_createCStream();
        if (!cs[s]) { fprintf(stderr, "ZSTD_createCStream failed\n"); exit(1); }
        ZSTD_initCStream(cs[s], 3);
        const char *hdr = outs[s].kmer_only ? "#kmer\n"
                                            : "#kmer\tlist_scrub_id\n";
        zstd_write(cs[s], fp[s], obuf, obuf_sz, hdr, strlen(hdr));
    }

    uint64_t cur_kmer = 0;
    int      have_cur = 0;
    size_t   n_ids    = 0;

    kmer_record rec;
    while (merge_iter_next(&it, &rec)) {
        if (have_cur && rec.kmer != cur_kmer) {
            emit_kmer_to_sinks(outs, n_outs, fp, cs, nr, obuf, obuf_sz,
                               &row, &row_cap, kmer_len, cur_kmer, ids, n_ids);
            n_ids = 0;
        }
        cur_kmer = rec.kmer;
        have_cur = 1;

        /* IDs arrive ascending (merge orders by sample_id within a kmer),
         * so the row's list is sorted; skip a duplicate trailing id. */
        if (n_ids == 0 || ids[n_ids-1] != rec.sample_id) {
            if (n_ids == id_cap) {
                id_cap *= 2;
                uint32_t *grown = realloc(ids, id_cap * sizeof *ids);
                if (!grown) { perror("realloc ids"); exit(1); }
                ids = grown;
            }
            ids[n_ids++] = rec.sample_id;
        }
    }
    if (have_cur && n_ids > 0)
        emit_kmer_to_sinks(outs, n_outs, fp, cs, nr, obuf, obuf_sz,
                           &row, &row_cap, kmer_len, cur_kmer, ids, n_ids);

    /* flush + finalize each zstd frame, then report */
    for (size_t s = 0; s < n_outs; s++) {
        for (;;) {
            ZSTD_outBuffer out = { obuf, obuf_sz, 0 };
            size_t remain = ZSTD_endStream(cs[s], &out);
            fwrite(obuf, 1, out.pos, fp[s]);
            if (ZSTD_isError(remain)) {
                fprintf(stderr, "zstd end error: %s\n", ZSTD_getErrorName(remain));
                break;
            }
            if (remain == 0) break;
        }
        ZSTD_freeCStream(cs[s]);
        fclose(fp[s]);

        pthread_mutex_lock(&g_stderr_lock);
        if (outs[s].min_count == 0 && outs[s].max_count == 0)
            fprintf(stderr, "merge: wrote %" PRIu64 " kmers (full) to %s\n",
                    nr[s], outs[s].path);
        else if (outs[s].min_count == 0)
            fprintf(stderr, "merge: wrote %" PRIu64 " kmers (count < %" PRIu32
                    ") to %s\n", nr[s], outs[s].max_count, outs[s].path);
        else
            fprintf(stderr, "merge: wrote %" PRIu64 " kmers (count >= %" PRIu32
                    ") to %s\n", nr[s], outs[s].min_count, outs[s].path);
        pthread_mutex_unlock(&g_stderr_lock);
    }

    free(fp); free(cs); free(nr);
    free(obuf); free(ids); free(row);
    merge_iter_free(&it);
    return 0;
}

/* Intermediate pass: merge inputs into one sorted binary temp file (still
 * (kmer, sample_id) records, just globally ordered and deduped). */
static int merge_to_binary(char *const *paths, size_t n, const char *outpath)
{
    merge_iter it;
    if (merge_iter_init(&it, paths, n, 1) != 0) { merge_iter_free(&it); return -1; }

    FILE *out = fopen(outpath, "wb");
    if (!out) { perror("fopen intermediate"); merge_iter_free(&it); return -1; }

    kmer_record wbuf[CURSOR_BUF_RECORDS];
    size_t      wn = 0;
    kmer_record rec;
    while (merge_iter_next(&it, &rec)) {
        wbuf[wn++] = rec;
        if (wn == CURSOR_BUF_RECORDS) {
            fwrite(wbuf, sizeof *wbuf, wn, out);
            wn = 0;
        }
    }
    if (wn) fwrite(wbuf, sizeof *wbuf, wn, out);

    fclose(out);
    merge_iter_free(&it);
    return 0;
}

/* ════════════════════════════════════════════════════════════
 * Phase 2 driver — multi-pass when inputs exceed the fd budget
 * ════════════════════════════════════════════════════════════ */

int mdb_merge(char *const *temp_paths, size_t n_files,
              int kmer_len, const char *tmpdir, int fanout,
              const mdb_output *outputs, size_t n_outputs)
{
    if (fanout < 2) fanout = 2;

    if (n_files == 0)                       /* nothing to merge: header(s) only */
        return merge_to_tsv_multi(NULL, 0, kmer_len, outputs, n_outputs);

    /* Own a working copy of the path list so passes can freely swap it. */
    size_t  cur_n = n_files;
    char  **cur   = malloc(cur_n * sizeof *cur);
    if (!cur) { perror("malloc cur"); return -1; }
    for (size_t i = 0; i < cur_n; i++) {
        cur[i] = strdup(temp_paths[i]);
        if (!cur[i]) { perror("strdup"); return -1; }
    }

    int pass = 0;

    /* Intermediate passes: collapse `fanout` files at a time into
     * intermediate binary files until few enough remain for a final pass. */
    while (cur_n > (size_t)fanout) {
        size_t  next_cap = (cur_n + (size_t)fanout - 1) / (size_t)fanout;
        char  **next     = malloc(next_cap * sizeof *next);
        if (!next) { perror("malloc next"); return -1; }
        size_t  next_n   = 0;

        pthread_mutex_lock(&g_stderr_lock);
        fprintf(stderr, "merge pass %d: %zu inputs -> %zu groups (fanout %d)\n",
                pass, cur_n, next_cap, fanout);
        pthread_mutex_unlock(&g_stderr_lock);

        for (size_t off = 0; off < cur_n; off += (size_t)fanout) {
            size_t batch = cur_n - off;
            if (batch > (size_t)fanout) batch = (size_t)fanout;

            char inter[4096];
            snprintf(inter, sizeof inter, "%s/mdb_merge_p%d_%zu.tmp.bin",
                     tmpdir, pass, next_n);

            if (merge_to_binary(&cur[off], batch, inter) != 0)
                return -1;

            next[next_n] = strdup(inter);
            if (!next[next_n]) { perror("strdup inter"); return -1; }
            next_n++;
        }

        for (size_t i = 0; i < cur_n; i++) free(cur[i]);
        free(cur);
        cur   = next;
        cur_n = next_n;
        pass++;
    }

    pthread_mutex_lock(&g_stderr_lock);
    fprintf(stderr, "merge pass %d (final): %zu inputs -> %zu output(s)\n",
            pass, cur_n, n_outputs);
    pthread_mutex_unlock(&g_stderr_lock);

    int rc = merge_to_tsv_multi(cur, cur_n, kmer_len, outputs, n_outputs);

    for (size_t i = 0; i < cur_n; i++) free(cur[i]);
    free(cur);
    return rc;
}

/* ════════════════════════════════════════════════════════════
 * manifest
 * ════════════════════════════════════════════════════════════ */

void mdb_write_manifest_line(FILE *manifest_fp, uint32_t sample_id,
                             const char *filepath)
{
    fprintf(manifest_fp, "%" PRIu32 "\t%s\n", sample_id, filepath);
}
