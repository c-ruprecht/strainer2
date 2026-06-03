#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <sys/resource.h>
#include "metagenome_db.h"

#define DEFAULT_KMER_LEN   31
#define DEFAULT_THREADS     8
#define DEFAULT_TMPDIR    "/tmp"
#define PATH_BUF         4096
#define MAX_FANOUT       1024   /* cap on temp files open at once in the merge;
                                 * keeps merge RAM bounded (~fanout * 48 KiB)
                                 * and stays well under any sane ulimit -n   */

static void usage(void)
{
    fprintf(stderr,
        "Usage: metagenome_db_build\n"
        "         -f <file listing FASTQ/FASTA paths, one per line>\n"
        "         -o <output BASENAME — suffixes are appended, see below>\n"
        "        [-p <prevalence_keep> | --prevalence-keep <x>]\n"
        "        [-b | --keep-both]\n"
        "        [-c | --common-list]\n"
        "        [-t <threads, default 8>]\n"
        "        [-T <tmpdir for per-sample sort files, default /tmp>]\n"
        "        [-k <kmer length, default 31>]\n"
        "\n"
        "  Two-phase build:\n"
        "    1. parallel workers extract+sort each sample to a binary\n"
        "       temp file <tmpdir>/<sample_id>.tmp.bin\n"
        "    2. a k-way merge streams them into zstd-compressed TSV output(s).\n"
        "\n"
        "  Prevalence cleanup (-p x): keep only k-mers found in FEWER than x\n"
        "  samples (id-list count < x), dropping high-prevalence k-mers to\n"
        "  shrink the DB. With -b/--keep-both, the full index is written too.\n"
        "  With -c/--common-list, the dropped high-prevalence k-mers (count\n"
        "  >= x) are written as a bare #kmer list (one k-mer per line), so a\n"
        "  later merge across scrubs can recognize them instead of mistaking\n"
        "  them for rare.\n"
        "\n"
        "  Output files (BASENAME from -o; a trailing .tsv[.zst] is ignored):\n"
        "    <BASENAME>.full.tsv.zst    full index   (no -p, or --keep-both)\n"
        "    <BASENAME>.prev<x>.tsv.zst  rare index   (with -p x)\n"
        "    <BASENAME>.common<x>.tsv.zst common kmers (with --common-list)\n"
        "    <BASENAME>.manifest.tsv    sample_id<TAB>filepath\n"
        "\n"
        "  TSV rows:  #kmer<TAB>list_scrub_id  /  ACGT...<TAB>1,5,23,...\n"
        "             common list: #kmer  /  ACGT...\n");
    exit(1);
}

/* Strip a known suffix from str in-place. Returns 1 if stripped. */
static int strip_suffix(char *str, const char *suffix)
{
    size_t slen = strlen(str);
    size_t xlen = strlen(suffix);
    if (slen >= xlen && strcmp(str + slen - xlen, suffix) == 0) {
        str[slen - xlen] = '\0';
        return 1;
    }
    return 0;
}

/* ── work queue ──────────────────────────────────────────────
 * Read-only `paths` array; workers atomically claim the next index.
 * Each worker writes only its own slot of `temp_paths`, so phase 1
 * has no shared mutable state beyond the claim counter. */
typedef struct {
    char          **paths;
    uint32_t        n;
    uint32_t        next;
    pthread_mutex_t lock;
    int             kmer_len;
    const char     *tmpdir;
    char          **temp_paths;   /* out: temp file path per sample, or NULL */
} work_queue;

static void *worker_main(void *arg)
{
    work_queue *q = arg;
    for (;;) {
        uint32_t i;
        pthread_mutex_lock(&q->lock);
        i = q->next++;
        pthread_mutex_unlock(&q->lock);
        if (i >= q->n) break;

        char     tp[PATH_BUF];
        uint32_t sample_id = i + 1;     /* 1-based, matches manifest */
        int rc = mdb_sort_sample(q->paths[i], sample_id, q->kmer_len,
                                 q->tmpdir, tp, sizeof tp);
        q->temp_paths[i] = (rc == 0) ? strdup(tp) : NULL;
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    char    *f_file  = NULL;
    char    *o_file  = NULL;
    char    *tmpdir  = NULL;
    int      kmer_len = DEFAULT_KMER_LEN;
    int      threads  = DEFAULT_THREADS;
    long     prev_x   = 0;     /* prevalence_keep threshold (0 = unset)    */
    int      prev_set = 0;
    int      keep_both = 0;
    int      common_set = 0;   /* also emit the common (count>=x) list      */
    int      c;

    static const struct option long_opts[] = {
        { "prevalence-keep", required_argument, 0, 'p' },
        { "keep-both",       no_argument,       0, 'b' },
        { "common-list",     no_argument,       0, 'c' },
        { "threads",         required_argument, 0, 't' },
        { "tmpdir",          required_argument, 0, 'T' },
        { "kmer",            required_argument, 0, 'k' },
        { "help",            no_argument,       0, 'h' },
        { 0, 0, 0, 0 }
    };

    while ((c = getopt_long(argc, argv, "f:o:k:t:T:p:bch",
                            long_opts, NULL)) != -1)
        switch (c) {
            case 'f': f_file    = strdup(optarg); break;
            case 'o': o_file    = strdup(optarg); break;
            case 'k': kmer_len  = atoi(optarg);   break;
            case 't': threads   = atoi(optarg);   break;
            case 'T': tmpdir    = strdup(optarg); break;
            case 'p': prev_x    = atol(optarg); prev_set = 1; break;
            case 'b': keep_both = 1;              break;
            case 'c': common_set = 1;             break;
            case 'h':
            default:  usage();
        }

    if (!f_file || !o_file) usage();
    if (!tmpdir) tmpdir = strdup(DEFAULT_TMPDIR);

    if (kmer_len < 1 || kmer_len > 32) {
        fprintf(stderr, "error: kmer length must be 1-32 (got %d)\n", kmer_len);
        exit(1);
    }
    if (threads < 1) threads = 1;

    if (prev_set && (prev_x < 1 || prev_x > 0xFFFFFFFFL)) {
        fprintf(stderr, "error: --prevalence-keep must be >= 1 (got %ld)\n", prev_x);
        exit(1);
    }
    if (keep_both && !prev_set) {
        fprintf(stderr, "error: --keep-both requires --prevalence-keep <x>\n");
        exit(1);
    }
    if (common_set && !prev_set) {
        fprintf(stderr, "error: --common-list requires --prevalence-keep <x>\n");
        exit(1);
    }
    /* Which outputs to write:
     *   no -p             -> full only
     *   -p x              -> rare/prev index (count < x)
     *   -p x --keep-both  -> also the full index
     *   -p x --common-list-> also the common list (count >= x, count-only) */
    int do_prev   = prev_set;
    int do_full   = (!prev_set) || keep_both;
    int do_common = common_set;

    /* Raise the open-file soft limit to the hard limit so the merge can
     * keep more temp files open per pass; derive a safe fanout from it. */
    int fanout = MAX_FANOUT;
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        rl.rlim_cur = rl.rlim_max;
        setrlimit(RLIMIT_NOFILE, &rl);          /* best effort */
        getrlimit(RLIMIT_NOFILE, &rl);
        if (rl.rlim_cur != RLIM_INFINITY) {
            long avail = (long)rl.rlim_cur - 16; /* headroom for stdio etc. */
            if (avail < 2) avail = 2;
            if (avail < fanout) fanout = (int)avail;
        }
    }

    /* ── read the file list (skip blanks) ───────────────────── */
    FILE *flist = fopen(f_file, "r");
    if (!flist) {
        fprintf(stderr, "error: cannot open file list %s\n", f_file);
        exit(1);
    }

    size_t  cap   = 1024;
    size_t  n     = 0;
    char  **paths = malloc(cap * sizeof *paths);
    if (!paths) { perror("malloc paths"); exit(1); }

    char path[PATH_BUF];
    while (fgets(path, sizeof path, flist) != NULL) {
        int len = (int)strlen(path);
        while (len > 0 && (path[len-1] == '\n' || path[len-1] == '\r'))
            path[--len] = '\0';
        if (len == 0) continue;

        if (n == cap) {
            cap *= 2;
            char **grown = realloc(paths, cap * sizeof *paths);
            if (!grown) { perror("realloc paths"); exit(1); }
            paths = grown;
        }
        paths[n] = strdup(path);
        if (!paths[n]) { perror("strdup path"); exit(1); }
        n++;
    }
    fclose(flist);

    if (n == 0) {
        fprintf(stderr, "error: no input files listed in %s\n", f_file);
        exit(1);
    }
    if (n > 0xFFFFFFFFu) {
        fprintf(stderr, "error: too many samples (%zu); id space is uint32\n", n);
        exit(1);
    }

    /* ── derive output paths from the -o basename ─────────────
     * -o is a basename; a trailing .tsv[.zst] is ignored so both
     * "out" and "out.tsv.zst" yield the same set of files. */
    char base[PATH_BUF];
    strncpy(base, o_file, sizeof base - 1);
    base[sizeof base - 1] = '\0';
    if (!strip_suffix(base, ".tsv.zst"))
        if (!strip_suffix(base, ".zst"))
            strip_suffix(base, ".tsv");

    char manifest_path[PATH_BUF + 32];
    char full_path[PATH_BUF + 32];
    char prev_path[PATH_BUF + 32];
    char common_path[PATH_BUF + 32];
    snprintf(manifest_path, sizeof manifest_path, "%s.manifest.tsv", base);
    snprintf(full_path, sizeof full_path, "%s.full.tsv.zst", base);
    snprintf(prev_path, sizeof prev_path, "%s.prev%ld.tsv.zst", base, prev_x);
    snprintf(common_path, sizeof common_path, "%s.common%ld.tsv.zst", base, prev_x);

    /* ── validate the output path(s) up front ────────────────
     * Phase 1 is expensive, and the merge cleans up its temp files even
     * on failure — so a bad -o discovered only at merge time wastes the
     * entire extract+sort. Probe each output now: catches a directory
     * basename, a missing parent dir, or no write permission in ms. */
    const char *probe_paths[3];
    int n_probe = 0;
    if (do_full)   probe_paths[n_probe++] = full_path;
    if (do_prev)   probe_paths[n_probe++] = prev_path;
    if (do_common) probe_paths[n_probe++] = common_path;
    for (int i = 0; i < n_probe; i++) {
        FILE *probe = fopen(probe_paths[i], "wb");
        if (!probe) {
            fprintf(stderr, "error: cannot write output '%s': %s\n",
                    probe_paths[i], strerror(errno));
            fprintf(stderr, "       -o must be a writable file basename "
                            "(e.g. .../index), not a directory\n");
            exit(1);
        }
        fclose(probe);
    }

    /* ── manifest (written up front, in file-list order) ──────
     * sample_id is assigned here (index + 1), so writing the manifest
     * now — rather than from workers that finish out of order — yields a
     * complete, sample_id-sorted, durable mapping regardless of phase-1
     * scheduling, and avoids sharing a FILE* across threads. */
    FILE *mf = fopen(manifest_path, "w");
    if (!mf) {
        fprintf(stderr, "error: cannot open manifest %s\n", manifest_path);
        exit(1);
    }
    fprintf(mf, "sample_id\tfilepath\n");
    for (size_t i = 0; i < n; i++)
        mdb_write_manifest_line(mf, (uint32_t)(i + 1), paths[i]);
    fflush(mf);
    fclose(mf);

    /* ── banner ──────────────────────────────────────────────── */
    if ((size_t)threads > n) threads = (int)n;

    fprintf(stderr, "Building metagenome kmer DB (k=%d)\n", kmer_len);
    fprintf(stderr, "  file list : %s (%zu samples)\n", f_file, n);
    if (do_full) fprintf(stderr, "  full out  : %s\n", full_path);
    if (do_prev) fprintf(stderr, "  prev out  : %s  (keep count < %ld)\n",
                         prev_path, prev_x);
    if (do_common) fprintf(stderr, "  common out: %s  (count >= %ld, kmers only)\n",
                         common_path, prev_x);
    fprintf(stderr, "  manifest  : %s\n", manifest_path);
    fprintf(stderr, "  tmpdir    : %s\n", tmpdir);
    fprintf(stderr, "  threads   : %d\n", threads);
    fprintf(stderr, "  fanout    : %d temp files / merge pass\n\n", fanout);

    /* ── phase 1: parallel extract + sort ────────────────────── */
    char **temp_paths = calloc(n, sizeof *temp_paths);
    if (!temp_paths) { perror("calloc temp_paths"); exit(1); }

    work_queue q;
    q.paths      = paths;
    q.n          = (uint32_t)n;
    q.next       = 0;
    q.kmer_len   = kmer_len;
    q.tmpdir     = tmpdir;
    q.temp_paths = temp_paths;
    pthread_mutex_init(&q.lock, NULL);

    pthread_t *tids = malloc((size_t)threads * sizeof *tids);
    if (!tids) { perror("malloc tids"); exit(1); }
    for (int i = 0; i < threads; i++)
        pthread_create(&tids[i], NULL, worker_main, &q);
    for (int i = 0; i < threads; i++)
        pthread_join(tids[i], NULL);
    pthread_mutex_destroy(&q.lock);
    free(tids);

    /* ── phase 2: collect temp files and merge ───────────────── */
    char **merge_inputs = malloc(n * sizeof *merge_inputs);
    if (!merge_inputs) { perror("malloc merge_inputs"); exit(1); }
    size_t n_inputs = 0;
    for (size_t i = 0; i < n; i++)
        if (temp_paths[i]) merge_inputs[n_inputs++] = temp_paths[i];

    /* Configure the merge outputs:
     *   full   : count range [0, inf), id-list schema
     *   prev   : count range [0, x),   id-list schema  (the rare/kept set)
     *   common : count range [x, inf), bare-kmer list  (the dropped set) */
    mdb_output outputs[3];
    size_t n_outputs = 0;
    if (do_full)   outputs[n_outputs++] = (mdb_output){ full_path,   0, 0, 0 };
    if (do_prev)   outputs[n_outputs++] = (mdb_output){ prev_path,   0,
                                                        (uint32_t)prev_x, 0 };
    if (do_common) outputs[n_outputs++] = (mdb_output){ common_path,
                                                        (uint32_t)prev_x, 0, 1 };

    fprintf(stderr,
            "\nPhase 1 done: %zu/%zu samples produced kmers.\n", n_inputs, n);
    fprintf(stderr, "Phase 2: merging -> %zu output(s)\n", n_outputs);

    int rc = mdb_merge(merge_inputs, n_inputs, kmer_len, tmpdir, fanout,
                       outputs, n_outputs);

    /* ── cleanup ─────────────────────────────────────────────── */
    for (size_t i = 0; i < n; i++) free(paths[i]);
    free(paths);
    for (size_t i = 0; i < n; i++) free(temp_paths[i]);   /* strdup'd by workers */
    free(temp_paths);
    free(merge_inputs);
    free(f_file);
    free(o_file);
    free(tmpdir);

    fprintf(stderr, "%s\n", rc == 0 ? "Done." : "Failed.");
    return rc == 0 ? 0 : 1;
}
