#define _GNU_SOURCE
#include "kmdb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <zstd.h>

/* ── 2-bit tables ─────────────────────────────────────────────────── */

#define X4 4,4,4,4
#define X16 X4,X4,X4,X4
const uint8_t KMDB_NT2BIT[256] = {
	X16, X16, X16, X16,                                   /*   0.. 63 */
	4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4,                      /*  64.. 79 @ABCDEFGHIJKLMNO */
	4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,                      /*  80.. 95 PQRSTUVWXYZ      */
	4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4,                      /*  96..111 `abcdefghijklmno */
	4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,                      /* 112..127 pqrstuvwxyz      */
	X16, X16, X16, X16, X16, X16, X16, X16                /* 128..255 */
};
#undef X16
#undef X4

static const char TWOBIT2NT[4] = { 'A', 'C', 'G', 'T' };

int kmdb_encode(const char *s, int k, uint64_t *out)
{
	uint64_t x = 0;
	for (int i = 0; i < k; i++) {
		uint8_t c = KMDB_NT2BIT[(unsigned char)s[i]];
		if (c > 3) return -1;
		x = (x << 2) | c;
	}
	*out = x;
	return 0;
}

void kmdb_decode(uint64_t x, int k, char *out)
{
	for (int i = k - 1; i >= 0; i--) {
		out[i] = TWOBIT2NT[x & 3];
		x >>= 2;
	}
	out[k] = '\0';
}

int kmdb_is_kmdb_path(const char *path)
{
	size_t n = strlen(path);
	return n > 5 && strcmp(path + n - 5, ".kmdb") == 0;
}

/* ── little-endian + varint helpers ───────────────────────────────── */

static void put_u32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put_u64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint32_t get_u32(const uint8_t *p) { uint32_t v = 0; for (int i = 3; i >= 0; i--) v = (v << 8) | p[i]; return v; }
static uint64_t get_u64(const uint8_t *p) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = (v << 8) | p[i]; return v; }

static inline size_t put_varint(uint8_t *p, uint64_t v)
{
	size_t n = 0;
	while (v >= 0x80) { p[n++] = (uint8_t)(v | 0x80); v >>= 7; }
	p[n++] = (uint8_t)v;
	return n;
}

/* Returns bytes consumed, or 0 if the buffer ends mid-varint / overlong. */
static inline size_t get_varint(const uint8_t *p, const uint8_t *end, uint64_t *v)
{
	uint64_t x = 0;
	for (size_t i = 0; i < 10 && p + i < end; i++) {
		x |= (uint64_t)(p[i] & 0x7f) << (7 * i);
		if (!(p[i] & 0x80)) { *v = x; return i + 1; }
	}
	return 0;
}

static int write_all(int fd, const void *buf, size_t len, off_t off)
{
	const uint8_t *p = buf;
	while (len > 0) {
		ssize_t w = pwrite(fd, p, len, off);
		if (w < 0) { if (errno == EINTR) continue; return -1; }
		p += w; len -= (size_t)w; off += w;
	}
	return 0;
}

static int read_all(int fd, void *buf, size_t len, off_t off)
{
	uint8_t *p = buf;
	while (len > 0) {
		ssize_t r = pread(fd, p, len, off);
		if (r < 0) { if (errno == EINTR) continue; return -1; }
		if (r == 0) { errno = EIO; return -1; }  /* truncated file */
		p += r; len -= (size_t)r; off += r;
	}
	return 0;
}

/* ── writer ───────────────────────────────────────────────────────── */

struct kmdb_writer_s {
	int              fd;
	char            *path;
	int              k;
	uint32_t         block_n;
	ZSTD_CCtx       *cctx;

	uint64_t        *kbuf;
	uint32_t        *cbuf;
	uint32_t         nbuf;

	uint8_t         *raw;
	size_t           raw_cap;
	uint8_t         *comp;
	size_t           comp_cap;

	uint64_t         offset;     /* next write position */
	uint64_t         n_kmers;
	int              have_last;
	uint64_t         last_kmer;

	kmdb_block_info *blocks;
	uint64_t         n_blocks, blocks_cap;

	char            *meta;
	size_t           meta_len, meta_cap;
	int              failed;
};

kmdb_writer *kmdb_writer_open(const char *path, int k, uint32_t block_n,
                              int zstd_level)
{
	if (k < 1 || k > KMDB_MAX_K) {
		fprintf(stderr, "kmdb_writer_open: k=%d out of range 1..%d\n",
		        k, KMDB_MAX_K);
		return NULL;
	}
	if (block_n == 0) block_n = KMDB_DEFAULT_BLOCK_N;
	if (zstd_level == 0) zstd_level = 3;

	kmdb_writer *w = calloc(1, sizeof(*w));
	if (!w) return NULL;
	w->fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
	if (w->fd < 0) {
		fprintf(stderr, "kmdb_writer_open: cannot open %s: %s\n",
		        path, strerror(errno));
		free(w);
		return NULL;
	}
	w->path    = strdup(path);
	w->k       = k;
	w->block_n = block_n;
	w->kbuf    = malloc(sizeof(uint64_t) * block_n);
	w->cbuf    = malloc(sizeof(uint32_t) * block_n);
	w->raw_cap = (size_t)block_n * 15;            /* <=10 + <=5 bytes per entry */
	w->raw     = malloc(w->raw_cap);
	w->comp_cap = ZSTD_compressBound(w->raw_cap);
	w->comp    = malloc(w->comp_cap);
	w->cctx    = ZSTD_createCCtx();
	if (!w->path || !w->kbuf || !w->cbuf || !w->raw || !w->comp || !w->cctx) {
		fprintf(stderr, "kmdb_writer_open: out of memory\n");
		exit(EXIT_FAILURE);
	}
	ZSTD_CCtx_setParameter(w->cctx, ZSTD_c_compressionLevel, zstd_level);
	ZSTD_CCtx_setParameter(w->cctx, ZSTD_c_checksumFlag, 1);
	w->offset = KMDB_HEADER_SIZE;

	/* placeholder header; rewritten at close */
	uint8_t zero[KMDB_HEADER_SIZE] = {0};
	if (write_all(w->fd, zero, sizeof zero, 0) != 0) w->failed = 1;
	return w;
}

static int flush_block(kmdb_writer *w)
{
	if (w->nbuf == 0) return 0;
	size_t n = 0;
	uint64_t prev = w->kbuf[0];
	for (uint32_t i = 0; i < w->nbuf; i++) {
		n += put_varint(w->raw + n, w->kbuf[i] - prev);
		prev = w->kbuf[i];
	}
	for (uint32_t i = 0; i < w->nbuf; i++)
		n += put_varint(w->raw + n, w->cbuf[i]);

	size_t c = ZSTD_compress2(w->cctx, w->comp, w->comp_cap, w->raw, n);
	if (ZSTD_isError(c)) {
		fprintf(stderr, "kmdb: zstd compression failed: %s\n",
		        ZSTD_getErrorName(c));
		return -1;
	}
	if (write_all(w->fd, w->comp, c, (off_t)w->offset) != 0) {
		fprintf(stderr, "kmdb: write to %s failed: %s\n", w->path,
		        strerror(errno));
		return -1;
	}

	if (w->n_blocks == w->blocks_cap) {
		w->blocks_cap = w->blocks_cap ? w->blocks_cap * 2 : 64;
		w->blocks = realloc(w->blocks, w->blocks_cap * sizeof(*w->blocks));
		if (!w->blocks) { perror("realloc kmdb index"); exit(EXIT_FAILURE); }
	}
	kmdb_block_info *b = &w->blocks[w->n_blocks++];
	b->first_kmer = w->kbuf[0];
	b->last_kmer  = w->kbuf[w->nbuf - 1];
	b->offset     = w->offset;
	b->comp_size  = (uint32_t)c;
	b->n          = w->nbuf;

	w->offset += c;
	w->nbuf = 0;
	return 0;
}

int kmdb_writer_add(kmdb_writer *w, uint64_t kmer, uint32_t count)
{
	if (w->failed) return -1;
	if (w->have_last && kmer <= w->last_kmer) {
		fprintf(stderr, "kmdb_writer_add: keys not strictly increasing\n");
		w->failed = 1;
		return -1;
	}
	w->have_last = 1;
	w->last_kmer = kmer;
	w->kbuf[w->nbuf] = kmer;
	w->cbuf[w->nbuf] = count;
	w->nbuf++;
	w->n_kmers++;
	if (w->nbuf == w->block_n && flush_block(w) != 0) {
		w->failed = 1;
		return -1;
	}
	return 0;
}

void kmdb_writer_meta(kmdb_writer *w, const char *key, const char *value)
{
	size_t need = strlen(key) + strlen(value) + 3;
	if (w->meta_len + need + 1 > w->meta_cap) {
		w->meta_cap = (w->meta_len + need + 1) * 2;
		w->meta = realloc(w->meta, w->meta_cap);
		if (!w->meta) { perror("realloc kmdb meta"); exit(EXIT_FAILURE); }
	}
	/* newlines inside values would break the line format */
	size_t start = w->meta_len;
	w->meta_len += (size_t)snprintf(w->meta + w->meta_len,
	                                w->meta_cap - w->meta_len,
	                                "%s=%s\n", key, value);
	for (size_t i = start; i + 1 < w->meta_len; i++)
		if (w->meta[i] == '\n' || w->meta[i] == '\r') w->meta[i] = ' ';
}

void kmdb_writer_meta_u64(kmdb_writer *w, const char *key, uint64_t value)
{
	char buf[32];
	snprintf(buf, sizeof buf, "%llu", (unsigned long long)value);
	kmdb_writer_meta(w, key, buf);
}

int kmdb_writer_close(kmdb_writer *w)
{
	int rc = w->failed ? -1 : 0;
	if (rc == 0 && flush_block(w) != 0) rc = -1;

	uint64_t meta_offset = w->offset;
	if (rc == 0 && w->meta_len > 0) {
		if (write_all(w->fd, w->meta, w->meta_len, (off_t)w->offset) != 0) rc = -1;
		w->offset += w->meta_len;
	}

	uint64_t index_offset = w->offset;
	if (rc == 0 && w->n_blocks > 0) {
		size_t isz = (size_t)w->n_blocks * KMDB_INDEX_ENTRY_SIZE;
		uint8_t *ib = malloc(isz);
		if (!ib) { perror("malloc kmdb index"); exit(EXIT_FAILURE); }
		for (uint64_t i = 0; i < w->n_blocks; i++) {
			uint8_t *p = ib + i * KMDB_INDEX_ENTRY_SIZE;
			put_u64(p,      w->blocks[i].first_kmer);
			put_u64(p + 8,  w->blocks[i].last_kmer);
			put_u64(p + 16, w->blocks[i].offset);
			put_u32(p + 24, w->blocks[i].comp_size);
			put_u32(p + 28, w->blocks[i].n);
		}
		if (write_all(w->fd, ib, isz, (off_t)w->offset) != 0) rc = -1;
		w->offset += isz;
		free(ib);
	}

	if (rc == 0) {
		uint8_t h[KMDB_HEADER_SIZE] = {0};
		memcpy(h, KMDB_MAGIC, 4);
		put_u32(h + 4,  KMDB_VERSION);
		put_u32(h + 8,  (uint32_t)w->k);
		put_u32(h + 12, w->block_n);
		put_u64(h + 16, w->n_kmers);
		put_u64(h + 24, w->n_blocks);
		put_u64(h + 32, index_offset);
		put_u64(h + 40, meta_offset);
		put_u64(h + 48, (uint64_t)w->meta_len);
		put_u64(h + 56, KMDB_FLAG_CANON_MAX);
		if (write_all(w->fd, h, sizeof h, 0) != 0) rc = -1;
	}
	if (rc != 0)
		fprintf(stderr, "kmdb_writer_close: failed writing %s: %s\n",
		        w->path, strerror(errno));
	if (close(w->fd) != 0) rc = -1;

	ZSTD_freeCCtx(w->cctx);
	free(w->kbuf); free(w->cbuf); free(w->raw); free(w->comp);
	free(w->blocks); free(w->meta); free(w->path);
	free(w);
	return rc;
}

/* ── reader ───────────────────────────────────────────────────────── */

kmdb_t *kmdb_open(const char *path)
{
	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		fprintf(stderr, "kmdb_open: cannot open %s: %s\n", path, strerror(errno));
		return NULL;
	}
	struct stat st;
	uint8_t h[KMDB_HEADER_SIZE];
	if (fstat(fd, &st) != 0 || (uint64_t)st.st_size < KMDB_HEADER_SIZE ||
	    read_all(fd, h, sizeof h, 0) != 0 || memcmp(h, KMDB_MAGIC, 4) != 0) {
		fprintf(stderr, "kmdb_open: %s is not a kmdb file (or is truncated)\n",
		        path);
		close(fd);
		return NULL;
	}
	uint32_t version = get_u32(h + 4);
	if (version != KMDB_VERSION) {
		fprintf(stderr, "kmdb_open: %s has version %u, this build reads %u\n",
		        path, version, KMDB_VERSION);
		close(fd);
		return NULL;
	}

	kmdb_t *db = calloc(1, sizeof(*db));
	db->fd       = fd;
	db->path     = strdup(path);
	db->k        = get_u32(h + 8);
	db->block_n  = get_u32(h + 12);
	db->n_kmers  = get_u64(h + 16);
	db->n_blocks = get_u64(h + 24);
	uint64_t index_offset = get_u64(h + 32);
	uint64_t meta_offset  = get_u64(h + 40);
	uint64_t meta_len     = get_u64(h + 48);
	db->flags    = get_u64(h + 56);

	uint64_t fsize = (uint64_t)st.st_size;
	if (db->k < 1 || db->k > KMDB_MAX_K ||
	    meta_offset + meta_len > fsize ||
	    index_offset + db->n_blocks * KMDB_INDEX_ENTRY_SIZE > fsize) {
		fprintf(stderr, "kmdb_open: %s has an inconsistent header "
		        "(was the build interrupted?)\n", path);
		kmdb_close(db);
		return NULL;
	}

	db->meta = malloc(meta_len + 1);
	if (meta_len && read_all(fd, db->meta, meta_len, (off_t)meta_offset) != 0) {
		fprintf(stderr, "kmdb_open: cannot read metadata of %s\n", path);
		kmdb_close(db);
		return NULL;
	}
	db->meta[meta_len] = '\0';

	if (db->n_blocks > 0) {
		size_t isz = (size_t)db->n_blocks * KMDB_INDEX_ENTRY_SIZE;
		uint8_t *ib = malloc(isz);
		db->blocks = malloc((size_t)db->n_blocks * sizeof(*db->blocks));
		if (!ib || !db->blocks ||
		    read_all(fd, ib, isz, (off_t)index_offset) != 0) {
			fprintf(stderr, "kmdb_open: cannot read index of %s\n", path);
			free(ib);
			kmdb_close(db);
			return NULL;
		}
		uint64_t total = 0;
		for (uint64_t i = 0; i < db->n_blocks; i++) {
			const uint8_t *p = ib + i * KMDB_INDEX_ENTRY_SIZE;
			kmdb_block_info *b = &db->blocks[i];
			b->first_kmer = get_u64(p);
			b->last_kmer  = get_u64(p + 8);
			b->offset     = get_u64(p + 16);
			b->comp_size  = get_u32(p + 24);
			b->n          = get_u32(p + 28);
			total += b->n;
			if (b->comp_size > db->max_comp_size) db->max_comp_size = b->comp_size;
			if (b->n > db->max_n) db->max_n = b->n;
			if (b->offset + b->comp_size > fsize) {
				fprintf(stderr, "kmdb_open: %s block %llu points past EOF\n",
				        path, (unsigned long long)i);
				free(ib);
				kmdb_close(db);
				return NULL;
			}
		}
		free(ib);
		if (total != db->n_kmers) {
			fprintf(stderr, "kmdb_open: %s index sums to %llu k-mers, "
			        "header says %llu\n", path, (unsigned long long)total,
			        (unsigned long long)db->n_kmers);
			kmdb_close(db);
			return NULL;
		}
	}
	return db;
}

void kmdb_close(kmdb_t *db)
{
	if (!db) return;
	if (db->fd >= 0) close(db->fd);
	free(db->path);
	free(db->blocks);
	free(db->meta);
	free(db);
}

char *kmdb_meta_get(const kmdb_t *db, const char *key)
{
	if (!db->meta) return NULL;
	size_t klen = strlen(key);
	const char *found = NULL;
	const char *line = db->meta;
	while (*line) {
		const char *nl = strchr(line, '\n');
		size_t len = nl ? (size_t)(nl - line) : strlen(line);
		if (len > klen && strncmp(line, key, klen) == 0 && line[klen] == '=')
			found = line;
		if (!nl) break;
		line = nl + 1;
	}
	if (!found) return NULL;
	const char *v = found + klen + 1;
	const char *nl = strchr(v, '\n');
	size_t vlen = nl ? (size_t)(nl - v) : strlen(v);
	char *out = malloc(vlen + 1);
	memcpy(out, v, vlen);
	out[vlen] = '\0';
	return out;
}

void kmdb_block_buf_init(kmdb_block_buf *b)
{
	memset(b, 0, sizeof(*b));
	b->dctx = ZSTD_createDCtx();
	if (!b->dctx) { fprintf(stderr, "ZSTD_createDCtx failed\n"); exit(EXIT_FAILURE); }
}

void kmdb_block_buf_free(kmdb_block_buf *b)
{
	ZSTD_freeDCtx((ZSTD_DCtx *)b->dctx);
	free(b->comp); free(b->raw); free(b->kmers); free(b->counts);
	memset(b, 0, sizeof(*b));
}

static void grow(void **p, size_t *cap, size_t need, size_t elem)
{
	if (*cap >= need) return;
	void *q = realloc(*p, need * elem);
	if (!q) { perror("realloc kmdb buffer"); exit(EXIT_FAILURE); }
	*p = q;
	*cap = need;
}

long kmdb_read_block(const kmdb_t *db, uint64_t i, kmdb_block_buf *b)
{
	if (i >= db->n_blocks) return -1;
	const kmdb_block_info *bi = &db->blocks[i];

	grow((void **)&b->comp, &b->comp_cap, bi->comp_size, 1);
	if (read_all(db->fd, b->comp, bi->comp_size, (off_t)bi->offset) != 0) {
		fprintf(stderr, "kmdb: read of block %llu in %s failed: %s\n",
		        (unsigned long long)i, db->path, strerror(errno));
		return -1;
	}
	unsigned long long rsz = ZSTD_getFrameContentSize(b->comp, bi->comp_size);
	if (rsz == ZSTD_CONTENTSIZE_ERROR || rsz == ZSTD_CONTENTSIZE_UNKNOWN ||
	    rsz > (unsigned long long)bi->n * 15) {
		fprintf(stderr, "kmdb: block %llu in %s is corrupt\n",
		        (unsigned long long)i, db->path);
		return -1;
	}
	grow((void **)&b->raw, &b->raw_cap, rsz ? rsz : 1, 1);
	size_t got = ZSTD_decompressDCtx((ZSTD_DCtx *)b->dctx, b->raw, b->raw_cap,
	                                 b->comp, bi->comp_size);
	if (ZSTD_isError(got) || got != rsz) {
		fprintf(stderr, "kmdb: block %llu in %s failed to decompress: %s\n",
		        (unsigned long long)i, db->path,
		        ZSTD_isError(got) ? ZSTD_getErrorName(got) : "size mismatch");
		return -1;
	}

	if (b->n_cap < bi->n) {
		size_t cap = bi->n;
		b->kmers  = realloc(b->kmers,  cap * sizeof(uint64_t));
		b->counts = realloc(b->counts, cap * sizeof(uint32_t));
		if (!b->kmers || !b->counts) { perror("realloc kmdb decode"); exit(EXIT_FAILURE); }
		b->n_cap = cap;
	}

	const uint8_t *p = b->raw, *end = b->raw + got;
	uint64_t key = bi->first_kmer, v;
	for (uint32_t j = 0; j < bi->n; j++) {
		size_t c = get_varint(p, end, &v);
		if (!c) goto corrupt;
		p += c;
		key += v;
		b->kmers[j] = key;
	}
	for (uint32_t j = 0; j < bi->n; j++) {
		size_t c = get_varint(p, end, &v);
		if (!c || v > UINT32_MAX) goto corrupt;
		p += c;
		b->counts[j] = (uint32_t)v;
	}
	if (p != end || (bi->n && b->kmers[bi->n - 1] != bi->last_kmer)) goto corrupt;
	return (long)bi->n;

corrupt:
	fprintf(stderr, "kmdb: block %llu in %s has an invalid payload\n",
	        (unsigned long long)i, db->path);
	return -1;
}
