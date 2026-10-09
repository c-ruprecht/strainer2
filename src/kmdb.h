#ifndef KMDB_H
#define KMDB_H

/*
	kmdb — sorted k-mer database file, one per metagenome.

	Holds canonical k-mers (k <= 31) packed 2 bits per base
	(A=0 C=1 G=2 T=3, first base in the most significant position) with a
	uint32 count each, sorted ascending by the packed value. Canonical means
	max(forward, reverse complement) as packed integers, which is the same
	choice orient_string()/rc_strcmp() make on ASCII strings, so a decoded
	k-mer is byte-identical to the corresponding BIO_hash key.

	File layout (all integers little-endian):

	  [0..63]    header
	               char     magic[4]   "KMDB"
	               uint32   version    (1)
	               uint32   k
	               uint32   block_n    max k-mers per block
	               uint64   n_kmers
	               uint64   n_blocks
	               uint64   index_offset
	               uint64   meta_offset
	               uint64   meta_len
	               uint64   flags      (bit 0: canonical = max(fwd, rc))
	  [64..]     blocks, back to back. Each block is one zstd frame (with
	             content checksum) whose decompressed payload is
	               n varints: key deltas (first is 0, relative to
	                          index.first_kmer)
	               n varints: counts
	  [meta]     "key=value\n" text lines
	  [index]    n_blocks x 32 bytes
	               uint64 first_kmer, uint64 last_kmer, uint64 offset,
	               uint32 comp_size,  uint32 n

	Blocks are independent, so readers can decode any block from any
	thread; kmdb_read_block() uses pread() and is thread-safe as long as
	each thread passes its own kmdb_block_buf.
*/

#include <stdint.h>
#include <stddef.h>

#define KMDB_MAGIC            "KMDB"
#define KMDB_VERSION          1u
#define KMDB_HEADER_SIZE      64u
#define KMDB_INDEX_ENTRY_SIZE 32u
#define KMDB_FLAG_CANON_MAX   1ull
#define KMDB_DEFAULT_BLOCK_N  (1u << 20)
#define KMDB_MAX_K            31

/* ── 2-bit helpers ─────────────────────────────────────────────────── */

/* 0..3 for ACGT/acgt, 4 for anything else */
extern const uint8_t KMDB_NT2BIT[256];

static inline uint64_t kmdb_revcomp(uint64_t x, int k)
{
	/* complement, then reverse the order of the 2-bit groups */
	x = ~x;
	x = ((x >> 2)  & 0x3333333333333333ull) | ((x & 0x3333333333333333ull) << 2);
	x = ((x >> 4)  & 0x0F0F0F0F0F0F0F0Full) | ((x & 0x0F0F0F0F0F0F0F0Full) << 4);
	x = ((x >> 8)  & 0x00FF00FF00FF00FFull) | ((x & 0x00FF00FF00FF00FFull) << 8);
	x = ((x >> 16) & 0x0000FFFF0000FFFFull) | ((x & 0x0000FFFF0000FFFFull) << 16);
	x = (x >> 32) | (x << 32);
	return x >> (64 - 2 * k);
}

static inline uint64_t kmdb_canonical(uint64_t x, int k)
{
	uint64_t r = kmdb_revcomp(x, k);
	return r > x ? r : x;
}

/* Packs s[0..k-1]. Returns -1 if any base is not ACGT. */
int  kmdb_encode(const char *s, int k, uint64_t *out);
/* Writes k bases + NUL into out (size >= k+1). */
void kmdb_decode(uint64_t x, int k, char *out);

/* ── writer ────────────────────────────────────────────────────────── */

typedef struct kmdb_writer_s kmdb_writer;

/* block_n == 0 -> KMDB_DEFAULT_BLOCK_N; zstd_level == 0 -> 3 */
kmdb_writer *kmdb_writer_open(const char *path, int k, uint32_t block_n,
                              int zstd_level);
/* Keys must be strictly increasing. Returns 0, or -1 on error. */
int  kmdb_writer_add(kmdb_writer *w, uint64_t kmer, uint32_t count);
/* Metadata is written at close. Keys/values must not contain '\n' or '='
   in keys. Later calls with the same key are appended, not replaced. */
void kmdb_writer_meta(kmdb_writer *w, const char *key, const char *value);
void kmdb_writer_meta_u64(kmdb_writer *w, const char *key, uint64_t value);
/* Flushes, writes metadata/index/header, closes. Frees w. 0 or -1. */
int  kmdb_writer_close(kmdb_writer *w);

/* ── reader ────────────────────────────────────────────────────────── */

typedef struct {
	uint64_t first_kmer;
	uint64_t last_kmer;
	uint64_t offset;
	uint32_t comp_size;
	uint32_t n;
} kmdb_block_info;

typedef struct {
	int              fd;
	char            *path;
	uint32_t         k;
	uint32_t         block_n;
	uint64_t         n_kmers;
	uint64_t         n_blocks;
	uint64_t         flags;
	kmdb_block_info *blocks;
	char            *meta;       /* NUL-terminated "key=value\n" text */
	uint32_t         max_comp_size;
	uint32_t         max_n;
} kmdb_t;

/* Per-thread decode buffers. */
typedef struct {
	void     *dctx;      /* ZSTD_DCtx* */
	uint8_t  *comp;
	size_t    comp_cap;
	uint8_t  *raw;
	size_t    raw_cap;
	uint64_t *kmers;
	uint32_t *counts;
	size_t    n_cap;
} kmdb_block_buf;

/* Returns NULL (and prints why) on failure. */
kmdb_t     *kmdb_open(const char *path);
void        kmdb_close(kmdb_t *db);
/* Returns a malloc'd copy of the value of the LAST line with this key,
   or NULL. */
char       *kmdb_meta_get(const kmdb_t *db, const char *key);

void kmdb_block_buf_init(kmdb_block_buf *b);
void kmdb_block_buf_free(kmdb_block_buf *b);
/* Decodes block i into b->kmers / b->counts. Returns n, or -1 on error. */
long kmdb_read_block(const kmdb_t *db, uint64_t i, kmdb_block_buf *b);

/* True if path ends in ".kmdb". */
int kmdb_is_kmdb_path(const char *path);

#endif /* KMDB_H */
