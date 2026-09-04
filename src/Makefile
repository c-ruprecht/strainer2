CC       ?= gcc
CFLAGS   ?= -std=gnu17 -O2 -Wall -Wextra -Wshadow -Wcast-qual -g
LDLIBS    = -lz -lm -lpthread

# ---------------------------------------------------------------------------
# libzstd detection
#
# Only kmer_scrub_count_individual actually needs zstd (kmer_scrub_streaming.c
# writes the compressed presence index), and genome_compare_presence uses it
# only when you ask for a .zst output. The old Makefile put -lzstd in the
# global LDFLAGS, so a machine without libzstd failed to link *every* target
# with "cannot find -lzstd" -- even the tools that never touch it.
#
# Order of resolution:
#   1. ZSTD_HOME=/path  (e.g. a conda env, or the prefix from `module show zstd`)
#   2. pkg-config libzstd
#   3. a bare -lzstd link test
#   4. nothing -> zstd targets are skipped, everything else still builds
#
# Override the probe with:  make HAVE_ZSTD=0            (force off)
#                           make ZSTD_HOME=$CONDA_PREFIX
# ---------------------------------------------------------------------------
ifdef ZSTD_HOME
  ZSTD_CFLAGS := -I$(ZSTD_HOME)/include
  ZSTD_LIBS   := -L$(ZSTD_HOME)/lib -Wl,-rpath,$(ZSTD_HOME)/lib -lzstd
  HAVE_ZSTD   ?= 1
else ifeq ($(shell pkg-config --exists libzstd 2>/dev/null && echo yes),yes)
  ZSTD_CFLAGS := $(shell pkg-config --cflags libzstd)
  ZSTD_LIBS   := $(shell pkg-config --libs libzstd)
  HAVE_ZSTD   ?= 1
else
  HAVE_ZSTD   ?= $(shell printf 'int main(void){return 0;}' | \
                   $(CC) -x c - -lzstd -o /dev/null 2>/dev/null && echo 1 || echo 0)
  ZSTD_CFLAGS :=
  ZSTD_LIBS   := -lzstd
endif

ifeq ($(HAVE_ZSTD),1)
  ZSTD_DEF     := -DHAVE_ZSTD $(ZSTD_CFLAGS)
  ZSTD_TARGETS := kmer_scrub_count_individual
else
  ZSTD_DEF     :=
  ZSTD_LIBS    :=
  ZSTD_TARGETS :=
endif

shared_obj = BIO_sequence.o BIO_hash.o genome_compare.o up2bit.o

BASE_TARGETS = genome_compare genome_compare_presence kmer_scrub_count \
               strain_detect kmer_strain_detect kmer_strain_detect_genome

all: $(BASE_TARGETS) $(ZSTD_TARGETS) zstd-note

zstd-note:
ifneq ($(HAVE_ZSTD),1)
	@echo ""
	@echo "  NOTE: libzstd was not found, so kmer_scrub_count_individual was"
	@echo "        NOT built, and genome_compare_presence was built without"
	@echo "        .zst output (-z falls back to .gz). Everything else is fine."
	@echo "        To get zstd back, install it and rebuild:"
	@echo "          conda install -c conda-forge zstd zstd-static"
	@echo "          make clean && make ZSTD_HOME=\$$CONDA_PREFIX"
	@echo "          (Debian/Ubuntu: apt install libzstd-dev)"
	@echo "          (HPC module:    module load zstd && make ZSTD_HOME=\$$ZSTD_ROOT)"
	@echo ""
endif

%.o: %.c
	$(CC) $(CFLAGS) $(ZSTD_DEF) -c -o $@ $<

genome_compare: main.o $(shared_obj)
	$(CC) -o $@ $^ $(LDLIBS)

genome_compare_presence: genome_compare_presence.o
	$(CC) -o $@ $^ $(LDLIBS) $(ZSTD_LIBS)

kmer_scrub_count: kmer_scrub_count.o $(shared_obj)
	$(CC) -o $@ $^ $(LDLIBS)

kmer_scrub_count_individual: kmer_scrub_count_individual.o kmer_scrub_streaming.o $(shared_obj)
	$(CC) -o $@ $^ $(LDLIBS) $(ZSTD_LIBS)

strain_detect: strain_detect.o $(shared_obj)
	$(CC) -o $@ $^ $(LDLIBS)

kmer_strain_detect: kmer_strain_detect.o $(shared_obj)
	$(CC) -o $@ $^ $(LDLIBS)

kmer_strain_detect_genome: kmer_strain_detect_genome.o $(shared_obj)
	$(CC) -o $@ $^ $(LDLIBS)

# quick end-to-end check on the toy genomes (poly-A / poly-T / half-and-half)
.PHONY: test
test: genome_compare_presence
	@bash test/run_test.sh

.PHONY: clean
clean:
	$(RM) *.o $(BASE_TARGETS) kmer_scrub_count_individual
	$(RM) -r test/out

.PHONY: all zstd-note
