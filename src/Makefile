CFLAGS = -O2 -Wall -Wextra -Wshadow -Wcast-qual -g

LDFLAGS = -lz -lm -lpthread

src = $(wildcard *.c)
obj = $(src:.c=.o)
shared_obj = BIO_sequence.o BIO_hash.o genome_compare.o up2bit.o


all: genome_compare kmer_scrub_count kmer_scrub_count_individual strain_detect kmer_strain_detect kmer_strain_detect_genome

genome_compare: main.o $(shared_obj)
	$(CC) -o $@ $^ $(LDFLAGS)

kmer_scrub_count: kmer_scrub_count.o $(shared_obj)
	$(CC) -o $@ $^ $(LDFLAGS)

kmer_scrub_count_individual: kmer_scrub_count_individual.o kmer_scrub_streaming.o $(shared_obj)
	$(CC) -o $@ $^ $(LDFLAGS)

strain_detect: strain_detect.o $(shared_obj)
	$(CC) -o $@ $^ $(LDFLAGS)

kmer_strain_detect: kmer_strain_detect.o $(shared_obj)
	$(CC) -o $@ $^ $(LDFLAGS)

kmer_strain_detect_genome: kmer_strain_detect_genome.o $(shared_obj)
	$(CC) -o $@ $^ $(LDFLAGS)


.PHONY: clean
clean:
	$(RM) $(obj) genome_compare kmer_scrub_count kmer_scrub_count_individual strain_detect kmer_strain_detect kmer_strain_detect_genome