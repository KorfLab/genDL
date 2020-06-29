#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

int edit_distance(const char *s1, const char *s2, int len) {
	int i, d;
	d = 0;
	for (i = 0; i < len; i++) {
		if (s1[i] != s2[i]) d++;
	}
	return d;
}
typedef struct _seq_data_t {
	int len;
	int count;
	char *name;
	char **seqs;
} seq_data_t;

seq_data_t read_seqs(char *filename) {
	FILE *fp;
	char line[100];
	int i, count, len;
	char **seqs;
	seq_data_t data;
	
	/* get the number of sequences and length */
	count = 0;
	len = 0;
	fp = fopen(filename, "r");
	while(fgets(line, sizeof(line), fp)) {
		count += 1;
		len = strlen(line);
	}
	fclose(fp);
	
	/* allocate storage */
	seqs = malloc(count * sizeof(char*));
	for (i = 0; i < count; i++) {
		seqs[i] = malloc(len * sizeof(char));
	}
	
	/* store all sequences */
	fp = fopen(filename, "r");
	i = 0;
	while(fgets(line, sizeof(line), fp)) {
		strcpy(seqs[i], line);
		i++;
	}
	fclose(fp);
	
	data.name = filename;
	data.len = len;
	data.count = count;
	data.seqs = seqs;
	return data;
}

long *compare_seqs(const seq_data_t s1, const seq_data_t s2, char mode) {
	int i, j, start;
	long *hist = malloc(s1.len * sizeof(long));
	
	for (i = 0; i < s1.len; i++) hist[i] = 0;
	
	for (i = 0; i < s1.count; i++) {
		start = (mode == 'h') ? i + 1 : 0;
		for (j = start; j < s2.count; j++) {
			hist[edit_distance(s1.seqs[i], s2.seqs[j], s1.len)]++;
		}
	}
	
	printf("%s vs. %s\n", s1.name, s2.name);
	for (i = 0; i < s1.len; i++) {
		printf("%d\t%zu\n", i, hist[i]);
	}
	
	return hist;
}

int main (int argc, char ** argv) {

	/* usage */
	if (argc != 3) {
		fprintf(stderr, "usage: %s <file1> <file2>\n", argv[0]);
		exit(1);
	}
	char *file1 = argv[1];
	char *file2 = argv[2];
	seq_data_t seqs1, seqs2;
	long *h1, *h2, *hx;
	
	/* get sequences */
	seqs1 = read_seqs(file1);
	seqs2 = read_seqs(file2);
	assert(seqs1.len == seqs2.len);
	
	printf("seqs1 %d\n", seqs1.count);
	printf("seqs2 %d\n", seqs2.count);
	
	/* compare sequences */
	h1 = compare_seqs(seqs1, seqs1, 'h');
	h2 = compare_seqs(seqs2, seqs2, 'h');
	hx = compare_seqs(seqs1, seqs2, 'f');
	
	exit(0);
}

