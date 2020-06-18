#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

int THREADS;

int edit_distance(char *s1, char *s2, int len) {
	int i, d;
	d = 0;
	for (i = 0; i < len; i++) {
		if (s1[i] != s2[i]) d++;
	}
	return d;
}

typedef struct _thread_data_t {
	int tid;
	int job;
	int count;
	int len;
	char ** seqs;
	long sum;
	long comps;
} thread_data_t;

void *thr_func (void *arg) {
	int i, j, d;
	thread_data_t *data = (thread_data_t *)arg;
	
	printf("working in thread id: %d\n", data->tid);
	data->sum = 0;
	data->comps = 0;
	for (i = 0; i < data->count; i++) {
		if (i % THREADS != data->tid) continue;		
		for (j = i + 1; j < data->count; j++) {
			d = edit_distance(data->seqs[i], data->seqs[j], data->len-1);
			data->sum += d;
			data->comps++;
		}
	}
	
	printf("%d: %zu %zu %f\n", data->tid, data->sum, data->comps, (double)data->sum/data->comps);
	
	
	pthread_exit(NULL);
}

int main (int argc, char ** argv) {
	/* usage */
	if (argc != 3) {
		fprintf(stderr, "usage: %s <file> <threads>\n", argv[0]);
		exit(1);
	}
	char *filename = argv[1];
	THREADS = atoi(argv[2]);

	/* variables */
	FILE *fp;
	char line[100];
	int i, count, len;
	char ** seqs;
	long sum, comps;
	pthread_t thr[THREADS];
	int rc;
	thread_data_t thr_data[THREADS];
	
	/* get the number of sequences and length */
	count = 0;
	len = 0;
	fp = fopen(argv[1], "r");
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
	
	/* working */
	for (i = 0; i < THREADS; ++i) {
		thr_data[i].tid = i;
		thr_data[i].count = count;
		thr_data[i].seqs = seqs;
		thr_data[i].len = len;
		if ((rc = pthread_create(&thr[i], NULL, thr_func, &thr_data[i]))) {
			fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
			return EXIT_FAILURE;
		}
	}
	
	/* collecting */
	sum = 0;
	comps = 0;
	for (i = 0; i < THREADS; ++i) {
		pthread_join(thr[i], NULL);
		sum += thr_data[i].sum;
		comps += thr_data[i].comps;
	}

	printf("%d %zu %zu %.3f\n", count, sum, comps, (double)sum/comps);

}

