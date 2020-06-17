#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int edit_distance(char *s1, char *s2, int len) {
	int i, d;
	d = 0;
	for (i = 0; i < len; i++) {
		if (s1[i] != s2[i]) d++;
	}
	return d;
}

int main (int argc, char ** argv) {
	FILE *fp;
	char line[100];
	int i, j, count, len;
	char ** seqs;
	long d, sum, comps;
	
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
	fp = fopen(argv[1], "r");
	i = 0;
	while(fgets(line, sizeof(line), fp)) {
		strcpy(seqs[i], line);
		i++;
	}
	fclose(fp);
	
	/* do comparisons */
	sum = 0;
	comps = 0;
	for (i = 0; i < count; i++) {
		for (j = i + 1; j < count; j++) {
			d = edit_distance(seqs[i], seqs[j], len-1);
			sum += d;
			comps++;
		}
	}
	
	/* report stats */
	printf("%zu %zu %f\n", sum, comps, (double)sum/comps);

}



