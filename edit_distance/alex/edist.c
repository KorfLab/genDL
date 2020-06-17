#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int encode_nt(char nt) {
	/* 
	 *  Each nucleotide is represented by 4 bits.
	 *  Any of the encoded values XORed with any other of these values
	 *  gives a result with exactly two "1" bits set.
	 * 
	 *  e.g. AT ^ AA = 00010100 ^ 00010001 = 00000101 
	 *  -> 2 bits set -> distance = 2/1 = 1
	 */
	switch(nt) {
		case 'A':	return 1;
		case 'C':	return 2;
		case 'G':	return 4;
		case 'T':	return 8;
	}
	printf("broken %c \n", nt);
	return 0;
}

int main(int argc, char **argv) {

	clock_t begin_pre = clock();

	FILE *fp;
	char line[100];
	int i, j, k, count, len, chunks;

	/* 
	 *  Each sequence is represented by chunks of 8-byte integers containing
	 *  16 4-bit nucleotides. A 64-bit processor can XOR a pair of such chunks
	 *  in one cycle. Counting the number of 1s in the result gives the edit distance.
	 */
	uint64_t ** seqs;
	long d, sum, comps;
	
	/* get the number of sequences and length */
	count = 0;
	len = 0;
	fp = fopen(argv[1], "r");
	while(fgets(line, sizeof(line), fp)) {
		count += 1;
		len = strlen(line);
	}
	len -= 1; // ignore '\n'
	fclose(fp);
	

	/* compute the number of chunks per sequence */
	if (len % 16 == 0) {
		chunks = len / 16;
	} else {
		chunks = (int) ceil(((double) len) / 16);
	}

	/* allocate storage */
	seqs = malloc(count * sizeof(uint64_t *));

	for (i = 0; i < count; i++) {
		seqs[i] = malloc(chunks * sizeof(uint64_t));
	}
	
	/* store all sequences */
	fp = fopen(argv[1], "r");
	i = 0;

	uint64_t mask, new, chunk, nt;
	int pos;
	while (fgets(line, sizeof(line), fp)) {
		for (j = 0; j < chunks; j++) {
		chunk = 0;
			for (k=0; k < 16; k++) { // for each nucleotide in this chunk
				pos = 16*j + k; // absolute position in the sequence
				if (pos >= len) {
					break;
				}
				/* set the k-th 4-bit segment of the chunk to the encoded nt */
				nt = encode_nt(line[pos]);
				mask = (uint64_t) 0b1111 << 4*k;
				new =  nt << 4*k;
				chunk = (chunk & ~mask) | (new & mask);
			}
			seqs[i][j] = chunk;
		}
		i++;
	}
	fclose(fp);

	/* perform comparisons */
	clock_t begin_comp = clock();

	sum = 0;
	comps = 0;
	for (i = 0; i < count; i++) {
		for (j = i + 1; j < count; j++) {
			d = 0;
			for (k = 0; k < chunks; k++) {
				//  gcc built-in function (counts the number of 1s in the string)
				//  divide by 2 because each mismatched base pair causes two 1s to appear
				d += __builtin_popcountll(seqs[i][k] ^ seqs[j][k]) / 2;
			}
			sum += d;
			comps++;
		}
		free(seqs[i]);
	}
	free(seqs);

	clock_t end = clock();

	 /* report stats */
	printf("%zu %zu %f\n", sum, comps, (double)sum/comps);
	printf("Preprocessing time: %.6f s\n",  (double)(begin_comp - begin_pre) / CLOCKS_PER_SEC);
	printf("Comparison time: %.6f s\n",  (double)(end - begin_comp) / CLOCKS_PER_SEC);
	printf("Total time: %.6f s\n", (double)(end - begin_pre) / CLOCKS_PER_SEC);

}
