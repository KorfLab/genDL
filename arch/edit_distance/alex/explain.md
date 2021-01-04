The main idea, given two sequences, is to compare more nucleotides of  each sequence at a time. This is accomplished in two ways -- by using  XOR operations to compare more things at once, and by reducing the size  of a sequence in memory.



### Encoding scheme

Given two binary digits, the XOR (exclusive-or) operation, denoted with `^` , returns 1 when exactly one of the inputs are 1.

i.e.

`0 ^ 0 = 0`

`1 ^ 0 = 1`

`0 ^ 1 = 1`

`1 ^ 1 = 0`



When applied to each bit of two binary strings, this has the effect of returning a 1 in each position that the strings differ.

e.g.

`1010 ^ 1010 = 0000`

`0000 ^ 1111 = 1111`

`1000 ^ 0001 = 1001`

The number of 1s in the result is the number of mismatches between the strings (the edit distance). 



#### Two bits?

If DNA had only two nucleotides, we could just call one of them 1 and the other 0, and then XOR two sequences represented as binary numbers. Counting the mismatches would give us the edit distance. But there are four unique values in DNA (A, C, G, T) so we need more than one bit to represent them.

We could use just two bits to represent a nucleotide, for example:

`00 = A`

`01 = C`

`10 = G`

`11 = T`



However, if we use this scheme, then when we apply the XOR idea above we run into an issue. For example,

`A ^ C = 00 ^ 01 = 01`

`A ^ T = 00 ^ 11 = 11`

The result of `A ^ C` has one "1" and the result of `A ^ T` has two. If we were to compare two DNA sequences encoded this way with a single XOR  operation, we would have something like this:

`ACAT ^ AGAC = 00 01 00 11 ^ 00 10 00 01 = 00 11 00 10`

Each mismatched base causes at least one 1 to appear in the result, but some cause two 1s. This means we can't calculate the number of mismatches by counting the number of 1s. The only solution to this is to XOR each base pair separately, and count it as a mismatch if there are any number of 1s in the result. But that brings us back to the approach of iterating over every base pair.



#### Four bits

If we use four bits instead of two to represent nucleotides, we can encode them as follows:

`0001 = A`

`0010 = C`

`0100 = G`

`1000 = T`

Note that the 1 is in a different position in each nucleotide. Because of this, if we XOR any two of the above numbers, the result will either contain all zeros (if they are the same) or exactly two 1s (if they are different). Now, when we chain multiple nucleotides together into sequences, then XOR them, each base mismatch will cause two 1s to appear in the result. The edit distance can then be found by dividing the total number of 1s in the result by two.

e.g.

`AG ^ AT = 0001 0100 ^ 0001 1000 = 0000 0101`, which contains two 1s, giving an edit distance of `2/2 = 1`.



#### Space savings

In C, a `char` variable (a single character) uses 8 bits of memory. The encoding above uses 4 bits per nucleotide instead of the 8 that would be used if each base were stored as a character.



### How many at once?

Now that we have a binary encoding for nucleotides, we want to compare as many of them at once as possible. Most processors today are "64-bit", meaning they operate on 64 bits at a time (per CPU cycle). A 64-bit processor can XOR two 64-bit values in a single cycle. 

So we should split each DNA sequence into 64-bit chunks, and operate on pairs of those chunks at a time. We can store 16 nucleotides (4 bits each) in one 64-bit integer. If we do this, theoretically we can compare 16 base pairs per CPU cycle (but this might not necessarily be true due to the compiler's decisions).

