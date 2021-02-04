README for clustring methods
============================
For genDL, three clustering methods were created. Two of the methods are using k-means clustering, while another the last is based on apriori algorithm. K-means algorithm is an unsupervised learning algorithm that tries to partition n observations into k clusters. Apriori algorithm is another unsupervised learning algoriithm but unlike k-means clustering that looks thorughout the each dataset to figure out association rules present.

#k-means #
For k-means script, k-means clustering is performed on data. For the data, it is required to provide two files. After those two files are placed into the same set on which k-means clustering is performed.

Example command line:
    python3

Output:

Results:

| file1   | file0   | mm | accuracy | strawman | notes
|:--------|:--------|:--:|----------|:---------|:-----
|

#k-means pwm#
Unlike k-means script, here we perform k-means clustering methods on two files separately. After the k-means is done, we go through each dataset and create separate dataset for each k-label. After, we use those datasets to create pwm. Test sequences are then run against those pwms; output for each pwm is a score, which is then compared to other scores. This test seq is then labeled as either belogning to either of the files based on whichever pwm is score as a highest. Output is the accuracy.

Example command line:
    python3

Output:

Results:

| file1   | file0   | mm | accuracy | strawman | notes
|:--------|:--------|:--:|----------|:---------|:-----
|

#Apriori#
Apriori is initially performed on two provided datasets (file1 and file0) to figure out the rules present in each of the datasets. After we use those rules to split the test datasets into two groups: seqs that had this rule present and seqs that did not. Then, they were used to create two pwm. After that, the test seqs, which contain test seqs from both of the sets were run against pwm and scored. The accuracy was the output. Did the same for each rule. The output for this function was the rule and the accuracy.

Example command line:
    python3

Output:

Results:

| file1   | file0   | mm | accuracy | strawman | notes
|:--------|:--------|:--:|----------|:---------|:-----
|
