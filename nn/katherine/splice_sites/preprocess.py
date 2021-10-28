import seqio
import pandas as pd

# X is a list of strings
X = seqio.fasta2binary("cdsi.fa.gz",1)

# Convert list of strings into csv
f = open("data.csv", 'w')
for string in X:
    string = string.replace('',',')
    f.write(string + "\n")
f.close()

# Clean up unnecessary commas
df = pd.read_csv("data.csv")
# Drop first and last columns (which are null)
df = df.drop(columns=[df.columns[0],
                      df.columns[-1]])

# Rename columns
cols = []
for num in range (1, 103):
    cols.append(num)
cols.append("Label")
df.columns = cols

# Output as csv
df.to_csv("cdsi_true.csv", index=False)
