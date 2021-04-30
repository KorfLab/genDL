import matplotlib.pyplot as plt
import csv
import argparse
import pandas as pd
import sys


parser = argparse.ArgumentParser(description='CNN model')
parser.add_argument('--file', required=True, type=str,
	metavar='<path>', help='path to cvs file')
arg = parser.parse_args()

df = pd.read_csv(arg.file)
print(df.columns)
#sys.exit()

df = df[df.Fare != 512.3292]
#graphs
# sex vs age
colours = {0:'red', 1:'blue'}
df1 = df[df.Age > 50]
plt.scatter(df1['Sex'], df1['Age'], c=df1['Survived'].map(colours))
plt.show()

plt.scatter(df['Pclass'], df['Fare'], c=df['Survived'].map(colours))
xint = [1, 2, 3]
plt.xticks(xint)
plt.show()

df2 = df[df.Fare < 200]
df2['Embarked'] = df2['Embarked'].astype(str)
plt.scatter(df2['Embarked'], df2['Fare'], c=df2['Survived'].map(colours))
plt.show()



#survived vs parch
#survived vs sibsp
''' Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object') '''

