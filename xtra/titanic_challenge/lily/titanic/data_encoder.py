#!/usr/bin/python3
import csv
import random

#add argparse, 
#turn into a csv file after encoding, use csv writer function (need comma after every digit)
#dont need passenger id for training 

#file = 'train.csv'

#this is for training!!!!!!!
def one_hot(file):
	with open(file) as fp:
		reader = csv.DictReader(fp)
		#next(reader, None)
		new_data = []
		#labels = []
		for row in reader:
			#print(row)
			encoding = []
			#encoding += str(row[0])

			
			if row['Pclass'] == '1': 
				encoding.append(1)
				encoding.append(0)
				encoding.append(0)
			elif row['Pclass'] == '2': 
				encoding.append(0)
				encoding.append(1)
				encoding.append(0)
			elif row['Pclass'] == '3':  
				encoding.append(0)
				encoding.append(0)
				encoding.append(1)
			else: 
				raise Error('no pclass')
			if row['Sex'] == 'female': 
				encoding.append(1)
				encoding.append(0)
			else:                  
				encoding.append(0)
				encoding.append(1)
			if row['Age'] == '': 
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				#row['Age'] = random.choice(['1','10','20', '40', '60','70']) 
			elif float(row['Age']) < 5: 
				encoding.append(1)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
			elif float(row['Age']) < 15: 
				encoding.append(0)
				encoding.append(1)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
			elif float(row['Age']) < 25: 
				encoding.append(0)
				encoding.append(0)
				encoding.append(1)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
			elif float(row['Age']) < 45: 
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(1)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
			elif float(row['Age']) < 65: 
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(1)
				encoding.append(0)
				encoding.append(0)
			elif float(row['Age']) < 75: 
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(1)
				encoding.append(0)
			else:                 
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(0)
				encoding.append(1)
			
			if float(row['SibSp']) == 0: 
				encoding.append(1)
				encoding.append(0)
				encoding.append(0)
			elif float(row['SibSp']) >= 1 and float(row['SibSp']) < 3:
				encoding.append(0)
				encoding.append(1)
				encoding.append(0)
			else: 
				encoding.append(0)
				encoding.append(0)
				encoding.append(1)
			if float(row['Parch']) == 0: 
				encoding.append(1)
				encoding.append(0)
				encoding.append(0)
			elif float(row['Parch']) == 1: 
				encoding.append(0)
				encoding.append(1)
				encoding.append(0)
			else: 
				encoding.append(0)
				encoding.append(0)
				encoding.append(1)
				
			if row['Survived'] == '1':
				encoding.append(1)
			else:
				encoding.append(0)
			new_data.append(encoding)
			#row['PassengerID]
			#labels.append(row['Survived'])
	return new_data #, labels
	
	
training  = one_hot('train.csv')
#testing = one_hot('test.csv')

#print(training)
#print(row[4])
#print(testing)		

with open('data.csv','w') as file:
	writer = csv.writer(file)
	for l in training:
		writer.writerow(l)
		