#!/usr/bin/python3

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

file_path = '~/.kaggle/titanic/train.csv'

titan_data = pd.read_csv(file_path)

titan_test_fp = '~/.kaggle/titanic/test.csv'

titan_test = pd.read_csv(titan_test_fp)


#print(titan_data)
#print(titan_data.columns)

titan_data = titan_data.replace(to_replace="male",value="0")
titan_data = titan_data.replace(to_replace="female",value="1")
titan_data = titan_data.replace(to_replace="S",value="0")
titan_data = titan_data.replace(to_replace="Q",value="1")
titan_data = titan_data.replace(to_replace="C",value="2")


#print(titan_data)

titan_vars = ['PassengerId','Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']

titan_data = titan_data[titan_vars]

#print(titan_data)

titan_data = titan_data.dropna(axis=0)

#print(titan_data)

y = titan_data.Survived

#print(y)



titan_vars = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']

x = titan_data[titan_vars]
#print(x)
x = x.dropna(axis=0)
#print(x)
#y = x.Survived
#print(y)


titan_model = DecisionTreeRegressor(random_state=1)

# Fit model
titan_model.fit(x, y)
#print("The predictions are")
#print(titan_model.predict()

print("The predictions are (training data)")
predictions = titan_model.predict(x)
print('MAE',mean_absolute_error(y, predictions))


titan_test_fp = '~/.kaggle/titanic/test.csv'

titan_test = pd.read_csv(titan_test_fp)

titan_test = titan_test.replace(to_replace="male",value="0")
titan_test = titan_test.replace(to_replace="female",value="1")
titan_test = titan_test.replace(to_replace="S",value="0")
titan_test = titan_test.replace(to_replace="Q",value="1")
titan_test = titan_test.replace(to_replace="C",value="2")

titan_vars = ['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']
titan_test = titan_test[titan_vars]

#print(titan_data)

titan_test = titan_test.dropna(axis=0)

print(titan_test)

#y = titan_test.Survived

#print(y)



titan_vars = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']

x = titan_test[titan_vars]
print(x)
x = x.dropna(axis=0)
print(x)
#y = x.Survived
#print(y)


#titan_ = DecisionTreeRegressor(random_state=1)

# Fit model
#print(titan_model.fit(x, y))
#print("The predictions are")
#print(titan_model.predict()

print("The predictions are")
print(titan_model.predict(x))

