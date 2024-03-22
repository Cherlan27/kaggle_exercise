# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:57:25 2024

@author: supex623
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Import data from source

test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")

print(train_data.keys())
print(train_data.head(5))
print(train_data["Survived"].head(5))

# Data Cleaning
age_train = train_data["Sex"].replace("male",0).replace("female",1)
y_train = train_data["Survived"]
#x_train = pd.concat([train_data["PassengerId"], train_data["Pclass"], train_data["Name"], train_data['Sex'], train_data['Age'], train_data['SibSp'],
#       train_data['Parch'], train_data['Ticket'], train_data['Fare'], train_data['Cabin'], train_data['Embarked']], axis = 1)
x_train = pd.concat([train_data["PassengerId"], train_data["Pclass"], age_train, train_data['Age'], train_data['SibSp'],
       train_data['Parch'], train_data['Fare']], axis = 1)

print(x_train.head(10))

clf = RandomForestClassifier(max_depth = 5, random_state = 0)
clf.fit(x_train, y_train)

# Test prediction
print("\nTest prediction:")
print(clf.predict([[1, 3, 0, 22.0, 1, 0, 7.25]]))
print(clf.predict([[2, 1, 1, 38.0, 1, 0, 71.2833]]))

age_test = test_data["Sex"].replace("male",0).replace("female",1)
x_test = pd.concat([test_data["PassengerId"], test_data["Pclass"], age_test, test_data['Age'], test_data['SibSp'],
       test_data['Parch'], test_data['Fare']], axis = 1)


y_test = clf.predict(x_test)
prediction = pd.concat([test_data["PassengerId"], pd.DataFrame(y_test, header = "Survived")], axis = 1)
prediction.to_csv("prediction.csv", index = False)