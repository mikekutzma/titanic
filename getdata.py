import numpy as np
import pandas as pd
import csv as csv

def get_features_array(data_file_name):
	df = pd.read_csv(data_file_name,header=0)
	df['Gender'] = df.Sex.map( {'female':0,'male':1} ).astype(int)
	mean_embarked = df.Embarked.mean
	df.Embarked = df.Embarked.fillna(df.Embarked.value_counts().index[0])
	df.Embarked = df.Embarked.map( {'S': 0, 'C':1, 'Q':2} ).astype(int)

	#Find median age within each [gender,class]
	med_ages = np.zeros((2,3))
	for i in range(0,2):		#Gender
		for j in range(0,3):	# Pclass
			med_ages[i,j] = df[(df.Gender==i) & (df.Pclass == j+1)].Age.dropna().median()

	#Guess unknown ages based on [gender,class] medians
	for i in range(0,2):
		for j in range(0,3):
			df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'Age'] = med_ages[i,j]

	df = df.drop(['Name','Sex','Ticket','Cabin','PassengerId','Survived'],axis=1)

	return df.values

def get_ids(data_file_name):
	df = pd.read_csv(open(data_file_name,'r'))
	return df.PassengerId.values

def get_survived(data_file_name):
	df = pd.read_csv(open(data_file_name,'r'))
	return df.Survived.values






