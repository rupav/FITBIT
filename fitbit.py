# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:45:09 2017

@author: MARK
"""

import numpy as np
import pandas as pd

df = pd.read_csv('pml-training.csv')
#print(df.head())
print(df.columns)
del df['Unnamed: 0']
print(df.columns)
print(df['user_name'].unique()) ## 6 users
print(df.shape)  ## (19622,159)
print(df.describe())

df.loc[df.classe=='A','classe']= 0
df.loc[df.classe=='B','classe']= 1
df.loc[df.classe == 'C','classe']=2
df.loc[df.classe=='D','classe']=3
df.loc[df.classe=='E','classe']=4

x = df.columns
print(pd.isnull(df).sum())
df.dropna(axis = 1, inplace = True)
print(df.shape)
print(df.describe())
print(pd.isnull(df).sum())
df.drop(['user_name'],axis=1,inplace = True)
print(df.head())

## printing unique values of different columns
print(df.head())
### to iterate over columns!!!
for columns in df:
    print("column: ",columns," : ",df[columns].unique().shape[0])   

#deleting     
'''
for columns in df.iteritems():   ## 2nd method for columns travel... for row we can use iterrows()
    print(columns)
'''
''' 
to filter rows according to their data types!
'''
print(df.describe(include=['O',np.number]))  ##'O' for object type(in pandas it refers to a string or characters):: returns count,unique,top,freq

print(df.dtypes) ##dtypes is not callable!
print(df.select_dtypes(include = [], exclude=['float64','int64']).head())

## timestamp will not be usefull for dataset behaviour, so ignoring them..
df.drop(['raw_timestamp_part_1','raw_timestamp_part_2','cvtd_timestamp'],axis=1,inplace=True)
print(df.head())

new_window_no = df.loc[df.new_window=='no','new_window']
print(new_window_no.shape[0]) ## 19216 similar to earlier 100 nan columns .. i.e. not useful
df.drop(['new_window','num_window'],axis=1,inplace = True)

print(df.shape)
print(df.describe())

from sklearn.tree import DecisionTreeClassifier
#from tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree 
from sklearn.model_selection import cross_val_score, train_test_split 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pydotplus


le = LabelEncoder()
df['classe'] = le.fit_transform(df['classe'])



inputDF = df.drop('classe',axis=1)
targetDF = df.classe
clf = DecisionTreeClassifier()
clf_R = RandomForestClassifier()



#x_train,x_test,y_train,y_test = train_test_split(inputDF, targetDF, test_size = 0.2 ) 
#clf.fit(x_train,y_train)
#clf_R.fit(x_train,y_train)

score_D = cross_val_score(clf,inputDF,targetDF,cv = 10)
print("Decision tree score: ",score_D.mean())

score_R = cross_val_score(clf_R,inputDF,targetDF,cv = 10)
print("Random forest Score: ",score_R.mean())

'''
dot_data = tree.export_graphviz(clf,feature_names = inputDF.columns,class_names = 'classe', 
                                filled = True, 
                                rounded= True,
                                special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("fitbit.pdf")
'''






























    
    
    
    
    


















