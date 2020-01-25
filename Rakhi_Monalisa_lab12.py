# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:06:31 2019

@author: 300997447
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',30) # set the maximum width
# Load the dataset in a dataframe object 
df = pd.read_csv('C:/Users/300997447/Desktop/New folder/titanic3.csv')
# Explore the data check the column values
print(df.columns.values)
print (df.head())
categories = []
for col, col_type in df.dtypes.iteritems():
     if col_type == 'O':
          categories.append(col)
     else:
          df[col].fillna(0, inplace=True)
print(categories)
print(df.columns.values)
print(df.head())
df.describe()
df.dtypes
#check for null values
print(len(df) - df.count())  #Cabin , boat, home.dest have so many missing values

######################################################

include = ['age','sex', 'embarked', 'survived']
df_ = df[include]
print(df_.columns.values)
print(df_.head())
df_.describe()
df_.dtypes
df_['sex'].unique()
df_['embarked'].unique()
df_['age'].unique()
df_['survived'].unique()
# check the null values
print(df_.isnull().sum())
print(df_['sex'].isnull().sum())
print(df_['embarked'].isnull().sum())
print(len(df_) - df_.count())

#################################################

df_.dropna(axis=0,how='any',inplace=True)  

###############################################

categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)
print(categoricals)

###########################################

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=False)
pd.set_option('display.max_columns',30)
print(df_ohe.head())
print(df_ohe.columns.values)
print(len(df_ohe) - df_ohe.count())

############################################

from sklearn import preprocessing
# Get column names first
names = df_ohe.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df['age'].describe())
print(scaled_df['sex_male'].describe())
print(scaled_df['sex_female'].describe())
print(scaled_df['embarked_C'].describe())
print(scaled_df['embarked_Q'].describe())
print(scaled_df['embarked_S'].describe())
print(scaled_df['survived'].describe())
print(scaled_df.dtypes)

########################################

from sklearn.linear_model import LogisticRegression
dependent_variable = 'survived'
# Another way to split the three features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = scaled_df[dependent_variable]
#convert the class back into integer
y = y.astype(int)
# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)
#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)

#####################################

testY_predict = lr.predict(testX)
testY_predict.dtype
#print(testY_predict)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))

#################################

import joblib 
joblib.dump(lr, 'C:/Users/300997447/Desktop/New folder/model_lr2.pkl')
print("Model dumped!")

##############################

model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, 'C:/Users/300997447/Desktop/New folder/model_columns.pkl')
print("Models columns dumped!")

###############################

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
# Your API definition
app = Flask(__name__)

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to titanic model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load('C:/Users/300997447/Desktop/New folder/model_lr2.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('C:/Users/300997447/Desktop/New folder/model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)



