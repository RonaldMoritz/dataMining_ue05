#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.naive_bayes import GaussianNB
import os
import datetime

st.write("""
### App predicting whether there will or won't be rain on the next day for given parameters
    """)

st.sidebar.header("User input parameters")

def user_input_data():
    date = st.sidebar.date_input('Date', datetime.date(2022,1,1))
    location = st.sidebar.text_input('Location', placeholder='location')
    minTemp = st.sidebar.number_input('MinTemp')
    maxTemp = st.sidebar.number_input('MaxTemp')
    rainfall = st.sidebar.number_input('Rainfall')
    evaporation = st.sidebar.number_input('Evaporation')
    sunshine = st.sidebar.number_input('Sunshine')
    windGustDir = st.sidebar.text_input('WindGustDir', placeholder='NW')
    windGustSpeed = st.sidebar.number_input('WindGustSpeed')
    windDir9am = st.sidebar.text_input('WindDir9am', placeholder='NW')
    windDir3pm = st.sidebar.text_input('WindDir3pm', placeholder='NW')
    windSpeed9am = st.sidebar.number_input('WindSpeed9am')
    windSpeed3pm = st.sidebar.number_input('WindSpeed3pm')
    humidity9am = st.sidebar.number_input('Humidity9am')
    humidity3pm = st.sidebar.number_input('Humidity3pm')
    pressure9am = st.sidebar.number_input('Pressure9am')
    pressure3pm = st.sidebar.number_input('Pressure3pm')
    cloud9am = st.sidebar.number_input('Cloud9am')
    cloud3pm = st.sidebar.number_input('Cloud3pm')
    temp9am = st.sidebar.number_input('Temp9am')
    temp3pm = st.sidebar.number_input('Temp3pm')
    rainToday = st.sidebar.text_input('RainToday', placeholder='Yes or No')
    data={'Date': date,
          'Location': location,
          'MinTemp': minTemp,
          'MaxTemp': maxTemp,
          'Rainfall': rainfall,
          'Evaporation': evaporation,
          'Sunshine': sunshine,
          'WindGustDir': windGustDir,
          'WindGustSpeed': windGustSpeed,
          'WindDir9am': windDir9am,
          'WindDir3pm': windDir3pm,
          'WindSpeed9am': windSpeed9am,
          'WindSpeed3pm': windSpeed3pm,
          'Humidity9am': humidity9am,
          'Humidity3pm': humidity3pm,
          'Pressure9am': pressure9am,
          'Pressure3pm': pressure3pm,
          'Cloud9am': cloud9am,
          'Cloud3pm': cloud3pm,
          'Temp9am': temp9am,
          'Temp3pm': temp3pm,
          'RainToday': rainToday
         }

    features=pd.DataFrame(data,index=[0])
    return features

df=user_input_data()
del df['Date']
del df['Location']
del df['WindGustDir']
del df['WindDir9am']
del df['WindDir3pm']
del df['RainToday']

# load and preprocess data
trainingData = pd.read_csv("data\weather_train_data.csv", encoding= 'unicode_escape', delimiter = ',')
trainingLabel = pd.read_csv("data\weather_train_label.csv", encoding= 'unicode_escape', delimiter = ',')
preprocessed_trainingData = trainingData.copy()
del preprocessed_trainingData['Date']
del preprocessed_trainingData['Location']
del preprocessed_trainingData['WindGustDir']
del preprocessed_trainingData['WindDir9am']
del preprocessed_trainingData['WindDir3pm']
del preprocessed_trainingData['RainToday']
for column in trainingData:
    if trainingData[column].dtype == np.float64:
        upper_bound = trainingData[column].mean() + 3*trainingData[column].std()
        lower_bound = trainingData[column].mean() - 3*trainingData[column].std()
        
        preprocessed_trainingData[column] = np.where(
            trainingData[column]>upper_bound,
            upper_bound,
            np.where(
                trainingData[column]<lower_bound,
                lower_bound,
                trainingData[column]
            )
        )
preprocessed_trainingData['label'] = trainingLabel
preprocessed_trainingData = preprocessed_trainingData.dropna()
trainingLabel_ = preprocessed_trainingData.pop("label")
trainingLabel_ = trainingLabel_.to_numpy()      
X= preprocessed_trainingData
Y= trainingLabel_

# Prediction
clf=GaussianNB(var_smoothing=0.8)
clf.fit(X,Y)

prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)

st.subheader('Prediction [Yes or No]:')
st.write(prediction)

