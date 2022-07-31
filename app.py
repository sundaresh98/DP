# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:27:51 2022

@author: LENOVO
"""

import numpy as np
import pickle
import streamlit as st
import sklearn
from sklearn.preprocessing import StandardScaler
loaded_model=pickle.load(open('trained_model.sav','rb'))

    


def diabetes_prediction(input_data):
    
    scaler=StandardScaler()
        # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)


    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    scaler.fit(input_data_reshaped)
    std_data = scaler.transform(input_data_reshaped)
    print(std_data)
    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    st.title('Diabetes Prediction Web App')
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Number of Glucose')
    BloodPressure = st.text_input('BloodPressure Value')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction Value')
    Age = st.text_input('Age of the person')
    
    diagnosis = ''
    
    #creat a button 
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    if __name__ == "__main__":
        main()
    



    
 

    
