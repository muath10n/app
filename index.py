import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import joblib
@st.cache_data
def load_models():
    model = joblib.load("Pmodel.pgz")
    return model
def predict(input_dataframe: pd.DataFrame):
    model = load_models()               
    y_test_predicted_inj = model.predict(input_dataframe.values)
    with st.container():
        st.header("Output")
        if len(y_test_predicted_inj) == 1:
            output = 'injured' if y_test_predicted_inj[0] == 1 else 'Non-injured'
            st.write(f"The predicted output is '{output}'")

st.title('Machine learning on traffic accident dataset')
st.write ("This is an app for predicting a person's injury after an accident.")
input_dataframe = None
dead= st.number_input('Was the driver dead or not?', value=0,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
airbag = st.number_input('Is the airbag out or not??', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
seatbelt = st.number_input('Was the driver wearing a seat belt or not??', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
frontal = st.number_input('Was the impact frontal or lateral?', value=1, min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
sex = st.number_input('What is your gender?', value=0,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Male' and 0 for 'Female'")
occRole= st.number_input('Was the effect on the driver or the passenger?', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
AgeCar = st.number_input('Was the car more than 5 years old?', value=0,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
ageOFocc = st.number_input('In what category is your age? (1 = 18-24 ,9 = 60-64 ,13 = 80 or older)',value= 9,min_value=1,max_value=13,step=1,help='''
1 Age 18 to 24\n
2 Age 25 to 29\n
3 Age 30 to 34\n
4 Age 35 to 39\n
5 Age 40 to 44\n
6 Age 45 to 49\n
7 Age 50 to 54\n
8 Age 55 to 59\n
9 Age 60 to 64\n
10 Age 65 to 69\n
11 Age 70 to 74\n
12 Age 75 to 79\n
13 Age 80 or older''')

l = [[ dead, airbag,seatbelt,frontal, sex,ageOFocc,occRole , AgeCar]]
input_dataframe = pd.DataFrame(l, columns=['dead', 'airbag','seatbelt','frontal','sex','ageOFocc','occRole','AgeCar'])
submit = st.button('Submit')
if submit:
    predict(input_dataframe)