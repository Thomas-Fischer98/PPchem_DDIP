import streamlit as st
import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from src.ppchem_ddip.functions import lipinski1, descriptors1 

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn import preprocessing

from streamlit_ketcher import st_ketcher


selector = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/notebooks/selection.joblib')
model_rf = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/notebooks/rf_model.joblib')
#model_gbm = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/notebooks/model_gbm2.joblib')
#model_nn = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/notebooks/model_nn2.joblib')

st.image('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/image_dipp.png', caption='DIPP', width=300)



st.title('Is it active against mTOR ?')
st.write('Welcome to the DIPP app where the pIC50 of your molecule will be predicted')

smiles = st.text_input("Please input your candidate SMILE", "")

st.write('OR')
st.write('You can simply draw a candidate molecule in the box bellow:')


smiles = st_ketcher(height=400)

#f_desc = lipinski1(smiles)

df_desc1 = descriptors1(smiles)

#fused1 = pd.concat([df_desc, df_desc1], axis=1)

#fused_f = selector.transform(fused1)

fused_f = selector.transform(df_desc1)

def predict():
    
    prediction1 = model_rf.predict(fused_f)
    #prediction2 = model_rf.predict(smiles)
    #prediction3 = model_rf.predict(smiles)

    #prediction_list = [prediction1, prediction2, prediction3]

    #mean_value = statistics.mean(prediction_list)
    #std_dev_value = statistics.stdev(prediction_list)

    #approx = mean_value + std_dev_value

    st.write(f"Predicted pIC50 value: {prediction1[0]}")

    if prediction1 <= 3:
        
        st.succes('Congratulations :partying_face: !!! The molecule you proposed is active :thumbsup:')
        st.ballons()

    else:
        
        st.error('The molecule is inactive :thumbsdown:')
    

st.button('Predict', on_click=predict)