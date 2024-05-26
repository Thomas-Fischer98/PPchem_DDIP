import streamlit as st
import joblib
import numpy as np
import pandas as pd
import statistics

from rdkit import Chem
from rdkit.Chem import Descriptors

from src.ppchem_ddip.functions import descriptor_smiles

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn import preprocessing

from streamlit_ketcher import st_ketcher


selector = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/notebooks/selection.joblib')
model_rf = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/models/rf_model.joblib')
model_gbm = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/models/gbm_model.joblib')
model_fcn = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/models/fcn_model.joblib')
model_fcnn = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/models/fcnn_model.joblib')
model_svm = joblib.load('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/models/fcnn_model.joblib')


st.image('/Users/davidsegura/git/practical_programming_in_chemistry/PPchem_DDIP/image_dipp.png', caption='DIPP', width=300)



st.title('Is it active against mTOR ?')
st.write('Welcome to the DIPP app where the bioactivity of your molecule will be predicted')

smiles = st.text_input("Please input your candidate SMILE", "")

st.write('OR')
st.write('You can simply draw a candidate molecule in the box bellow:')

#add bioactivity will be there to convert pIC50 to bioactivaty score
#datacleaner will clean the NaN etc
#look descriptor_df


smiles = st_ketcher(height=400)

df_desc1 = descriptor_smiles(smiles)

fused_f = selector.transform(df_desc1)

def predict():
    
    prediction1 = model_rf.predict(fused_f)
    prediction2 = model_fcn.predict(fused_f)
    prediction3 = model_svm.predict(fused_f)
    prediction4 = model_fcnn.predict(fused_f)
    prediction5 = model_gbm.predict(fused_f)

    prediction_list = [prediction1, prediction2, prediction3, prediction4, prediction5]

    mean_value = statistics.mean(prediction_list)
    std_dev_value = statistics.stdev(prediction_list)

    approx = mean_value - std_dev_value

    st.write(f"Predicted bioactivity score: {prediction1[0]}")

    if prediction1 >= 1.5:
        
        st.succes('Congratulations :partying_face: !!! The molecule you proposed is active :thumbsup:')
        st.ballons()

    elif 0.5 <= prediction1 <= 1.5:

        st.succes('Congratulations ! The molecule you proposed is moderately active :thumbsup:')
       
    else:
        
        st.error('The molecule is inactive :thumbsdown:')
    

st.button('Predict', on_click=predict)