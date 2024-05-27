import streamlit as st
import pickle
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


with open('/content/PPchem_DDIP/models/fcn_model.pkl', 'rb') as file:
    model_fcn = pickle.load(file)
with open('/content/PPchem_DDIP/models/rf_model.pkl', 'rb') as file:
    model_rf = pickle.load(file)
with open('/content/PPchem_DDIP/models/svm_model.pkl', 'rb') as file:
    model_svm = pickle.load(file)
with open('/content/PPchem_DDIP/models/variance_threshold_selector.pkl', 'rb') as file:
    selector = pickle.load(file)


st.image('/content/PPchem_DDIP/assets/image_dipp.png', caption='DIPP', width=300)



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

df_dropped = df_desc1.drop(df_desc1.columns[0], axis=1)

fused_f = selector.transform(df_dropped)

def predict():
    
    prediction1 = model_rf.predict(fused_f)
    prediction2 = model_fcn.predict(fused_f)
    prediction3 = model_svm.predict(fused_f)

    prediction_list = [prediction1[0], prediction2[0], prediction3[0]]

    mean_value = statistics.mean(prediction_list)
    std_dev_value = statistics.stdev(prediction_list)

    approx = mean_value - std_dev_value

    st.write(f"Predicted bioactivity score: {prediction1[0]}")

    if prediction1 >= 1.5:
        
        st.success('Congratulations :partying_face: !!! The molecule you proposed is active :thumbsup:')
        st.ballons()

    elif 0.5 <= prediction1 <= 1.5:

        st.success('Congratulations ! The molecule you proposed is moderately active :thumbsup:')
       
    else:
        
        st.error('The molecule is inactive :thumbsdown:')
    

st.button('Predict', on_click=predict)