import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import sklearn
from itertools import cycle

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error, max_error,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report,
    roc_curve, auc
)

from scipy.stats import randint, uniform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import skorch
from skorch import NeuralNetClassifier

from chembl_webresource_client.new_client import new_client
from joblib import dump, load


# FUNCTION DEFINTIONS

def lipinski1(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    MW = Descriptors.MolWt(mol)
    LogP = Descriptors.MolLogP(mol)
    NHDonors = Lipinski.NumHDonors(mol)
    NHAcceptors = Lipinski.NumHAcceptors(mol)
    
    desc_data = pd.DataFrame({
        "Mw": [MW],
        "H donors": [NHDonors],
        "H acceptors": [NHAcceptors],
        "Log P": [LogP]
    })
    
    return desc_data

def lipinski(smiles):

    """
    smile: smile (column) 

    The function calculate the Lipinski descriptors from the smiles, the Lipinski descriptors are the MW, the LogP (solubility),
    the number of H donors and the number of H acceptors
    
    """ 
    molec = []

    for i in smiles:
        mol = Chem.MolFromSmiles(i) 
        molec.append(mol)

    MW = []
    LogP = []
    NHDonors = []
    NHAcceptors = []

    for n in molec:        
        MW.append(Descriptors.MolWt(n))
        LogP.append(Descriptors.MolLogP(n))
        NHDonors.append(Lipinski.NumHDonors(n))
        NHAcceptors.append(Lipinski.NumHAcceptors(n))
        
    desc_data = pd.DataFrame({
        "Mw": MW,
        "H donors": NHDonors,
        "H acceptors": NHAcceptors,
        "Log P": LogP
    })
    
    return desc_data

def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if i > 100000000:
            i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop(columns=['standard_value'])  # Drop the 'standard_value' column

    return x

def pIC50(input):

    """
    input: clean dataframe of the candidate molecules 

    The function calculate and add the pIC50 (a measure of bioactivity)
    
    """ 
     
    if 'standard_value_norm' not in input.columns:
        raise ValueError("Column 'standard_value_norm' does not exist in the DataFrame.")

    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i * (10**-9)  # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop(columns=['standard_value_norm'])  # Drop the 'standard_value' column

    return x

def descriptors1(smiles):
    molec = Chem.MolFromSmiles(smiles)
    if molec is None:
        raise ValueError("Invalid SMILES string")
    
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    descrs = []
    for desc in Descriptors.descList:
        
        try:
            descrs.append(desc[1](molec))
        except Exception as e:
            descrs.append(None)
     
    df_descr = pd.DataFrame([descrs], columns=descriptor_names)
    
    return df_descr

def descriptors(smiles):

    """
    smile: smile (column) 

    The function calculate the rdkit descriptors from the smiles, the rdkit descriptors are composed of a lot of different descriptors
    
    """ 

    mols = []

    for i in smiles:
        molec = Chem.MolFromSmiles(i)
        mols.append(molec)
    descrs = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    df_descr = pd.DataFrame(descrs)

    return df_descr

def data_cleaner(dataframe):
    """
    dataframe: dataframe of all the candidate proteins 

    The function only keep the entries of interest: 'molecule_chembl_id','canonical_smiles','standard_value'
    and also get rid of the incomplete data
    
    """ 
    dataframe = dataframe = dataframe[['molecule_chembl_id','canonical_smiles','standard_value']]
    dataframe_cleaned = dataframe[dataframe.standard_value.notna()]
    dataframe_cleaned = dataframe_cleaned[dataframe.canonical_smiles.notna()]
    dataframe_cleaned = dataframe_cleaned.drop_duplicates(['canonical_smiles'])
    
    return dataframe_cleaned

def add_bioactivity(dataframe):
    """
    dataframe: (cleaned) dataframe of all the candidate proteins 

    The function labels the molecules as inactive, partialy active or active depending on their standard value
    """ 
    bioactivity = []
    for i in dataframe.standard_value:
      if float(i) >= 10000:
        bioactivity.append(0) #inactive
      elif float(i) <= 1000:
        bioactivity.append(2) #active
      else:
        bioactivity.append(1) #depends
    dataframe['Bioactivity'] = bioactivity
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe

""""
def lipinski(smiles):
    
    molec = []
    for i in smiles:
        mol = Chem.MolFromSmiles(i) 
        molec.append(mol)

    MW = []
    LogP = []
    NHDonors = []
    NHAcceptors = []

    for n in molec:        
        MW.append(Descriptors.MolWt(n))
        LogP.append(Descriptors.MolLogP(n))
        NHDonors.append(Lipinski.NumHDonors(n))
        NHAcceptors.append(Lipinski.NumHAcceptors(n))
        
    desc_data = pd.DataFrame({
        "Mw": MW,
        "H donors": NHDonors,
        "H acceptors": NHAcceptors,
        "Log P": LogP
    })
    
    return desc_data
"""


def norm_value(input):
    """
    input: clean dataframe of the candidate molecules 

    The function cnormalize the standard value 
    
    """ 
    
    norm = []

    for i in input['standard_value']:
        if i > 100000000:
            i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop(columns=['standard_value'])  # Drop the 'standard_value' column

    return x

"""
def pIC50(input):
    
    if 'standard_value_norm' not in input.columns:
        raise ValueError("Column 'standard_value_norm' does not exist in the DataFrame.")

    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i * (10**-9)  # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop(columns=['standard_value_norm'])  # Drop the 'standard_value' column

    return x
"""

def lipinski_df(dataframe):
    """
    dataframe: clean dataframe of the candidate molecules 

    The function calculate and add the the Lipinski descriptors, normalize the standard values and the pIC50
    
    """ 

    
    dataframe_lipinski = pd.concat([dataframe, lipinski(dataframe.canonical_smiles)], axis = 1)
    dataframe_lipinski['standard_value'] = pd.to_numeric(dataframe_lipinski['standard_value'], errors='coerce')
    dataframe_lipinski = dataframe_lipinski.loc[pd.notna(dataframe_lipinski['standard_value'])]
    dataframe_lipinski = norm_value(dataframe_lipinski)
    dataframe_lipinski = pIC50(dataframe_lipinski)

    return dataframe_lipinski

#### DEFINITION FUNCTION ####

"""
def descriptors(smiles):
    
    mols = []

    for i in smiles:
        molec = Chem.MolFromSmiles(i)
        mols.append(molec)
    descrs = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    df_descr = pd.DataFrame(descrs)

    return df_descr
"""

def descriptor_df(dataframe):
    """
    dataframe: clean dataframe of the candidate molecules 

    The function calculate and add the the rdkit descriptors, normalize the standard values and the pIC50
    
    """ 
    
    dataframe_descriptors = pd.concat([dataframe, descriptors(dataframe.canonical_smiles)], axis = 1)
    dataframe_descriptors['standard_value'] = pd.to_numeric(dataframe_descriptors['standard_value'], errors='coerce')
    dataframe_descriptors = dataframe_descriptors.loc[pd.notna(dataframe_descriptors['standard_value'])]
    dataframe_descriptors = norm_value(dataframe_descriptors)
    dataframe_descriptors = pIC50(dataframe_descriptors)
    
    #dataframe_descriptors.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe_descriptors.dropna(inplace=True)
    dataframe_descriptors.reset_index(drop=True, inplace=True)
    
    return dataframe_descriptors

#### DEFINITION FUNCTION ####

def data_prep(dataframe):
    
    dataframe = dataframe.drop('molecule_chembl_id', axis=1)
    #dataframe = dataframe.drop('Bioactivity', axis=1)
    dataframe = dataframe.drop('canonical_smiles', axis=1)

    selection = VarianceThreshold(threshold=(.9 * (1 - .3)))

    X = dataframe.drop('pIC50', axis=1)
    X = selection.fit_transform(X)
    Y = dataframe['Bioactivity']

    X_train, X_test, Y_train, Y_test = data_split_scale(X,Y)

    return X_train, X_test, Y_train, Y_test

def data_split_scale(X,Y):
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

#### DEFINITION FUNCTION ####


def data_prep(dataframe):
    """
    dataframe: clean dataframe of the candidate molecules with the descriptors

    The function remove the 'molecule_chembl_id', 'pIC50' and 'canonical_smiles', it centers the labels around zero. It separates 
    the features in a 'X' matrix and the bioactivity label in a 'Y' vector. Also keep the features with a sufficient threshold and 
    then split the data into a training set and a test set.
    
    """ 

    
    dataframe = dataframe.drop('molecule_chembl_id', axis=1)
    dataframe = dataframe.drop('pIC50', axis=1)
    dataframe = dataframe.drop('canonical_smiles', axis=1)
    dataframe['Bioactivity'] = dataframe['Bioactivity'] - 1 


    selection = VarianceThreshold(threshold=(.9 * (1 - .3)))

    X = dataframe.drop('Bioactivity', axis=1)
    X = selection.fit_transform(X)
    Y = dataframe['Bioactivity']

    X_train, X_test, Y_train, Y_test = data_split_scale(X,Y)

    return X_train, X_test, Y_train, Y_test

def data_split_scale(X,Y):
    """
    X: features of the data
    Y: label of the data

    The function split the data into a test and a training set and scale the features on the training set.
    """
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test


def plot_multiclass_roc(y_test, y_score, n_classes):
    """
    y_test: true features of the test data
    y_score

    The function create a plot of the ROC curve for multiclass.
    """

    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'darkred'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def optimize_hyperparameters_random_search(dataframe, model_type):
    """
    dataframe: clean dataframe of the candidate molecules with the descriptors
    model_type: 'rf', 'svm', or 'fcn'

    The function does the data preparation and then trains the chosen model using a randomized search.
    The model is then evaluated using accuracy, precision, recall, F1 Score, and confusion matrix.
    The best parameter found are then returned.
    """

    X_train, X_test, y_train, y_test = data_prep(dataframe)
    np.random.seed(100)

    if model_type == 'rf':
        model = RandomForestClassifier()
        param_space = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
        }
    elif model_type == 'svm':
        model = SVC(probability=True)
        param_space = {
            'C': uniform(0.1, 10),
            'gamma': uniform(0.01, 1),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        }
    elif model_type == 'fcn':
        model = MLPClassifier(hidden_layer_sizes=(100,), solver='adam')
        param_space = {
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': uniform(0.0001, 0.9999),
            'learning_rate_init': uniform(0.001, 0.099),
        }
    else:
        raise ValueError("Invalid model_type. Choose 'rf', 'svm', or 'fcn'.")

    opt = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=25,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0,
        random_state=42,
    )

    opt.fit(X_train, y_train)
    best_params = opt.best_params_
    best_model = opt.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted')
    test_recall = recall_score(y_test, y_pred, average='weighted')
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


    return best_params, test_accuracy

def randomize_smiles(smiles):
    """
    Generate a randomized SMILES string
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, doRandom=True)

def augment_data(df, num_augments=5):
    augmented_data = []

    for idx, row in df.iterrows():
        smiles = row['canonical_smiles']
        for _ in range(num_augments):
            random_smiles = randomize_smiles(smiles)
            if random_smiles:
                augmented_row = row.copy()
                augmented_row['canonical_smiles'] = random_smiles
                augmented_data.append(augmented_row)

    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df

def plot_f1_scores(y_pred, y_test , n_classes):
    """
    y_pred: predicted label
    y_test: true label
    n_classes: number of classes
    """

    f1_scores = f1_score(y_test, y_pred, average=None)
    plt.figure()
    classes = list(range(n_classes))
    plt.bar(classes, f1_scores, color=['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'darkred'][:n_classes])
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for each class')
    plt.xticks(classes)
    plt.ylim([0.0, 1.0])
    plt.show()
