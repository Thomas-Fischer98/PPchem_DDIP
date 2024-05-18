import numpy as np
import seaborn as sns
import pandas as pd
import rdkit
import matplotlib.pyplot as plt

from rdkit import Chem 
from rdkit.Chem import Descriptors, Lipinski

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

selection = VarianceThreshold(threshold=(.8 * (1 - .8)))  

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

import keras_tuner as kt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import randint as sp_randint
from torch.utils.data import TensorDataset, DataLoader

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
    mols = []

    for i in smiles:
        molec = Chem.MolFromSmiles(i)
        mols.append(molec)
    descrs = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    df_descr = pd.DataFrame(descrs)

    return df_descr

"""""

def split_data(X, Y):
   
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    return X_train, X_test, Y_train, Y_test

def test_data(X,Y,model):

    Y_pred = model(X)
    mae = mean_absolute_error(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    r_squared = r2_score(Y, Y_pred)

def create_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=(X_train_scaled.shape[1],)))
    for i in range(hp.Int('n_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'units_layer_{i}', min_value=32, max_value=512, step=32),
                        activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_'+str(i), min_value=0.0, max_value=0.5, default=0.2, step=0.1)))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd']),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    
    return model

def train_model(X_train, y_train, X_val, y_val, input_size, hidden_size1, hidden_size2, num_epochs=100, lr=0.001):
    # Instantiate the model
    model = FNN(input_size, hidden_size1, hidden_size2)

    # Define loss function and optimizer for regression
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert data to PyTorch DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

    # Evaluate the model on the validation set
    with torch.no_grad():
        outputs = model(X_val)
        val_loss = criterion(outputs, y_val.unsqueeze(1))
        val_r2 = r2_score(y_val.numpy(), outputs.numpy())
    
    return val_loss.item(), val_r2


def optimize_hyperparameters(X, y, model_type):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Define the objective function to optimize
    def objective_function(params):
        model.set_params(**params)  # Set model hyperparameters
        model.fit(X_train, y_train)  # Fit the model
        y_pred = model.predict(X_test)  # Make predictions
        r2 = r2_score(y_test, y_pred)  # Calculate R-squared
        return -r2  # Minimize negative R-squared (maximize R-squared)

    if model_type == 'rf':
        model = RandomForestRegressor()
        param_space = {
            'n_estimators': (100, 500),
            'max_depth': (1, 50),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 20),
        }
    elif model_type == 'gbm':
        model = GradientBoostingRegressor()
        param_space = {
            'n_estimators': (100, 500),
            'learning_rate': (0.01, 0.5),
            'max_depth': (1, 10),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 20),
        }
    elif model_type == 'fcn':
        model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam')
        param_space = {
            'alpha': (0.0001, 0.01),
            'learning_rate_init': (0.001, 0.1),
        }
    else:
        raise ValueError("Invalid model_type. Choose 'rf', 'gbm', or 'fcn'.")

    # Initialize BayesianOptimization
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=50,  # Number of iterations
        cv=5,       # Cross-validation folds
        n_jobs=-1,  # Use all available CPU cores
        scoring='r2',  # Metric to optimize
        verbose=0,  # Verbosity level
        random_state=42,  # Random seed
    )

    # Perform hyperparameter optimization
    opt.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = opt.best_params_

    # Get the best estimator
    best_model = opt.best_estimator_

    # Evaluate the best model on test data
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    return best_params, test_r2

"""""