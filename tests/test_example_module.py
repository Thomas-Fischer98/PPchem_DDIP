import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from rdkit.Chem import Descriptors

from ppchem_ddip.functions import (
    lipinski1, lipinski, norm_value, pIC50, descriptors1, descriptors,
    data_cleaner, add_bioactivity, lipinski_df, descriptor_df, data_prep, data_split_scale, plot_multiclass_roc,
    randomize_smiles, augment_data, plot_f1_scores
)

# Mock data for testing
@pytest.fixture
def valid_smiles():
    return ["CCO", "CC", "CCC"]

@pytest.fixture
def invalid_smiles():
    return ["InvalidSMILES", "AnotherInvalid"]

@pytest.fixture
def dataframe():
    return pd.DataFrame({
        'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
        'canonical_smiles': ["CCO", "CC", "CCC"],
        'pIC50': [5000, 100000, 150000],
        'standard_value': [5000, 100000, 150000],
        'standard_value_norm': [5000, 100000, 150000]
    })

def test_lipinski1(valid_smiles):
    smiles = valid_smiles[0]
    result = lipinski1(smiles)
    assert isinstance(result, pd.DataFrame)
    assert "Mw" in result.columns
    assert "H donors" in result.columns
    assert "H acceptors" in result.columns
    assert "Log P" in result.columns

    with pytest.raises(ValueError):
        lipinski1("InvalidSMILES")
 
def test_lipinski(valid_smiles):
    result = lipinski(valid_smiles)
    assert isinstance(result, pd.DataFrame)
    assert "Mw" in result.columns
    assert "H donors" in result.columns
    assert "H acceptors" in result.columns
    assert "Log P" in result.columns

def test_norm_value(dataframe):
    normed_df = norm_value(dataframe)
    assert "standard_value_norm" in normed_df.columns
    assert "standard_value" not in normed_df.columns
    assert normed_df['standard_value_norm'].max() <= 100000000

def test_pIC50(dataframe):
    normed_df = norm_value(dataframe)
    result = pIC50(normed_df)
    assert "pIC50" in result.columns
    assert "standard_value_norm" not in result.columns

    dataframe_without_norm = dataframe.drop(columns=['standard_value_norm'], errors='ignore')
    
    assert 'standard_value_norm' not in dataframe_without_norm.columns
    
    with pytest.raises(ValueError):
        pIC50(dataframe_without_norm)

def test_descriptors1(valid_smiles):
    smiles = valid_smiles[0]
    result = descriptors1(smiles)
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) > 0

    with pytest.raises(ValueError):
        descriptors1("InvalidSMILES")

def test_descriptors(valid_smiles):
    result = descriptors(valid_smiles)
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) > 0

def test_data_cleaner(dataframe):
    cleaned_df = data_cleaner(dataframe)
    assert "molecule_chembl_id" in cleaned_df.columns
    assert "canonical_smiles" in cleaned_df.columns
    assert "standard_value" in cleaned_df.columns
    assert cleaned_df.isnull().sum().sum() == 0

def test_add_bioactivity(dataframe):
    bioactivity_df = add_bioactivity(dataframe)
    assert "Bioactivity" in bioactivity_df.columns
    assert all(bioactivity_df["Bioactivity"].isin([0, 1, 2]))

def test_lipinski_df(dataframe):
    result = lipinski_df(dataframe)
    assert isinstance(result, pd.DataFrame)
    assert "pIC50" in result.columns
    assert "standard_value_norm" not in result.columns

def test_descriptor_df(dataframe):
    result = descriptor_df(dataframe)
    assert isinstance(result, pd.DataFrame)
    assert "pIC50" in result.columns
    assert "standard_value_norm" not in result.columns

def test_data_prep(dataframe):
    bioactivity_df = add_bioactivity(dataframe)
    X_train, X_test, Y_train, Y_test = data_prep(bioactivity_df)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert Y_train.shape[0] > 0
    assert Y_test.shape[0] > 0

def test_data_split_scale():
    np.random.seed(42)
    num_samples = 100
    num_features = 10
    X = np.random.rand(num_samples, num_features)
    Y = np.random.randint(0, 2, size=num_samples)

    X_train, X_test, Y_train, Y_test = data_split_scale(X, Y)

    assert X_train.shape[0] == int(0.8 * num_samples)
    assert X_test.shape[0] == int(0.2 * num_samples)
    assert X_train.shape[1] == num_features
    assert X_test.shape[1] == num_features
    assert len(Y_train) == int(0.8 * num_samples)
    assert len(Y_test) == int(0.2 * num_samples)

    assert np.allclose(np.mean(X_train, axis=0), 0)
    assert np.allclose(np.std(X_train, axis=0), 1)  


def test_plot_multiclass_roc():
    n_classes = 5
    y_test = np.random.randint(0, n_classes, size=100)
    y_score = np.random.rand(100, n_classes)
    plot_multiclass_roc(y_test, y_score, n_classes)

def test_optimize_hyperparameters_random_search():
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_classes = 3)
    dataframe = pd.DataFrame(X)
    dataframe['target'] = y
    model_type = 'rf'
    best_params, test_accuracy = test_optimize_hyperparameters_random_search(dataframe, model_type)
    assert isinstance(best_params, dict)
    assert isinstance(test_accuracy, float)

def test_randomize_smiles():
    smiles = 'CCO'
    randomized_smiles = randomize_smiles(smiles)
    assert isinstance(randomized_smiles, str)

def test_augment_data():
    np.random.seed(42)
    num_samples = 100
    df = pd.DataFrame({'canonical_smiles': ['CCO']*num_samples})
    augmented_df = augment_data(df)
    assert augmented_df.shape[0] == num_samples*5

def test_plot_f1_scores():
    n_classes = 5
    y_test = np.random.randint(0, n_classes, size=100)
    y_pred = np.random.randint(0,n_classes, size=100)
    plot_f1_scores(y_pred, y_test, n_classes)





if __name__ == "__main__":
    pytest.main()