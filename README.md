![Project Logo](assets/image_dipp.png)

![Coverage Status](assets/coverage-badge.svg)

<h1 align="center">
PPchem_DDIP
</h1>

<br>



## Overview

DDIP (Drug Developpment and pIC50 Prediction) is a programm based in a jupyter notebook format that can predict pIC50 values for different drugs acting on a target protein. To do so, different machine learning models (such as RandomForest, Gradient Boosting Machines (GBM) and Fully Connected Neural Network) were tested in order to have the best prediction on pIC50 values. These pIC50 values are then used to determine if a drug is bioactive on the target protein. 

## 👩‍💻 Installation

### Setup environment. 

To setup the enviromnent, you first you need to clone the repository on your local storage. 

```
git clone https://github.com/Thomas-Fischer98/PPchem_DDIP.git
```

You can create the environment. To do so, you need to go in the folder .github/workflow 
Then you can enter the following code line.

```
conda env create -f environment.yml 
```
Then you can activate the environment. 

```
conda activate myenv
```

If you need jupyter lab, install it 

```
(ppchem_ddip) $ pip install jupyterlab
```
### Install packages

To install the needed packages, you can enter the following command. 

```
pip install ppchem_ddip
```

## 🔥 Usage

```python
from mypackage import main_func

# One line to rule them all
result = main_func(data)
```

This usage example shows how to quickly leverage the package's main functionality with just one line of code (or a few lines of code). 
After importing the `main_func` (to be renamed by you), you simply pass in your `data` and get the `result` (this is just an example, your package might have other inputs and outputs). 
Short and sweet, but the real power lies in the detailed documentation.

## License

This project is licensed under the MIT license

## Authors

Kelian Gaedecke : https://github.com/KelianGaedecke
David Segura : https://github.com/Davidmingsegura
Thomas Fischer : https://github.com/Thomas-Fischer98

This project was created for the class "Practical Programming in Chemistry" at EPFL.



