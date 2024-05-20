![Project Logo](image_dipp.png)

![Coverage Status](assets/coverage-badge.svg)

<h1 align="center">
PPchem_DDIP
</h1>

<br>



## Overview

DDIP (Drug Developpment and pIC50 Prediction) is a programm based in a jupyter notebook format that can predict pIC50 values for different drugs acting on a target protein. To do so, different machine learning models (such as Forest, ... and ...) were tested in order to have the best prediction on pIC50 values. 


## 🔥 Usage

```python
from mypackage import main_func

# One line to rule them all
result = main_func(data)
```

This usage example shows how to quickly leverage the package's main functionality with just one line of code (or a few lines of code). 
After importing the `main_func` (to be renamed by you), you simply pass in your `data` and get the `result` (this is just an example, your package might have other inputs and outputs). 
Short and sweet, but the real power lies in the detailed documentation.

## 👩‍💻 Installation

Create a new environment, you may also give the environment a different name. 

```
conda create -n ppchem_ddip python=3.10 
```

```
conda activate ppchem_ddip
(conda_env) $ pip install .
```

If you need jupyter lab, install it 

```
(ppchem_ddip) $ pip install jupyterlab
```


## 🛠️ Development installation

Initialize Git (only for the first time). 

Note: You should have create an empty repository on `https://github.com:Thomas-Fischer98/PPchem_DDIP`.

```
git init
git add * 
git add .*
git commit -m "Initial commit" 
git branch -M main
git remote add origin git@github.com:Thomas-Fischer98/PPchem_DDIP.git 
git push -u origin main
```

Then add and commit changes as usual. 

To install the package, run

```
(ppchem_ddip) $ pip install -e ".[test,doc]"
```

### Run tests and coverage

```
(conda_env) $ pip install tox
(conda_env) $ tox
```

## Authors

Kelian Gaedecke :
David Segura :
Thomas Fischer :

This project was created for the class "Practical Programming in Chemistry" at EPFL.



