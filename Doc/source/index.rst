.. Project for Chemistry documentation master file, created by
   sphinx-quickstart on Sun May  5 11:11:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Drug Discovery and :math:`pIC_{50}` Prediction (DDIP) program !
=================================================


.. toctree::
   :maxdepth: 2

   usage/installation
   usage/quickstart

   intro
   strings
   datatypes
   numeric
   SMILES

 :caption: Contents:
   




Introduction 
------------

This page is here to help users go through our program. This program was made by David Segura, Kelian Gaedecke and Thomas Fischer for the class "Practical Programming in Chemistry" at EPFL. 

Indices and tables
------------------

* :doc:`SMILES`
* :doc:`IC50`

Our Program
-----------
Our program can calculate :math:`pIC_{50}` (logarithm of :math:`IC_{50}`) for different drugs on mTOR (mechanistic target of rapamycin). 
By inserting the SMILES of the molecule of interest, :math:`pIC_{50}` values can be aquired. 
Using this value, it's then possible to determine if the drug is bioactive on the target protein.

To do so, we have implemeted three different machine learning models such as **RandomForest**, **Gradient Boosting Machines** (GBMs) and **Fully Connected Neural Network**. 
:math:`pIC_{50}` predictions are made using descriptors and fingerprints from rdkit. 
DDIP program was also adapted as an application that people can use. To get to the application, follow the steps explained in the **How to use it** section below.

How to use it
-------------

In order to use the application, you need to clone the DDIP repository. To do so, you will need to enter certain lines of code in your terminal. 
All these installation steps are explained in the README file of the project. You can access the file via the following link : https://github.com/Thomas-Fischer98/PPchem_DDIP.git
This section will serve to show users how to get access to the application, how to run and use it. 

To access the application, the following steps are needed. 

When the environment was setup as explained in the README file mentionned above, the application can be opened with : 

- streamlit run app.py &dev/null/&

Once the application opened, the SMILES of a molecule can be either entered or the molecule can be drawn. 
Once one of the two input is completed, the application output will tell you if the drug inserted is bioactive or not for mTOR. 