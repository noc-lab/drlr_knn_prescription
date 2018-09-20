# DRLR kNN prescription model

This repo contains codes for DRLR kNN prescription models.

## Environment

python package ``dist_robust_regress`` is used for fitting DRLR and must be included.

``requirements.txt`` is provided for set up the python environment. By default the python environment in BMC server is good. 

The multi-threading computing is disabled by default. One can enable it once determined how much computation resource should be used.  

## To run the code

All codes that are used for generate the prescription model are in folder ``examples`` and some helper functions 
and frequently used functions are defined in ``drlr_knn_prescription`` package


In details,

a. The **preprocessed** data is store in the BMC server and the scripts will automatically locate them once run in the BMC
server. This repo dose not contain the code used for preprocessing. Detailed path can be found in the script
``drlr_knn_prescription/load_table.py``. The corresponding function will be called automatically
 and there is no need to run this script. 
 
b. the model and the result will be generated in the folder specified by the flag ``--save_dir``. The detailed name of
the output can be found at scripts in the examples.

c. the flags of the scripts are defined in the next section.

Before running the code:

1. clone this repo

2. append the path to python package ``dist_robust_regress`` to the environment variable ``PYTHONPATH``

The order of the codes to run

1. run scripts ``drlr_knn_prescription.py``, ``ols_knn_prescription.py`` and ``lasso_and_cart.py`` that are used to 
generate  the prescription model. The hyperparameter are defined in the corresponding scripts. Running in any order is fine.

1. run ``evaluate_performance.py`` is used to evaluate the performance of the model after collecting all the prescription
 model.
  
Optionally, run ``best_prescription_epsilon.py`` servers are a helper script for determine which the soften factor of the randomize
prescription rules.



## Flags

All scripts use the same set of flags

1. ``--trial``: ID of the trial, used for generating multiple trial to evaluate the average performance.

1. ``--test_ratio``: proportion of the examples that used for as the test dataset that evaluate the prescription performance

1. ``--save_dir``: folder for the model checkpoints one want to save. For different test ratio and different datasets, it is better to set different folders.

1. ``--diabetes``: load diabetes dataset if it is ``True`` and load hypertension dataset if it is ``False``
