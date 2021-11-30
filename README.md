This is the code repository for the graph representation learning for the relationship between brain structural and functional networks, as described in our paper:

Yang Li, Gonzalo Mateos and Zhengwu Zhang, Learning to Model the Relationship Between Structural and Functional Brain Connectomes. 

Please refer to the paper regarding any details of the implementation of the code.

### Scheme of the proposed Graph Representation Learning framework

![Alt text](https://github.com/yli131/brainGRL/blob/a30079007b4fecebc438f7959bf0e25a5d73e66f/scheme3.png)
==========================================================

#### Step 1: Load and prepare data

Code to run: step1_load_data.py

The data used in this work is the structural connectivity (SC) network and the functional connectivity (FC) network of altogether 1058 subjects from the Human Connectome Project. Each brain network is a 68x68 matrix that represents the connections between 68 brain regions-of-interest. The code script is used to read the network data from csv files. If user data comes in different format, the code needs to be modified accordingly. The data needs to be put in a separate folder which will be visited by this code script.

All the processed and prepared data will be saved in the 'data' folder for later use.

==========================================================

#### Step 2: Low resolution search for the optimal model architecture

Code to run: step2_main_SC_2_FC.py

This script is used to determine the optimal graph convolutional network (GCN) architecture for the problem solved in this work. 10-fold cross validation is carried out and the λ is set to be 0.1 as a temporary tuning paramter between the FC reconstruction and subject-level classification. For each choice of model architecture, the results including mean squared error for FC regression, F1-score for subject classification and etc. across all 10 folds will be automatically saved.

==========================================================

#### Step 3: Re-format the results in Step 2

Code to run: step3_post_analysis_SC_2_FC.py

The major usage of this script is to re-organize the results from Step 2 due to large file size.

==========================================================

#### Step 4: Analysis of the results and identifying optimal model architectures

Code to run: step4_post_analysis2_SC_2_FC.py

Comprehensive investigatation of the results from the 10-fold experiments w.r.t each model architecture. Plot the results regarding FC reconstruction MSE, subject classification precision, recall and F1-score, together with the training time. Introduce a criteria to compare the performance of all the architectures trained and determine 3 models with top performance for the high-resolution λ search in the next step.

==========================================================

#### Step 5: λ search

Code to run: step5_main_lambda_search.py

Do an extensive search by testing the λ values within a range on the same 10 folds used previously of the 3 selected models in the previous step.

==========================================================

#### Step 6: Finalize model architecture and parameter setting

Code to run: step6_main_lambda_search2.py

With 3 selected model architecutures and several λ values, introduce another criteria to determine the one and only optimal model architecture and the value of λ.

### Contact

Any questions regarding the code, please refer to Yang Li at yli131@ur.rochester.edu

### REFERENCE

Please cite our papers if you use this code in your own work.
