This is the code repository for the graph representation learning for the relationship between brain structural and functional networks, as described in our paper:

Yang Li, Gonzalo Mateos and Zhengwu Zhang, Graph Representation Learning for Modeling the Relationship between Structural and Functional Brain Connectivities. 

Please refer to the paper regarding any details of the implementation of the code.

### Scheme of the proposed Graph Representation Learning framework

![Alt text](https://github.com/yli131/brainGRL/blob/f4a25b445edc28c6350dea9a69cab1496f0da03e/scheme.png)

==========================================================

#### Step 1: Load and prepare data

Code to run: step1_load_data.py

The data used in this work is the structural connectivity (SC) network and the functional connectivity (FC) network of altogether 1058 subjects from the Human Connectome Project. Each brain network is a 68x68 matrix that represents the connections between 68 brain regions-of-interest. The code script is used to read the network data from csv files. If user data comes in different format, the code needs to be modified accordingly. The data needs to be put in a separate folder which will be visited by this code script.

All the processed and prepared data will be saved in the 'data' folder for later use.

==========================================================

#### Step 2: Low resolution search for the optimal model architecture

Code to run: step2_main_SC_2_FC.py

This script is used to determine the optimal graph convolutional network (GCN) architecture for the problem solved in this work. 10-fold cross validation is carried out and the \lambda is set to be 0.1 as a temporary tuning paramter between the FC reconstruction and subject-level classification. For each choice of model architecture, the results including mean squared error for FC regression, F1-score for subject classification and etc. across all 10 folds will be automatically saved.

==========================================================

#### Step 3: Re-format the results in Step 2

Code to run: step3_post_analysis_SC_2_FC.py

The major usage of this script is to re-organize the results from Step 2 due to large file size.


