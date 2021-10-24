This is the code repository for the graph representation learning for the relationship between brain structural and functional networks, as described in our paper:

Yang Li, Gonzalo Mateos and Zhengwu Zhang, Graph Representation Learning for Modeling the Relationship between Structural and Functional Brain Connectivities. 

Please refer to the paper regarding any details of the implementation of the code.

### Scheme of the proposed Graph Representation Learning framework

![Alt text](https://github.com/yli131/brainGRL/blob/f4a25b445edc28c6350dea9a69cab1496f0da03e/scheme.png)

#### Step 1: Load and prepare data

The data used in this work is the structural connectivity (SC) network and the functional connectivity (FC) network of altogether 1058 subjects from the Human Connectome Project. Each brain network is a 68x68 matrix that represents the connections between 68 brain regions-of-interest. The code script is used to read the network data from csv files. If user data comes in different format, the code needs to be modified accordingly. The data needs to be put in a separate folder which will be visited by this code script.


