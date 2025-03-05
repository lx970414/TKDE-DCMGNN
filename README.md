# DCMGNN
This is for TKDE-submission:
> Dual-Channel Multiplex Graph Neural Networks for Recommendation

## Dependencies
Recent versions of the following packages for Python 3 are required:
* numpy==1.21.2
* torch==1.9.1 + cuda11.1
* scipy==1.7.1
* scikit-learn==0.24.2
* pandas==0.25.0
* tqdm==4.61.2

## Datasets
All of the datasets we use are publicly available datasets.
### Link
The used datasets are available at:
* Retail_Rocket https://tianchi.aliyun.com/competition/entrance/231719/information/
* Tmall https://tianchi.aliyun.com/dataset/dataDetail?dataId=649
* Yelp https://www.yelp.com/dataset/download

### Preprocess
We compress the data set into a mat format file, which includes the following contents.
* edges: array of subnetworks after coupling, each element in the array is a subnetwork.
* features: attributes of each node (user\item) in the network.
* labels: label of labeled points.
* train: index of training set points. 
* valid: index of validation set points.
* test: index of test set points.

In addition, we also sample the positive and negative edges in the network, and divide them into training, validation, and test fields for recommendations.

## Notice
First, you need to choose the specific dataset for recommendation tasks in `Recommendation.py`. Second, you need to modify the number of weights in `Model.py`. The number of weights should be the number of sub-networks after decoupling. Finally, you need to determine the sub-network and the number of sub-networks in `Decoupling_matrix_aggregation.py`. Commonly hyperparameters can be reset in `args.py`. `relation_chain.py` and `rc_aware_encoder.py` construct the relation chains and the relation chain-aware encoder in DCMGNN that captures the relation sequence, and the correlations and dependencies between different auxiliary relations and the target relations, respectively.
