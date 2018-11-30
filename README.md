# Credit-Card-Fraud-Detection
Recognize fraudulent credit card transactions

   It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

Throughout the financial sector, machine learning algorithms are being developed to detect fraudulent transactions. In this project, that is exactly what we are going to be doing as well. Using a dataset of of nearly 28,500 credit card transactions and multiple unsupervised anomaly detection algorithms, I have identified transactions with a high probability of being credit card fraud. In this project, I built and deployed the following two machine learning algorithms:

Local Outlier Factor (LOF)
Isolation Forest Algorithm
Furthermore, using metrics suchs as precision, recall, and F1-scores, I have investigated why the classification accuracy for these algorithms can be misleading.

In addition, I have explored the use of data visualization techniques common in data science, such as parameter histograms and correlation matrices, to gain a better understanding of the underlying distribution of data in our data set.

## 1. Importing Necessary Libraries:

1. Numpy
2. Pandas
3. Matplotlib
4. Seaborn
5. Scipy

## 2. The Data Set

I have imported our dataset from a .csv file as a Pandas DataFrame. Furthermore, I began exploring the dataset to gain an understanding of the type, quantity, and distribution of data in our dataset. For this purpose, I have used Pandas' built-in describe feature, as well as parameter histograms and a correlation matrix.

### Columns:

1) Time- Number of seconds elapsed between this transaction and the first transaction in the dataset

2) V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features

3) Amount - Transaction Amount

4) Class- 1 for fraudulent transactions, 0 otherwise

## 3. Unsupervised Outlier Detection

After processing our data, deploy our machine learning algorithms.

##### Local Outlier Factor (LOF)

The anomaly score of each sample is called Local Outlier Factor. It measures the local deviation of density of a given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood.

##### Isolation Forest Algorithm

The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.

This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.

Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.

   
   Link for the Dataset--->[creditcard.csv](https://www.kaggle.com/samkirkiles/credit-card-fraud/data)
   
  
