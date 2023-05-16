# DMTS-PUL
This is a TENSORFLOW 2 implementation of the method proposed in [1] for Multivariate Time Series Classification With Positive and Unlabeled Data.
The main file is DeepPUL.py that implements the main procedure.

The Classification model is implemented in ResNet.py while the RNN AutoEncoder is implemented in BIRNNAE.py

In the DATA folder there three paris of files:

- POSITIVE DATA files:
          + P_30_X.npy: It contains 30 positive multivariate time series. The file size is (30, 23, 6). It contains 30 multivariate time series. Each time series has 23 timestamps and 6 variables
          + P_30_Y.npy: It contains the positive labels, one for each positive time series. The file size is (30,).
- UNLABELLED DATA files:
          + U_30_X.npy: It contains unlabelled multivariate time series that can belong to both positive and negative classes. The file size is (2098, 23, 6).
          + U_30_Y.npy: It contains the labels (positive or negative) associated to each unlabelled multivariate time series. This information is not used in the training stage but only to keep   
                        compute statistics about the behaviour of the model
- TEST DATA files:
          + T_30_X.npy: It contains the test multivaraite time series that can belong to both positive and negative classes. The file size is (2129, 23, 6).
          + T_30_Y.npy: It contains the labels (positive or negative) associated to each test multivariate time series. This information is used to evaluate the performances of the trained model.

DATA correspond to the Dordogne datasets used in [1]
The ResNet.py files is a modification of the code provided in [2]

To run the script on the data available in the DATA directory, the command is:

python DeepPUL.py DATA 30

The first argument (DATA) is the name of the directory in which data are available.
The second argument (30) is the number of positive samples available. This parameter is used to load the files (P/U/T)_30_(X/Y).npy


[1] Dino Ienco: A Deep Neural Network Framework for Multivariate Time Series Classification With Positive and Unlabeled Data. IEEE Access 11: 20877-20884 (2023)

[2] https://github.com/hfawaz/dl-4-tsc
