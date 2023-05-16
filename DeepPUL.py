'''
  Copyright (C) 2023 Dino Ienco

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; see the file COPYING. If not, write to the
  Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
'''

import numpy as np
import tensorflow as tf
import os
import sys
from sklearn.metrics import f1_score, r2_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as pyplot
from ResNet import Classifier_RESNET
from scipy.stats import entropy

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from BIRNNAE import RNNAE

def getBestElementsByClass(pred, unlabeled, toSelect, classVal=1 ):
	hardLabel = np.argmax(pred,axis=1)
	idx = np.where(hardLabel == classVal)[0]
	subset_examples = unlabeled[idx]
	subset_pred = pred[idx]

	entro = entropy( subset_pred, base=2.0, axis=1)
	new_idx = np.argsort(entro)
	return subset_examples[new_idx][0:toSelect]



def IPL(model, positive, unlabeled, incrSamples, nIterations, OUTPUT_FOLDER, test_data, n_epochs_supervised):
	pred = model.predict(unlabeled)
	nClasses = 2
	tempModel = None
	newPositive = None
	for i in range(nIterations):
		addPositive = getBestElementsByClass(pred, unlabeled, incrSamples*(i+1), classVal=1 )
		negative = getBestElementsByClass(pred, unlabeled, positive.shape[0]+(incrSamples*(i+1)), classVal=0 )
		tempModel = Classifier_RESNET(positive.shape[1::], nClasses, verbose=False)
		newPositive = np.concatenate((positive, addPositive),axis=0)
		pos_labels = np.ones((newPositive.shape[0]))
		neg_labels = np.zeros((negative.shape[0]))
		new_data = np.concatenate((newPositive, negative),axis=0)
		new_labels = np.concatenate((pos_labels,neg_labels))
		tempModel.fit(new_data, tf.keras.utils.to_categorical(new_labels), nb_epochs=n_epochs_supervised)
		tempModel.model.save_weights(dirName+"/RESNET_"+str(i+1))
		pred_c = tempModel.predict(test_data)
		np.save(dirName+"/results_"+str(i+1)+".npy", pred_c)
		fscore = f1_score(test_labels, np.argmax(pred_c,axis=1), average="weighted")

	return tempModel

def getRNExample(rec_error_unl, unlabeled_data, n_examples):
	idx = np.argsort(rec_error_unl)[::-1]
	reliable_negative = unlabeled_data[idx][0:n_examples]
	return reliable_negative


def getDistrib(oneH, treeH, zeroH, initialPos):
	lOneH = np.ones(oneH.shape) * 0
	lTreeH = np.ones(treeH.shape)
	lZeroH = np.ones(zeroH.shape) * 2
	pos = np.stack((oneH, lOneH),axis=1)
	pos_unl = np.stack((treeH, lTreeH),axis=1)
	neg_unl = np.stack((zeroH, lZeroH),axis=1)
	tot =np.concatenate((pos, pos_unl, neg_unl),axis=0)
	tot = tot[tot[:,0].argsort()[::-1]]
	subtot = tot[0:initialPos]
	subtot = subtot[:,1]
	return np.bincount(subtot.astype("int"))


def recErrors(x, rec):
	vals = np.sum( np.square(x - rec), axis=1 )
	return vals

def plotDistrib(hist_a, hist_b, hist_c, bin_edges):
	x_axis = np.array(range(len(hist_a)))
	pyplot.bar(x_axis-0.2, hist_a,width=0.2,color="r",align="center")
	pyplot.bar(x_axis, hist_b,width=0.2,color="b",align="center")
	pyplot.bar(x_axis+0.2, hist_c,width=0.2,color="g",align="center")
	pyplot.draw()
	pyplot.pause(0.1)
	pyplot.clf()


def computeRecoDistrib(x, rec, bin_input=None):
	x = [el.flatten() for el in x]
	x = np.array(x)

	rec = [el.flatten() for el in rec]
	rec = np.array(rec)
	hist = None
	bin_edges = None
	distrib = np.sum(np.abs(x - rec), axis=1)
	if bin_input is None:
		hist, bin_edges = np.histogram(distrib, bins=10, density=False)
	else:
		hist, bin_edges = np.histogram(distrib, bins=bin_input, density=False)
		bigger = [el for el in distrib if el > bin_edges[-1]]
		hist[-1] = hist[-1] + len(bigger)
		smaller = [el for el in distrib if el < bin_edges[0]]
		hist[0] = hist[0] + len(smaller)
	return hist, bin_edges

def plot2DFeatures(features_t, features_s):
	feat_val = 100
	features_t = shuffle(features_t)
	features_s = shuffle(features_s)
	sub_t = features_t[0:feat_val]
	sub_s = features_s[0:feat_val]
	X_embedded = TSNE(n_components=2).fit_transform( np.concatenate([sub_t,sub_s], axis=0))
	pyplot.scatter(X_embedded[0:feat_val,0], X_embedded[0:feat_val,1],marker='o')
	pyplot.scatter(X_embedded[feat_val::,0], X_embedded[feat_val::,1],marker='+')
	pyplot.draw()
	#pyplot.savefig("images/"+str(e)+".png")
	pyplot.pause(0.1)
	pyplot.clf()


def getBatch(X, i, batch_size):
	start_id = i*batch_size
	t = (i+1) * batch_size
	end_id = min( (i+1) * batch_size, X.shape[0])
	batch_x = X[start_id:end_id]
	return batch_x


def trainClassif(model, x_train, loss_object, optimizer, BATCH_SIZE, e, rad_constr=False, centroid=None):
	loss_iteration = 0
	tot_loss = 0.0
	iterations = x_train.shape[0] / BATCH_SIZE
	if x_train.shape[0] % BATCH_SIZE != 0:
		iterations += 1
	for ibatch in range(int(iterations)):
		batch_x = getBatch(x_train, ibatch, BATCH_SIZE)
		batch_x_corrupted = batch_x * np.random.normal(loc=1.0, scale=0.05, size=batch_x.shape)
		with tf.GradientTape() as tape:
			estimation, estimationR, emb = model(batch_x_corrupted, training=True)
			loss_rec = loss_object(batch_x, estimation)
			grads = tape.gradient(loss_rec, model.trainable_variables, 
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			tot_loss+=loss_rec
	return (tot_loss / iterations)


def trainRNNAE(model, positiveData, loss_huber, optimizer, BATCH_SIZE, n_epochs):
	for e in range(n_epochs):
		positiveData = shuffle(positiveData)
		trainLoss = trainClassif(model, positiveData, loss_huber, optimizer, BATCH_SIZE, e)
	return model


###### INPUT INFORMATION ##########
# DATA DIRECTORY that CONTAINS THE DATA FILE
dataDir = sys.argv[1]
# PREFIX OF THE DATA FILE CONTAINING THE POSITIVE DATA. THE NUMBER INDICATES HOW MANY POSITIVE DATA ARE AVAILABLE IN THE NUMPY FILE
initialPos = int(sys.argv[2])

###### HARD CODED VALUES FOR THE TRAINING STAGE AS WELL AS TO SAVE THE RESULTS/MODELS ##########
BATCH_SIZE = 8
n_epochs = 300
n_epochs_supervised = 300
quantile_threshold = 0.9
nClasses = 2
OUTPUT_FOLDER = "OUTPUT"


#### HARD CODED VALUES FOR THE INCREMENTAL PSEUDO LABELING PROCEDURE ######
incrSamples = 30
nIterations = 7



positive = np.load(dataDir+"/P_"+str(initialPos)+"_X.npy")
positive_labels = np.load(dataDir+"/P_"+str(initialPos)+"_Y.npy")

unlabeled = np.load(dataDir+"/U_"+str(initialPos)+"_X.npy")
unlabeled_labels = np.load(dataDir+"/U_"+str(initialPos)+"_Y.npy")

test = np.load(dataDir+"/T_"+str(initialPos)+"_X.npy")
test_labels = np.load(dataDir+"/T_"+str(initialPos)+"_Y.npy")

idx = np.where(unlabeled_labels==1)
unl_pos = unlabeled[idx[0]]

idx = np.where(unlabeled_labels==0)
unl_neg = unlabeled[idx[0]]

dirName = dataDir+"/"+str(initialPos)
if not os.path.exists(dirName):
    os.makedirs(dirName)


RNNAE_model = RNNAE(128, unl_neg.shape[-1], dropout_rate=0.2)
loss_huber = tf.keras.losses.Huber()
loss_object2 = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.005)



#Build a model for the positive data
print("TRAINING RNN-AE")
RNNAE_model = trainRNNAE(RNNAE_model, positive, loss_huber, optimizer, BATCH_SIZE, n_epochs)
RNNAE_model.save_weights(dirName+"/RNNAE")

print("END TRAINING RNN-AE")
#COMPUTE RECONSTRUCTION STATISTICS
rec_pos, _, _ = RNNAE_model.predict(positive)
rec_unl, _, _ = RNNAE_model.predict(unlabeled)
rec_error_pos = np.sum(loss_object2(positive, rec_pos),axis=1)
rec_error_unl = np.sum(loss_object2(unlabeled, rec_unl), axis=1)
thr = np.quantile(rec_error_pos, quantile_threshold)
#EXTRACT Reliable Negative Examples
print("GET Reliable Negative Examples")
rn_examples = getRNExample(rec_error_unl, unlabeled, initialPos)
pos_labels = np.ones(positive.shape[0])
neg_labels = np.zeros(rn_examples.shape[0])

#BUILD (initial) BINARY TASK dataset
new_data = np.concatenate((positive, rn_examples),axis=0)
new_labels = np.concatenate((pos_labels, neg_labels),axis=0)
print("Train RESNET classifier for binary task")


print("new_data ",new_data.shape)
print("new_labels ",new_labels.shape)
model = Classifier_RESNET(new_data.shape[1::], nClasses, verbose=False)
model.fit(new_data, tf.keras.utils.to_categorical(new_labels), nb_epochs=n_epochs_supervised)
model.model.save_weights(dirName+"/RESNET_0")
pred = model.predict(test)
np.save(dirName+"/results_0.npy", pred)
########ITERATIVE PSEUDO LABELING###########
print("START IPL PROCEDURE")

model = IPL(model, positive, unlabeled, incrSamples, nIterations, dirName, test, n_epochs_supervised)

pred = model.predict(test)
fscore = f1_score(test_labels, np.argmax(pred,axis=1), average="weighted")
print("F-score on TEST data %f" % fscore)