# !/usr/bin/python
# _*_ coding:utf8 _*_

'''
description: deep learning for CNN used for recognizing optical remote
sensing images
code data: 2018/06/14
modified data:2018/07/26
athor: TTang
'''

# switch to Theano or Tensorflow
import os
os.environ['KERAS_BACKEND']='theano'

# import some libs
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,Dense,Activation,Flatten,Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
import scipy.io as sio
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def decode_predictions_custom(preds, top=5):
    CLASS_CUSTOM = ["0","1","2","3","4","5","6","7","8","9"]
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        # result = [tuple(CLASS_CUSTOM[i]) + (pred[i]*100,) for i in top_indices]
        result = [tuple(CLASS_CUSTOM[i]) + (pred[i]*100,) for i in top_indices]
        results.append(result)
    return results

def start(K_fold):

	# initialize parameters
	seed=7
	np.random.seed(seed)
	nb_classes=7

	# load dataset
	X=np.load(os.getcwd()+'/datasets/X.npy')
	Y=np.load(os.getcwd()+'/datasets/Y.npy')

	# define 5-fold cross validation test harness
	skf=StratifiedKFold(n_splits=k_fold,shuffle=True,random_state=seed)
	splitted_indices=skf.split(X,Y)
	cvscores=[]

	for index,(train_indices,val_indices) in enumerate(splitted_indices):
		i=index+1
		print('This is the %dth training'%(i))
		nb_val=val_indices.size
		X_train=X[train_indices].astype('float64')/255
		X_val=X_train[0:nb_val,:,:,:]
		Y_train=Y[train_indices]
		Y_val=Y_train[0:nb_val,:]
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
		Y_val=np_utils.to_categorical(Y_val,nb_classes)

		# build neural net
		print("Building neural net......")
		model=Sequential()

		# set 1st convolution layer
		model.add(Convolution2D(
			filters=32,
			kernel_size=(2,2),
			padding='same', 
			input_shape=X_train.shape[1:]
			))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(
			pool_size=(2,2),
			strides=(2,2),
			padding='same'
			))
		model.add(Dropout(0.25))

		# set 2nd convolution layer
		model.add(Convolution2D(
			filters=64,
			kernel_size=(2,2),
			padding='same', 
			input_shape=(1,28,28)
			))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(
			pool_size=(2,2),
			strides=(2,2),
			padding='same'
			))
		model.add(Dropout(0.25))

		# set dense layer
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.25))
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		# set optimizier
		adam=Adam(lr=1e-4)

		# compile neural net
		model.compile(
			loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy']
			)

		# train neural net
		print("Training.....")
		history=model.fit(X_train,Y_train,epochs=100,batch_size=20)

		# plot the loss and accuracy of training
		fig=plt.figure(i)
		plt.plot(range(0,len(history.history['loss'])),(history.history['loss']))
		plt.plot(range(0,len(history.history['loss'])),(history.history['acc']))
		plt.xlabel('Epochs')
		plt.ylabel('Loss and Accuracy')
		plt.title('Loss and accuracy of training')
		plt.legend(['Loss','Acc'],loc='center right')
		# plt.axis([0,len((history.history['loss'])),0,max((history.history['loss']))])
		plt.grid(True)
		# plt.show()
		fig.savefig(os.getcwd()+'/'+'performance_images/'+'optical_recognition_performance_'+str(i)+'.png')

		print("Test......")
		accuracy=model.evaluate(X_val,Y_val,batch_size=100)
		print("The accuracy is : %.2f%%" % (accuracy[1]*100))
		cvscores.append(accuracy[1]*100)
		model.save(os.getcwd()+'/models/model_IR_recognition_'+str(i)+'.h5')

	print("The mean accuracy and std are : %.2f%% (+/- %.2f%%)" % (np.mean(cvscores),np.std(cvscores)))

	# save trained model
	model.save(os.getcwd()+'/models/model_optical_recognition.h5')
	print('Finishing saving model')

	# plot the structure of net
	plot_model(
		model,show_shapes=True,show_layer_names=True,
		to_file=os.getcwd()+'/structure_optical_recognition_model.png'
		)

if __name__=='__main__':
	k_fold=10
	start(k_fold)	