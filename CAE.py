#DCGAN Using Lasagne (for CelebA)
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, flatten, reshape, batch_norm, Upscale2DLayer
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import sigmoid
from lasagne.layers import get_output, get_all_params, get_output_shape, get_all_layers
from lasagne.objectives import binary_crossentropy as bce
from lasagne.objectives import squared_error
from lasagne.updates import adam

import numpy as np
import theano
from theano import tensor as T
import time
from matplotlib import pyplot as plt 

from skimage.io import imsave

floatX=theano.config.floatX

def get_args():
	print 'getting args...'

def save_args():
	print 'saving args...'

def build_net(nz=100):
	# nz = size of latent code
	#N.B. using batch_norm applies bn before non-linearity!
	F=32
	enc = InputLayer(shape=(None,1,28,28))
	enc = Conv2DLayer(incoming=enc, num_filters=F*2, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)
	enc = Conv2DLayer(incoming=enc, num_filters=F*4, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)
	enc = Conv2DLayer(incoming=enc, num_filters=F*4, filter_size=5,stride=1, nonlinearity=lrelu(0.2),pad=2)
	enc = reshape(incoming=enc, shape=(-1,F*4*7*7))
	enc = DenseLayer(incoming=enc, num_units=nz, nonlinearity=sigmoid)
	#Generator networks
	dec = InputLayer(shape=(None,nz))
	dec = DenseLayer(incoming=dec, num_units=F*4*7*7)
	dec = reshape(incoming=dec, shape=(-1,F*4,7,7))
	dec = Deconv2DLayer(incoming=dec, num_filters=F*4, filter_size=4, stride=2, nonlinearity=relu, crop=1)
	dec = Deconv2DLayer(incoming=dec, num_filters=F*4, filter_size=4, stride=2, nonlinearity=relu, crop=1)
	dec = Deconv2DLayer(incoming=dec, num_filters=1, filter_size=3, stride=1, nonlinearity=sigmoid, crop=1)

	return enc, dec



def prep_train(alpha=0.0002, nz=100):
	E,D=build_net(nz=nz)

	x = T.tensor4('x')

	#Get outputs z=E(x), x_hat=D(z)
	encoding = get_output(E,x)
	decoding = get_output(D,encoding)

	#Get parameters of G and D
	params_e=get_all_params(E, trainable=True)
	params_d=get_all_params(D, trainable=True)
	params = params_e + params_d

	#Calc cost and updates
	cost = T.mean(squared_error(x,decoding))
	grad=T.grad(cost,params)

	updates = adam(grad,params, learning_rate=alpha)

	train = theano.function(inputs=[x], outputs=cost, updates=updates)
	rec = theano.function(inputs=[x], outputs=decoding)
	test = theano.function(inputs=[x], outputs=cost)


	return train ,test, rec, E, D

def train(trainData,testData, nz=100, alpha=0.001, batchSize=64, epoch=1):
	train, test, rec, E, D = prep_train(nz=nz, alpha=alpha)
	print np.shape(trainData)
	sn,sc,sx,sy=np.shape(trainData)
	print sn,sc,sx,sy
	batches=int(np.floor(float(sn)/batchSize))

	#keep training info
	trainCost_=[]
	testCost_=[]
	print 'batches=',batches

	timer=time.time()
	#Train D (outerloop)
	print 'epoch \t batch \t train cost \t\t test cost \t\t time (s)'
	for e in range(epoch):
		#random re-order of data (no doing for now cause slow)
		#Do for all batches

		try:

			for b in range(batches):
				trainCost=train(trainData[b*batchSize:(b+1)*batchSize])
				testCost=test(testData[:100])
				print e,'\t',b,'\t',trainCost,'\t',testCost,'\t', time.time()-timer
				timer=time.time()
				trainCost_.append(trainCost)
				testCost_.append(testCost)
		except KeyboardInterrupt:
			print 'press cntl-c for each epoch that is left'

	#save plot of the cost
	plt.plot(trainCost_, label="train")
	plt.plot(testCost_, label="test")
	plt.legend()
	plt.xlabel('iter')
	plt.savefig('cost_regular.png')

	return test, rec, E, D

def test(x, rec):
	return rec(x)

def load_data():
	dataDir = '/home/ac2211/Documents/PhD_Yr1/Projects/Project17_DAAE/InData/mnist.pkl'  #/data/datasets/MNIST/mnist.pkl
	train,test,val = np.load(dataDir,mmap_mode='r')

	return train[0].reshape(-1,1,28,28).astype(floatX), train[1], test[0].reshape(-1,1,28,28).astype(floatX), test[1], val[0].astype(floatX), val[1]


enc, dec = build_net()
for l in get_all_layers(enc):
	print get_output_shape(l)
for l in get_all_layers(dec):
	print get_output_shape(l)


x_train, _,x_test,_,_,_=load_data()
test, rec, E, D =train(x_train, x_test)


#see if the output images look good:
#Save example reconstructions
REC = rec(x_test[:10])
print np.shape(REC), np.shape(x_test[:10])
try:
	test_rec = np.mean((REC - x_test[:rec.shape[0]])**2)
	print 'Mean Test Reconstruction Error:', test_rec  #may no rec all
except:
	print 'error calc mse'

#Save generated samples (by category)
fig=plt.figure()
newDir=''
montageRow1 = np.hstack(x_test[:10].reshape(-1,28,28))
montageRow2 = np.hstack(REC[:10].reshape(-1,28,28))
montage = np.vstack((montageRow1, montageRow2))
plt.imshow(montage, cmap='gray')
plt.savefig('rec.png')
