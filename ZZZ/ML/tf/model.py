#from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def create_model():


	"""
	inp = Input((28, 28), "images")
	top = Flatten()(inp)
	top = Tanh()(Linear(28*28*10, name="l1")(top))
	top = Tanh()(Linear(1000, name="l2")(top))
	top = Linear(10, name="out")(top)
	loss = CrossEntropyLoss("LogLoss")(top)
	"""


	num_classes = 10
	model = Sequential()
	input_shape = (28, 28)
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(28*28*10, activation="tanh"))
	model.add(Dense(1000, activation="tanh"))
	model.add(Dense(num_classes, activation="softmax"))
	opt = keras.optimizers.SGD(lr=0.001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
		      optimizer=opt,
		      metrics=['accuracy'])

	return model


