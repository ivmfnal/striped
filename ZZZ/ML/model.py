#from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def create_model():

	num_classes = 10
	model = Sequential()
	input_shape = (32, 32, 3)
	model.add(Conv2D(32, (3, 3), padding='same',
			 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
	opt = keras.optimizers.SGD(lr=0.001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
		      optimizer=opt,
		      metrics=['accuracy'])

	return model


