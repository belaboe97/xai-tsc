# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 
import os

from utils.utils import save_logs_mtl
from utils.utils import calculate_metrics


class Classifier_FCN_MT_NN:

	def __init__(self, output_directory, input_shape, nb_classes_1, lossf, gamma, epochs, batch_size, verbose=False, build=True):
		self.output_directory = output_directory
		self.gamma = gamma
		self.epochs = epochs
		self.batch_size = batch_size
		self.latent_inputs = None
		self.output_2_loss = lossf
		if build == True:
			self.model = self.build_model(input_shape, nb_classes_1)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory+'model_init.hdf5')
		return
	
	
	def custom_loss(y_true, y_pred):
		# Compute the squared error between true and predicted values
		squared_error = tf.square(y_true - y_pred)
		
		# Mask for non-zero predictions when true value is zero
		mask = tf.cast(tf.equal(y_true, 0) & tf.not_equal(y_pred, 0), dtype=tf.float32)
		modified_mask = tf.keras.backend.switch(tf.keras.backend.equal(mask, 0), 0.5, mask)

		# Multiply the squared error with the mask
		masked_error = squared_error * mask
		
		# Compute the mean of the masked error as the loss
		loss = tf.reduce_mean(masked_error)
		
		return loss


	def build_model(self, input_shape, nb_classes_1):
		"""
		Main branch, shared features. 
		"""
		input_layer = keras.layers.Input(input_shape)


		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same',trainable=True, name="shared_l1")(input_layer)
		conv1 = keras.layers.BatchNormalization(trainable=True, name="shared_l2")(conv1)
		conv1 = keras.layers.Activation(activation='relu',trainable=True, name="shared_l3")(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same',trainable=True, name="shared_l4")(conv1)
		conv2 = keras.layers.BatchNormalization(trainable=True, name="shared_l5")(conv2)
		conv2 = keras.layers.Activation('relu',trainable=True, name="shared_l6")(conv2)

		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same',trainable=True, name="shared_l7")(conv2)
		conv3 = keras.layers.BatchNormalization(trainable=True, name="shared_l8")(conv3)
		conv3 = keras.layers.Activation('relu',trainable=True, name="shared_l9")(conv3)
		
		#gap_layer = keras.layers.AveragePooling1D()(conv3)

		output_for_task_1 = keras.layers.GlobalAveragePooling1D()(conv3)  # alternative to GlobalAveragePooling1D

		flatten = keras.layers.Flatten()(conv3)

		#conv1d = keras.layers.Conv1DTranspose(filters=1, kernel_size=3,padding='same',activation="linear")(conv3)
		#conv1d_flatten =  keras.layers.Flatten()(conv1d)

		interm_function_1 = keras.layers.Dense(2*input_shape[0], activation='relu')(flatten)
		interm_function_2 = keras.layers.Dense(2*input_shape[0], activation='relu')(interm_function_1)
		interm_function_3 = tf.keras.layers.Dense(2*input_shape[0], activation='relu')(interm_function_2)


		"""
		Specific Output layers: 
		"""
		output_layer_1 = keras.layers.Dense(nb_classes_1, activation='softmax', name='task_1_output', trainable=True)(output_for_task_1)

		output_layer_2 = keras.layers.Dense(input_shape[0], activation='linear', name='task_2_output')(interm_function_3)

		#output_layer_2 = keras.layers.Conv1DTranspose(filters=input_shape[1], kernel_size=1, padding='same', activation='linear', name='task_2_output')(conv6)

		model = keras.models.Model(inputs=[input_layer], outputs=[output_layer_1, output_layer_2])

		#print(model.summary())

		model.compile(
			optimizer = keras.optimizers.Adam(), 
			loss={'task_1_output': 'categorical_crossentropy', 'task_2_output': self.output_2_loss},
			loss_weights={'task_1_output': self.gamma, 'task_2_output': 1 -  self.gamma},
			metrics=['accuracy']) #mae

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)
				
		#early_stop = keras.callbacks.EarlyStopping(patience = 3)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
		
		self.callbacks = [reduce_lr,model_checkpoint] #g, early_stop]

		return model 

	def fit(self, x_train, y_train_1,y_train_2, x_val, y_val_1, y_val_2, y_true_1, y_true_2):
			
		print("SHAPES", y_train_1.shape, y_train_2.shape)
	

		#x_val and y_val are only used to monitor the test loss and NOT for training  

		batch_size = self.batch_size

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(
		{'input_1': x_train},
        {'task_1_output': y_train_1, 'task_2_output': y_train_2},
		batch_size=mini_batch_size, 
		epochs=self.epochs,
		verbose=self.verbose, 
		validation_data=(
			x_val,
			{'task_1_output': y_val_1, 'task_2_output': y_val_2}), 
		callbacks=self.callbacks)
		
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')
		
		if os.getenv("COLAB_RELEASE_TAG"):
			model = keras.models.load_model(self.output_directory+'best_model.hdf5')
		else:
			model = keras.models.load_model(self.output_directory+'best_model.hdf5', compile=False)

		# convert the predicted from binary to integer 
		# Multitask output 
		y_pred = model.predict(x_val)

		#Predictions for task1 and task2
		y_pred_1 = np.argmax(y_pred[0] , axis=1)
		y_pred_2 = np.argmax(y_pred[1] , axis=1)

		"""
		save_logs: 
		Calculate metrics and saves as csv. 
		Input format: 
		save_logs(output_directory, hist, y_pred_1, y_pred_2, y_true_1, y_true_2, duration, lr=True, y_true_val=None, y_pred_val=None)
		"""

		#print(y_pred_1.shape, y_pred_1, y_pred_2)
		save_logs_mtl(self.output_directory, hist, y_pred_1, y_pred_2, y_true_1, y_true_2, duration)

		keras.backend.clear_session()

	def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
		model_path = self.output_directory + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred