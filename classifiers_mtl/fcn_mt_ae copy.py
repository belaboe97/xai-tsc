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

class Classifier_FCN_MT_AE:

	def __init__(self, output_directory, input_shape, nb_classes_1, gamma, epochs, batch_size, verbose=False, build=True):
		self.output_directory = output_directory
		self.gamma = gamma
		self.epochs = epochs
		self.batch_size = batch_size
		self.latent_inputs = None
		if build == True:
			self.model = self.build_model(input_shape, nb_classes_1)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory+'model_init.hdf5')
		return


	def build_model(self, input_shape, nb_classes_1):



		print("input_shape",input_shape)
		"""
		Main branch, shared features. 
		"""
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		print(conv1.shape)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		print(conv2.shape)


		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		print(conv3.shape)

		
		gap_layer = keras.layers.AveragePooling1D()(conv3)



		print("POOL size", gap_layer.shape)

		output_for_task_1 = keras.layers.GlobalAveragePooling1D()(gap_layer)  # alternative to GlobalAveragePooling1D

		print( keras.layers.GlobalAveragePooling1D()(conv3).shape)

		"""a
		Decoder 
		"""
	

		#latent_inputs = keras.layers.Input(gap_layer.shape[-1])
		#dense_layer = keras.layers.Dense(128, activation='relu')(gap_layer)
		#dense_layer = keras.layers.Reshape((1, 128))(dense_layer)

		upsample = keras.layers.UpSampling1D()(gap_layer)

		print("upsample", upsample.shape)
		
		#reshaped_gap_layer = keras.layers.Reshape((128,1))(gap_layer)

		conv4 = keras.layers.Conv1DTranspose(filters=128, kernel_size=3, padding='same')(upsample)
		conv4 = keras.layers.BatchNormalization()(conv4)
		conv4 = keras.layers.Activation('relu')(conv4)

		conv5 = keras.layers.Conv1DTranspose(filters=256, kernel_size=5, padding='same')(conv4)
		conv5 = keras.layers.BatchNormalization()(conv5)
		conv5 = keras.layers.Activation('relu')(conv5)

		conv6 = keras.layers.Conv1DTranspose(filters=128, kernel_size=8, padding='same')(conv5)
		conv6 = keras.layers.BatchNormalization()(conv6)
		conv6 = keras.layers.Activation('relu')(conv6)

		print("Conv6",conv6.shape)

		flat_layer = keras.layers.Flatten()(conv6) 
		print(flat_layer.shape)

		#decoder = keras.Model(decoder_input, decoder_output, name="decoder")

		"""
		Specific Output layers: 
		"""
		output_layer_1 = keras.layers.Dense(nb_classes_1, activation='softmax', name='task_1_output')(output_for_task_1)

		print("INPUT SHAPE", input_shape[0],input_shape[1])
		output_layer_2 = keras.layers.Conv1DTranspose(filters=input_shape[1], kernel_size=8, padding='same', activation='sigmoid', name='task_2_output')(conv6)


		print("SHAPE OUTPUT",output_layer_2.shape)


		"""
		Define model: 

		"""

		model = keras.models.Model(inputs=[input_layer], outputs=[output_layer_1, output_layer_2])

		print(model.summary())

		model.compile(
			optimizer = keras.optimizers.Adam(), 
			loss={'task_1_output': 'categorical_crossentropy', 'task_2_output': 'mae'},
			loss_weights={'task_1_output': self.gamma, 'task_2_output': 1 -  self.gamma},
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)
				
		#early_stop = keras.callbacks.EarlyStopping(patience = 3)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)
		

		self.callbacks = [reduce_lr,model_checkpoint] #g, early_stop]

		return model 

	def fit(self, x_train, y_train_1,y_train_2, x_val, y_val_1, y_val_2, y_true_1, y_true_2):
			
		print("SHAPES", y_train_1.shape, y_train_2.shape)
		"""
				
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		"""

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