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

from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, AveragePooling1D, UpSampling1D, Conv1DTranspose, GlobalAveragePooling1D, Dense


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

		print("VERSION",tf.__version__)

		# Define input shape
		input_shape = (150, 1)

		# Define input layer
		input_layer = Input(shape=input_shape, name='input_1')

		# Encoder
		x = Conv1D(filters=128, kernel_size=3, padding='same')(input_layer)
		print("SHPAPE", x.name,x.shape)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = AveragePooling1D()(x)
		print(x.shape)
		# Decoder
		x = UpSampling1D()(x)
		x = Conv1DTranspose(filters=128, kernel_size=3, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv1DTranspose(filters=256, kernel_size=3, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv1DTranspose(filters=1, kernel_size=3, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('sigmoid')(x)

		# Output layers
		y = GlobalAveragePooling1D()(x)
		y1 = Dense(units=2, activation='softmax', name='task_1_output')(y)
		y2 = Conv1DTranspose(filters=1, kernel_size=3, padding='same', name='task_2_output')(x)

		# Define the model
		model = Model(inputs=input_layer, outputs=[y1, y2])

		"""
		Define model: 

		"""

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

		#model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
		#	save_best_only=True)
		

		#self.callbacks = [reduce_lr,model_checkpoint] #g, early_stop]

		return model 

	def fit(self, x_train, y_train_1,y_train_2, x_val, y_val_1, y_val_2, y_true_1, y_true_2):
			
		print("SHAPES",x_train.shape, y_train_1.shape, y_train_2.shape)
		"""
				
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		"""


		print("TYPE", type(x_train[0]))
		#x_val and y_val are only used to monitor the test loss and NOT for training  

		batch_size = self.batch_size

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))


		start_time = time.time() 
		num_epochs = 1
		gamma = 0.5

		hist = self.model.fit(x_train,[y_train_1,y_train_2], batch_size=batch_size, epochs=1, verbose=1)

		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')
		
		if os.getenv("COLAB_RELEASE_TAG"):
			model = keras.models.load_model(self.output_directory+'best_model.hdf5')
		else:
			model = keras.models.load_model(self.output_directory+'best_model.hdf5', compile=False)

		# convert the predicted from binary to integer 
		# Multitask output 
		y_pred = model.predict(x_val)

		print(y_pred)
		"""
		#Predictions for task1 and task2
		y_pred_1 = np.argmax(y_pred[0] , axis=1)
		y_pred_2 = np.argmax(y_pred[1] , axis=1)

		save_logs: 
		Calculate metrics and saves as csv. 
		Input format: 
		save_logs(output_directory, hist, y_pred_1, y_pred_2, y_true_1, y_true_2, duration, lr=True, y_true_val=None, y_pred_val=None)
	

		#print(y_pred_1.shape, y_pred_1, y_pred_2)
		save_logs_mtl(self.output_directory, hist, y_pred_1, y_pred_2, y_true_1, y_true_2, duration)

		"""

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