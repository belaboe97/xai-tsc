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



class Classifier_FCN_MT_TEST:

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


	def build_model(self, input_shape, nb_classes_1):
		
		"""
		Main branch, shared features. 
		"""
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)


		# Reshape the output for compatibility with dense layers
		#reshaped_output = tf.keras.layers.Reshape((1, 1, -1))(gap_layer)
		
		# Fully connected layers for generating heatmap
		#RES 1
		res1 = keras.layers.Dense(128)(conv3)
		res1 = keras.layers.BatchNormalization()(res1)
		res1 = keras.layers.Activation('relu')(res1)
		
		res1 = keras.layers.Dense(128)(res1)
		res1 = keras.layers.BatchNormalization()(res1)

		res1 = keras.layers.Add()([res1, conv3])  # Adding the residual connection

		output_res_1 = keras.layers.Activation('relu')(res1)

		res2 = keras.layers.Dense(128)(output_res_1)
		res2 = keras.layers.BatchNormalization()(res2)
		res2 = keras.layers.Activation('relu')(res2)
		
		res2 = keras.layers.Dense(128)(res2)
		res2 = keras.layers.BatchNormalization()(res2)
		res2 = keras.layers.Add()([res2, res1])
		
		output_res_2 = keras.layers.Activation('relu')(res2)

		output_2_l = keras.layers.Flatten()(output_res_2)
	
	
		"""
		Specific Output layers: 
		"""

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

		print(gap_layer.shape)
		print(output_2_l.shape)
		output_layer_1 = keras.layers.Dense(nb_classes_1, activation='softmax', name='task_1_output')(gap_layer)
		output_layer_2 = keras.layers.Dense(units=input_shape[0], activation='linear', name='task_2_output')(output_2_l)
		#linear
		"""
		Define model: 

		"""

		model = keras.models.Model(inputs=[input_layer], outputs=[output_layer_1, output_layer_2])

		

		#print(model.summary())
		#'task_2_output': 'mae'

		model.compile(
			optimizer = keras.optimizers.Adam(), 
			loss={'task_1_output': 'categorical_crossentropy','task_2_output': self.output_2_loss},
			loss_weights={'task_1_output': self.gamma, 'task_2_output': 1 -  self.gamma},
			metrics=['accuracy']) #mae

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)
				
		#early_stop = keras.callbacks.EarlyStopping(patience = 3)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)
		

		self.callbacks = [reduce_lr,model_checkpoint] 

		return model 

	def fit(self, x_train, y_train_1,y_train_2, x_val, y_val_1, y_val_2, y_true_1, y_true_2):
			
		print("SHAPES", y_train_1.shape, y_train_2.shape)
		"""
				
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		"""

		#x_val and y_val are only used to monitor the test loss and NOT for training  
		import sys
		sys.path.append("...") 
		from utils.constants import CAM_LAYERS
		from utils.explanations import get_layer_index
		import sklearn
		import numpy as np 

		"""
		# transform to binary labels
		enc = sklearn.preprocessing.OneHotEncoder()
		enc.fit(np.concatenate((y_train_1.copy()), axis=0).reshape(-1, 1))
		y_train_binary = enc.transform(y_train_1.copy().reshape(-1, 1)).toarray()
		"""
			

		for epoch in range(self.epochs):
			print(epoch)
			print(len(x_train), x_train.shape, len(y_train_1), y_train_1.shape, len(y_train_2),y_train_2.shape)

			batch_size = self.batch_size

			mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

			start_time = time.time() 

			hist = self.model.fit(
			{'input_1': x_train},
			{'task_1_output': y_train_1, 'task_2_output': y_train_2},
			batch_size=mini_batch_size, 
			verbose=self.verbose, 
			validation_data=(
				x_val,
				{'task_1_output': y_val_1, 'task_2_output': y_val_2}), 
			callbacks=self.callbacks)


			# Make predictions using model.predict


			#get gap and output layer
			gap = CAM_LAYERS["fcn"]["gap_layer"]
			gap = get_layer_index(self.model, gap)
			out = CAM_LAYERS["fcn"]["task_1"]
			out = get_layer_index(self.model, out)

				
			w_k_c = self.model.layers[out].get_weights()[0]  # weights for each filter k for each class c

			# the same input
			new_input_layer = self.model.inputs
			new_output_layer = [self.model.layers[gap].output, self.model.layers[out].output]
			new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

		
			# Calculate classwise attribution gap to output
			idx = 0
			for x_value ,y_vals_1 in zip(x_train,y_train_1):
	
				ts = x_value.reshape(1, -1, 1)
				[conv_out, predicted] = new_feed_forward([ts])
				cas = np.zeros(dtype=np.float64, shape=(conv_out.shape[1]))
				pred_label = np.argmax(predicted[0])
				orig_label = int(np.argmax(y_vals_1))
				if pred_label == orig_label:
					for k, w in enumerate(w_k_c[:,orig_label]):
						cas += w * conv_out[0, :, k] 

				y_train_2[idx] = cas
				idx += 1

			
			#print(len(x_train), x_train.shape, len(y_train_1), y_train_1.shape, len(y_train_2),y_train_2.shape)
						

			
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')
		
		if os.getenv("COLAB_RELEASE_TAG"):
			model = keras.models.load_model(self.output_directory+'best_model.hdf5', compile=False)
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