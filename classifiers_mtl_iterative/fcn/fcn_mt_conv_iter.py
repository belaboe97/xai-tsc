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
from utils.explanations import integrated_gradients


class Classifier_FCN_MT_CONV_ITER:

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
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(filters=128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
	

		output_layer_1 = keras.layers.Dense(nb_classes_1, activation='softmax', name='task_1_output')(gap_layer)

		#output_layer_2 = keras.layers.Flatten()(conv)
		#output_layer_2  = keras.layers.Conv1D(filters=1, kernel_size=1,padding='same',activation="linear",name='task_2_output')(conv3)
		#linear

		Define model: 

		"""
		
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(filters=128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)


		conv3 = keras.layers.Conv1D(filters=128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)
	
		"""
		Specific Output layers: 
		"""
		output_layer_1 = keras.layers.Dense(nb_classes_1, activation='softmax', name='task_1_output',trainable=False)(gap_layer)
		#output_layer_2 = keras.layers.Conv1D(filters=input_shape[1], kernel_size=1, padding='same', activation='linear')(conv3)
		#print(output_layer_2.shape,output_layer_1.shape)
		#output_layer_2 = keras.layers.Flatten()(output_layer_2)
		flatten_conv3 = keras.layers.Flatten()(conv3)
		concat_input_2 = keras.layers.Concatenate()([flatten_conv3, output_layer_1])
		output_layer_2 = keras.layers.Dense(units=input_shape[0], activation="relu")(concat_input_2)
		output_layer_2 = keras.layers.Dense(units=input_shape[0], activation="linear")(output_layer_2)
		scale_flatten = keras.layers.Flatten(trainable=False)(input_layer)
		output_layer_2 = keras.layers.Multiply(name='task_2_output')([output_layer_2, scale_flatten])
		

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
			
			#print(epoch)
			#print(len(x_train), x_train.shape, len(y_train_1), y_train_1.shape, len(y_train_2),y_train_2.shape)

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

			#Conditions  
			if epoch == 0 and epoch + 1 % 1 == 0: #epoch > 0 and epoch % 1 == 0

				if epoch % 500  == 0: 
					self.gamma #- 0.25
					print("Gamma reducded", self.gamma)
					weights = self.model.get_weights()
					model.compile( 
						optimizer = keras.optimizers.Adam(), 
						loss={'task_1_output': 'categorical_crossentropy','task_2_output': self.output_2_loss},
						loss_weights={'task_1_output': self.gamma, 'task_2_output': 1 -  self.gamma},
						metrics=['accuracy']) 
					self.model.set_weights(weights)
					print("model weights retrieved and set")
				# Calculate classwise attribution gap to output
				baseline = tf.zeros(len(x_train[0]))#

				for mode , [xvalues,yvalues] in enumerate([[x_train,y_train_1],[x_val,y_val_1]]):

					idx = 0
					pred = self.model.predict(x_train)
					for x,y in zip(xvalues,yvalues):#

						print(x, x.shape)
						#print("YVALUES",np.argmax(y))
						if True: #np.argmax(pred[0][idx]) == np.argmax(y): 
							series = x.flatten()
							#print(series.shape,pred[0][idx])
							#series = x.flatten()
							ig_att = integrated_gradients(self.model,baseline,series.astype('float32'),
														np.argmax(pred[0][idx]),
														task=1)
							if mode == 0: 
								y_train_2[idx] = ig_att

							if mode == 1: 
								y_val_2[idx] = ig_att


						idx += 1

		np.savetxt(self.output_directory+f"test{epoch}_XTEST", x_val.flatten(), delimiter=',')
		np.savetxt(self.output_directory+f"test{epoch}_TRAIN", y_train_2, delimiter=',')
		np.savetxt(self.output_directory+f"test{epoch}_TEST", y_val_2, delimiter=',')	

		#print(len(x_train), x_train.shape, len(y_train_1), y_train_1.shape, len(y_train_2),y_train_2.shape)

		#create_directory(dir_path)
			
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