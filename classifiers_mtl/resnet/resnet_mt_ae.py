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

class Classifier_RESNET_MT_AE:

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

		n_feature_maps = 64
		
		"""
		Main branch, shared features. 
		"""
		input_layer = keras.layers.Input(input_shape)

		# BLOCK 1

		conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same',  name="shared_l1")(input_layer)
		conv_x = keras.layers.BatchNormalization(name="shared_l2")(conv_x)
		conv_x = keras.layers.Activation('relu',name="shared_l3")(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same',name="shared_l4")(conv_x)
		conv_y = keras.layers.BatchNormalization(name="shared_l5")(conv_y)
		conv_y = keras.layers.Activation('relu',name="shared_l6")(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same',name="shared_l7")(conv_y)
		conv_z = keras.layers.BatchNormalization(name="shared_l8")(conv_z)

		# expand channels for the sum
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same',name="shared_l9")(input_layer)
		shortcut_y = keras.layers.BatchNormalization(name="shared_l10")(shortcut_y)

		output_block_1 = keras.layers.add([shortcut_y, conv_z])
		output_block_1 = keras.layers.Activation('relu',name="shared_l11")(output_block_1)

		# BLOCK 2

		conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same',name="shared_l12")(output_block_1)
		conv_x = keras.layers.BatchNormalization(name="shared_l13")(conv_x)
		conv_x = keras.layers.Activation('relu',name="shared_l15")(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same',name="shared_l16")(conv_x)
		conv_y = keras.layers.BatchNormalization(name="shared_l17")(conv_y)
		conv_y = keras.layers.Activation('relu',name="shared_l18")(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same',name="shared_l19")(conv_y)
		conv_z = keras.layers.BatchNormalization(name="shared_20")(conv_z)

		# expand channels for the sum
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same',name="shared_l21")(output_block_1)
		shortcut_y = keras.layers.BatchNormalization(name="shared_l22")(shortcut_y)

		output_block_2 = keras.layers.add([shortcut_y, conv_z])
		output_block_2 = keras.layers.Activation('relu',name="shared_l23")(output_block_2)

		# BLOCK 3

		conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same',name="shared_l24")(output_block_2)
		conv_x = keras.layers.BatchNormalization(name="shared_l25")(conv_x)
		conv_x = keras.layers.Activation('relu',name="shared_l26")(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same',name="shared_l27")(conv_x)
		conv_y = keras.layers.BatchNormalization(name="shared_l28")(conv_y)
		conv_y = keras.layers.Activation('relu',name="shared_l29")(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same',name="shared_l30")(conv_y)
		conv_z = keras.layers.BatchNormalization(name="shared_l31")(conv_z)

		# no need to expand channels because they are equal
		shortcut_y = keras.layers.BatchNormalization(name="shared_l32")(output_block_2)

		output_block_3 = keras.layers.add([shortcut_y, conv_z])
		output_block_3 = keras.layers.Activation('relu',name="shared_l33")(output_block_3)

		###
		# BLOCK 3 (mirrored)
		conv_z_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps * 2, kernel_size=3, padding='same')(output_block_3)
		conv_z_dec = keras.layers.BatchNormalization()(conv_z_dec)
		conv_z_dec = keras.layers.Activation('relu')(conv_z_dec)

		conv_y_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_z_dec)
		conv_y_dec = keras.layers.BatchNormalization()(conv_y_dec)
		conv_y_dec = keras.layers.Activation('relu')(conv_y_dec)

		conv_x_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps * 2, kernel_size=8, padding='same')(conv_y_dec)
		conv_x_dec = keras.layers.BatchNormalization()(conv_x_dec)
		conv_x_dec = keras.layers.Activation('relu')(conv_x_dec)

		shortcut_y_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2)
		shortcut_y_dec = keras.layers.BatchNormalization()(shortcut_y_dec)

		output_block_3_dec = keras.layers.add([shortcut_y_dec, conv_x_dec])
		output_block_3_dec = keras.layers.Activation('relu')(output_block_3_dec)

		# BLOCK 2 (mirrored)
		conv_z_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps, kernel_size=3, padding='same')(output_block_3_dec)
		conv_z_dec = keras.layers.BatchNormalization()(conv_z_dec)
		conv_z_dec = keras.layers.Activation('relu')(conv_z_dec)

		conv_y_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps, kernel_size=5, padding='same')(conv_z_dec)
		conv_y_dec = keras.layers.BatchNormalization()(conv_y_dec)
		conv_y_dec = keras.layers.Activation('relu')(conv_y_dec)

		conv_x_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps, kernel_size=8, padding='same')(conv_y_dec)
		conv_x_dec = keras.layers.BatchNormalization()(conv_x_dec)
		conv_x_dec = keras.layers.Activation('relu')(conv_x_dec)

		shortcut_y_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps, kernel_size=1, padding='same')(output_block_1)
		shortcut_y_dec = keras.layers.BatchNormalization()(shortcut_y_dec)

		output_block_2_dec = keras.layers.add([shortcut_y_dec, conv_x_dec])
		output_block_2_dec = keras.layers.Activation('relu')(output_block_2_dec)

		# BLOCK 1 (mirrored)
		conv_z_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps, kernel_size=3, padding='same')(output_block_2_dec)
		conv_z_dec = keras.layers.BatchNormalization()(conv_z_dec)
		conv_z_dec = keras.layers.Activation('relu')(conv_z_dec)

		conv_y_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps, kernel_size=5, padding='same')(conv_z_dec)
		conv_y_dec = keras.layers.BatchNormalization()(conv_y_dec)
		conv_y_dec = keras.layers.Activation('relu')(conv_y_dec)

		conv_x_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps, kernel_size=8, padding='same')(conv_y_dec)
		conv_x_dec = keras.layers.BatchNormalization()(conv_x_dec)
		conv_x_dec = keras.layers.Activation('relu')(conv_x_dec)

		shortcut_y_dec = keras.layers.Conv1DTranspose(filters=n_feature_maps, kernel_size=1, padding='same')(output_block_1)
		shortcut_y_dec = keras.layers.BatchNormalization()(shortcut_y_dec)

		output_block_1_dec = keras.layers.add([shortcut_y_dec, conv_x_dec])
		output_block_1_dec = keras.layers.Activation('relu')(output_block_1_dec)

		# FINAL

		gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
		

		#flatten_l = keras.layers.Flatten()(input_layer)
		#constants = keras.layers.Identity()(input_layer)  # Make a copy of the input_tensor


		# Apply stop_gradients to the constants tensor
		#constants = tf.stop_gradient(constants)
	
		"""
		Specific Output layers: 
		"""
		output_layer_1 = keras.layers.Dense(nb_classes_1, activation='softmax', name='task_1_output')(gap_layer)
		#output_layer_2 = keras.layers.Dense(units=input_shape[0], activation='linear', name='task_2_output')(gap_layer)
		#linearkeras.layers.LeakyReLU(alpha=0.03)
		output_layer_2 = keras.layers.Conv1DTranspose(filters=input_shape[1], kernel_size=1, padding='same', activation='linear',name='task_2_output')(output_block_1_dec)
		#output_layer_2 = tf.keras.layers.Multiply(name='task_2_output')([output_layer_2, constants])
		#keras.layers.LeakyReLU(alpha=0.03)
		#activation='linear'
		"""
		Define model: 

		"""
		model = keras.models.Model(inputs=[input_layer], outputs=[output_layer_1, output_layer_2])

		#print(model.summary())
		#'task_2_output': 'mae'
		def custom_loss(y_true, y_pred):
			# Compute the squared error between true and predicted values
			squared_error = tf.square(y_true - y_pred)
			
			# Mask for non-zero predictions when true value is zero
			mask = tf.cast(tf.equal(y_true, 0) & tf.not_equal(y_pred, 0), dtype=tf.float32)
		
			#mask_with_constant = tf.add(mask, 1)
			#modified_mask = tf.keras.backend.switch(tf.keras.backend.equal(mask, 0), 0.5, mask)

			# Multiply the squared error with the mask
			#masked_error = squared_error * mask_with_constant
			
			masked_loss = squared_error * mask
			penalized_loss = tf.reduce_mean(masked_loss)
			
			# Calculate the non-penalized loss
			non_penalized_loss = squared_error
			
			# Combine the penalized and non-penalized losses
			combined_loss = 2*penalized_loss + non_penalized_loss

			# Compute the mean of the masked error as the loss
			loss = tf.reduce_mean(combined_loss)
			
			return loss

		model.compile(
			optimizer = keras.optimizers.Adam(), 
			loss={'task_1_output': 'categorical_crossentropy','task_2_output': 'mse'},
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
			model = keras.models.load_model(self.output_directory+'best_model.hdf5', compile=False)
		else:
			model = keras.models.load_model(self.output_directory+'best_model.hdf5', compile=False)

		# convert the predicted from binary to integer 
		# Multitask output 
		y_pred = model.predict(x_val)

		#Predictions for task1 and task2
		y_pred_1 = np.argmax(y_pred[0] , axis=1)
		y_pred_2 = np.argmax(y_pred[1] , axis=1)

	
		#save_logs: 
		#Calculate metrics and saves as csv. 
		#Input format: 
		#save_logs(output_directory, hist, y_pred_1, y_pred_2, y_true_1, y_true_2, duration, lr=True, y_true_val=None, y_pred_val=None)


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
		
	
		"""
		
		

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

		"""

		self.model.save(self.output_directory+'last_model.hdf5')
		

		"""
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

	
		#save_logs: 
		#Calculate metrics and saves as csv. 
		#Input format: 
		save_logs(output_directory, hist, y_pred_1, y_pred_2, y_true_1, y_true_2, duration, lr=True, y_true_val=None, y_pred_val=None)


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
		
	"""