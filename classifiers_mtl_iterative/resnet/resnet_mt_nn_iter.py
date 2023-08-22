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

class Classifier_RESNET_MT_NN_ITER:

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

		# FINAL
	
		gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

		flatten =  keras.layers.Flatten()(output_block_3)

		interm_function_1 = keras.layers.Dense(2*input_shape[0], activation='relu')(flatten)
		interm_function_2 = keras.layers.Dense(2*input_shape[0], activation='relu')(interm_function_1)
		interm_function_3 = tf.keras.layers.Dense(2*input_shape[0], activation='relu')(interm_function_2)

		"""
		Specific Output layers: 
		"""
		output_layer_1 = keras.layers.Dense(nb_classes_1, activation='softmax', name='task_1_output')(gap_layer)

		#interm_layer_2 = keras.layers.Dense(activation='sigmoid')(gap_layer)

		output_layer_2 = keras.layers.Dense(input_shape[0], activation='linear', name='task_2_output')(interm_function_3)

		#keras.layers.Dense(units=input_shape[0], activation=keras.layers.LeakyReLU(alpha=0.03), name='task_2_output')(flatten_layer)
		#linear


		print("SHAPE OUTPUT",output_layer_2.shape)


		"""
		Define model: 

		"""

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

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)
		

		self.callbacks = [reduce_lr,model_checkpoint] #g, early_stop]

		return model 


	def fit(self, x_train, y_train_1,y_train_2, x_val, y_val_1, y_val_2, y_true_1, y_true_2):

		from utils.explanations import norm, integrated_gradients	

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

		loss =  []
		val_loss = [] 

		for epoch in range(self.epochs):
			

			batch_size = self.batch_size

			mini_batch_size = int(min(x_train.shape[0]/10, batch_size))


			if  epoch > 200 and epoch % 20==0:
				baseline = tf.zeros(len(x_train[0]))
				for mode , [xvalues,yvalues] in enumerate([[x_train,y_train_1],[x_val,y_val_1]]):
					idx = 0
					pred = self.model.predict(xvalues)
					for x,y in zip(xvalues,yvalues):
						series = x.flatten()
						ig_att = integrated_gradients(self.model,baseline,series.astype('float32'),
													np.argmax(pred[0][idx]),
													task=1)
						if mode == 0: 
							y_train_2[idx] = norm(ig_att)
						if mode == 1: 
							y_val_2[idx] = norm(ig_att)
						idx += 1


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


			metric = "loss"
			loss.append(hist.history[metric][0])
			val_loss.append(hist.history['val_' + metric][0])

		np.savetxt(self.output_directory+f"test{epoch}_Loss", loss, delimiter=',')
		np.savetxt(self.output_directory+f"test{epoch}_Val_Loss", val_loss, delimiter=',')

		np.savetxt(self.output_directory+f"test{epoch}_TRAIN", y_train_2, delimiter=',')
		np.savetxt(self.output_directory+f"test{epoch}_TEST", y_val_2, delimiter=',')	

		
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

		
		save_logs: 
		Calculate metrics and saves as csv. 
		Input format: 
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