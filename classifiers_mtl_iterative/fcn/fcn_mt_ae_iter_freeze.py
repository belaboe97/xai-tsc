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


class Classifier_FCN_MT_AE_ITER_FREEZE:

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

		"""
		Decoder 
		"""

		conv4 = keras.layers.Conv1DTranspose(filters=128, kernel_size=3, padding='same')(conv3)
		conv4 = keras.layers.BatchNormalization()(conv4)
		conv4 = keras.layers.Activation('relu')(conv4)

		conv5 = keras.layers.Conv1DTranspose(filters=256, kernel_size=5, padding='same')(conv4)
		conv5 = keras.layers.BatchNormalization()(conv5)
		conv5 = keras.layers.Activation('relu')(conv5)

		conv6 = keras.layers.Conv1DTranspose(filters=128, kernel_size=8, padding='same')(conv5)
		conv6 = keras.layers.BatchNormalization()(conv6)
		conv6 = keras.layers.Activation('relu')(conv6)

		#print("Conv6",conv6.shape)

		flat_layer = keras.layers.Flatten()(conv6) 

		#decoder = keras.Model(decoder_input, decoder_output, name="decoder")

		"""
		Specific Output layers: 
		"""
		output_layer_1 = keras.layers.Dense(nb_classes_1, activation='softmax', name='task_1_output', trainable=True)(output_for_task_1)

		output_layer_2 = keras.layers.Conv1DTranspose(filters=input_shape[1], kernel_size=1, padding='same', activation='linear', name='task_2_output')(conv6)

		#linear
		"""
		Define model: 

		"""

		model = keras.models.Model(inputs=[input_layer], outputs=[output_layer_1, output_layer_2])

		model.compile(
			optimizer = keras.optimizers.Adam(), 
			loss={'task_1_output': 'categorical_crossentropy','task_2_output': self.output_2_loss},
			loss_weights={'task_1_output': self.gamma, 'task_2_output': 1 -  self.gamma},
			metrics=['accuracy']) 


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

		ecount = 0

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

			ecount = epoch
			if epoch == 299: 
				break


		self.model.save(self.output_directory+'classifier_model.hdf5')

			# Make predictions using model.predict
		np.savetxt(self.output_directory+f"test{'first'}_TRAIN", y_train_2, delimiter=',')
		np.savetxt(self.output_directory+f"test{'first'}_TEST", y_val_2, delimiter=',')	

			#Conditions  
		if ecount + 2 > 300: #epoch > 0:

			# Calculate classwise attribution gap to output
			baseline = tf.zeros(len(x_train[0]))
			for mode , [xvalues,yvalues] in enumerate([[x_train,y_train_1],[x_val,y_val_1]]):

				idx = 0
				pred = self.model.predict(x_train)
				for x,y in zip(xvalues,yvalues):#
					#print("YVALUES",np.argmax(y))
					#if np.argmax(pred[0][idx]) == np.argmax(y): 
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

			print("attributions calcualted")

			# set gamma 0
			self.gamma = 0
			print("Gamma reducded", self.gamma)

			weights = self.model.get_weights()
			# Set layer to non trainable 

			self.model.compile( 
				optimizer = keras.optimizers.Adam(), 
				loss={'task_1_output': 'categorical_crossentropy','task_2_output': self.output_2_loss},
				loss_weights={'task_1_output': self.gamma, 'task_2_output': 1 -  self.gamma},
				metrics=['accuracy']) 
			
			
			self.model.set_weights(weights)

			for layer in self.model.layers:
				if 'shared' in layer.name or 'task_1_output' in layer.name: 
					layer.trainable = False

			print("model weights retrieved and set")
			
			hist = self.model.fit(
			{'input_1': x_train},
			{'task_1_output': y_train_1, 'task_2_output': y_train_2},
			batch_size=mini_batch_size, 
			epochs= 30,
			verbose=self.verbose, 
			validation_data=(
				x_val,
				{'task_1_output': y_val_1, 'task_2_output': y_val_2}), 
			callbacks=self.callbacks)


			baseline = tf.zeros(len(x_train[0]))
			for mode , [xvalues,yvalues] in enumerate([[x_train,y_train_1],[x_val,y_val_1]]):

				idx = 0
				pred = self.model.predict(x_train)
				for x,y in zip(xvalues,yvalues):#
					#print("YVALUES",np.argmax(y))
					#if np.argmax(pred[0][idx]) == np.argmax(y): 
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


		

		np.savetxt(self.output_directory+f"test{'second'}_TRAIN", y_train_2, delimiter=',')
		np.savetxt(self.output_directory+f"test{'second'}_TEST", y_val_2, delimiter=',')	

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
		model_path = self.output_directory + 'last_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred