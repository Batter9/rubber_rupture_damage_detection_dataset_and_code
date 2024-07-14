# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 20:17:24 2024

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 10:52:56 2023

@author: admin
"""

from sklearn.metrics import precision_score,roc_auc_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

import h5py

import os
import numpy as np
import h5py
import random
import math
import tensorflow.keras
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from tensorflow.keras import optimizers
from keras import regularizers

from scipy.optimize import minimize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import datasets, linear_model,svm
from sklearn.model_selection import train_test_split
from sklearn import tree
import urllib
from tensorflow.keras.models import model_from_json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
import time
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length
from numpy import percentile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix,recall_score,f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator
import pandas as pd
from bayes_opt import BayesianOptimization
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def list2nurray(data):
    data = np.asarray(data)
    data = np.asarray(data.astype('float32'))
    return data


def load_small_wavelet_axial(filename,dataname,num_layer,num_r,num_base):
    data = h5py.File(filename)
    waveletcoeff_layer_axial_c_v_sensor_r_base = np.transpose(data[dataname])
    dataset = []
    label_damage = []
    label_axial = []
    label_shear = []
    num = 1
    wavelet_len = waveletcoeff_layer_axial_c_v_sensor_r_base.shape
    print(waveletcoeff_layer_axial_c_v_sensor_r_base.shape)
    layer = num_layer
    # for layer in range(0,num_layer):
    for axial in range(0,wavelet_len[1]):
        for c in range(0,wavelet_len[2]):
            for v in range(0,wavelet_len[3]):
                for sensor in range(0,wavelet_len[4]):
                    for r in range(0,num_r):
                        for base in range(0,num_base):
                            dataset.append(waveletcoeff_layer_axial_c_v_sensor_r_base[layer,axial,c,v,sensor,r,base,:]) 
                            label_damage.append(0)
                            label_axial.append(axial * 1)
                            label_shear.append(0)
                            num = num + 1
    return dataset, label_damage,label_axial,label_shear

def load_small_wavelet_shear(filename,dataname,num_layer,num_r,num_base):
    data = h5py.File(filename)
    waveletcoeff_layer_axial_c_v_sensor_r_base = np.transpose(data[dataname])
    dataset = []
    label_damage = []
    label_axial = []
    label_shear = []
    num = 1
    wavelet_len = waveletcoeff_layer_axial_c_v_sensor_r_base.shape
    print(waveletcoeff_layer_axial_c_v_sensor_r_base.shape)
    layer = num_layer
    # for layer in range(0,num_layer):
    for shear in range(0,wavelet_len[1]):
        for p in range(0,wavelet_len[2]):
            for c in range(0,wavelet_len[3]):
                for v in range(0,wavelet_len[4]):
                    for sensor in range(0,wavelet_len[5]):
                        for r in range(0,num_r):
                            for base in range(0,num_base):
                                dataset.append(waveletcoeff_layer_axial_c_v_sensor_r_base[layer,shear,p,c,v,sensor,r,base,:]) 
                                label_damage.append(0)
                                label_axial.append(0)
                                label_shear.append(shear * 10)
                                num = num + 1
    return dataset, label_damage,label_axial,label_shear

def load_small_wavelet_damage(filename,dataname,num_layer,num_v,num_r,num_base):
    data = h5py.File(filename)
    waveletcoeff_layer_axial_c_v_sensor_r_base = np.transpose(data[dataname])
    dataset = []
    label_damage = []
    label_axial = []
    label_shear = []
    label_layer = []
    num = 1
    wavelet_len = waveletcoeff_layer_axial_c_v_sensor_r_base.shape
    print(waveletcoeff_layer_axial_c_v_sensor_r_base.shape)
    layer = num_layer
    # for layer in range(num_layer,wavelet_len[0]):
    for loadi in range(0,wavelet_len[1]):
        if loadi == 0:
            load = 0;
        else:
            load = loadi *2 - 1        
        # for damagei in range(0,wavelet_len[2]):
        for damagei in range(0,wavelet_len[2]):
            # if damagei == 3:
            #     damage = 1
            # else:
            #     damage = damagei / 4     
            for c in range(0,wavelet_len[3]):
                for v in range(0,num_v):
                    for sensor in range(0,wavelet_len[5]):
                        for r in range(0,num_r):
                            for base in range(0,num_base):
                                dataset.append(waveletcoeff_layer_axial_c_v_sensor_r_base[layer,loadi,damagei,c,v,sensor,r,base,:]) 
                                label_damage.append(damagei)
                                label_axial.append(load)
                                label_shear.append(0)
                                label_layer.append(layer - 1)
                                num = num + 1
    return dataset, label_damage,label_layer,label_axial,label_shear

def load_small_wavelet_damageGAN(filename,dataname,num_layer,damage,num_r,num_base):
    data = h5py.File(filename)
    waveletcoeff_layer_axial_c_v_sensor_r_base = np.transpose(data[dataname])
    dataset = []
    label_damage = []
    label_axial = []
    label_shear = []
    label_layer = []
    num = 1
    wavelet_len = waveletcoeff_layer_axial_c_v_sensor_r_base.shape
    print(waveletcoeff_layer_axial_c_v_sensor_r_base.shape)
    layer = num_layer
    # for layer in range(num_layer,wavelet_len[0]):
    for loadi in range(0,wavelet_len[1]):
        if loadi == 0:
            load = 0;
        else:
            load = loadi *2 - 1        
        damagei = damage
        for c in range(0,wavelet_len[3]):
            for v in range(0,wavelet_len[4]):
                for sensor in range(0,wavelet_len[5]):
                    for r in range(0,num_r):
                        for base in range(0,num_base):
                            dataset.append(waveletcoeff_layer_axial_c_v_sensor_r_base[layer,loadi,damagei,c,v,sensor,r,base,:]) 
                            label_damage.append(damagei)
                            label_axial.append(load)
                            label_shear.append(0)
                            label_layer.append(layer - 1)
                            num = num + 1
    return dataset, label_damage,label_layer,label_axial,label_shear


dataname_list = ['waveletcoeff8']

dataname1 = dataname_list[0]

#############################################################################

num_damageR = 1
dataset1,label_damage1,label_axial1,label_shear1 = load_small_wavelet_axial('small_axial_20231028.mat',dataname1, 0,1,1)
dataset2,label_damage2,label_axial2,label_shear2 = load_small_wavelet_shear('smallShear20231028.mat',dataname1, 0,1,1)
dataset3,label_damage3,label_layer3,label_axial3,label_shear3 = load_small_wavelet_damage('smallDamage20231028.mat',dataname1, 1,4,1,num_damageR)
dataset3_3,label_damage3_3,label_layer3_3,label_axial3_3,label_shear3_3 = load_small_wavelet_damage('smallDamage20231028.mat',dataname1, 1,4,1,3)

dataset4,label_damage4,label_axial4,label_shear4 = load_small_wavelet_axial('small_axial_20231028.mat',dataname1, 1,1,1)
dataset5,label_damage5,label_axial5,label_shear5 = load_small_wavelet_shear('smallShear20231028.mat',dataname1, 1,1,1)
dataset6,label_damage6,label_layer6,label_axial6,label_shear6 = load_small_wavelet_damage('smallDamage20231028.mat',dataname1, 2,4,1,num_damageR)
dataset6_3,label_damage6_3,label_layer6_3,label_axial6_3,label_shear6_3 = load_small_wavelet_damage('smallDamage20231028.mat',dataname1, 2,4,1,3)


dataset_all = dataset1 + dataset2 + dataset3 + dataset4 + dataset5 + dataset6
label_damage_all = label_damage1 + label_damage2 + label_damage3 + label_damage4 + label_damage5 + label_damage6


dataset_all_3 = dataset1 + dataset2 + dataset3_3 + dataset4 + dataset5 + dataset6_3
label_damage_all_3 = label_damage1 + label_damage2 + label_damage3_3 + label_damage4 + label_damage5 + label_damage6_3


dataset_all_ori = list2nurray(dataset_all)
label_damage_all_ori = list2nurray(label_damage_all)

dataset_all = list2nurray(dataset_all)
label_damage_all = list2nurray(label_damage_all)

dataset_all_3 = list2nurray(dataset_all_3)
label_damage_all_3 = list2nurray(label_damage_all_3)
################################################################################################

# label_damage_all_ori = to_categorical(label_damage_all_ori, num_classes=4)


# # Split the dataset into train, validation, and test sets
# split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
# split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

# for train_index, temp_index in split1.split(dataset_all_ori, label_damage_all_ori):
#     x_train, x_temp = dataset_all_ori[train_index], dataset_all_ori[temp_index]
#     y_train, y_temp = label_damage_all_ori[train_index], label_damage_all_ori[temp_index]

# for val_index, test_index in split2.split(x_temp, y_temp):
#     x_val, x_test = x_temp[val_index], x_temp[test_index]
#     y_val, y_test = y_temp[val_index], y_temp[test_index]

# # y_train = to_categorical(y1, num_classes=4)
# # y_test = to_categorical(y2, num_classes=4)

# # # Define a list to store the results
# # # Define a DataFrame to store the results
# results_df = pd.DataFrame(columns=['Iteration', 'Best Num Layers', 'Best Num Units', 'Val Accuracy','Test Accuracy','Time'])

# # Define the 1D-CNN model
# def create_model(num_layers, num_units):
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(256, 1)))  # Input shape for 1D data

#     for _ in range(int(num_layers)):
#         model.add(layers.Conv1D(filters=int(num_units), kernel_size=3, activation='relu'))
#         model.add(layers.MaxPooling1D(pool_size=2))
    
#     model.add(layers.Flatten())
#     model.add(layers.Dense(4, activation='softmax'))
    
#     optimizer = keras.optimizers.Adam(learning_rate=0.001)

#     model.compile(optimizer=optimizer,
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
    
#     return model

# # Modify the input data shape to include the channel dimension
# x_train = x_train[:, :, np.newaxis]
# x_val = x_val[:, :, np.newaxis]
# x_test = x_test[:, :, np.newaxis]

# # Define the objective function to optimize
# def objective(num_layers, num_units):
#     num_layers = int(num_layers)
#     num_units = int(num_units)
    
#     num_units = 2 ** num_units
#     # Create the final model with the specified hyperparameters
#     final_model = create_model(num_layers, num_units)
    
#     start_time = time.time()

#     # Train the final model on the entire training set
#     final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=0)
    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
    
#     # Evaluate the model on the val set
#     y_pred_val = final_model.predict(x_val, verbose=0)
#     val_accuracy = accuracy_score(y_val.argmax(axis=1), y_pred_val.argmax(axis=1))
    
#     # Calculate the confusion matrix
#     cm = confusion_matrix(y_val.argmax(axis=1), y_pred_val.argmax(axis=1))
    
#     # Plot the confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#                 xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
#                 yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(f'Confusion Matrix ori val\nAccuracy: {val_accuracy:.2f}')
#     # 
#     plt.show()
    

#     # Evaluate the model on the test set
#     y_pred = final_model.predict(x_test, verbose=0)
#     test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
#     # Calculate the confusion matrix
#     cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
#     # Plot the confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#                 xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
#                 yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(f'Confusion Matrix ori test\nAccuracy: {test_accuracy:.2f}')
#     # 
#     plt.show()
    
    
    
#     # Store the results in the DataFrame
#     results_df.loc[len(results_df)] = [len(results_df) + 1, num_layers, num_units, val_accuracy, test_accuracy,elapsed_time]
    
#     return test_accuracy  # Negative because BayesianOptimization seeks to minimize

# # Rest of the code remains the same
# # Define the hyperparameter search space
# pbounds = {'num_layers': (1, 5), 'num_units': (5, 12)}

# # Create the BayesianOptimization object
# optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

# # Perform Bayesian optimization to find the best hyperparameters
# optimizer.maximize(init_points=5, n_iter=20)

# # Save the results to a CSV file
# results_df.to_csv('bayesian_optimization_results20240713.csv', index=False)


# plt.figure(figsize=(10, 6))
# plt.plot(results_df['Iteration'], results_df['Test Accuracy'], marker='o', linestyle='-')
# plt.title('Accuracy vs. Iteration during Bayesian Optimization L2L3 gan ')
# plt.xlabel('Iteration')
# plt.ylabel('Test Accuracy')
# plt.grid(True)
# plt.show()

################################################################################################
# #using GAN seperately to do 
import time
elapsed_times = []
for layer in range(1,3):
    for damagei in range(0,4):
        
        model_name = f"generator_model_layer_{layer}_damage_{damagei}.h5"
        weights_name = f"generator_weights_layer_{layer}_damage_{damagei}.h5"
        signals_name = f"generated_signals_layer_{layer}_damage_{damagei}.npy"
        
        num_damageR = 1 
        dataset3,label_damage3,label_layer3,label_axial3,label_shear3 = load_small_wavelet_damageGAN('smallDamage20231028.mat',dataname1, layer,damagei,1,num_damageR)
        # dataset,label = load_mat_wavelet_axial('axial_0831_20230907.mat',dataname, 1,1)
        
        dataset = list2nurray(dataset3)
        label = list2nurray(label_damage3)
        
        target_signal = dataset
        
        initial_learning_rate = 0.001  # Set your initial learning rate
        lr_schedule = ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,  # Adjust decay steps as needed
            decay_rate=0.9  # Adjust decay rate as needed
        )
        
        # Compute mean and standard deviation of the target signals
        mean_target = np.mean(target_signal)
        std_target = np.std(target_signal)
        
        # Normalize the target signals
        # normalized_target_signals = (target_signal - mean_target) / std_target
        
        l2_value = 0.01
        
        # Define early stopping callback
        # early_stopping = EarlyStopping(
        #     monitor='val_loss',  # Monitor the validation loss
        #     patience=10,          # Number of epochs with no improvement before stopping
        #     restore_best_weights=True  # Restore the model with the best weights
        # )
        
        # Define the generator
        generator = keras.Sequential([
            keras.layers.Dense(256, input_shape=(target_signal.shape[1],), activation='relu'),  # Adjust input shape
            
            keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(l2_value)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            
            keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(l2_value)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            
            keras.layers.Dense(1024, kernel_regularizer=keras.regularizers.l2(l2_value)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            
            # keras.layers.Dense(2048, kernel_regularizer=keras.regularizers.l2(l2_value)),
            # keras.layers.BatchNormalization(),
            # keras.layers.LeakyReLU(alpha=0.2),
            
            # keras.layers.Dense(2048, kernel_regularizer=keras.regularizers.l2(l2_value)),
            # keras.layers.BatchNormalization(),
            # keras.layers.LeakyReLU(alpha=0.2),
            
            # keras.layers.Dense(2048, kernel_regularizer=keras.regularizers.l2(l2_value)),
            # keras.layers.BatchNormalization(),
            # keras.layers.LeakyReLU(alpha=0.2),a
            
            keras.layers.Dense(1024, kernel_regularizer=keras.regularizers.l2(l2_value)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            
            # Hidden layer 4 with L2 regularization and BatchNormalization
            keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(l2_value)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
        
            # Hidden layer 5 with L2 regularization and BatchNormalization
            keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(l2_value)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            
            keras.layers.Dense(target_signal.shape[1], activation='linear')  # Output shape for 1D signal
            
        ])
        print("Generator Summary:")
        generator.summary()
        # Define the discriminator
        discriminator = keras.Sequential([
            # keras.layers.Dense(256, input_shape=(target_signal.shape[1],), activation='relu'),  # Adjust input shape
            # keras.layers.Dense(256, activation='relu'),
            # keras.layers.Dense(128, activation='relu'),
            # keras.layers.Dense(64, activation='relu'),
            # keras.layers.Dense(1, activation='sigmoid')  # Output a single scalar value
            
            keras.layers.Dense(256, input_shape=(target_signal.shape[1],), activation='relu'),  # Adjust input shape
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Output a single scalar value
        ])
        print("Discriminator Summary:")
        
        discriminator.summary()
        
        custom_learning_rate = 0.001  # Replace with your desired learning rate
    
        def custom_loss(y_true, y_pred):
            
            # Calculate the squared errors
            # squared_errors = tf.square(y_true - y_pred)

            # # Importance coefficients for the first 64 features (set to 1)
            # importance_coefficients = tf.concat([tf.ones(128, dtype=tf.float32), 
            #                                      tf.linspace(0.1, 0.0, 128)], axis=0)

            # # Apply importance coefficients to squared errors
            # weighted_errors = squared_errors * importance_coefficients

            # # Calculate the weighted mean of the squared errors
            # mse_loss = tf.reduce_mean(weighted_errors)

            # # Calculate the mean squared error (MSE) loss
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # Ensure all features are larger than 0
            min_feature_value = 0.0
            feature_min_loss = tf.reduce_mean(tf.nn.relu(min_feature_value - y_pred))
            
            # Encourage the last 192 features to be near zero
            # last_192_loss = tf.reduce_mean(tf.square(y_pred[:, -192:]))
            
            # Gradually increase the penalty on the last 192 features
            # last_192_loss = tf.reduce_mean(tf.square(y_pred[:, -192:]) * tf.range(1, 193, dtype=tf.float32))
            
            # Encourage the maximum values to be similar
            # max_value_loss = tf.reduce_mean(tf.square(tf.reduce_max(y_true) - tf.reduce_max(y_pred)))
            
            # Calculate the L2 regularization loss on the last 192 features
            last_192_features = y_pred[:, -128:]
            l2_loss = tf.reduce_sum(tf.square(last_192_features))

            # Combine the MSE loss and L2 regularization loss
            total_loss = mse_loss +  feature_min_loss

            return total_loss
        
        # initial_learning_rate = 0.001  # Set your initial learning rate
        # lr_schedule = ExponentialDecay(
        #     initial_learning_rate,
        #     decay_steps=500,  # Adjust decay steps as needed
        #     decay_rate=0.5  # Adjust decay rate as needed
        # )
        # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=custom_loss)
        # discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Define the GAN model
        discriminator.trainable = False
        gan_input = keras.Input(shape=(target_signal.shape[1],))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = keras.Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        
        discriminator_losses = []
        generator_losses = []
        
        # Training loop
        epochs =5000
        batch_size = 96
        
        # min_loss = float('inf')
        # patience = 5000
        # early_stopping_counter = 0
        start_time = time.time()
        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (batch_size, target_signal.shape[1]))
            
            generated_signals = generator.predict(noise)
            real_signals = target_signal[np.random.choice(len(target_signal), batch_size)]
        
            # Train the discriminator
            discriminator_loss_real = discriminator.train_on_batch(real_signals, np.ones(batch_size))
            discriminator_loss_generated = discriminator.train_on_batch(generated_signals, np.zeros(batch_size))
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_generated)
        
            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, target_signal.shape[1]))

            generator_loss = gan.train_on_batch(noise, np.ones(batch_size))
            
            discriminator_losses.append((discriminator_loss_real[0] + discriminator_loss_generated[0]) / 2)
            generator_losses.append(generator_loss)
        
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: D Loss = {discriminator_loss[0]}, G Loss = {generator_loss}")
                
            
        # Save generator model
        generator.save(model_name)

        # Save generator weights
        generator.save_weights(weights_name)
        
        noise = np.random.normal(0, 1, (batch_size * 3, target_signal.shape[1]))
        
        new_signals = generator.predict(noise)
        new_label = vector = np.full((batch_size * 3,), damagei)
        
        dataset_all = np.vstack((dataset_all, new_signals))
        label_damage_all = np.concatenate((label_damage_all, new_label), axis=0)
        
        target_signal = np.transpose(target_signal)
        new_signals = np.transpose(new_signals)
        
        # Save generated signals
        np.save(signals_name, new_signals)

        
        plt.plot(target_signal)  # Plot the first target signal as an example
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title((f'Target Signal (layer: {layer})\nDamage: {damagei:.2f}'))
        plt.show()
        
        plt.plot(new_signals)  # Plot the first generated signal as an example
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title((f'Generated Signal (layer: {layer})\nDamage: {damagei:.2f}'))
        plt.show()
        
        
        # Plot discriminator and generator losses in one figure
        plt.plot(discriminator_losses, label="Discriminator Loss", color='blue')
        # plt.plot(generator_losses, label="Generator Loss", color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Discriminator Losses")
        plt.show()
        
        plt.plot(generator_losses, label="Generator Loss", color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Generator Losses")
        plt.show()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)
        print(f"Time = {elapsed_time}")
        

label_damage_all = to_categorical(label_damage_all, num_classes=4)


# Split the dataset into train, validation, and test sets
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for train_index, temp_index in split1.split(dataset_all, label_damage_all):
    x_train, x_temp = dataset_all[train_index], dataset_all[temp_index]
    y_train, y_temp = label_damage_all[train_index], label_damage_all[temp_index]

for val_index, test_index in split2.split(x_temp, y_temp):
    x_val, x_test = x_temp[val_index], x_temp[test_index]
    y_val, y_test = y_temp[val_index], y_temp[test_index]

# y_train = to_categorical(y1, num_classes=4)
# y_test = to_categorical(y2, num_classes=4)

# # Define a list to store the results
# # Define a DataFrame to store the results
results_df = pd.DataFrame(columns=['Iteration', 'Best Num Layers', 'Best Num Units', 'Val Accuracy','Test Accuracy','Time'])

# Define the 1D-CNN model
def create_model(num_layers, num_units):
    model = keras.Sequential()
    model.add(layers.Input(shape=(256, 1)))  # Input shape for 1D data

    for _ in range(int(num_layers)):
        model.add(layers.Conv1D(filters=int(num_units), kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Modify the input data shape to include the channel dimension
x_train = x_train[:, :, np.newaxis]
x_val = x_val[:, :, np.newaxis]
x_test = x_test[:, :, np.newaxis]

# Define the objective function to optimize
def objective(num_layers, num_units):
    num_layers = int(num_layers)
    num_units = int(num_units)
    
    num_units = 2 ** num_units
    # Create the final model with the specified hyperparameters
    final_model = create_model(num_layers, num_units)
    
    start_time = time.time()

    # Train the final model on the entire training set
    final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=0)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Evaluate the model on the val set
    y_pred_val = final_model.predict(x_val, verbose=0)
    val_accuracy = accuracy_score(y_val.argmax(axis=1), y_pred_val.argmax(axis=1))
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_val.argmax(axis=1), y_pred_val.argmax(axis=1))
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
                yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix Gan val\nAccuracy: {val_accuracy:.2f}')
    # 
    plt.show()
    

    # Evaluate the model on the test set
    y_pred = final_model.predict(x_test, verbose=0)
    test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
                yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix Gan test\nAccuracy: {test_accuracy:.2f}')
    # 
    plt.show()
    
    
    
    # Store the results in the DataFrame
    results_df.loc[len(results_df)] = [len(results_df) + 1, num_layers, num_units, val_accuracy, test_accuracy,elapsed_time]
    
    return test_accuracy  # Negative because BayesianOptimization seeks to minimize

# Rest of the code remains the same
# Define the hyperparameter search space
pbounds = {'num_layers': (1, 5), 'num_units': (5, 12)}

# Create the BayesianOptimization object
optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

# Perform Bayesian optimization to find the best hyperparameters
optimizer.maximize(init_points=5, n_iter=20)

# Save the results to a CSV file
results_df.to_csv('bayesian_optimization_gan_results20240713.csv', index=False)


plt.figure(figsize=(10, 6))
plt.plot(results_df['Iteration'], results_df['Test Accuracy'], marker='o', linestyle='-')
plt.title('Accuracy vs. Iteration during Bayesian Optimization L2L3 gan ')
plt.xlabel('Iteration')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.show()

################################################################################################
#
# ############################################################################################
# ##1D-CNN model to detect the L2 and L3,data augmentation by gan


# label_damage_all = to_categorical(label_damage_all, num_classes=4)

# split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
# for train_index,test_index in split1.split(dataset_all,label_damage_all):
#     x_train,x_test = dataset_all[train_index],dataset_all[test_index]
#     y_train,y_test = label_damage_all[train_index],label_damage_all[test_index]

# # y_train = to_categorical(y1, num_classes=4)
# # y_test = to_categorical(y2, num_classes=4)

# # # Define a list to store the results
# # Define a DataFrame to store the results
# results_df = pd.DataFrame(columns=['Iteration', 'Best Num Layers', 'Best Num Units', 'Test Accuracy'])
# 
# # Define the 1D-CNN model
# def create_model(num_layers, num_units):
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(256, 1)))  # Input shape for 1D data

#     for _ in range(int(num_layers)):
#         model.add(layers.Conv1D(filters=int(num_units), kernel_size=3, activation='relu'))
#         model.add(layers.MaxPooling1D(pool_size=2))
    
#     model.add(layers.Flatten())
#     model.add(layers.Dense(4, activation='softmax'))
    
#     optimizer = keras.optimizers.Adam(learning_rate=0.001)

#     model.compile(optimizer=optimizer,
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
    
#     return model

# # Modify the input data shape to include the channel dimension
# x_train = x_train[:, :, np.newaxis]
# x_test = x_test[:, :, np.newaxis]

# # Define the objective function to optimize
# def objective(num_layers, num_units):
#     num_layers = int(num_layers)
#     num_units = int(num_units)
    
#     # Create the final model with the specified hyperparameters
#     final_model = create_model(num_layers, num_units)

#     # Train the final model on the entire training set
#     final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=1)

#     # Evaluate the model on the test set
#     y_pred = final_model.predict(x_test)
#     test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
#     # Store the results in the DataFrame
#     results_df.loc[len(results_df)] = [len(results_df) + 1, num_layers, num_units, test_accuracy]
    
#     return test_accuracy  # Negative because BayesianOptimization seeks to minimize

# # # Rest of the code remains the same
# # # Define the hyperparameter search space
# pbounds = {'num_layers': (1, 5), 'num_units': (32, 1024)}

# # Create the BayesianOptimization object
# optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

# # Perform Bayesian optimization to find the best hyperparameters
# optimizer.maximize(init_points=5, n_iter=20)

# # Save the results to a CSV file
# results_df.to_csv('bayesian_optimization_results.csv', index=False)

# # Get the best hyperparameters
# best_params = optimizer.max['params']
# best_num_layers = int(best_params['num_layers'])
# best_num_units = int(best_params['num_units'])

# # Create the final model with the best hyperparameters
# final_model = create_model(best_num_layers, best_num_units)

# # Train the final model on the entire training set
# final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=1)

# # Evaluate the model on the test set
# y_pred = final_model.predict(x_test)
# test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# print(f"Test Accuracy: {test_accuracy}")

# # Calculate the confusion matrix
# cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

# # Calculate precision, recall, and F1-score for each class
# precision = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None)
# recall = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None)
# f1 = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None)

# # Calculate the accuracy for each class
# class_accuracy = cm.diagonal() / cm.sum(axis=1)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
#             yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Confusion Matrix L2L3 gan (Repeats: {num_damageR})\nAccuracy: {test_accuracy:.2f}')

# plt.show()

# # Plot class-wise accuracy, precision, recall, and F1-score as text on the heatmap
# for i in range(len(class_accuracy)):
#     plt.text(i, i, f'Accuracy: {class_accuracy[i]:.2f}\nPrecision: {precision[i]:.2f}\nRecall: {recall[i]:.2f}\nF1: {f1[i]:.2f}', ha='center', va='center', color='red')

# plt.show()


# plt.figure(figsize=(10, 6))
# plt.plot(results_df['Iteration'], results_df['Test Accuracy'], marker='o', linestyle='-')
# plt.title('Accuracy vs. Iteration during Bayesian Optimization L2L3 gan ')
# plt.xlabel('Iteration')
# plt.ylabel('Test Accuracy')
# plt.grid(True)
# plt.show()


# ############################################################################################
# ##1D-CNN model to detect the L2 and L3,data augmentation by repeat 


# label_damage_all_3 = to_categorical(label_damage_all_3, num_classes=4)

# split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
# for train_index,test_index in split1.split(dataset_all_3,label_damage_all_3):
#     x_train,x_test = dataset_all_3[train_index],dataset_all_3[test_index]
#     y_train,y_test = label_damage_all_3[train_index],label_damage_all_3[test_index]

# # y_train = to_categorical(y1, num_classes=4)
# # y_test = to_categorical(y2, num_classes=4)

# # # Define a list to store the results
# # # Define a DataFrame to store the results
# results_df = pd.DataFrame(columns=['Iteration', 'Best Num Layers', 'Best Num Units', 'Test Accuracy'])

# # Define the 1D-CNN model
# def create_model(num_layers, num_units):
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(256, 1)))  # Input shape for 1D data

#     for _ in range(int(num_layers)):
#         model.add(layers.Conv1D(filters=int(num_units), kernel_size=3, activation='relu'))
#         model.add(layers.MaxPooling1D(pool_size=2))
    
#     model.add(layers.Flatten())
#     model.add(layers.Dense(4, activation='softmax'))
    
#     optimizer = keras.optimizers.Adam(learning_rate=0.001)

#     model.compile(optimizer=optimizer,
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
    
#     return model

# # Modify the input data shape to include the channel dimension
# x_train = x_train[:, :, np.newaxis]
# x_test = x_test[:, :, np.newaxis]

# # Define the objective function to optimize
# def objective(num_layers, num_units):
#     num_layers = int(num_layers)
#     num_units = int(num_units)
    
#     # Create the final model with the specified hyperparameters
#     final_model = create_model(num_layers, num_units)

#     # Train the final model on the entire training set
#     final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=1)

#     # Evaluate the model on the test set
#     y_pred = final_model.predict(x_test)
#     test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
#     # Store the results in the DataFrame
#     results_df.loc[len(results_df)] = [len(results_df) + 1, num_layers, num_units, test_accuracy]
    
#     return test_accuracy  # Negative because BayesianOptimization seeks to minimize

# # Rest of the code remains the same
# # Define the hyperparameter search space
# pbounds = {'num_layers': (1, 5), 'num_units': (32, 1024)}

# # Create the BayesianOptimization object
# optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

# # Perform Bayesian optimization to find the best hyperparameters
# optimizer.maximize(init_points=5, n_iter=20)

# # Save the results to a CSV file
# results_df.to_csv('bayesian_optimization_results.csv', index=False)

# # Get the best hyperparameters
# best_params = optimizer.max['params']
# best_num_layers = int(best_params['num_layers'])
# best_num_units = int(best_params['num_units'])

# # Create the final model with the best hyperparameters
# final_model = create_model(best_num_layers, best_num_units)

# # Train the final model on the entire training set
# final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=1)

# # Evaluate the model on the test set
# y_pred = final_model.predict(x_test)
# test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# print(f"Test Accuracy: {test_accuracy}")

# # Calculate the confusion matrix
# cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

# # Calculate precision, recall, and F1-score for each class
# precision = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None)
# recall = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None)
# f1 = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None)

# # Calculate the accuracy for each class
# class_accuracy = cm.diagonal() / cm.sum(axis=1)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
#             yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Confusion Matrix L2L3 R3 (Repeats: {num_damageR})\nAccuracy: {test_accuracy:.2f}')

# plt.show()

# # Plot class-wise accuracy, precision, recall, and F1-score as text on the heatmap
# for i in range(len(class_accuracy)):
#     plt.text(i, i, f'Accuracy: {class_accuracy[i]:.2f}\nPrecision: {precision[i]:.2f}\nRecall: {recall[i]:.2f}\nF1: {f1[i]:.2f}', ha='center', va='center', color='red')

# plt.show()


# plt.figure(figsize=(10, 6))
# plt.plot(results_df['Iteration'], results_df['Test Accuracy'], marker='o', linestyle='-')
# plt.title('Accuracy vs. Iteration during Bayesian Optimization L2L3 R3 ')
# plt.xlabel('Iteration')
# plt.ylabel('Test Accuracy')
# plt.grid(True)
# plt.show()

############################################################################################
#doing transfer
# split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
# for train_index,test_index in split1.split(dataset_l2,label_damage_l2):
#     x_train,x_test = dataset_l2[train_index],dataset_l2[test_index]
#     y_train,y_test = label_damage_l2[train_index],label_damage_l2[test_index]

# # y_train = to_categorical(y1, num_classes=4)
# # y_test = to_categorical(y2, num_classes=4)

# # # Define a list to store the results
# # # Define a DataFrame to store the results
# results_df = pd.DataFrame(columns=['Iteration', 'Best Num Layers', 'Best Num Units', 'Test Accuracy'])

# # Define the DNN model
# def create_model(num_layers, num_units):
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(256,)))

#     for _ in range(int(num_layers)):
#         model.add(layers.Dense(int(num_units), activation='relu'))
    
#     model.add(layers.Dense(4, activation='softmax'))
    
#     optimizer = keras.optimizers.Adam(learning_rate=0.001)  # You can adjust the learning rate

#     model.compile(optimizer=optimizer,
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
    
#     return model

# # Define the objective function to optimize
# def objective(num_layers, num_units):
#     num_layers = int(num_layers)
#     num_units = int(num_units)
    
#     # Create the final model with the specified hyperparameters
#     final_model = create_model(num_layers, num_units)

#     # Train the final model on the entire training set
#     final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=0)

#     # Evaluate the model on the test set
#     y_pred = final_model.predict(x_test)
#     test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))  # Calculate accuracy
    
#     # Store the results in the DataFrame
#     results_df.loc[len(results_df)] = [len(results_df) + 1, num_layers, num_units, test_accuracy]
    
#     return test_accuracy  # Negative because BayesianOptimization seeks to minimize

# # Define the hyperparameter search space
# pbounds = {'num_layers': (1, 10), 'num_units': (32, 1024)}

# # Create the BayesianOptimization object
# optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

# # Perform Bayesian optimization to find the best hyperparameters
# optimizer.maximize(init_points=10, n_iter=30)

# # Save the results to a CSV file
# results_df.to_csv('bayesian_optimization_results.csv', index=False)

# # Get the best hyperparameters
# best_params = optimizer.max['params']
# best_num_layers = int(best_params['num_layers'])
# best_num_units = int(best_params['num_units'])

# # Create the final model with the best hyperparameters
# final_model = create_model(best_num_layers, best_num_units)

# # Train the final model on the entire training set
# final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=2)

# # Evaluate the model on the test set
# y_pred = final_model.predict(x_test)
# test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# print(f"Test Accuracy: {test_accuracy}")

# # Calculate the confusion matrix
# cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
#             yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Confusion Matrix L2\nAccuracy: {test_accuracy:.2f}')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(results_df['Iteration'], results_df['Test Accuracy'], marker='o', linestyle='-')
# plt.title('Accuracy vs. Iteration during Bayesian Optimization For l2')
# plt.xlabel('Iteration')
# plt.ylabel('Test Accuracy')
# plt.grid(True)
# plt.show()

# # Save the final model and its weights
# final_model.save('final_model.h5')

# ##
# #transfer 
# split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
# for train_index,test_index in split1.split(dataset_l3,label_damage_l3):
#     x_train_new,x_test_new = dataset_l3[train_index],dataset_l3[test_index]
#     y_train_new,y_test_new = label_damage_l3[train_index],label_damage_l3[test_index]
    

# ##


# # Load the saved model
# loaded_model = keras.models.load_model('final_model.h5')


# y_pred_new1 = loaded_model.predict(x_test_new)
# test_accuracy = accuracy_score(y_test_new.argmax(axis=1), y_pred_new1.argmax(axis=1))
# # Calculate the confusion matrix
# cm = confusion_matrix(y_test_new.argmax(axis=1), y_pred_new1.argmax(axis=1))

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
#             yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Confusion Matrix L2 transfer L3 raw\nAccuracy: {test_accuracy:.2f}')
# plt.show()

# print(f"Test Accuracy: {test_accuracy}")


# for layer in loaded_model.layers[:-2]:
#     layer.trainable = False
    
# # Compile the model with an appropriate optimizer and loss function
# loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model on your new dataset, allowing the last few layers to be fine-tuned
# # You may need to one-hot encode your target labels if they are not already in that format
# # Use the categorical cross-entropy loss for multi-class classification
# loaded_model.fit(x_train_new, y_train_new, epochs=100, batch_size=32, validation_data=(x_test_new, y_test_new))

# # Evaluate the model on the new dataset
# y_pred_new = loaded_model.predict(x_test_new)
# test_accuracy = accuracy_score(y_test_new.argmax(axis=1), y_pred_new.argmax(axis=1))
# print(f"Test Accuracy: {test_accuracy}")
# # Calculate the confusion matrix
# cm = confusion_matrix(y_test_new.argmax(axis=1), y_pred_new.argmax(axis=1))

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
#             yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Confusion Matrix L2 transfer L3 after fitting\nAccuracy: {test_accuracy:.2f}')
# plt.show()


# accuracy = loaded_model.evaluate(x_test_new, y_test_new)
# print(f"Test Accuracy on the New Dataset: {accuracy[1]}")






















#############################################################################
#先利用小支座的数据，进行模型预测
#dataset split
# split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
# for train_index,test_index in split1.split(dataset_small,label_damage_small):
#     x_train,x_test = dataset_small[train_index],dataset_small[test_index]
#     y_train,y_test = label_damage_small[train_index],label_damage_small[test_index]
    
# # Define a list to store the results
# # Define a DataFrame to store the results
# results_df = pd.DataFrame(columns=['Iteration', 'Best Num Layers', 'Best Num Units', 'Test Accuracy'])

# # Define the DNN model
# def create_model(num_layers, num_units):
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(256,)))

#     for _ in range(int(num_layers)):
#         model.add(layers.Dense(int(num_units), activation='relu'))
    
#     model.add(layers.Dense(1, activation='sigmoid'))
    
#     optimizer = keras.optimizers.Adam(learning_rate=0.001)  # You can adjust the learning rate

#     model.compile(optimizer=optimizer,
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
    
#     return model

# # Define the objective function to optimize
# def objective(num_layers, num_units):
#     num_layers = int(num_layers)
#     num_units = int(num_units)
    
#     # Create the final model with the specified hyperparameters
#     final_model = create_model(num_layers, num_units)

#     # Train the final model on the entire training set
#     final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=0)

#     # Evaluate the model on the test set
#     y_pred = (final_model.predict(x_test) > 0.5).astype(int)
#     test_accuracy = accuracy_score(y_test, y_pred)
    
#     # Store the results in the DataFrame
#     results_df.loc[len(results_df)] = [len(results_df) + 1, num_layers, num_units, test_accuracy]
    
#     return test_accuracy  # Negative because BayesianOptimization seeks to minimize

# # Define the hyperparameter search space
# pbounds = {'num_layers': (6, 10), 'num_units': (32, 1024)}

# # Create the BayesianOptimization object
# optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

# # Perform Bayesian optimization to find the best hyperparameters
# optimizer.maximize(init_points=10, n_iter=30)

# # Save the results to a CSV file
# results_df.to_csv('bayesian_optimization_results.csv', index=False)

# # Get the best hyperparameters
# best_params = optimizer.max['params']
# best_num_layers = int(best_params['num_layers'])
# best_num_units = int(best_params['num_units'])

# # Create the final model with the best hyperparameters
# final_model = create_model(best_num_layers, best_num_units)

# # Train the final model on the entire training set
# final_model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=2)

# # Evaluate the model on the test set
# y_pred = (final_model.predict(x_test) > 0.5).astype(int)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Test Accuracy: {accuracy}")

# # Calculate the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=[' Healthy', ' Damage'], 
#             yticklabels=[' Healthy', ' Damage'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Save the final model and its weights
# final_model.save('final_model.h5')
# y_pred = (final_model.predict(x_test) > 0.5).astype(int)
# accuracy = accuracy_score(y_test, y_pred)


#############################################################################
#直接训练大支座的模型
# dataset split

# split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
# for train_index,test_index in split1.split(dataset_big_all,label_damage_big_all):
#     x_train,x_test = dataset_big_all[train_index],dataset_big_all[test_index]
#     y_train,y_test = label_damage_big_all[train_index],label_damage_big_all[test_index]
    
    
# # Define a list to store the results
# # Define a DataFrame to store the results
# results_df = pd.DataFrame(columns=['Iteration', 'Best Num Layers', 'Best Num Units', 'Test Accuracy'])

# # Define the DNN model
# def create_model(num_layers, num_units):
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(256,)))

#     for _ in range(int(num_layers)):
#         model.add(layers.Dense(int(num_units), activation='relu'))
    
#     model.add(layers.Dense(1, activation='sigmoid'))
    
#     optimizer = keras.optimizers.Adam(learning_rate=0.001)  # You can adjust the learning rate

#     model.compile(optimizer=optimizer,
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
    
#     return model

# # Define the objective function to optimize
# def objective(num_layers, num_units):
#     num_layers = int(num_layers)
#     num_units = int(num_units)
    
#     # Create the final model with the specified hyperparameters
#     final_model = create_model(num_layers, num_units)

#     # Train the final model on the entire training set
#     final_model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=0)

#     # Evaluate the model on the test set
#     y_pred = (final_model.predict(x_test) > 0.5).astype(int)
#     test_accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
    
#     # Store the results in the DataFrame
#     results_df.loc[len(results_df)] = [len(results_df) + 1, num_layers, num_units, precision]
    
#     return test_accuracy  # Negative because BayesianOptimization seeks to minimize

# # Define the hyperparameter search space 
# pbounds = {'num_layers': (1, 10), 'num_units': (32, 1024)}

# # Create the BayesianOptimization object
# optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

# # Perform Bayesian optimization to find the best hyperparameters
# optimizer.maximize(init_points=2, n_iter=10)

# # Save the results to a CSV file
# results_df.to_csv('bayesian_optimization_results.csv', index=False)

# # Get the best hyperparameters
# best_params = optimizer.max['params']
# best_num_layers = int(best_params['num_layers'])
# best_num_units = int(best_params['num_units'])

# # Create the final model with the best hyperparameters
# final_model = create_model(best_num_layers, best_num_units)

# # Train the final model on the entire training set
# final_model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2)

# # Evaluate the model on the test set
# y_pred = (final_model.predict(x_test) > 0.5).astype(int)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# print(f"Test Accuracy: {accuracy}")

# # Calculate the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=[' Healthy', ' Damage'], 
#             yticklabels=[' Healthy', ' Damage'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()



#############################################################################
# Load the saved model for transfer learning
# loaded_model = keras.models.load_model('final_model.h5')

# # split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
# # for train_index,test_index in split1.split(dataset_big,label_damage_big):
# #     x_new_train,x = dataset_big[train_index],dataset_big[test_index]
# #     y_new_train,y = label_damage_big[train_index],label_damage_big[test_index]

# # x_new_test = np.concatenate((x, dataset7), axis=0)
# # y_new_test = np.concatenate((y, label_damage7), axis=None)
# split1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
# for train_index,test_index in split1.split(dataset_big_all,label_damage_big_all):
#     x_new_train,x_new_test = dataset_big_all[train_index],dataset_big_all[test_index]
#     y_new_train,y_new_test = label_damage_big_all[train_index],label_damage_big_all[test_index]



# y_new_pred = (loaded_model.predict(x_new_test) > 0.5).astype(int)
# accuracy = accuracy_score(y_new_test, y_new_pred)
# print(f"Test Accuracy: {accuracy}")

# # Calculate the confusion matrix
# cm_new = confusion_matrix(y_new_test, y_new_pred)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=[' Healthy', ' Damage'], 
#             yticklabels=[' Healthy', ' Damage'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# #############################################################################
# # # Assuming it's a binary classification task (1 output unit)
# new_output_layer = layers.Dense(1, activation='sigmoid')

# # Create a new model for fine-tuning by combining the loaded model and the new output layer
# fine_tuned_model = keras.Sequential()
# fine_tuned_model.add(loaded_model)  # Use the layers from the loaded model
# # fine_tuned_model.add(new_output_layer)  # Add the new output layer

# # # Unfreeze specific layers in the loaded model for fine-tuning
# # # Example: Unfreeze layers from index 5 onwards
# for layer in loaded_model.layers[:-1]:
# # # for layer in loaded_model.layers:
#     layer.trainable = True

# # Compile the fine-tuned model
# optimizer = keras.optimizers.Adam(learning_rate=0.001)  # You can adjust the learning rate
# fine_tuned_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Fine-tune the model on the new dataset
# fine_tuned_model.fit(x_new_train, y_new_train, epochs=100, batch_size=64, verbose=2)

# # Evaluate the fine-tuned model on the new test set
# anomaly_scores = fine_tuned_model.predict(x_new_test)
# threshold = 0.2  # You can adjust the threshold
# y_new_pred = (anomaly_scores > threshold).astype(int)
# # y_new_pred = (fine_tuned_model.predict(x_new_test) > 0.5).astype(int)
# accuracy_new = accuracy_score(y_new_test, y_new_pred)
# print(f"New Dataset Test Accuracy: {accuracy_new}")

# # # Calculate the confusion matrix
# cm = confusion_matrix(y_new_test, y_new_pred)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=[' Healthy', ' Damage'], 
#             yticklabels=[' Healthy', ' Damage'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Save the fine-tuned model and its weights
# fine_tuned_model.save('fine_tuned_model.h5')




##############################################################################################
# Create a new output layer for the anomaly detection task
# new_output_layer = layers.Dense(1, activation='sigmoid')

# # Create a new model for anomaly detection by combining the loaded model and the new output layer
# anomaly_detection_model = keras.Sequential()
# # anomaly_detection_model.add(loaded_model)  # Use the layers from the loaded model

# # Add the new output layer (optional, you can replace it with your custom layer)
# anomaly_detection_model.add(new_output_layer)  # Add the new output layer

# # Compile the anomaly detection model
# optimizer = keras.optimizers.Adam(learning_rate=0.0001)  # You can adjust the learning rate
# anomaly_detection_model.compile(optimizer=optimizer, loss='binary_crossentropy')

# # # Freeze all layers in the loaded model (optional)
# # for layer in loaded_model.layers:
# #     layer.trainable = False

# # Unfreeze specific layers in the loaded model for fine-tuning
# # Example: Unfreeze layers from index 5 onwards
# for layer in loaded_model.layers:
#     layer.trainable = True

# # Train the model on the data labeled as class 0 (x_class0)
# anomaly_detection_model.fit(x_new_train, y_new_train, epochs=10, batch_size=32, verbose=2)

# # Evaluate the anomaly detection model on the test set with both classes (x_test)
# anomaly_scores = anomaly_detection_model.predict(x_new_test)
# # Anomaly scores can be used to identify anomalies or classify instances based on a threshold

# # For example, you can classify instances with scores above a threshold as class 1 (anomaly) and below as class 0
# threshold = 0.2  # You can adjust the threshold
# predictions = (anomaly_scores > threshold).astype(int)

# # Evaluate the predictions on the test set
# accuracy_new = accuracy_score(y_new_test, predictions)
# print(f"Anomaly Detection Test Accuracy: {accuracy_new}")

# # Calculate the confusion matrix
# cm = confusion_matrix(y_new_test, predictions)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=[' Healthy', ' Damage'], 
#             yticklabels=[' Healthy', ' Damage'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Save the anomaly detection model and its weights
# anomaly_detection_model.save('anomaly_detection_model.h5')