from typing import Callable
import tensorflow as tf
import numpy as np
import tensorflow.keras.initializers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import SGD
import pandas as pd
from collections import defaultdict

import tensorflow_probability as tfp

from loss import quantile_loss

tf.random.set_seed(0)


class OneClassNeuralNetwork:

    def __init__(self,
                 input_dim: int,
                 hidden_layer_size: int,
                 n_inputs: int,
                 r: float = 1.0,
                 g: Callable[[tf.Tensor], tf.Tensor] = tf.nn.sigmoid):

        self.input_dim = input_dim
        self.hidden_size = hidden_layer_size
        self.r = r
        self.g = g
        self.n=n_inputs

    def custom_ocnn_loss(self, nu: float) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        def custom_hinge(_: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
            loss = 0.5 * tf.norm(self.w) + \
                   0.5 * tf.norm(self.V) + \
                   quantile_loss(self.r, y_hat, nu)
            self.r = tfp.stats.percentile(y_hat,
                                          q=100 * nu/self.n,
                                          interpolation='linear')
            return loss

        return custom_hinge

    def build_model(self):
        h_size = self.hidden_size
        model = Sequential()
        input_hidden = Dense(h_size, input_dim=self.input_dim, kernel_initializer=tf.keras.initializers.RandomNormal(0.5), name="input_hidden")
        model.add(input_hidden)
        model.add(Activation(self.g))

        # Define Dense layer from hidden to output
        hidden_output = Dense(1, name="hidden_output", kernel_initializer=tf.keras.initializers.RandomNormal(0.5))
        model.add(hidden_output)
        model.add(Activation("linear"))

        self.V = input_hidden.get_weights()[0]  # "V is the weight matrix from input to hidden units"
        self.w = hidden_output.get_weights()[0]  # "w is the scalar output obtained from the hidden to output layer"

        return model

    def train_model(self, X: np.array, y_actual: np.array,epochs: int = 50, nu: float = 1e-2, init_lr: float = 1e-2, save: bool = True):

        train_history=defaultdict(list)

        def on_epoch_end(epoch, logs):
            self.w = model.get_layer('hidden_output').get_weights()[0]
            self.V = model.get_layer('input_hidden').get_weights()[0]
          
            # Detection result
            data_y=  model.predict(X)
            data_y = pd.DataFrame(data_y)
            result = np.concatenate((data_y,y_actual), axis=1)
            result = pd.DataFrame(result, columns=['p', 'y'])
            result = result.sort_values('p', ascending=True)

            # Calculate the AUC
            inlier_parray = result.loc[lambda df: df.y == 1, 'p'].values
            outlier_parray = result.loc[lambda df: df.y == -1, 'p'].values
            sum = 0.0
            for o in outlier_parray:
                for i in inlier_parray:
                    if o < i:
                        sum += 1.0
                    elif o == i:
                        sum += 0.5
                    else:
                        sum += 0
            AUC = '{:.4f}'.format(sum / (len(inlier_parray) * len(outlier_parray)))
            TP=0
            for o in outlier_parray:
                if o<=self.r:
                    TP += 1

            train_history['auc'].append((sum / (len(inlier_parray) * len(outlier_parray))))
            train_history['recall'].append(TP/len(outlier_parray))
            outlier_cnt=0
            for i in range(20):
                if result.iloc[i]['y']==-1:
                    outlier_cnt+=1
                if i==4:
                    train_history['precision@5'].append(outlier_cnt/(i+1))
                if i==9:
                    train_history['precision@10'].append(outlier_cnt/(i+1))
                if i==19:
                    train_history['precision@20'].append(outlier_cnt/(i+1))

            print('AUC:{}'.format(AUC))
            
  
        model = self.build_model()
        learning_rate = 0.01
        sgd_optimizer = SGD(learning_rate=learning_rate)
        model.compile(optimizer=sgd_optimizer,
                      loss=self.custom_ocnn_loss(nu),
                     run_eagerly=True)

        history = model.fit(X, np.zeros((X.shape[0],)),
                            batch_size=10,
                            shuffle=True,
                            callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)],
                            epochs=epochs)

        return model, history, train_history


