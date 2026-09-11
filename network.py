import cupy

import numpy as np
from tqdm import tqdm

from utils import Layer, FLOAT_TYPE
from layers import *
from transformer_adapters import *


class CrossEntropy:

    def __init__(self, num_classes):
        self.one_hot = cupy.eye(num_classes, dtype = FLOAT_TYPE)

    def gradients(self, logits, labels):
        return logits - self.one_hot[labels] # Combined Softmax + Cross Entropy Loss
      
    def loss(self, logits, labels):
        labels = self.one_hot[labels]
        eta = 1e-7
        return -1 * cupy.sum(labels * cupy.log(logits + eta))
    
    def num_samples(self, labels):
        return labels.size


class Network:

    def __init__(self, layers:list[Layer]):
        self.layers = layers
        
    def predict(self, input):
        return self._forward(input)

    def _forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def _backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def set_eval(self, eval_mode):
        for layer in self.layers:
            layer.set_eval(eval_mode)
            layer.zero_grad()
            
    def _zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def _zero_adam(self):
        for layer in self.layers:
            layer.zero_adam()

    def _update(self, learning_rate, weight_decay, t, num_samples):

        beta1, beta2 = 0.9, 0.999

        for layer in self.layers:
            if layer.is_embed_layer_back:
                continue
            
            for param, grad, moment, variance in zip(layer.parameters,
                                                     layer.gradients,
                                                     layer.moments,
                                                     layer.variances):
                grad /= num_samples
                lmda = weight_decay
                
                if len(param.shape) == 1 or isinstance(layer, (VitProjector, VitMLPHead, 
                                                               GPTEmbedFront, GPTEmbedBack)):
                    lmda = 0.0
                    
                moment *= beta1
                moment += (1 - beta1)*grad

                variance *= beta2
                variance += (1 - beta2)*grad**2

                mom_hat = moment / (1 - beta1**t)
                var_hat = variance / (1 - beta2**t)

                param -= learning_rate * (mom_hat / (var_hat**0.5 + 1e-7) + lmda * param)

    def train(self, criterion, train_data, train_labels, test_data = None, test_labels = None,
                    augments = None, epochs = 1, batch_size = 64, batches_per_step = 1,
                    learning_rate = 0.001, weight_decay = 0.01):

        step_count = 1
        self._zero_adam()
        
        for i in range(epochs):
            
            self.set_eval(False)

            samples_in_step = 0
            batches_seen = 0
            
            train_loss, train_correct = 0, 0

            shuffle = np.random.permutation(len(train_labels))
            train_data   = train_data[shuffle]
            train_labels = train_labels[shuffle]

            for j in tqdm(range(0, len(train_data), batch_size)):
                
                x = train_data  [j: min(j + batch_size, len(train_data))]
                y = train_labels[j: min(j + batch_size, len(train_data))]
                
                if augments is not None:
                    x = augments(x)

                x = cupy.array(x)
                y = cupy.array(y)
                
                y_hat = self._forward(x)
                
                grad = criterion.gradients(y_hat, y)
                self._backward(grad)
                
                samples_in_step += criterion.num_samples(y)
                batches_seen += 1
                
                if batches_seen % batches_per_step == 0:
                    self._update(learning_rate, weight_decay, step_count, samples_in_step)
                    self._zero_grad()
                    step_count += 1
                    samples_in_step = 0

                train_loss += criterion.loss(y_hat, y)
                train_correct += cupy.equal(cupy.argmax(y_hat, axis = -1), y).astype(cupy.int32).sum()
                
            train_loss     = train_loss    / np.prod(train_labels.shape)
            train_accuracy = train_correct / np.prod(train_labels.shape)
            print("epoch:", i + 1, "train loss:", train_loss, "train accuracy:", train_accuracy)
            
            if test_data is not None and test_labels is not None:
                test_loss, test_accuracy = self.evaluate(test_data, test_labels, criterion, batch_size=batch_size)
                print("epoch:", i + 1, "test loss:", test_loss, "test accuracy:", test_accuracy, "\n")

    def evaluate(self, test_data, test_labels, criterion, batch_size = 64):
        
        self.set_eval(True)
        loss, correct = 0, 0

        for i in tqdm(range(0, len(test_data), batch_size)):

            x = cupy.array(test_data  [i: min(i + batch_size, len(test_data))])
            y = cupy.array(test_labels[i: min(i + batch_size, len(test_data))])

            y_hat = self.predict(x)

            loss += criterion.loss(y_hat, y)
            correct += cupy.equal(cupy.argmax(y_hat, axis = -1), y).astype(cupy.int32).sum()

        return loss / np.prod(test_labels.shape), correct / np.prod(test_labels.shape)