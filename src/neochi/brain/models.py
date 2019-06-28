# MIT License
#
# Copyright (c) 2019 Morning Project Samurai (MPS)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


__author__ = 'Junya Kaneko <junya@mpsamurai.org>'


import os
import json
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self, *args, **kwargs):
        print('INITIALIZE MODEL %s' % self.__class__.__name__)
        self._model = None
        print('MODEL %s INITIALIZED' % self.__class__.__name__)

    def fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass


class BehaviorClassifier(Model):
    def __init__(self, shape=None, fps=None, labels=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape = shape
        self._fps = fps
        self.labels = labels

    @property
    def shape(self):
        return self._shape

    @property
    def shape_with_batch(self):
        return -1, self._shape[0], self._shape[1], self._shape[2]

    @property
    def image_size(self):
        return self._shape[1], self._shape[0]

    @property
    def time_steps(self):
        return self._shape[2]

    @property
    def fps(self):
        return self._fps

    @property
    def config(self):
        return {
            'image_size': self.image_size,
            'time_steps': self.time_steps,
            'fps': self.fps
        }

    def _create_model(self):
        ipt = keras.layers.Input(shape=self._shape)
        cnn1_1 = keras.layers.SeparableConv2D(16, (3, 3), activation='relu')(ipt)
        pool1_1 = keras.layers.MaxPool2D()(cnn1_1)
        cnn2_1 = keras.layers.SeparableConv2D(32, (3, 3), activation='relu')(pool1_1)
        pool2_1 = keras.layers.MaxPool2D()(cnn2_1)
        fc1 = keras.layers.Dense(128, activation='relu')(keras.layers.Flatten()(pool2_1))
        fc2 = keras.layers.Dense(len(self.labels), activation='softmax')(fc1)
        self._model = keras.models.Model(inputs=[ipt, ], outputs=[fc2, ])

    def fit(self, X, y, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', ], train_size=0.75, epochs=20):
        self._create_model()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._model.fit(X_train, y_train, epochs=epochs)
        test_loss, test_acc = self._model.evaluate(X_test, y_test)
        print('TEST LOSS:', test_loss)
        print('TEST ACCURACY:', test_acc)
        return self

    def predict(self, X):
        return np.argmax(self._model.predict(X), axis=1)

    def score(self, X, y):
        return self._model.evaluate(X, y)

    def predict_probs(self, X):
        return self.predict(X)

    def predict_labels(self, X):
        return [self.labels[i] for i in self.predict(X)]

    def save(self, save_dir):
        print('SAVE MODEL %s' % self.__class__.__name__)
        with open(os.path.join(save_dir, 'model.json'), 'w') as f:
            json.dump({'shape': list(self._shape), 'fps': self._fps, 'labels': self.labels}, f)
        self._model.save(os.path.join(save_dir, 'model.h5'))
        print('MODEL %s SAVED' % self.__class__.__name__)

    def load(self, save_dir):
        print('LOAD MODEL %s' % self.__class__.__name__)
        with open(os.path.join(save_dir, 'model.json'), 'r') as f:
            params = json.load(f)
            self._shape, self._fps, self.labels = (params[key] for key in ['shape', 'fps', 'labels'])
        self._model = keras.models.load_model(os.path.join(save_dir, 'model.h5'))
        print('MODEL %s LOADED' % self.__class__.__name__)


class SleepDetector(Model):
    def __init__(self, time_steps=None, weights=None, fps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._time_steps = time_steps
        if isinstance(weights, np.ndarray):
            self._weights = weights
        elif isinstance(weights, list):
            self._weights = np.array(weights)
        elif weights is None:
            self._weights = weights
        else:
            raise ValueError('weights must be np.ndarray, list or None')
        self._fps = fps

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.average(X, axis=1, weights=self._weights)

    def save(self, save_dir):
        print('SAVE MODEL %s' % self.__class__.__name__)
        with open(os.path.join(save_dir, 'model.json'), 'w') as f:
            json.dump({'time_steps': self._time_steps, 'weights': self._weights.tolist(), 'fps': self._fps}, f)
        print('MODEL %s SAVED' % self.__class__.__name__)

    def load(self, save_dir):
        print('LOAD MODEL %s' % self.__class__.__name__)
        with open(os.path.join(save_dir, 'model.json'), 'r') as f:
            params = json.load(f)
            self._time_steps = params['time_steps']
            self._weights = np.array(params['weights'])
            self._fps = params['fps']
        print('MODEL %s LOADED' % self.__class__.__name__)
