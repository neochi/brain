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
import cv2
import time
import pickle
import redis
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from neochi.core import dataflow
from neochi.brain import settings


class Model:
    name = 'model'

    def __init__(self, *args, **kwargs):
        print('INITIALIZE MODEL %s' % self.__class__.__name__)
        self._model = None
        self._model_timestamp = -1
        print('MODEL %s INITIALIZED' % self.__class__.__name__)

    def fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def save(self, path):
        pass

    def load(self, path):
        pass

    def reload(self, path):
        try:
            if os.path.getmtime(path) > self._model_timestamp:
                print('RELOAD MODEL %s' % self.__class__.__name__)
                self.load(path)
                return True
            else:
                return False
        except FileNotFoundError:
            return False


class PickleModelMixin:
    def save(self, path):
        print('SAVE MODEL %s' % self.__class__.__name__)
        with open(path, 'wb') as f:
            pickle.dump(self._model, f)
        print('MODEL %s SAVED' % self.__class__.__name__)

    def load(self, path):
        print('LOAD MODEL %s' % self.__class__.__name__)
        with open(path, 'rb') as f:
            self._model = pickle.load(f)
        self._model_timestamp = os.path.getmtime(path)
        print('MODEL %s LOADED' % self.__class__.__name__)


class SkLearnModelMixin:
    def fit(self, X, y):
        return self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)


class KerasModelMixin:
    def _create_model(self):
        raise NotImplementedError

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        return self._model.evaluate(X, y)

    def save(self, path):
        print('SAVE MODEL %s' % self.__class__.__name__)
        self._model.save(path)
        print('MODEL %s SAVED' % self.__class__.__name__)

    def load(self, path):
        print('LOAD MODEL %s' % self.__class__.__name__)
        self._model = keras.models.load_model(path)
        self._model_timestamp = os.path.getmtime(path)
        print('MODEL %s LOADED' % self.__class__.__name__)


class KerasClassifierMixin(KerasModelMixin):
    class_labels = []

    def fit(self, X, y,
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', ], train_size=0.75):
        self._create_model()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._model.fit(X_train, y_train, epochs=20)
        test_loss, test_acc = self._model.evaluate(X_test, y_test)
        print('TEST LOSS:', test_loss)
        print('TEST ACCURACY:', test_acc)
        return self

    def predict_class(self, X):
        return np.argmax(self.predict(X))

    def predict_class_label(self, X):
        return self.class_labels[self.predict_class(X)]


class TrainDataLoader:
    def __init__(self, base_data_dir, model_cls):
        self._data_dir = os.path.join(base_data_dir, model_cls.name)
        self._model_cls = model_cls

    def load(self):
        pass


class ClassifierTrainDataLoader(TrainDataLoader):
    def load(self):
        X, y = [], []
        for i, label in enumerate(self._model_cls.class_labels):
            label_data_dir = os.path.join(self._data_dir, label)
            for file_name in os.listdir(label_data_dir):
                if file_name.endswith('.npy'):
                    X.append(np.load(os.path.join(label_data_dir, file_name)))
                    y.append(i)
        return np.array(X), np.array(y)


class DataReceiver:
    data_class = dataflow.data.base.BaseData

    def __init__(self, redis_server, fps):
        self._data = self.data_class(redis_server)
        self._fps = fps
        self.quit = False

    def receive(self):
        last_time = None
        while True:
            if self.quit:
                break
            if last_time is not None and 1. / self._fps - (time.time() - last_time) > 0:
                yield False, None
                continue
            last_time = time.time()
            yield True, self._data.value


class SequentialDataReceiver(DataReceiver):
    data_class = dataflow.data.base.BaseData

    def __init__(self, redis_server, fps=0.5, length=5):
        super().__init__(redis_server, fps)
        self._length = length

    def receive(self):
        last_time = None
        data_seq = []
        while True:
            if self.quit:
                break
            if last_time is not None:
                sleep_duration = 1. / self._fps - (time.time() - last_time) > 0
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            last_time = time.time()
            data_seq.append(self._data.value)
            if len(data_seq) < self._length:
                yield False, None
                continue
            elif len(data_seq) > self._length:
                data_seq = data_seq[1:]
            yield True, data_seq


class SequentialGrayedEyeImageReceiver(SequentialDataReceiver):
    data_class = dataflow.data.eye.Image

    def receive(self):
        for received, images in super().receive():
            if self.quit:
                break
            if not received:
                yield received, images
                continue
            images = np.dstack([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])
            yield received, images


def fit(model_class, train_data_loader_class, data_dir=None, models_dir=None, model_kwargs={}):
    model = model_class(**model_kwargs)
    if data_dir is None:
        data_dir = settings.DATA_DIR
    train_data_loader = train_data_loader_class(data_dir, model)
    X, y = train_data_loader.load()
    model.fit(X, y)
    if models_dir is None:
        models_dir = settings.MODELS_DIR
    model_dir = os.path.join(models_dir, model.name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model.save(os.path.join(model_dir, 'model'))
    return model


def predict(model_class, data_receiver_class, data_class, redis_server=None, fps=None, models_dir=None, model_kwargs={}):
    if redis_server is None:
        redis_server = redis.StrictRedis(settings.REDIS_HOST, settings.REDIS_PORT)
    if fps is None:
        fps = float(os.environ.get('NEOCHI_BRAIN_MODEL_%s_FPS' % model_class.name, settings.FPS))
    data_receiver = data_receiver_class(redis_server, fps)
    data = data_class(redis_server)
    model = model_class(**model_kwargs)
    if models_dir is None:
        models_dir = settings.MODELS_DIR
    model_dir = os.path.join(models_dir, model.name)
    model.load(os.path.join(model_dir, 'model'))
    for received, X in data_receiver.receive():
        start_time = time.time()
        if received:
            data.value = model.predict(X.reshape((-1, ) + X.shape))
        sleep_duration = 1. / fps - (time.time() - start_time)
        if sleep_duration > 0:
            time.sleep(sleep_duration)
