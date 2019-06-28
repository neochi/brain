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


from tensorflow import keras
from neochi.brain import base
from neochi.core.dataflow import data


class Model(base.KerasClassifierMixin):
    name = 'movement_classifier'
    class_labels = ['move_laying', 'nomove_laying', 'move', 'nomove', 'none']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_model(self):
        ipt = keras.layers.Input(shape=(32, 32, 5))
        cnn1_1 = keras.layers.SeparableConv2D(16, (3, 3), activation='relu')(ipt)
        pool1_1 = keras.layers.MaxPool2D()(cnn1_1)
        cnn2_1 = keras.layers.SeparableConv2D(32, (3, 3), activation='relu')(pool1_1)
        pool2_1 = keras.layers.MaxPool2D()(cnn2_1)
        fc1 = keras.layers.Dense(128, activation='relu')(keras.layers.Flatten()(pool2_1))
        fc2 = keras.layers.Dense(5, activation='softmax')(fc1)
        self._model = keras.models.Model(inputs=[ipt, ], outputs=[fc2, ])


class TrainDataLoader(base.ClassifierTrainDataLoader):
    def __init__(self, base_data_dir, model):
        super().__init__(base_data_dir, model)


class DataReceiver(base.SequentialGrayedEyeImageReceiver):
    pass


def fit(data_dir=None, models_dir=None, model_kwargs={}):
    base.fit(Model, TrainDataLoader, data_dir=data_dir, models_dir=models_dir, model_kwargs=model_kwargs)


def predict(redis_server=None, fps=None, models_dir=None, model_kwargs={}):
    base.predict(Model, DataReceiver, data.brain.MovementClassProbabilities,
                 redis_server=redis_server, fps=fps, models_dir=models_dir, model_kwargs=model_kwargs)