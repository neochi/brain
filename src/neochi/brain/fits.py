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
from neochi.core.utils import environ
from . import data_loaders
from . import models


def fit_behavior_classifier():
    kwargs = environ.get_kwargs('BRAIN_BEHAVIOR_CLASSIFIER')
    X, y = data_loaders.load_classification_data(kwargs['data_dir'], kwargs['labels'])
    model = models.BehaviorClassifier(kwargs['shape'], kwargs['fps'], kwargs['labels'])
    model.fit(X, y, kwargs['optimizer'], kwargs['loss'], kwargs['metrics'], kwargs['train_size'], kwargs['epochs'])
    if not os.path.exists(kwargs['save_dir']):
        os.mkdir(kwargs['save_dir'])
    model.save(kwargs['save_dir'])


def fit_sleep_detector():
    kwargs = environ.get_kwargs('BRAIN_SLEEP_DETECTOR')
    model = models.SleepDetector(kwargs['time_steps'], kwargs['weights'], kwargs['fps'])
    if not os.path.exists(kwargs['save_dir']):
        os.mkdir(kwargs['save_dir'])
    model.save(kwargs['save_dir'])
