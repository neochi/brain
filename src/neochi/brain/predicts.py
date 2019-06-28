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


import numpy as np
import redis
from neochi.core.utils import environ
from neochi.core.dataflow.receivers import eye
from neochi.core import dataflow
from . import models


def predict_behavior():
    pass


def predict_sleep():
    def load_behavior_classifier():
        kwargs = environ.get_kwargs('BRAIN_BEHAVIOR_CLASSIFIER')
        behavior_classifier = models.BehaviorClassifier()
        behavior_classifier.load(kwargs['save_dir'])
        return behavior_classifier

    def load_sleep_detector():
        kwargs = environ.get_kwargs('BRAIN_SLEEP_DETECTOR')
        sleep_detector = models.SleepDetector()
        sleep_detector.load(kwargs['save_dir'])
        return sleep_detector

    def get_eye_redis_server():
        kwargs = environ.get_kwargs('EYE')
        return redis.StrictRedis(kwargs['redis_host'], kwargs['redis_port'])

    def get_brain_redis_server():
        kwargs = environ.get_kwargs('BRAIN')
        return redis.StrictRedis(kwargs['redis_host'], kwargs['redis_port'])

    behavior_classifier = load_behavior_classifier()
    sleep_detector = load_sleep_detector()

    image_receiver = eye.SequentialGrayImages(get_eye_redis_server(), behavior_classifier.fps, behavior_classifier.time_steps)

    brain_redis_server = get_brain_redis_server()
    sleeping_possibility = dataflow.data.brain.SleepingPossibility(brain_redis_server)
    sleeping_notification = dataflow.notifications.brain.DetectedSleep(brain_redis_server)

    X_sleep = []
    for updated, X_behavior in image_receiver.receive():
        if updated:
            X_behavior = X_behavior.reshape(behavior_classifier.shape_with_batch)
            y_behavior = behavior_classifier.predict(X_behavior)
            X_sleep.append(y_behavior[0])
            print('BEHAVIOR:', behavior_classifier.labels[y_behavior[0]])
            if len(X_sleep) < sleep_detector.time_steps:
                continue
            elif len(X_sleep) > sleep_detector.time_steps:
                X_sleep = X_sleep[1:]
            y_sleep = sleep_detector.predict_probs(np.array(X_sleep).reshape((-1, sleep_detector.time_steps)))
            sleeping_possibility.value = y_sleep[0]
            print('SLEEP POSSIBILITY:', y_sleep[0])
            if y_sleep > sleep_detector.threshold:
                print('NOTIFY DETECTED SLEEP')
                sleeping_notification.notify()
