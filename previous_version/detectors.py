import os
import copy
import time
import pickle
import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier
# try:
#     CLAP_DETECTOR = True
#     import alsaaudio, audioop
# except ImportError:
#     CLAP_DETECTOR = False
from neochi.eye import caches as eye_caches
from neochi.neochi import settings
from neochi.brain import caches
import click


class Detector:
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


class DictModelMixin:
    def update_params(self, params):
        model_updated = copy.deepcopy(self._model)
        model_updated.update(params)
        if model_updated != self._model:
            print('PARAM UPDATED: %s' % self.__class__.__name__)
            self._model = model_updated
            return True
        else:
            return False


class PersonDetector(Detector):
    def __init__(self, max_iter=200, verbose=False):
        super().__init__()
        self._max_iter = max_iter
        self._verbose = verbose

    def fit(self, X, y):
        self._model = MLPClassifier(hidden_layer_sizes=(100, 100), tol=0.000001,
                                    max_iter=self._max_iter, verbose=self._verbose)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        return self._model.score(X, y)


class MovementDetector(Detector, DictModelMixin):
    NO_MOVE = 0
    MOVE = 1

    def __init__(self):
        super().__init__()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._fgbg = cv2.createBackgroundSubtractorMOG2()
        self._model = {'cutoff': 180, 'detection_threshold': 10}

    def predict(self, X):
        predictions = []
        for x in X:
            fgmask = self._fgbg.apply(x)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self._kernel)
            fgmask[fgmask < self._model['cutoff']] = 0
            if np.mean(fgmask) < self._model['detection_threshold']:
                predictions.append(self.NO_MOVE)
            else:
                predictions.append(self.MOVE)
        return np.array(predictions)


class SleepDetector(Detector, DictModelMixin):
    AWAKE = 0
    SLEEP = 1

    def __init__(self):
        super().__init__()
        self._model = {'detection_threshold': 0.9}

    def predict(self, X):
        y = 1. - np.mean(X, axis=0)
        y[y > self._model['detection_threshold']] = self.SLEEP
        y[y < self._model['detection_threshold']] = self.AWAKE
        return y


class MovingAverage:
    def __init__(self, n, initial_value):
        super().__init__()
        self._initial_value = initial_value
        self._history = [self._initial_value] * n

    def reset(self):
        self._history = [self._initial_value for _ in self._history]

    def push(self, value):
        self._history.insert(0, value)
        self._history = self._history[:-1]

    def average(self):
        return np.mean(self._history)


class DetectorExecutor:
    def __init__(self, detector, detector_settings, detection):
        print('INITIALIZE %s...' % detector.__class__.__name__)
        self._detector = detector
        self._detector_settings = detector_settings
        self._detection = detection
        self._ma = MovingAverage(self._detector_settings.get()['ma']['n'],
                                 self._detector_settings.get()['ma']['initial_value'])
        print('%s INITIALIZED.' % detector.__class__.__name__)

    def reload_detector(self):
        self._detector.reload(self._detector_settings.get()['model']['save_path'])

    def reload_ma(self):
        self._ma = MovingAverage(self._detector_settings.get()['ma']['n'],
                                 self._detector_settings.get()['ma']['initial_value'])

    def update_params(self):
        if hasattr(self._detector, 'update_params') \
                and self._detector.update_params(self._detector_settings.get()['model']):
            self._detector.save(self._detector_settings.get()['model']['save_path'])

    def get_X(self):
        raise NotImplementedError

    def force_continue(self):
        return False

    def detect(self):
        print('%s STARTED.' % self._detector.__class__.__name__)
        while True:
            self.reload_detector()
            if hasattr(self._detector, 'update_params') \
                    and self._detector.update_params(self._detector_settings.get()['model']):
                self.reload_ma()
                self._detector.save(self._detector_settings.get()['model']['save_path'])
            if self.force_continue():
                time.sleep(1. / self._detector_settings.get()['executor']['fps'])
                continue
            self._ma.push(self._detector.predict(self.get_X())[0])
            self._detection.set(self._ma.average())
            # print('%s: %s' % (self.__class__.__name__, self._detection.get()))
            time.sleep(1. / self._detector_settings.get()['executor']['fps'])


class PersonDetectorExecutor(DetectorExecutor):
    def get_X(self):
        return np.array(
            [(cv2.cvtColor(eye_caches.server.image.get_body(), cv2.COLOR_RGB2GRAY) / 255.).ravel(), ])


class MovementDetectorExecutor(DetectorExecutor):
    def get_X(self):
        return np.array([eye_caches.server.image.get_body()])

    def force_continue(self):
        if caches.server.person_detection.get() < self._detector_settings.get()['executor']['lay_down_threshold']:
            self._ma.reset()
            self._detection.set(self._ma.average())
            return True
        else:
            return False


class SleepDetectorExecutor(DetectorExecutor):
    def __init__(self, detector, detector_settings, detection):
        super().__init__(detector, detector_settings, detection)
        self._movement_detection_history = [0.] * self._detector_settings.get()['executor']['history_length']

    def get_X(self):
        self._movement_detection_history = self._movement_detection_history[:-1]
        self._movement_detection_history.insert(0, caches.server.movement_detection.get())
        return np.array([self._movement_detection_history])

    def force_continue(self):
        if caches.server.person_detection.get() < self._detector_settings.get()['executor']['lay_down_threshold']:
            self._ma.reset()
            self._detection.set(self._ma.average())
            return True
        else:
            return False


@click.group()
def cmd():
    pass


@cmd.command()
def detect_person():
    executor = PersonDetectorExecutor(
        PersonDetector(), settings.person_detector_settings, caches.server.person_detection)
    executor.detect()


@cmd.command()
def detect_movement():
    executor = MovementDetectorExecutor(
        MovementDetector(), settings.movement_detector_settings, caches.server.movement_detection)
    executor.detect()


@cmd.command()
def detect_sleep():
    executor = SleepDetectorExecutor(
        SleepDetector(), settings.sleep_detector_settings, caches.server.sleep_detection)
    executor.detect()


# @cmd.command()
# def detect_clap():
#     print('INITIALIZE CLAP DETECTOR...')
#     model = ClapDetector('sysdefault:CARD=Device')
#     model.set_on_detect_func(clap_detector_on_detect_func)
#     model.start_detection_thread()
#     print('CLAP DETECTOR STARTED.')


if __name__ == '__main__':
    cmd()
