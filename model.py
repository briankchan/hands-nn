import pickle
import os.path
import glob
from abc import ABCMeta, abstractmethod
import inspect
import functools
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from misc import chunks, classproperty

class Model(metaclass=ABCMeta):
    _args = {}
    _args_filename = "params.json"
    _base_log_path = "runs"

    @property
    @classmethod
    @abstractmethod
    def _class_log_path_pattern(self):
        return "runs/run{}"

    @classproperty
    def _log_path_pattern(cls):
        return os.path.join(cls._base_log_path, cls._class_log_path_pattern)

    @classmethod
    def _format_log_path(cls, run_num, log_path_pattern=None):
        if log_path_pattern is None:
            log_path_pattern = cls._log_path_pattern
        return log_path_pattern.format(run_num)

    def _get_log_path(self):
        return self._format_log_path(self.run_num)

    @abstractmethod
    def __init__(self, run_num=None):
        self._build_model()
        self.reset()

    @abstractmethod
    def _build_model(self):
        pass

    def reset(self):
        self._reset_model()
        self.run_num = self._get_next_run_num()
        self.log_path = self._get_log_path()

    @abstractmethod
    def _reset_model(self):
        pass

    @classmethod
    def _get_next_run_num(cls):
        i = 0
        while True:
            i += 1
            log_path = cls._format_log_path(i)
            if not os.path.exists(log_path) and not glob.glob(log_path + "-*"):
                return i

    @classmethod
    def _get_prev_run(cls, path_pattern=None):
        i = 0
        prev_path = None
        multiple_paths = False
        while True:
            i += 1
            path = cls._format_log_path(i, path_pattern)
            if os.path.exists(path):
                prev_path = path
                multiple_paths = False
                continue
            paths = glob.glob(path + "-*")
            if paths:
                prev_path = paths[0]
                if len(paths) > 1:
                    multiple_paths = True
                continue

            if prev_path is None:
                raise ValueError("No such run.")
            if multiple_paths:
                print("Multiple possible runs; using", prev_path)
            return prev_path

    def save(self, path_pattern=None):
        if path_pattern is None:
            path = self.log_path
        elif os.path.exists(path_pattern):
            path = path_pattern
        else:
            path = self._format_log_path(self.run_num, path_pattern)
        self._save_args(path)
        self._save_model(path)

    def _save_args(self, path):
        with open(os.path.join(path, self._args_filename), "wb") as f:
            pickle.dump(self._args, f)

    @abstractmethod
    def _save_model(self, path):
        pass

    @classmethod
    def load(cls, run_num=None, path_pattern=None):
        path = cls._get_load_path(run_num, path_pattern)

        args = cls._load_args(path_pattern=path)
        output = cls(**args)
        output._load_run(path_pattern=path)
        return output

    @classmethod
    def _get_load_path(cls, run_num=None, path_pattern=None):
        if path_pattern is not None and os.path.exists(path_pattern):
            # load pathect path
            return path_pattern
        
        if run_num is None:
            # load previous run
            return cls._get_prev_run(path_pattern)

        # load specified run num
        path = cls._format_log_path(run_num, path_pattern)

        if not os.path.exists(path):
            paths = glob.glob(path + "-*")
            if len(paths) < 1:
                raise ValueError("No such run.")
            path = paths[0]
            if len(paths) > 1:
                print("Multiple possible runs; using", path)
        return path

    @classmethod
    def _load_args(cls, run_num=None, path_pattern=None):
        path = cls._get_load_path(run_num, path_pattern)
        with open(os.path.join(path, cls._args_filename), "rb") as f:
            return pickle.load(f)

    def _load_run(self, run_num=None, path_pattern=None):
        path = self._get_load_path(run_num, path_pattern)
        # self.run_num = run_num
        self.log_path = path
        self._load_model(path)

    @abstractmethod
    def _load_model(self, path):
        pass

    @abstractmethod
    def train(self, input, expected, indices, epochs=None, batch_size=None):
        pass

    @abstractmethod
    def predict(self, input, indices):
        pass

    def test(self, input, expected, indices):
        if indices is None:
            indices = range(len(input))
        else:
            indices = np.r_[tuple(indices)]
            expected = expected[indices]

        pred = self.predict(input, indices)

        pred_flat = pred.flatten()
        exp_flat = expected.flatten()

        conf_mat = confusion_matrix(exp_flat, pred_flat)
        accuracy = conf_mat.diagonal().sum() / conf_mat.sum()
        precision = conf_mat[1,1] / conf_mat[:,1].sum()
        recall = conf_mat[1,1] / conf_mat[1].sum()
        f1 = 0 if precision == 0 and recall == 0\
             else 2 * precision * recall / (precision + recall)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1)
        print("Confusion matrix")
        print(conf_mat)
        return pred


def save_args(method):
    """Stores provided method args as instance attributes."""
    argspec = inspect.getargspec(method)
    defaults = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    arg_names = argspec.args[1:]
    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Only run if args not already saved (i.e. dict is empty)
        if not self._args:
            # Get default arg values
            args = defaults.copy()
            # Add provided arg values
            list(map(args.update, (zip(arg_names, positional_args[1:]), keyword_args.items())))
            # Store values in instance as attributes
            # self.__dict__.update(args)
            vars(self).update(args)
            # Also store values to separate dict
            self._args = args
        return method(*positional_args, **keyword_args)

    return wrapper
