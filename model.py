from abc import ABCMeta, abstractmethod
import inspect
import functools

class Model(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def save(self):
        self._save_args()
        self._save_model()

    def _save_args(self):
        self.args

    @abstractmethod
    def _save_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    # @abstractmethod
    # def predict(self):
    #     pass

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def store_args(method):
    """Stores provided method args as instance attributes."""
    argspec = inspect.getargspec(method)
    defaults = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    arg_names = argspec.args[1:]
    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        list(map(args.update, (zip(arg_names, positional_args[1:]), keyword_args.items())))
        # Store values in instance as attributes
        self.__dict__.update(args)
        # Also store values to separate dict
        self._args = args
        return method(*positional_args, **keyword_args)

    return wrapper