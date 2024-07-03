import abc


class AutoInitMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            # Set all class-level attributes as instance attributes
            for key, value in cls.__dict__.items():
                if not key.startswith("__") and not callable(value):
                    setattr(self, key, value)
            # Set all keyword arguments as instance attributes
            for key, value in kwargs.items():
                setattr(self, key, value)
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls
