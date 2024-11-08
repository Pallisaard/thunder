import abc


class AutoInitMeta(abc.ABCMeta):
    """
    AutoInitMeta is a metaclass that automatically initializes instance attributes
    based on class-level attributes and keyword arguments passed to the constructor.

    When a class uses AutoInitMeta as its metaclass, the following behavior is added:
    - All class-level attributes that are not callable and do not start with "__"
        are set as instance attributes during initialization.
    - All keyword arguments passed to the constructor are set as instance attributes.

    Attributes:
            mcs: The metaclass.
            name: The name of the class being created.
            bases: A tuple containing the base classes of the class being created.
            namespace: A dictionary containing the class namespace.
            kwargs: Additional keyword arguments.

    Methods:
            __new__(mcs, name, bases, namespace, **kwargs): Creates a new class with the
            specified name, bases, and namespace, and modifies its __init__ method to
            automatically set instance attributes.
    """

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
