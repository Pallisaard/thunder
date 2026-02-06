# import abc


class AutoInit:
    """Lightweight mixin that auto-populates instance attributes.

    What it does on initialization:
    - Copies all non-callable, non-dunder class attributes from the subclass
      to the instance (useful for defaults like hyperparameters or schema).
    - Assigns any keyword arguments passed to the constructor as attributes on
      the instance.

    How to use:
    - Inherit from ``AutoInit`` and optionally define class attributes as
      defaults.
    - If you override ``__init__``, call ``super().__init__(**kwargs)`` to
      enable the automatic assignment.

    Example:
        class ModelConfig(AutoInit):
            lr: float = 1e-3
            momentum: float = 0.9

            def __init__(self, name: str, **kwargs):
                super().__init__(**kwargs)
                self.name = name

        cfg = ModelConfig("resnet", batch_size=64)
        # cfg.lr == 1e-3; cfg.momentum == 0.9; cfg.batch_size == 64; cfg.name == "resnet"

    Notes:
    - Mutable class attribute defaults are shared across instances; prefer
      immutables or override in ``__init__``.
    """

    def __init__(self, **kwargs):
        # Set all class-level attributes as instance attributes
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("__") and not callable(value):
                setattr(self, key, value)
        # Set all keyword arguments as instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
