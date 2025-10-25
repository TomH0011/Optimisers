# register initialisation
OPTIMISER_REGISTRY = {}


def register_optimiser(name):
    def decorator(cls):
        OPTIMISER_REGISTRY[name] = cls
        return cls

    return decorator
