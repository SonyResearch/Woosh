from .edm_sampler import EDMSampler

import sys
import inspect
import logging
import os

# get logger
log = logging.getLogger(__name__)

# import all modules from current edm_samplers directory
# WARNING: not checking subdirectories
# edm_sampler_dir = os.path.dirname(__file__)
# for file in os.listdir(edm_sampler_dir):
#     if file[-3:] == ".py" and not file == "__init__.py":
#         module_name = file[:-3]  # same without extension
#         exec(f"from .{module_name} import *")


def register_edm_samplers(edm_module_base_class):
    """
    Decoractor used to dynamically add EDM samplers to EDM module.
    """

    original_init = edm_module_base_class.__init__

    def __init__(self, *args, **kwargs):
        # Search all samplers in current dir
        sampler_module = sys.modules[__name__]
        # We iterate over classes
        for name, obj in inspect.getmembers(sampler_module, inspect.isclass):
            # search for strict subclasses
            # if not name == 'EDMSampler':
            if issubclass(obj, EDMSampler) and not name == "EDMSampler":
                # add method sample of obj to self
                assert obj.sampler_method_name is not None
                setattr(edm_module_base_class, obj.sampler_method_name, obj.sample)
                log.info(
                    f"register_edm_samplers: Added sampler {name} to class {type(self).__name__} as method {obj.sampler_method_name}"
                )

        original_init(self, *args, **kwargs)

    edm_module_base_class.__init__ = __init__
    return edm_module_base_class


def register_edm_samplers_on_object(self):
    # Search all samplers in current dir
    sampler_module = sys.modules[__name__]
    # We iterate over classes
    for name, obj in inspect.getmembers(sampler_module, inspect.isclass):
        # search for strict subclasses
        # if not name == 'EDMSampler':
        if issubclass(obj, EDMSampler) and not name == "EDMSampler":
            # add method sample of obj to self
            assert obj.sampler_method_name is not None
            setattr(self.__class__, obj.sampler_method_name, obj.sample)
            log.info(
                f"register_edm_samplers: Added sampler {name} to class {type(self).__name__} as method {obj.sampler_method_name}"
            )
