from time import perf_counter
import logging
from typing import Callable, Union


log = logging.getLogger(__name__)

# this flag should only be edit with the context lazy_loading
lazy_loading_enabled = False


class catchtime:
    # context to measure loading time: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    def __init__(
        self, debug_print="Time", logger: Union[logging.Logger, Callable] = log
    ):
        self.debug_print = debug_print
        self.logger = logger

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        readout = f"{self.debug_print}: {self.time:.3f} seconds"
        if isinstance(self.logger, logging.Logger):
            self.logger.info(readout)
        elif callable(self.logger):
            self.logger(readout)


class lazy_loading:
    # context to disable checkpoint loading  (e.g. for model loading whose checkpoint contains submodules loaded with external_module)
    def __init__(self, enabled=True, verbose=False):
        self.enabled = enabled
        self.verbose = verbose
        self.prev_state = None

    def __enter__(self):
        global lazy_loading_enabled
        self.prev_state = lazy_loading_enabled
        lazy_loading_enabled = self.enabled
        if self.verbose:
            log.info(
                f"Lazy loading Context (enter) enabled={lazy_loading_enabled}, prev_state={self.prev_state}"
            )
        return self

    def __exit__(self, type, value, traceback):
        global lazy_loading_enabled
        lazy_loading_enabled = self.prev_state
        if self.verbose:
            log.info(f"Lazy loading Context (exit) restored to {lazy_loading_enabled}")
