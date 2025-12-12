""" Cache directory related utility functions """

from pathlib import Path
from typing import Optional
import os
import logging
import shutil
import random
import time
from sfxfm.utils.dist import rank

current_rank = rank()

log = logging.getLogger(__name__)


class Cache:
    """
    Global cache system for experiment compute steps

    Two usage types:

    cache = Cache(config)

        if cache.start(filename):
            # compute whatever and create filename
            cache.end(filename)

    """

    def __init__(
        self,
        save_dir: Optional[Path],
        cache_dir: Optional[Path] = None,
        min_poll_time: float = 0.0,
        max_poll_time: float = 5.0,
        scale_poll_time: float = 1.5,
        cache_glob: Optional[str] = None,
        cache_max: Optional[int] = None,
    ):
        """Initialize experiment and cache directories"""
        self.save_dir = Path(save_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.last_cache_filename = None
        self.min_poll_time = min_poll_time
        self.max_poll_time = max_poll_time
        self.scale_poll_time = scale_poll_time
        self.cache_glob = cache_glob
        self.cache_max = cache_max
        self.cache_trace = None
        self.filename_inside_cache = False

    def copy(self):
        cache = Cache(
            save_dir=self.save_dir,
            cache_dir=self.cache_dir,
            min_poll_time=self.min_poll_time,
            max_poll_time=self.max_poll_time,
            scale_poll_time=self.scale_poll_time,
            cache_glob=self.cache_glob,
            cache_max=self.cache_max,
        )
        return cache

    def get_cache_filename(self, filename: Optional[Path] = None):
        """
        Get corresponding filename in cache for a regular file
        in experiment output
        """
        if filename is None:
            return self.last_cache_filename
        if self.cache_dir is not None:
            try:
                cache_filename = self.cache_dir / Path(filename).relative_to(
                    self.save_dir
                )
                self.filename_inside_cache = False
                return cache_filename
            except:
                try:
                    # file path might be in a different experiment id
                    # make relative to exp dir without experiment id
                    save_dir_root = self.save_dir.parent
                    # remove possibly mismatched exp tag at the beginning
                    rel_path = Path(
                        *Path(filename).relative_to(save_dir_root).parts[1:]
                    )
                    cache_filename = self.cache_dir / rel_path
                    self.filename_inside_cache = False
                    return cache_filename
                except:
                    # cache file/dir might be already inside cache
                    try:
                        cache_filename = self.cache_dir / Path(filename).relative_to(
                            self.cache_dir
                        )
                        self.filename_inside_cache = True
                        return cache_filename
                    except:
                        raise ValueError(
                            f"cannot cache files with prefix beyond save_dir or cache_dir"
                        )
        return None

    def get_done_filename(self, filename: Optional[Path] = None):
        """Get .done filename associated with a compute step"""
        if self.cache_dir is not None:
            cache_fn = self.get_cache_filename(filename)
            done_fn = Path(str(cache_fn) + ".done")
            return done_fn
        return None

    def get_inprogress_filename(self, filename: Optional[Path] = None):
        """Get .done filename associated with a compute step"""
        if self.cache_dir is not None:
            cache_fn = self.get_cache_filename(filename)
            done_fn = Path(str(cache_fn) + ".inprogress")
            return done_fn
        return None

    def is_done(self, filename, suffixes=None, done_only=False):
        """
        Check if a compute step has been already done. The .done file
        associated with the compute step is checked.
        """
        if suffixes is None:
            if self.cache_dir is not None:
                if not done_only:
                    cache_fn = self.get_cache_filename(filename)
                    cache_fn_exists = cache_fn.exists()
                done_fn = self.get_done_filename(filename)
                done_fn_exists = done_fn.exists()
                return (done_only and done_fn_exists) or (
                    not done_only and (done_fn_exists and cache_fn_exists)
                )
        else:
            if self.cache_dir is not None:
                for suffix in suffixes:
                    if not done_only:
                        cache_fn = self.get_cache_filename(Path(str(filename) + suffix))
                        cache_fn_exists = cache_fn.exists()
                    done_fn = self.get_done_filename(Path(str(filename) + suffix))
                    done_fn_exists = done_fn.exists()
                    if (done_only and not done_fn_exists) or (
                        not done_only and (not done_fn_exists or not cache_fn_exists)
                    ):
                        return False
                return True
            return None
        return None

    def is_inprogress(self, filename):
        """
        Check if a compute step has been already done. The .done file
        associated with the compute step is checked.
        """
        if self.cache_dir is not None:
            inprogress_fn = self.get_inprogress_filename(filename)
            return inprogress_fn.exists() and inprogress_fn.stat().st_size > 0
        return None

    def done(self, filename):
        """
        Mark that a compute step associated with the creation of filename
        has already been done
        """
        if self.cache_dir is not None:
            done_fn = self.get_done_filename(filename)
            done_fn.parent.mkdir(parents=True, exist_ok=True)
            with open(done_fn, "w") as f:  # pylint: disable=W1514
                f.write("done")

    def inprogress(self, filename):
        """
        Mark that a compute step associated with the creation of filename
        is in progress
        """
        if self.cache_dir is not None:
            inprogress_fn = self.get_inprogress_filename(filename)
            inprogress_fn.parent.mkdir(parents=True, exist_ok=True)
            with open(inprogress_fn, "w") as f:  # pylint: disable=W1514
                f.write("in progress")

    def enter(self, filename: Path, force=False, rank=0):
        """Start computing with outcome filename"""

        filename = Path(filename)

        if self.cache_dir is not None:

            cache_filename = self.get_cache_filename(filename)

            if force:
                # delete .done file while recomputing
                if current_rank == rank:
                    self.get_done_filename(filename).unlink(missing_ok=True)
                else:
                    # other ranks wait for done file to be removed
                    while self.get_done_filename(filename).exists():
                        time.sleep(0.5)
            else:
                if self.is_done(filename):
                    # file is present in cache and noone else is trying to compute
                    # the same thing
                    if current_rank == rank:
                        if not self.filename_inside_cache:
                            # copy file from cache
                            if cache_filename.is_file():
                                filename.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy(str(cache_filename), str(filename))
                            # copy whole directory from cache
                            elif cache_filename.is_dir():
                                if Path(filename).exists():
                                    shutil.rmtree(filename)
                                shutil.copytree(
                                    str(cache_filename),
                                    str(filename),
                                    dirs_exist_ok=True,
                                )
                    # update current cache filename, if we are computing the step
                    self.last_cache_filename = cache_filename
                    return False

            # update current cache filename, if we are computing the step
            self.last_cache_filename = cache_filename

            # need to compute
            # mark the compute step is computing
            if current_rank == rank:
                self.inprogress(filename)

            return True

        # compute, no cache set up
        return True

    def signal_done(self, filename, rank=0):
        """
        End section after step has been computed
        """

        def find_files_in_dir(globex, directory):
            matching_files = list(Path(directory).glob(globex))
            if len(matching_files) > 0:
                return matching_files
            return []

        # assume filename (file or directory) exists as
        # it has just been computed
        filename = Path(filename)

        if self.cache_dir is not None:

            # copy filename or directory to cache
            cache_fn = self.get_cache_filename(filename)

            if current_rank == rank:
                if not self.filename_inside_cache:
                    # store file to cache
                    if filename.is_file():
                        cache_fn.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(str(filename), str(cache_fn))
                    # store directory to cache
                    elif filename.is_dir():
                        # if Path(cache_fn).exists():
                        #     shutil.rmtree(cache_fn)
                        shutil.copytree(
                            str(filename),
                            str(cache_fn),
                            dirs_exist_ok=True,
                        )

                # mark compute step as done
                # remove inprogress file
                self.get_inprogress_filename(filename).unlink(missing_ok=True)
                # create done file
                self.done(filename)

                if (
                    self.cache_glob is not None
                    and self.cache_max is not None
                    and self.cache_max > 0
                ):
                    # leave at most keep_at_most latest files given a filename pattern
                    if Path(cache_fn.name).match(self.cache_glob):
                        if self.cache_trace is None:
                            self.cache_trace = sorted(
                                find_files_in_dir(self.cache_glob, cache_fn.parent),
                                key=os.path.getmtime,
                            )
                        else:
                            # add most recent file
                            self.cache_trace.append(cache_fn)

                        if len(self.cache_trace) > self.cache_max:
                            # remove oldest files
                            n_remove = len(self.cache_trace) - self.cache_max
                            for f in self.cache_trace[:n_remove]:
                                f.unlink(missing_ok=True)
                                Path(str(f) + ".done").unlink(missing_ok=True)
                            self.cache_trace = self.cache_trace[-self.cache_max :]

            else:
                # other ranks wait for done file to exist
                self.wait_done(filename)

    def wait_done(self, filename, suffixes=None, done_only=False):
        """Active wait to determine if done file is ready"""

        # use a random factor to avoid multiple processes access the file system at once
        poll_time = max(
            self.min_poll_time,
            min(self.max_poll_time, self.min_poll_time + random.random()),
        )

        time.sleep(poll_time)

        while not self.is_done(filename, suffixes=suffixes, done_only=done_only):

            poll_time = max(
                self.min_poll_time,
                min(
                    self.max_poll_time,
                    poll_time * self.scale_poll_time + random.random(),
                ),
            )

            time.sleep(poll_time)

    def __repr__(self):
        return f"{self.save_dir}_{self.cache_dir}"
