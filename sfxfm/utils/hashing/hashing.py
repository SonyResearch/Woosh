"""Hash string generation functions"""

import inspect
import os
import logging
from pathlib import Path
import hashlib
import random
import re
from typing import Optional, Dict, List, Callable, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
import git
import copy

log = logging.getLogger(__name__)


def init_git_repo():
    try:
        repo = git.Repo()
        # set safe directory
        os.system(f"git config --global --add safe.directory {repo.working_dir}")
        # repo.config_writer().set_value("safe", "directory", repo.working_dir).release()
    except:
        pass


def is_git_repo():
    try:
        _ = git.Repo()
        return True
    except:
        return False


def git_directory_hash(dirname: str, git_tree=None) -> str:
    """Get a SHA hex string for the given repo directory, HEAD commit"""

    if not is_git_repo():
        return None

    # split dirname in outer and its child parts
    dirname_parts = Path(dirname).parts
    dirname_outer = dirname_parts[0] if len(dirname_parts) > 0 else None
    dirname_outer_child = dirname_parts[1] if len(dirname_parts) > 1 else None

    # get git tree, either root or passed as argument
    tree = git.Repo().head.commit.tree if git_tree is None else git_tree
    trees = tree.trees
    if len(trees) > 0:
        # loop to find tree object that matchse outer name
        for t in trees:
            if dirname_outer is not None and t.name == dirname_outer:
                # matched subdir and associated git tree
                if dirname_outer_child is not None:
                    git_hash = git_directory_hash(dirname_outer_child, t)
                    if git_hash is not None:
                        return git_hash
                else:
                    return t.hexsha
        # did not find directory tree
        return None
    else:
        return None


def git_repo_hash() -> str:
    """Get a SHA hex string for the given repo, HEAD commit"""
    return git.Repo().head.object.hexsha


def get_ckpt_hash(config: DictConfig) -> str:
    """Get a hash string for datamodule + dataloader + trainer config"""

    if checkpoint_hash := config.get("checkpoint_hash"):
        log.info(
            f"checkpoint_hash=`{checkpoint_hash}` is provided. The hash will be used to store/resume: checkpoints and logging runs."
            "Make sure only one job is running with the same hash at the same time. For example, using --dependency=afterok:JOB_ID "
            "See more: https://www.gaia.sony.co.jp/manual/en/html/advanced/shared-partition-utilization.html"
        )
        return checkpoint_hash

    ckpt_config_hash, _ = get_hash_from_kwargs(
        config,
        include_keys=[
            # from datamodule
            "pool_datasets",
            "augmentation",
            "preprocessor",
            "sample_size",
            "sample_rate",
            "random_crop",
            "extra_left_context",
            "extra_right_context",
            # "dataloader",
            # from trainer
            # "accelerator",
            # "devices",
            # "precision",
            # from module
            "module",
        ],
        exclude_keys=[
            "num_workers",
            "trainer",
            "callbacks",
            "cache",
        ],
    )

    # get hash identifying git repo and config
    ckpt_hash = get_hash_from_strings(
        [
            git_directory_hash("sfxfm/datamodule"),
            git_directory_hash("sfxfm/module"),
            ckpt_config_hash,
        ]
    )

    return ckpt_hash


def get_datamodule_hash(kwargs) -> str:
    """Get a hash string for datamodule kwargs"""

    # get SHA hash for this datamodule
    # datamodule only caches preprocessor normalization params. this
    # depends on the choice of train,valid,test datasets (pool_datasets),
    # sample related params (sample_size,sample_rate,random_crop,extra_left_context,
    # extra_right_context), augmentation profile (augmentation) and preprocessor
    # config (preprocessor). changing the config of any of this will force the
    # preprocessor to recompute normalization params. the rest is not cached.
    dm_config_hash, clean_kwargs = get_hash_from_kwargs(
        kwargs=kwargs,
        include_keys=[
            "pool_datasets",
            "augmentation",
            "preprocessor",
            "sample_size",
            "sample_rate",
            "random_crop",
            "extra_left_context",
            "extra_right_context",
        ],
    )

    # get hash identifying git repo and config
    dm_hash = get_hash_from_strings(
        [
            git_directory_hash("sfxfm/datamodule"),
            dm_config_hash,
        ]
    )

    return dm_hash, clean_kwargs


def get_dataset_base_hash(kwargs) -> str:
    """Get a hash string for daataset base kwargs"""

    # get SHA hash for this dataset
    # the dataset only caches dataset metadata. this depends on
    # the raw dataset directory (root_dir), its name (name ) and the
    # metadata the might be required to be computed (extra_metadata)
    # changing the config of any of these will force the class to
    # re parse the whole dataset again to make anything stored on disk
    # (outputs or cache) consistent
    # hash string for base metadata config
    dset_base_hash, clean_kwargs = get_hash_from_kwargs(
        kwargs=kwargs,
        include_keys=[
            "name",
            "root_dir",
        ],
        exclude_keys=[
            "splits",
        ],
    )

    # get hash identifying git repo and config
    dset_hash = get_hash_from_strings(
        [
            git_directory_hash("sfxfm/datamodule/dataset"),
            dset_base_hash,
        ]
    )

    return dset_hash, clean_kwargs


def get_dataset_extra_hash(kwargs) -> str:
    """Get a hash string for dataset extra kwargs"""

    # get SHA hash for this dataset extra config
    # the dataset only caches dataset metadata. this depends on
    # the raw dataset directory (root_dir), its name (name ) and the
    # metadata the might be required to be computed (extra_metadata)
    # changing the config of any of these will force the class to
    # re parse the whole dataset again to make anything stored on disk
    # (outputs or cache) consistent
    # hash string for base metadata config
    dset_extra_hash, clean_kwargs = get_hash_from_kwargs(
        kwargs=kwargs,
        include_keys=[
            "name",
            "root_dir",
            "extra_metadata",
        ],
        exclude_keys=[
            "splits",
        ],
    )

    # get hash identifying git repo and config
    dset_hash = get_hash_from_strings(
        [
            git_directory_hash("sfxfm/datamodule/dataset"),
            dset_extra_hash,
        ]
    )

    return dset_hash, clean_kwargs


def clean_config_dict(
    d: Dict,
    exclude_keys=None,
    include_keys=None,
    include_propagate: bool = False,
):
    """
    Make a copy of input config dict with include_keys copied in and
    exclude_keys removed. It supports nested config dictionaries.

    The output clean dict is sorted by key, recursively, so that the dict is
    unique no matter the input order of the keys, no matter the level of the keys.

    Keys in include_keys list are included in the output dict, if include_keys is specified.
    Keys in exclude_keys list are removed in the output dict, if exclude_keys is specified.
    Included keys take precedence over ignored keys. A key may be included and some subkeys
    excluded.
    """
    # make a copy of a nested DictConfig, sorted by key:
    #
    #   - keys in include_keys list are included in the output dict, if include_keys is specified
    #   - keys in exclude_keys list are removed in the output dict, if exclude_keys is specified
    #   - included keys take precedence
    #   - can include a key and ignore certain subkeys
    #   - accepts values of basic, dict/DictConfig, or list/ListConfig types
    #
    dout = {}
    for k, v in sorted(d.items()):  # pylint: disable=R1702
        if k in ("self", "_target_"):
            continue

        if isinstance(v, (dict, DictConfig)):
            # dict of configs
            if include_propagate:
                # include this key and propagate to children config
                if exclude_keys is None or (
                    exclude_keys is not None and k not in exclude_keys
                ):
                    dtmp = clean_config_dict(
                        v,
                        exclude_keys=exclude_keys,
                        include_keys=include_keys,
                        include_propagate=True,
                    )
                    if len(dtmp) > 0:
                        dout[k] = dtmp
            else:
                if include_keys is None or (
                    include_keys is not None and k in include_keys
                ):
                    # include this key and propagate to children config
                    if exclude_keys is None or (
                        exclude_keys is not None and k not in exclude_keys
                    ):
                        # do not ignore key
                        dtmp = clean_config_dict(
                            v,
                            exclude_keys=exclude_keys,
                            include_keys=include_keys,
                            include_propagate=True,
                        )
                        # store this key,value if not empty
                        if len(dtmp) > 0:
                            dout[k] = dtmp
                else:
                    if exclude_keys is None or (
                        exclude_keys is not None and k not in exclude_keys
                    ):
                        # included keys could be in a subtree
                        dtmp = clean_config_dict(
                            v,
                            exclude_keys=exclude_keys,
                            include_keys=include_keys,
                            include_propagate=False,
                        )
                        # store this key,value if not empty
                        if len(dtmp) > 0:
                            dout[k] = dtmp

        elif isinstance(v, (list, ListConfig)):
            # list of configs
            newl = []
            for e in v:
                # do dictconfig branch above for each element in the list

                if isinstance(e, (dict, DictConfig)):
                    # dict of configs
                    if include_propagate:
                        # include this key and propagate to children config
                        if exclude_keys is None or (
                            exclude_keys is not None and k not in exclude_keys
                        ):
                            dtmp = clean_config_dict(
                                e,
                                exclude_keys=exclude_keys,
                                include_keys=include_keys,
                                include_propagate=True,
                            )
                            if len(dtmp) > 0:
                                newl.append(dtmp)
                    else:
                        if include_keys is None or (
                            include_keys is not None and k in include_keys
                        ):
                            # include this key and propagate to children config
                            if exclude_keys is None or (
                                exclude_keys is not None and k not in exclude_keys
                            ):
                                # do not ignore key
                                dtmp = clean_config_dict(
                                    e,
                                    exclude_keys=exclude_keys,
                                    include_keys=include_keys,
                                    include_propagate=True,
                                )
                                # store this key,value if not empty
                                if len(dtmp) > 0:
                                    newl.append(dtmp)
                        else:
                            if exclude_keys is None or (
                                exclude_keys is not None and k not in exclude_keys
                            ):
                                # include keys coudl be in a subtree
                                dtmp = clean_config_dict(
                                    e,
                                    exclude_keys=exclude_keys,
                                    include_keys=include_keys,
                                    include_propagate=False,
                                )
                                # store this key,value if not empty
                                if len(dtmp) > 0:
                                    newl.append(dtmp)

            if len(newl) > 0:
                dout[k] = newl

        else:
            # config leaf
            if not inspect.isclass(v):
                if include_propagate:
                    # include in principle
                    if exclude_keys is None or (
                        exclude_keys is not None and k not in exclude_keys
                    ):
                        # not ignored: include key
                        dout[k] = v
                else:
                    if include_keys is None or (
                        include_keys is not None and k in include_keys
                    ):
                        # include key unless it needs to be ignored
                        if exclude_keys is None or (
                            exclude_keys is not None and k not in exclude_keys
                        ):
                            dout[k] = v
    return dout


def get_hash_from_kwargs(
    kwargs: Dict,
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    length: int = 16,
) -> Tuple[str, Dict]:
    """
    Get a SHA hash key for a given nested config/kwargs dictionary. This
    config dict should uniquely identify the compute step that is
    to be protected/cached. Typically, an output file or directory
    is created after computation and its path contains such hash key
    in it, so its resulting file/dir can be checked for existence without
    collision with other compute steps from same/other experiment runs.

    The config dict is first cleaned and key sorted so that str(dict)
    results in a compact unique representation of its contents. A SHA hash
    key of desired length is computed over such string representation of the
    config dict. This means that all values in the config dict need to
    have a string representation. Class objects should have __repr__
    implemented to return a string uniquely identifying its content
    as a string.
    """

    # clean config dict based on keys we want ot not want
    kwargs = clean_config_dict(
        kwargs,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
    )

    # compute hash key and limit its length
    sha = hashlib.sha1(str(kwargs).encode("UTF-8")).hexdigest()
    # shortening a well-behaved hash key should result in a well-behaved hash key
    sha = sha[:length]

    return sha, kwargs


def get_hash_from_strings(
    strs: List[str],
    length: int = 16,
) -> str:
    """
    Get a SHA hash from a list of strings
    """

    if isinstance(strs, str):
        strs = [strs]

    # concatenate strings
    concat_str = "".join(s for s in strs if s is not None)

    # compute hash key and limit its length
    sha = hashlib.sha1(concat_str.encode("UTF-8")).hexdigest()
    # shortening a well-behaved hash key should result in a well-behaved hash key
    sha = sha[:length]

    return sha


def get_random_string(length: int = 16) -> str:
    """
    Get a random hash string
    """

    chars = "abcdefghijklmnopqrstuvwxyz0123456789"

    return "".join(random.choice(chars) for _ in range(length))


def log_dict(
    d: Dict,
    func: Callable = None,
    indent: str = None,
    match_str: str = None,
    match_regex: str = None,
    _first_indent: str = None,
    _indent: str = None,
    _parent_key=None,
    _parent_key_trace=None,
    _all_keys=None,
):
    """
    Print a dict in a pretty way. A log function can be
    specified as func. match_str allows substring key matching.
    match_regex allows for regex key matching.
    """

    indent = "" if indent is None else indent
    func = print if func is None else func
    _indent = indent if _indent is None else _indent
    _first_indent = indent if _first_indent is None else _first_indent

    if _all_keys is None:
        _all_keys = {}

    # depth-first traversal of the dict tree
    for n, (k, v) in enumerate(d.items()):
        if isinstance(v, (dict, DictConfig)):
            # traverse children dict

            # key filtering is delayed to leave nodes, making things tricky to be printed
            # store a dict of already_printed with all partial keys
            if _parent_key is not None:
                new_parent_key_trace = f"{_parent_key_trace}.{k}"
                _all_keys[new_parent_key_trace] = False
            else:
                new_parent_key_trace = k
                _all_keys[k] = False

            # in the tree traversal, we would print the parent key here
            #
            #    func(f"{_indent}{k}:")
            #
            # but we do not kow yet whether it needs to be printed. we wait
            # until we have the full dict key in the case below
            _all_keys = log_dict(
                v,
                func=func,
                indent=indent,
                match_str=match_str,
                match_regex=match_regex,
                _first_indent=_indent + "   ",
                _indent=_indent + "   ",
                _parent_key=k,
                _parent_key_trace=new_parent_key_trace,
                _all_keys=_all_keys,
            )

        elif isinstance(v, (list, ListConfig)):
            full_key = (
                f"{_parent_key_trace}.{k}" if _parent_key_trace is not None else k
            )

            # key filtering is delayed to leave nodes, making things tricky to be printed
            if match_str is not None and match_str not in full_key:
                continue
            if match_regex is not None and not re.search(match_regex, full_key):
                continue

            # access the hierarchy of parent keys for current full-key
            if _parent_key_trace in _all_keys:
                sub_parent_key_trace = None
                for parent_key in _parent_key_trace.split("."):
                    sub_parent_key_trace = (
                        f"{sub_parent_key_trace}.{parent_key}"
                        if sub_parent_key_trace is not None
                        else parent_key
                    )
                    sub_parent_indent = f"{indent}" + "   " * (
                        len(sub_parent_key_trace.split(".")) - 1
                    )
                    if not _all_keys[sub_parent_key_trace]:
                        num_prints = _all_keys[sub_parent_key_trace]
                        _all_keys[sub_parent_key_trace] = True
                        func(f"{sub_parent_indent}{parent_key}:")

            # print current key
            if n == 0:
                func(f"{_first_indent}{k}:")
            else:
                func(f"{_indent}{k}:")

            if _parent_key is not None:
                new_parent_key_trace = f"{_parent_key_trace}.{k}"
                _all_keys[new_parent_key_trace] = True
            else:
                new_parent_key_trace = k
                _all_keys[k] = True

            _indent_list = _indent + "   " + "- "

            for e in v:
                # import ipdb ; ipdb.set_trace()
                if isinstance(e, (dict, DictConfig)):
                    _all_keys = log_dict(
                        e,
                        func=func,
                        indent=indent + "  ",
                        match_str=match_str,
                        match_regex=match_regex,
                        _first_indent=_indent_list,
                        _indent=_indent + "   " + "  ",
                        _parent_key=k,
                        _parent_key_trace=new_parent_key_trace,
                        _all_keys=_all_keys,
                    )
                else:
                    func(f"{_indent_list}{e}")

        else:
            # dict leaves
            full_key = (
                f"{_parent_key_trace}.{k}" if _parent_key_trace is not None else k
            )
            # key filtering is delayed to leave nodes, making things tricky to be printed
            if match_str is not None and match_str not in full_key:
                continue
            if match_regex is not None and not re.search(match_regex, full_key):
                continue

            # access the hierarchy of parent keys for current full-key
            if _parent_key_trace in _all_keys:
                sub_parent_key_trace = None
                for parent_key in _parent_key_trace.split("."):
                    sub_parent_key_trace = (
                        f"{sub_parent_key_trace}.{parent_key}"
                        if sub_parent_key_trace is not None
                        else parent_key
                    )
                    sub_parent_indent = f"{indent}" + "   " * (
                        len(sub_parent_key_trace.split(".")) - 1
                    )
                    if not _all_keys[sub_parent_key_trace]:
                        num_prints = _all_keys[sub_parent_key_trace]
                        _all_keys[sub_parent_key_trace] = True
                        func(f"{sub_parent_indent}{parent_key}:")

            if n == 0:
                func(f"{_first_indent}{k}: {v}")
            else:
                func(f"{_indent}{k}: {v}")

    return _all_keys
