from collections.abc import Mapping
from functools import reduce
from pathlib import Path
from torch import nn
import typing as tp
from omegaconf import DictConfig, OmegaConf
import yaml
from hydra.utils import instantiate
import torch
import logging
# from sfxfm.utils.model_store import ModelStore
from sfxfm.utils.cache import Cache
from sfxfm.utils.dist import rank
import sfxfm.utils.loading

rank = rank()

# get logger
log = logging.getLogger(__name__)


def load_hydra_config(config_path: Path) -> DictConfig:
    with open(config_path, "r") as f:
        hydra_config = yaml.safe_load(f)
    return hydra_config


def get_checkpoint_filename(
    checkpoint_name_or_path,
    checkpoint_dir,
) -> tp.Optional[Path]:
    """Load a specific model/optim checkpoint
    close to the same method in config.py
    Returns None if nothing is found or if checkpoint_name_or_path is None
    """

    if checkpoint_name_or_path is None:
        return None

    def find_files_in_dir(globex, directory):
        matching_files = list(Path(directory).glob(globex))
        if len(matching_files) > 0:
            return matching_files
        return []

    checkpoint_name_or_path = Path(checkpoint_name_or_path)

    if checkpoint_name_or_path is not None and checkpoint_name_or_path.exists():
        return checkpoint_name_or_path

    if str(checkpoint_name_or_path) in ("last", "auto"):
        # find last.ckpt in ckpt_dir
        ckpt_last = find_files_in_dir("**/last*.ckpt", checkpoint_dir)
        if len(ckpt_last) > 0:
            # found last ckpt or similar, return this file
            return ckpt_last[0]
        # no last found, use the latest .ckpt
        ckpts = sorted(find_files_in_dir(r"**/[0-9\-]*.ckpt", checkpoint_dir))  # pylint: disable=W1401
        # get last one according to epoch-step sorting
        if len(ckpts) > 0:
            return ckpts[-1]

    raise ValueError(
        f"No checkpoint  {checkpoint_name_or_path} found in {checkpoint_dir}"
    )


def set_recursive_strict(strict, hydra_config):
    for k, v in hydra_config.items():
        if isinstance(v, bool) and k == "strict":
            hydra_config[k] = strict
        elif isinstance(v, (dict, DictConfig)):
            set_recursive_strict(strict, v)


def set_recursive_strict(strict, hydra_config):
    for k, v in hydra_config.items():
        if isinstance(v, bool) and k == "strict":
            hydra_config[k] = strict
        elif isinstance(v, (dict, DictConfig)):
            set_recursive_strict(strict, v)


def recursive_fix_cachedir(config, cache_dir, save_dir):
    """This function will recursively update the cachedir and savedir in a hydra configuration of the loaded model
    It's useful for loading diffusion models to avoid loading AE and CLAP multiple times. See Issue #757
    Args:
        config (Mapping): config to be updated
        cache_dir (str): cache_dir
        save_dir (str): save_dir
    """
    for k, v in config.items():
        if isinstance(v, Mapping):
            recursive_fix_cachedir(v, cache_dir, save_dir)
        if k == "cache_dir":
            config[k] = cache_dir
        if k == "save_dir":
            config[k] = save_dir
        if k == "output_dir":
            if config[k].endswith("dataset"):
                config[k] = f"{save_dir}/dataset"
            elif config[k].endswith("datamodule"):
                config[k] = f"{save_dir}/datamodule"
        # disable loading the sub modules (clap,AE..) multiple times. Instead use the weights from the diffusion checkpoint
        # COMMENTED OUT since prerpocessor is not saved any more see https://github.com/SonyResearch/project_mfm_sfxfm/issues/1019
        # if k == "chkp_path" and v == "last":
        #     config[k] = None


def lazy_load_checkpoint(instance, checkpoint, strict=True):
    r"""Do the actual loading of the checkpoint, but only load the missing parameters
    from the checkpoints of submodules that were loaded using `with lazy_loading(enabled=True):`


    Args:
        instance (nn.Module): instance
        checkpoint (dict):  pl.module checkpoint
        strict (bool, optional): strict. Defaults to True.
    """

    # find externel submodels
    unloaded_submodels = {
        k: v
        for k, v in instance.named_modules()
        if hasattr(v, "_hydra_external_config")
    }
    load_candidates = unloaded_submodels.keys()

    # find missing parameters from checkpoint
    while len(
        missing_params := [
            x for x in instance.state_dict().keys() if x not in checkpoint["state_dict"]
        ]
    ) and len(load_candidates):
        # there are still useful submodels to load
        # sorted from top level first
        load_candidates = [
            k
            for k in sorted(load_candidates, key=lambda x: len(x))
            if any(x.startswith(k) for x in missing_params)
        ]
        # load the top most candidate and loop
        if not len(load_candidates):
            break
        candidate = load_candidates.pop(0)
        module_submodule = unloaded_submodels[candidate]
        with sfxfm.utils.loading.catchtime(f"Lazy loading submodule {candidate}"):
            sub_checkpoint = load_checkpoint_instance(
                module_submodule,
                return_state_dict=True,
                **module_submodule._hydra_external_load_kwargs,  # type: ignore
            )
        for k, v in sub_checkpoint.items():
            # insert ONLY missing weights from the sub checkpoint
            checkpoint["state_dict"].setdefault(candidate + "." + k, v)

    if len(missing_params):
        log.warning(
            f"Missing parameters couldn't be loaded from checkpoints/submodules' checkpoints: {missing_params} not found in the submodules"
        )
    # insures that the checkpoint is backward compatible
    instance.on_load_checkpoint(checkpoint)
    load_results = instance.load_state_dict(checkpoint["state_dict"], strict=strict)
    if rank == 0:
        if load_results.missing_keys:
            log.warning(f"Missing keys: {load_results.missing_keys}")
        if load_results.unexpected_keys:
            log.warning(f"Unexpected_keys: {load_results.unexpected_keys}")


def load_checkpoint_instance(
    instance,
    checkpoint_filename,
    map_location=None,
    experiment_name=None,
    submodule_prefix_in_python="",
    strict=True,
    verbose=True,
    return_state_dict=False,
    weights_only=False,
):
    ckpt = torch.load(
        checkpoint_filename,
        map_location=(torch.device(map_location) if map_location is not None else None),
        # mmap=True, # avoid loading the whole module until needed. Edit: by testing mmap seems to be slower on crusoe
        weights_only=weights_only,
    )
    on_load_checkpoint_method = getattr(instance, "on_load_checkpoint", None)
    if callable(on_load_checkpoint_method):
        log.info(
            f"({experiment_name}) Calling external module instance on_load_checkpoint from class {type(instance)}"
        )
        instance.on_load_checkpoint(ckpt)
    else:
        log.warning(
            f"({experiment_name}) ExternalModule istance of type {type(instance)} has no on_load_checkpoint method, backward compitability is not guaranteed!"
            f" It's recommened to implement `on_load_checkpoint` method in the class  {type(instance)}"
            "so that loading old checkpoints (before changes in the module) can be converted to the current structure"
        )
    state_dict_all = ckpt["state_dict"]
    # keep only relevant entries
    added_dot = (
        "."
        if (
            len(submodule_prefix_in_python) > 0
            and submodule_prefix_in_python[-1] != "."
        )
        else ""
    )
    submodule_prefix_in_python = f"{submodule_prefix_in_python}{added_dot}"
    keys_to_keep = [
        f"{submodule_prefix_in_python}{short_key}"
        for short_key in instance.state_dict().keys()
    ]

    # we should use removeprefix if Python3.9+
    def removeprefix(s, prefix):
        l = len(prefix)
        assert s[:l] == prefix
        return s[l:]

    filtered_state_dict = {
        removeprefix(long_key, submodule_prefix_in_python): state
        for long_key, state in state_dict_all.items()
        if f"{long_key}" in keys_to_keep
    }
    if return_state_dict:
        return filtered_state_dict
    load_results = instance.load_state_dict(filtered_state_dict, strict=strict)
    if rank == 0 and verbose:
        log.info(
            f"({experiment_name})  ExternalModule weights copied from module.{submodule_prefix_in_python[:-1]} using checkpoint {checkpoint_filename} in strict={strict} mode"
        )
        if load_results.missing_keys:
            log.warning(f"Missing keys: {load_results.missing_keys}")
        if load_results.unexpected_keys:
            log.warning(f"Unexpected_keys: {load_results.unexpected_keys}")


class ExternalModule(nn.Module):
    """Classes that is used to instantiate a submodule from a resolved Hydra config"""

    def __new__(
        cls,
        experiment_name: tp.Optional[str] = None,
        experiment_dir: tp.Optional[Path] = None,
        chkp_path: tp.Optional[str] = None,
        strict: bool = True,
        strict_propagate: bool = False,
        submodule_path: str = "",
        submodule_sep: str = "/",
        submodule_prefix_in_python: str = "",
        map_location: str = None,
        cache: tp.Optional[Cache] = None,
        download_overwrite: bool = False,
        verbose: bool = True,
        config_override: tp.Any = None,
        recursive_fix_config_paths: bool = False,
        config_updates: tp.Optional[Mapping] = None,
        weights_only: bool = True,
    ) -> None:
        """Wrapper class able to instantiate a submodule from a previous experiment


        Args:
            experiment_name (tp.Optional[str]):
            experiment_dir (tp.Optional[Path]): No priority over experiment name. Must be a relative path relative to root_dir
            chkp_path (tp.Optional[str]): Path to checkpoint or relative path from the experiment_dir or in ("last", "auto")
            strict bool: flag with which we load the state_dict
            submodule_path: (tp.Optional[str]):
            submodule_sep: str = '/': separator used to navigate in the hierarchy
            recursive_fix_config_paths: bool = False, updates the cache, save_paths, in the loaded models. Helpful for diffusion models which have nested models.
            config_updates: tp.Optional[Mapping]: dictionary with updates to the **submodule** config. useful for overriding parameters in a loaded model, like removing the metrics from the loaded module.

        Raises:
            NotImplementedError: _description_
        """
        log.info(
            f"Trying to load model from ({experiment_name}) submodule_path={submodule_path}, chkp_path={chkp_path}"
        )
        # try finding experiment_dir from symlinks
        if experiment_name is not None:
            assert experiment_dir is None
            experiment_dir = Path("symlinks", experiment_name).resolve()
            if not experiment_dir.is_dir():
                if rank == 0 and verbose:
                    log.info(
                        f"({experiment_name}) Couldn't find experiment_dir={experiment_dir}"
                    )
                if cache is not None:
                    # try to download model from model storage
                    download_dir = (
                        cache.cache_dir / "download-native-models" / experiment_name
                    )
                    if cache.enter(download_dir, force=download_overwrite):
                        if rank == 0:
                            experiment_dir = ModelStore(log_fn=log.info).download(
                                experiment_name,
                                download_dir=download_dir.parent,
                                download_overwrite=download_overwrite,
                            )
                            # signal download step is done
                            cache.signal_done(download_dir)
                        else:
                            cache.wait_done(download_dir)
                    else:
                        experiment_dir = download_dir
                        if rank == 0 and verbose:
                            log.info(f"re-using model {download_dir}")
                    if experiment_dir is None or (
                        experiment_dir is not None and not experiment_dir.is_dir()
                    ):
                        raise ValueError(
                            f"({experiment_name}) could not download experiment {experiment_name}: aborting"
                        )
                else:
                    raise ValueError(
                        f"({experiment_name}) could not locate experiment {experiment_name} as {experiment_dir}: aborting"
                    )

        experiment_dir = Path(experiment_dir) if experiment_dir is not None else None
        if experiment_dir is None or (
            experiment_dir is not None and not experiment_dir.is_dir()
        ):
            raise ValueError(
                f"({experiment_name}) experiment directory {experiment_dir} is not a directory: aborting"
            )

        checkpoint_dir = experiment_dir / "checkpoints/"

        # if the chkp_path is None, find "last" in order to get the hash string
        checkpoint_filename = get_checkpoint_filename(
            checkpoint_name_or_path=chkp_path or "last", checkpoint_dir=checkpoint_dir
        )

        hydra_config: DictConfig = load_hydra_config(
            experiment_dir / "config_resolved.yaml"
        )
        if not strict:
            set_recursive_strict(strict, hydra_config)

        if config_override is not None:
            for k, v in config_override.items():
                hydra_config["module"][k] = v
            strict = False

        # propagate strict flag to all submodules
        if strict_propagate:
            set_recursive_strict(strict, hydra_config)

        if recursive_fix_config_paths:
            recursive_fix_cachedir(hydra_config, cache.cache_dir, cache.save_dir)

        module_config = reduce(
            lambda d, k: d[k], submodule_path.split(submodule_sep), hydra_config
        )

        if config_updates:
            module_config = OmegaConf.merge(module_config, config_updates)

        instance = instantiate(module_config)

        hash_str = Path(checkpoint_filename).parent.stem

        # this hydra config should suffice to initialize the module
        instance._hydra_external_config = module_config
        # this dict should suffice to load the checkpoint
        instance._hydra_external_load_kwargs = dict(
            checkpoint_filename=checkpoint_filename,
            map_location=map_location,
            experiment_name=experiment_name,
            submodule_prefix_in_python=submodule_prefix_in_python,
            strict=strict,
            verbose=verbose,
        )

        # load weights if necessary:
        if chkp_path is not None and checkpoint_filename is not None:
            # if lazy_loading_enabled, we don't load the checkpoints, but only set the attributes points to the checkpoint path
            if not sfxfm.utils.loading.lazy_loading_enabled:
                load_checkpoint_instance(
                    instance,
                    **instance._hydra_external_load_kwargs,  # type: ignore
                )
            else:
                log.warning(
                    f"({experiment_name})  Lazy loading enabled, not loading checkpoint for module.{submodule_prefix_in_python[:-1]}"
                )
        else:
            if rank == 0 and verbose:
                log.warning(
                    f"({experiment_name})  ExternalModule weights of module.{submodule_prefix_in_python[:-1]}  are **NOT** loaded from checkpoint!"
                )

        instance.checkpoint_hash = (
            hash_str if len(hash_str) > 0 and hash_str != experiment_name else None
        )
        instance.experiment_dir = experiment_dir
        instance.experiment_name = experiment_name

        if rank == 0 and verbose:
            log.info(
                f"({experiment_name}) loaded {submodule_path} of type {type(instance)} with ExternalModule based on Hydra config located in {experiment_dir}"
            )
            if instance.checkpoint_hash is not None:
                log.info(
                    f"({experiment_name})   using checkpoint hash {instance.checkpoint_hash}"
                )

        return instance


if __name__ == "__main__":
    import sys

    sys.path[0] = "/workspaces/project_mfm_sfxfm"
    module = ExternalModule(
        experiment_name="20240124-174325",
        chkp_path="last",
        submodule_path="module/model",
    )
    print(type(module))
