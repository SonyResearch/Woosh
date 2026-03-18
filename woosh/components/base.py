import logging
import os

from typing import Any, Dict, List, Mapping, Optional, Set, Union
import copy
import torch

from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from pydantic import BaseModel, ConfigDict


rank = 0
# get logger
log = logging.getLogger(__name__)


def _is_load_config(v) -> str:
    """
    Helper function to discrimniate between a LoadConfig and a config Args
    the only difference is the presence of the path field

    Acts on dictionaries or pydantic models
    Returns a string so that it can be used as a pydantic Discriminator

    """
    if "path" in v and v["path"] is not None:
        return "load_config"
    if hasattr(v, "path") and v.path is not None:
        return "load_config"
    return "component_args"


class ComponentConfig(BaseModel):
    """
    Main configuration class for all components.
    All components must define the following fields
    """

    # === special field to force not having extra arguments
    model_config = ConfigDict(extra="forbid")
    # all components must define exclude_from_checkpoint
    exclude_from_checkpoint: bool = False
    trainable: bool = True


class LoadConfig(ComponentConfig):
    # allow extra args
    # these will be used to overwrite the config
    path: str
    model_config = ConfigDict(
        extra="allow",
    )


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def recursive_update_config(config, update):
    r"""
    Recursively update a config dictionary with another dictionary.
    Return a new instance.
    """
    for k, v in update.items():
        if isinstance(v, Mapping):
            config[k] = recursive_update_config(config.get(k, {}), v)
        else:
            config[k] = v
    return config


def find_common_tensors_from_storage(state_dict_a, state_dict_b):
    """
    Partition state_dict_a := A and state_dict_b := B based on the memory address of the tensors
    in the state dicts.

    With C := A inter B:

    Returns
    A - C, B - C, A inter B

    Also returns the mapping
    state_dict_a.keys -> state_dict_b.keys on  A inter B

    C uses the keys from A
    """

    def mem(p):
        # return p.data.storage().data_ptr()
        return p.data.untyped_storage().data_ptr()

    mem_a = {mem(p) for p in state_dict_a.values()}
    mem_b = {mem(p) for p in state_dict_b.values()}
    intersection = mem_a & mem_b

    filtered_state_dict_a = {
        k: v for k, v in state_dict_a.items() if mem(v) not in intersection
    }
    filtered_state_dict_b = {
        k: v for k, v in state_dict_b.items() if mem(v) not in intersection
    }
    shared_state_dict_a = {
        k: v for k, v in state_dict_a.items() if mem(v) in intersection
    }

    inverse_mapping_b = {
        mem(v): k for k, v in state_dict_b.items() if mem(v) in intersection
    }

    a_to_b_mapping = {
        k: inverse_mapping_b[mem(v)] for k, v in shared_state_dict_a.items()
    }

    return (
        filtered_state_dict_a,
        filtered_state_dict_b,
        shared_state_dict_a,
        a_to_b_mapping,
    )


class BaseComponent:
    r"""
    Base class for all Components.

    [`BaseComponent`] handles storing and loading the configuration and the weights of the models.


    Class attributes (overridden by derived classes):

        - **config_class** ([`Any`]) -- the configuration class for the config that defines this model architecture.

    """

    config_class = ComponentConfig
    available_config_formats = ["yaml"]
    available_weight_formats = ["safetensors", "pt"]

    def __init__(self):
        assert isinstance(self, nn.Module)
        super(BaseComponent, self).__init__()

    def init_from_config(self, config):
        """
        __init__-like method that is to be called after the object is created
        and super.__init__ is called

        - Parse & resolve & validate config
        - Sets self.config (of type cls.config_class)

        """
        # Components must have an attached Pydantic BaseModel class
        assert self.config_class is not None
        assert issubclass(self.config_class, BaseModel)

        # if config is a ComponentConfig,
        # then the model will being loaded from a pretrained model
        # and _weights_path will be set
        self.config, self._weights_path = self.resolve_config(
            config, return_weights_path=True
        )
        # assert isinstance(self.config, ComponentConfig)

        self._subcomponents: Dict[str, BaseComponent] = {}
        self._subcomponents_configs = []

        # exclude_from_checkpoint is a boolean that indicates if the component
        # should be saved in the checkpoint WHEN part of a larger component
        # unused otherwise
        assert hasattr(self.config, "exclude_from_checkpoint"), (
            f"The configuration for {type(self).__name__} must have an "
            f"'exclude_from_checkpoint' attribute."
            f"Consider overriding it when loading the model from a checkpoint."
        )

        self._exclude_from_checkpoint = self.config.exclude_from_checkpoint
        self._trainable = self.config.trainable

        # exclude_subcomponents_from_checkpoint is a list of subcomponents that should not be saved in the checkpoint
        self._exclude_subcomponents_from_checkpoint = []

        # unset, will be computed after the model has been initialized
        # when registering subcomponents
        self._excluded_parameters_: Optional[Set[str]] = None
        self._included_parameters_: Optional[Set[str]] = None

    @property
    def _included_parameters(self):
        """
        This is to ensure a default behaviour if we don't register any component
        """
        assert isinstance(self, nn.Module)
        if self._included_parameters_ is None:
            log.debug(
                "Initializing included parameters, make sure this is called once all submodules & parameters are registered"
            )
            self._included_parameters_ = set(self.state_dict().keys())
            self._excluded_parameters_ = set()

        return self._included_parameters_

    @_included_parameters.setter
    def _included_parameters(self, value):
        self._included_parameters_ = value

    @property
    def _excluded_parameters(self):
        """
        This is to ensure a default behaviour if we don't register any component
        """
        assert isinstance(self, nn.Module)
        if self._excluded_parameters_ is None:
            log.debug(
                "Initializing excluded parameters, make sure this is called once all submodules & parameters are registered"
            )
            self._excluded_parameters_ = set()
            self._included_parameters_ = set(self.state_dict().keys())

        return self._excluded_parameters_

    @_excluded_parameters.setter
    def _excluded_parameters(self, value):
        self._excluded_parameters_ = value

    @classmethod
    def resolve_config(cls, config, return_weights_path=False):
        """
        If config is a LoadConfig, load the path and returns the args config
        Otherwise validates the config against the config_class
        Takes into account extra args

        This method is not recursive
        """
        overwrite_kwargs = {}
        weights_path = None

        # check if LoadConfig is passed
        # knowing that it can be given as a plain dict with path field
        # get the config with component_args and load the model
        # or a model parameters
        # this sets the _weights_path

        if _is_load_config(config) == "load_config":
            if isinstance(config, Mapping):
                config = LoadConfig(**config)
            # overwrite kwargs are in the config
            # everything else than path
            overwrite_kwargs = config.model_dump(exclude={"path"})
            config, weights_path = cls._config_and_weightspath_from_path(config.path)
            # config is now supposed to be a config_class, not LoadConfig

        # Validate the config using the config_class
        if not isinstance(config, cls.config_class):
            config = cls.config_class(**config)

        # add extra kwargs if present in LoadConfig
        # and validate
        # config = config.model_copy(update=overwrite_kwargs)
        # model_copy(update=overwrite_kwargs) doesn't support nested dicts
        # see https://github.com/pydantic/pydantic/issues/7387
        # see https://github.com/SonyResearch/project_mfm_sfxfm/issues/1302
        config = recursive_update_config(config.model_dump(), overwrite_kwargs)
        config = cls.config_class.model_validate(config)

        if return_weights_path:
            return config, weights_path
        else:
            return config

    def register_subcomponent(
        self,
        name: str,
        subcomponent: "BaseComponent",
        subkey: Optional[str] = None,
    ):
        """
        Register a subcomponent to the model.
        On the same model as self.register_buffer, self.register_parameter, etc.

        name: name of the attribute self."name"
        key: key in the config dict, use '.' as a separator for nested keys
        subkey: key in the subcomponent config dict

        if key is None, it is assumed that key=name
        """
        assert isinstance(self, nn.Module) and isinstance(subcomponent, nn.Module)
        self._subcomponents: Dict[str, BaseComponent]
        self._exclude_subcomponents_from_checkpoint: List[str]

        exclude_from_checkpoint = subcomponent._exclude_from_checkpoint

        self._subcomponents_configs.append(
            {
                "config": subcomponent.config,
                "subcomponent_path": name,  # place of the subcomponent relative to its parent component
                "exclude_from_checkpoint": exclude_from_checkpoint,
            }
        )

        # subcomponent = getattr(self, name)
        if subkey is not None:
            # subcomponent = getattr(subcomponent, subkey)
            name = f"{name}.{subkey}"
        self._subcomponents.update({name: subcomponent})

        # Compute parameters to remove from self.state_dict
        subcomponent_state_dict_to_exclude = subcomponent.state_dict()
        if exclude_from_checkpoint:
            self._exclude_subcomponents_from_checkpoint.append(name)
        else:
            # Do not exclude all parameters of the subcomponent
            # if exclude_from_checkpoint is False
            subcomponent_state_dict_to_exclude = {
                k: v
                for k, v in subcomponent_state_dict_to_exclude.items()
                if k not in subcomponent._included_parameters
            }

        A, B, C, A_to_B = find_common_tensors_from_storage(
            state_dict_a=self.state_dict(),
            state_dict_b=subcomponent_state_dict_to_exclude,
        )

        self._included_parameters = self._included_parameters - set(C.keys())
        assert self._excluded_parameters_ is not None
        self._excluded_parameters_ = self._excluded_parameters_ | set(C.keys())

    def register_subcomponent_dict(
        self,
        name: str,
        component_dict: Dict[str, Any] | nn.ModuleDict = {},
    ):
        """
        Register a subcomponent to the model.
        On the same model as self.register_buffer, self.register_parameter, etc.

        exclude_from_checkpoint_dict contains the subkeys and their exclude_from_checkpoint value
        We assume that subkeys have the same name in the configs structure and as attributes of the model
        """
        if component_dict == {}:
            log.warning(
                f"No subcomponents to register in register_subcomponent_dict for attribute {name} as component_dict is empty"
            )
        for k, component in component_dict.items():
            if isinstance(component, BaseComponent):
                self.register_subcomponent(name, subcomponent=component, subkey=k)

    def save(self, path, config_format="yaml", weights_format="safetensors"):
        """
        Save the configuration and the model's state dictionary to the save_directory.

        Args:
            save_directory (str): Directory to which to save the model.
        """
        assert isinstance(self, nn.Module)
        if rank != 0:
            return

        assert config_format in self.available_config_formats, (
            f"config_format={config_format} should be one of {(*self.available_config_formats,)}"
        )
        assert weights_format in self.available_weight_formats, (
            f"weights_format={weights_format} should be one of {(*self.available_weight_formats,)}"
        )

        try:
            umask = os.umask(0o002)

            # creat the save directory
            os.makedirs(path, exist_ok=True)
            config_dict = self.config.model_dump()  # type: ignore

            # if is_dataclass(config_dict):
            #     config_dict = asdict(config_dict)  # type: ignore
            # config_dict = OmegaConf.create(config_dict)

            # saving config
            save_config_path = os.path.join(path, f"config.{config_format}")
            log.info(f"Saving config of {type(self)} to {save_config_path}")
            if config_format == "yaml":
                with open(save_config_path, "w") as outfile:
                    OmegaConf.save(config_dict, outfile)
                    # yaml.dump(config_dict, outfile, default_flow_style=False)
            else:
                # should never happen
                raise NotImplementedError(
                    f"config_format={config_format} not implemented"
                )
            # saving weights
            save_wights_parh = os.path.join(path, f"weights.{weights_format}")
            log.info(f"Saving weights of {type(self)} to {save_wights_parh}")

            state_dict = self.state_dict()
            self.filter_state_dict_(state_dict)

            if weights_format == "safetensors":
                save_file(state_dict, save_wights_parh)
            elif weights_format == "pt":
                torch.save(state_dict, save_wights_parh)
            else:
                # should never happen
                raise NotImplementedError(
                    f"config_format={config_format} not implemented"
                )
        finally:
            os.umask(umask)  # type: ignore

    def filter_state_dict_(self, state_dict, prefix="") -> None:
        """
        In place
        Delete elements from a state dict to only keep relevant parameters

        This methods is based on the values stored in
        self._included_parameters

        prefix is the empty string or ends with a dot
        Can be used to filter larger state dicts

        prefix indicates the location of self in state_dict

        """
        if prefix != "":
            assert prefix.endswith(".")
        for k in copy.copy(list(state_dict.keys())):
            if (
                k.startswith(prefix)
                and k.removeprefix(prefix) in self._excluded_parameters
            ):
                del state_dict[k]

    def add_filtered_state_dict_keys_(self, incomplete_state_dict, prefix="") -> None:
        """
        In place
        Adds filtered out keys from the full (unfiltered) state dict to the incomplete_state_dict
        This is the opposite of filter_state_dict

        We only add the keys that were excluded from the incomplete_state_dict
        to ensure that other missing keys are still

        prefix: where to insert the keys in the incomplete_state_dict
        """
        assert isinstance(self, nn.Module)
        if prefix != "":
            assert prefix.endswith(".")
        state_dict = self.state_dict()
        incomplete_state_dict.update(
            {
                prefix + k: v
                for k, v in state_dict.items()
                if k in self._excluded_parameters
            }
        )

    def load_from_config(self):
        # load non excluded_from_checkpoints parameters of the component
        if self._weights_path is not None:
            self._load_statedict_from_disk()

        # load excluded_from_checkpoints subcomponents
        for component_name, component in self._subcomponents.items():
            if component_name in self._exclude_subcomponents_from_checkpoint:
                component.load_from_config()

    def _load_statedict_from_disk(self, only_return_state_dict=False, strict=True):
        """_summary_

        Args:
            only_return_state_dict (bool, optional): Don't loaded the state_dict into the model but rather return it. Can be useful for lazyloading. Defaults to False.

        Raises:
            ValueError: if the _weights_path is not set, can be due to not using the from_pretrained method.
            NotImplementedError: if the format of the weights file is not implemented.

        Returns:
            dict: loaded state_dict if only_return_state_dict is True
        """
        if self._weights_path is None:
            raise ValueError(
                f"Cannot load weights, _weights_path is not set. Did you load this model using {type(self)}.from_pretrained?"
            )
        weights_path = self._weights_path
        weights_format = weights_path.rsplit(".", 1)[-1]
        log.info(f"Loading weights for {type(self).__name__} from {weights_path}")
        if weights_format == "pt":
            state_dict = torch.load(weights_path)
            if only_return_state_dict:
                return state_dict
            self._load_state_dict(state_dict)
        elif weights_format == "safetensors":
            state_dict = {}
            with safe_open(weights_path, framework="pt") as f:  # type: ignore
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            if only_return_state_dict:
                return state_dict

            self._load_state_dict(state_dict)
        else:
            # should never happen
            raise NotImplementedError(
                f"weights_format={weights_format} not implemented"
            )

    def _load_state_dict(self, state_dict):
        """
        Tries to load_state_dict in strict mode, if it fails, retries in non-strict mode
        Logs the success or failure of the loading

        Components that have subcomponents excluded from the checkpoint
        are expected to fail with strict=True
        """
        assert isinstance(self, nn.Module)
        try:
            self.load_state_dict(state_dict, strict=True)
            log.info(f"Loaded state_dict for {type(self).__name__} in strict mode")
        except RuntimeError as e:
            log.info(f"Error loading state_dict in strict mode: {e}")
            log.info("Retrying in non-strict mode")
            self.load_state_dict(state_dict, strict=False)

    def _load_from_module_checkpoint(
        self,
        checkpoint,
        prefix="",
    ):
        """
        Select only the relevant keys as specified by prefix
        and adapt their names
        prefix must contain the trailing dot and be the full path to the component
        in the module e.g. "ldm."

        """
        if prefix != "":
            assert prefix.endswith(".")

        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix) :]] = v

        self._load_state_dict(new_state_dict)

    def config_from_pretrained(
        self,
        path: Optional[Union[str, os.PathLike]],
    ):
        """Loads a config from a path
        and sets the _weights_path to the weights file

        Args:
            path (Optional[Union[str, os.PathLike]]): path to the model

        Raises:
            FileNotFoundError: if the save path does not have config and weights.

        Returns:
            BaseComponent: the loaded model.
        """
        config, weights_path = self._config_and_weightspath_from_path(path)

        self._weights_path = weights_path

        return config

    @classmethod
    def _config_and_weightspath_from_path(cls, path):
        """
        Securely loads a config from a path
        and returns the config and the weights path
        """
        if path is None:
            raise NotImplementedError("path must be provided")
        # finding suitable config
        config_format = "invalid"
        for config_format in cls.available_config_formats:
            if os.path.isfile(os.path.join(path, f"config.{config_format}")):
                break
        else:
            # no config file
            raise FileNotFoundError(
                f"No config file found in {path}, make sure that config.{(*cls.available_config_formats,)} exists."
            )

        # finding suitable weights file
        for weights_format in cls.available_weight_formats:
            if os.path.isfile(os.path.join(path, f"weights.{weights_format}")):
                break
        else:
            # no config file
            raise FileNotFoundError(
                f"No weights file found in {path}, make sure that config.{(*cls.available_weight_formats,)} exists."
            )
        # loading config
        config_path = os.path.join(path, f"config.{config_format}")
        log.info(f"Loading config from {config_path}")
        # log.warning(
        #     f"Weights are not loaded for {config_path}, don't forget to call load_from_config"
        # )
        if config_format == "yaml":
            config = OmegaConf.load(config_path)
            # with open(config_path, "r") as infile:
            #     config = yaml.load(infile, Loader=yaml.FullLoader)
        else:
            # should never happen
            raise NotImplementedError(f"config_format={config_format} not implemented")

        # cast and verify config
        if cls.config_class is not None:
            config = cls.config_class(**config)

        weights_path = os.path.join(path, f"weights.{weights_format}")
        return config, weights_path

    @classmethod
    def from_pretrained(
        cls,
        path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        """Load a model from a path

        Args:
            path (Optional[Union[str, os.PathLike]]): path to the model
            *model_args: not supported yet.
            **kwargs: config updates to the pretrained model.

        Raises:
            FileNotFoundError: if the save path does not have config and weights.

        Returns:
            BaseComponent: the loaded model.
        """
        if path is None:
            raise NotImplementedError("path must be provided")
        # finding suitable config
        config_format = "invalid"
        for config_format in cls.available_config_formats:
            if os.path.isfile(os.path.join(path, f"config.{config_format}")):
                break
        else:
            # no config file
            raise FileNotFoundError(
                f"No config file found in {path}, make sure that config.{(*cls.available_config_formats,)} exists."
            )

        # finding suitable weights file
        for weights_format in cls.available_weight_formats:
            if os.path.isfile(os.path.join(path, f"weights.{weights_format}")):
                break
        else:
            # no config file
            raise FileNotFoundError(
                f"No weights file found in {path}, make sure that config.{(*cls.available_weight_formats,)} exists."
            )
        # loading config
        config_path = os.path.join(path, f"config.{config_format}")
        log.info(f"Loading config for {cls.__name__} from {config_path}")
        if config_format == "yaml":
            config = OmegaConf.load(config_path)
            # with open(config_path, "r") as infile:
            #     config = yaml.load(infile, Loader=yaml.FullLoader)
        else:
            # should never happen
            raise NotImplementedError(f"config_format={config_format} not implemented")

        # cast and verify config
        if cls.config_class is not None:
            config = cls.config_class(**config)

        # init object
        obj = cls(config, *model_args, **kwargs)

        # loading weights
        weights_path = os.path.join(path, f"weights.{weights_format}")
        obj._weights_path = weights_path

        # TODO: lazy loading?!
        # if not in a lazy loading, do actually load the weights
        if not woosh.utils.loading.lazy_loading_enabled:
            obj._load_statedict_from_disk()

        return obj

    def freeze_non_trainable_components(self):
        """
        Freezes all subcomponents that are not trainable
        If self._trainable is True, the WHOLE component is frozen,
        regardless of the values _trainable state of its subcomponents
        """
        assert isinstance(self, nn.Module)
        if not self._trainable:
            assert self._exclude_from_checkpoint, (
                "Strange case, where parameters are not trainable but should not be saved in the checkpoint"
            )
            self.requires_grad_(False)
            # self.eval()
        else:
            for k, subcomponent in self._subcomponents.items():
                subcomponent.freeze_non_trainable_components()

    def _component_summary(self, prefix="", depth=0):
        """
        prefix used to print properly the component tree
        """
        assert isinstance(self, nn.Module)

        filtered_state_dict = self.state_dict()
        self.filter_state_dict_(filtered_state_dict, prefix="")
        num_params = sum(
            [p.numel() for p in filtered_state_dict.values()]  # type: ignore
        )  # type: ignore

        if prefix != "":
            prefix = prefix + "."

        num_params_all = sum(p.numel() for p in self.state_dict().values())
        print(f"{' | ' * depth + '--- '}{prefix}{type(self).__name__}")
        print(f"{' | ' * (depth + 1) + '     * '}(from_weights={self._weights_path})")
        print(f"{' | ' * (depth + 1) + '     * '}(trainable={self._trainable})")
        print(
            f"{' | ' * (depth + 1) + '     * '}(excluded_from_checkpoint={self._exclude_subcomponents_from_checkpoint})"
        )
        print(
            f"{' | ' * (depth + 1) + '     * '}(Total number of parameters={human_format(num_params_all)})"
        )
        print(
            f"{' | ' * (depth + 1) + '     * '}(Component parameters={human_format(num_params)})"
        )
        print(
            f"{' | ' * (depth + 1) + '     * '}(num_included_tensors={len(self._included_parameters):,})"
        )
        assert self._excluded_parameters_ is not None
        print(
            f"{' | ' * (depth + 1) + '     * '}(num_excluded_tensors={len(self._excluded_parameters_):,})"
        )

        for component_name, component in self._subcomponents.items():
            component._component_summary(
                prefix=prefix + f"{component_name}", depth=depth + 1
            )
