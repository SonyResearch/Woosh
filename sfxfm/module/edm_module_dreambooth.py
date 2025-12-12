from typing import List, Dict, Any
from pathlib import Path
import copy
import re
import random
import sys
from tqdm import tqdm
import torchmetrics
import logging
from .edm_module import EDMModule
from .cfg_utils import BetaFunc
import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import ListConfig, DictConfig
import numpy as np
from tqdm import tqdm
import gc
from pytorch_lightning.utilities.model_summary import ModelSummary
import time
import json
try:
    import faiss
except ImportError:
    print("faiss not found, please install it for faster search")
from sfxfm.utils.dist import rank
from sfxfm.module.model import is_lora, set_requires_grad
import bisect

rank = rank()

# get logger
log = logging.getLogger(__name__)

# CLAP FAISS search/selection

def init_faiss_index(name, model_dir):
    model_state = {}
    model_dir = Path(model_dir)

    faiss_index_audio_path = model_dir / "faiss" / f"{name}_audio.index"
    faiss_index_text_path = model_dir / "faiss" / f"{name}_text.index"
    faiss_metadata_path = model_dir / "faiss" / f"{name}_metadata.json"

    if not (
        faiss_index_audio_path.exists()
        and faiss_index_text_path.exists()
        and faiss_metadata_path.exists()
    ):
        raise FileNotFoundError(
            "Couldn't find vector database files. Please create the database first."
        )
    log.info(
        f"Loading faiss index from {faiss_index_audio_path.as_posix()} and {faiss_index_text_path.as_posix()}"
    )
    faiss_metadata = json.loads(faiss_metadata_path.read_text())
    faiss_index_audio = faiss.read_index(faiss_index_audio_path.as_posix())
    faiss_index_text = faiss.read_index(faiss_index_text_path.as_posix())
    assert len(faiss_metadata) == faiss_index_audio.ntotal == faiss_index_text.ntotal
    log.info(f"Loaded database with {len(faiss_metadata)} vectors")

    ( model_state["index_audio"],
      model_state["index_text"],
      model_state["index_metadata"],
    ) = faiss_index_audio, faiss_index_text, faiss_metadata

    return model_state


def _faiss_search(index, query, k, subset, metadata):
    if subset is not None:
        id_selector = faiss.IDSelectorBatch(subset)  # type: ignore
        search_params = faiss.SearchParameters(sel=id_selector)  # type: ignore
    else:
        search_params = faiss.SearchParameters()

    D, I = index.search(query, k, params=search_params)  # type: ignore

    I = I[0].tolist()  # assume single query
    D = D[0]
    I = [metadata[str(i)] if i != -1 else -1 for i in I]  # type: ignore

    audio_ids = [f"{i['id']}" if i != -1 else "-" for i in I]  # type: ignore
    audio_descriptions = [f"{i['description']}" if i != -1 else "-" for i in I]  # type: ignore
    audio_categories = [f"{i['category']}" if i != -1 else "No result found" for i in I]  # type: ignore

    sfxclap_scores = [f"{i:.4f}" if i > -1000 else "-∞" for i in D.tolist()]

    return (
        audio_ids,
        sfxclap_scores,
    )


class EarlyStopTracker():

    """
    Track a loss overtime to determine early stopping of the training process

    2 phases:

      - search for loss maximum, with alpha_max smoothing factor
      - determine whether the loss is below a threshold, relative to the maximum, smoothed
        with alpha_stop factor

    stop_threshold is relative to the maximum detected.

    smoothed loss needs be below the threshold for stop_patience training steps to decide on stopping

    """

    def __init__(self, alpha_max=0.95, alpha_stop=0.98, stop_threshold=0.8, stop_patience=25, min_steps=None):
        self.alpha_max = alpha_max
        self.alpha_stop = alpha_stop
        self.state = 0.0
        self.count = 0
        self.found_peak = False
        self.stop_count = 0
        self.stop_patience = stop_patience
        self.step_count = 0
        self.stop_threshold = stop_threshold
        self.stop = False
        self.min_steps = min_steps
        log.info(f"EarlyStopTracker init: alpha_max={alpha_max}, alpha_stop={alpha_stop}, stop_threshold={stop_threshold} stop_patience={stop_patience}")

        self.max_state = -1e9
        self.last_max_state_step = -1

        self.state_threshold = 0.0

    def update(self, value):
        self.count += 1
        if not self.stop:

            # stop updating max_state after 50 steps not updating it
            if self.step_count<self.last_max_state_step+50:
                self.state = self.alpha_max * self.state + (1.0 - self.alpha_max) * value
                self.step_count += 1

                if self.state > self.max_state:
                    self.max_state = self.state
                    self.last_max_state_step = self.step_count
                log.info(f"EarlyStopTracker: state={self.state:.5f}, max={self.max_state:.5f} alpha_max={self.alpha_max:.5f}")
            else:
                self.state = self.alpha_stop * self.state + (1.0 - self.alpha_stop) * value
                self.step_count += 1
                # max_state was found and stable
                self.state_threshold = self.max_state * self.stop_threshold
                log.info(f"EarlyStopTracker: state={self.state:.5f}, max={self.max_state:.5f}, th={self.state_threshold:.5f}, alpha_stop={self.alpha_stop:.5f} stop_count={self.stop_count}")
                if self.state < self.state_threshold:
                    self.stop_count += 1
                    # need to be below threshold for stop_count_min steps to actually stop
                    if self.stop_count>self.stop_patience:
                        if self.min_steps is None or (self.min_steps is not None and self.count>=self.min_steps):
                            self.stop = True
                            log.info(f"EarlyStopTracker: STOP")
                else:
                    self.stop_count = 0

    def is_stop(self):
        return self.stop


def get_params_by_regex(model, include=None, exclude=None, output_names=False):
    """ 
    Retun list of params using include/exclude regex patterns matched against layer name
    """
    
    include = (
        set([])
        if include is None
        else (
            include
            if isinstance(include, set)
            else set(include) if isinstance(include, (list, ListConfig)) else set([include])
        )
    )

    exclude = (
        set([])
        if exclude is None
        else (
            exclude
            if isinstance(exclude, set)
            else set(exclude) if isinstance(exclude, (list, ListConfig)) else set([exclude])
        )
    )

    params = []
    names = []
    for name, param in model.named_parameters():
        include_ = any( re.search( s, name) is not None for s in include ) or len(include)==0
        exclude_ = any( re.search( s, name) is not None for s in exclude )
        if include_ and not exclude_ and param.requires_grad:
            params.append(param)
            names.append(name)

    if output_names:
        return params, names
    else:
        return params

def embedding_masking_hook(grad) -> torch.Tensor or None:
    global update_token_embeddings, tokenizer
    if update_token_embeddings is not None:
        # mask out any gradient for non-added tokens
        grad_out = grad.clone().detach()
        grad_out[:-len(update_token_embeddings),:] = 0.0
        # grad_idxs = torch.where(grad_out[:,0])
        # log.info(f"updating embedding gradients for tokens {'|'.join([ tokenizer.decode(i) for i in grad_idxs ])}")
        return grad_out
    return grad

class EDMModuleDreamBooth(EDMModule):
    """
    Elucidating Diffusion Models Module
    Args:
    """

    def __init__(
        self,
        prior_model,
        ema_decay,
        optim,
        t_schedule,
        t_schedule_prior = None,
        fixed_encoder=False,
        normalize_audio=True,
        condition_dropout=0.0,
        sigma_data=0.5,
        sigma_lo=0.0,
        sigma_hi=np.inf,
        beta=False,
        metric_collection_train=torchmetrics.MetricCollection([]),
        metric_collection_val=torchmetrics.MetricCollection([]),
        gen_edm_kwargs={},
        app_name: str = "Dreambooth generation",
        base_prompt: str = None,
        prompts: List[str] = [],
        token_keywords: Dict = {},
        split_tokens: int = 1,
        split_token_mode: str = "sequence",
        prior_datamodule = None,
        prior_datamodule_filter_model=None,
        prior_lambda: float = 1.0,
        prior_num_samples: int = 1000,
        basename: str = None,
        update_layers: str = None,
        # prior_nocond: bool = False, 
        # prior_nocond_prob: float = 0.5, 
        prior_dump_audio: bool = False,
        prior_edm_args: Any = None,
        augment_tokens: Dict = None,
        condition_processor_path: str = "condition_processor.condition_processor_dict.our_baseclap",
        monitor_gradient: bool = False,
        cache = None,
        early_stop = None,
        **kwargs,
    ):
        super().__init__(
            prior_model.diffusion_ema if hasattr(prior_model, "diffusion_ema") else prior_model.diffusion,
            prior_model.preprocessor,
            ema_decay,
            optim,
            prior_model.quantizer,
            t_schedule,
            fixed_encoder=fixed_encoder,
            normalize_audio=normalize_audio,
            condition_dropout=condition_dropout,
            sigma_data=sigma_data,
            sigma_lo=sigma_lo,
            sigma_hi=sigma_hi,
            beta=beta,
            metric_collection_train=metric_collection_train,
            metric_collection_val=metric_collection_val,
            gen_edm_kwargs=gen_edm_kwargs,
        )

        self.base_prompt = base_prompt
        self.prompts = prompts
        self.prior_lambda = prior_lambda
        self.prior_num_samples = prior_num_samples
        self.prior_datamodule = prior_datamodule
        if self.prior_datamodule is not None:
            self.prior_datamodule.setup()
        
        self.prior_datamodule_filter_model = prior_datamodule_filter_model
        self.prior_datamodule_filter_model_state = None
        if self.prior_datamodule_filter_model is not None:
            # load FAISS database for this clap model for fast sample selection/search
            model_name = self.prior_datamodule_filter_model.experiment_name
            self.prior_datamodule_filter_model_state = init_faiss_index(
                    model_name,
                    model_dir=Path(".") / "symlinks" / model_name
            )

        self.basename = basename
        # self.prior_nocond = prior_nocond
        # self.prior_nocond_prob = prior_nocond_prob
        self.cache = cache
        self.monitor_gradient = monitor_gradient
        self.prior_dump_audio = prior_dump_audio
        self.prior_edm_args = prior_edm_args
        self.augment_tokens = augment_tokens

        self.condition_processor = prior_model.condition_processor
        # set always drop (p-1.0) audio_energy conditioning
        # set no dropout (p=1e-4) for text conditioning
        envelope_cond_dropout = 1.0
        envelope_cond_dropout_multiplier = 1e4 * envelope_cond_dropout
        if hasattr(self.condition_processor, "condition_processor_dict"):
            if "audio_energy" in self.condition_processor.condition_processor_dict:
                self.condition_processor.condition_processor_dict["audio_energy"].dropout_multiplier = envelope_cond_dropout_multiplier
        elif hasattr(self.condition_processor, "audio_energy"):
                self.condition_processor.audio_energy.dropout_multiplier = envelope_cond_dropout_multiplier

        self.condition_dropout = 1e-4
        self.condition_processor_path = condition_processor_path

        # set noise sampling for fine-tunine purpose
        # typically higher mean and std, if few FT samples
        self.t_schedule_prior = t_schedule_prior if t_schedule_prior is not None else t_schedule

        # make dream booth prompts, indexed by the token types they reference in the prompt, e.g.
        #   "[ACTION_TOKEN]:[MATERIAL_TOKEN]": ["footsteps [ACTION_TOKEN] on [MATERIAL_TOKEN]", "[ACTION_TOKEN] footsteps on [MATERIAL_TOKEN]"]
        self.prompts = {}
        self.allowed_token_types = set()
        for prompt in prompts:
            prompt = prompt.replace("[BASE_PROMPT]", self.base_prompt)
            token_types = sorted(re.findall(r'\[.*?\]', prompt))
            self.allowed_token_types.update(token_types)
            token_types = ':'.join(token_types)
            if token_types not in self.prompts:
                self.prompts[token_types] = [prompt]
            else:
                self.prompts[token_types].append(prompt)


        self.split_tokens = split_tokens
        self.split_token_mode = split_token_mode
        
        # expand string or list of keyword string into an actual list
        self.token_keywords = {}
        for token_type, tokens in token_keywords.items():
            d = {}
            for token_id, name_keywords in tokens.items():
                if "keywords"in name_keywords:
                    d[token_id] = name_keywords["keywords"] \
                                    if isinstance(name_keywords["keywords"], (list, ListConfig)) \
                                    else [ k.strip() for k in name_keywords["keywords"].split(",") ]
                    self.token_keywords[token_type] = d

        self.token_description_keywords = {}
        for token_type, tokens in token_keywords.items():
            d = {}
            for token_id, name_keywords in tokens.items():
                if "description"in name_keywords:
                    d[token_id] = name_keywords["description"] \
                                    if isinstance(name_keywords["description"], (list, ListConfig)) \
                                    else [ k.strip() for k in name_keywords["description"].split(",") ]
                    self.token_description_keywords[token_type] = d
        # determine new tokens to be added for DreamBooth
        # consider we may update N different embedding tokens 
        # per given token in the config
      
        # self.new_tokens = [
        #         token_id
        #         for token_type, tokens in self.token_keywords.items()
        #         for token_id, name_keywords in tokens.items()
        #     ]

        self.new_tokens = [
                token_id.replace(">",f"_{n}>") if self.split_tokens>1 else token_id
                for token_type, tokens in self.token_keywords.items()
                for token_id, name_keywords in tokens.items()
                for n in range(self.split_tokens)
            ]

        if self.augment_tokens is not None:
            for token in self.augment_tokens:
                self.new_tokens.append(token)

        # clap = self.locate_clap_model(self.diffusion)
        # cprocessor = self.locate_cprocessor_model(self.diffusion)
        clap = self.locate_clap_model(self)
        cprocessor = self.locate_cprocessor_model(self)
        assert clap is not None
        
        
        # add required new tokens
        pre_vocab_size = len(clap.tokenizer)
        n_added_tokens = clap.tokenizer.add_tokens( self.new_tokens )
        # resize tokenizer embedding matrix
        # resizing sets requires_grad=True for the embeddings.word_embeddings matrix
        clap.sentence_frontend.resize_token_embeddings(len(clap.tokenizer))
        log.info(f"added {n_added_tokens} tokens to CLAP tokenizer: from {pre_vocab_size} to {len(clap.tokenizer)}")
        log.info(f"  {', '.join(self.new_tokens)}")
        clap.audio_config.frozen = True
        clap.sentence_config.frozen = False
        # IMPORTANT: no dropout in CLAP model, whether updating or not updating
        # if training mode (and lightning will set it as such), conditioning doe
        # does not work anymore at inference time
        cprocessor.freeze_clap = False
        cprocessor.eval_mode = True
        cprocessor.skip_new_token_normalization = True

        # set training parameters after token extension (token extension sets requires grad automatically)
        self.update_layers = update_layers
        self.update_layers_schedule = False
        if isinstance(self.update_layers, (dict, DictConfig)) and \
                all( isinstance(k, int) for k in self.update_layers ):
            # all keys are numbers => start global_step
            # sort dict of update layer configs by increasing start_epoch
            self.update_layers = dict(sorted(self.update_layers.items(), key=lambda x: x[0]))
            self.last_update_layers_key = None
            self.update_layers_schedule = True
        else:
            # no number keys => include and exclude keys directly
            # single layer update config throughout training
            update_include = update_layers.include if "include" in update_layers else None
            update_exclude = update_layers.exclude if "exclude" in update_layers else None
            set_requires_grad(
                    self,
                    include=update_include,
                    exclude=update_exclude,
                    )
        
        # store new tokens for embedding gradient masking in backward_hook
        global update_token_embeddings, tokenizer
        # None updates all embedding rows
        # a list of token names adapts those only
        # update_token_embeddings = None
        update_token_embeddings = self.new_tokens
        if update_token_embeddings is not None and \
            clap.sentence_frontend.embeddings.word_embeddings.weight.requires_grad:
            clap.sentence_frontend.embeddings.word_embeddings.weight.register_hook(embedding_masking_hook)

        # unload unused models
        self.unload_model(self.diffusion, remove=["audio_frontend$", "audio_head$" ])
        
        # set train mode and set whole model trainable by default
        # leave as is if the model contains LoRA layers
        self.diffusion.train()
        if not is_lora(self.diffusion):
            self.diffusion.requires_grad_(True)
       
        # create EMA model copy
        del self.diffusion_ema
        self.diffusion_ema = copy.deepcopy(self.diffusion)
        self.diffusion_ema.eval()
        self.diffusion_ema.requires_grad_(False)

        # eval no grad for rest of submodels
        self.preprocessor.eval()
        self.preprocessor.requires_grad_(False)
        if self.quantizer is not None:
            self.quantizer.eval()
            self.quantizer.requires_grad_(False)
        if self.metric_collection_train is not None:
            for m in self.metric_collection_train:
                m.eval()
                m.requires_grad_(False)
        if self.metric_collection_val is not None:
            for m in self.metric_collection_val:
                m.eval()
                m.requires_grad_(False)
        
        self.prior_sample_cache = None
        self.prior_audio_sample_cache = None
        self.prior_noise_cache = None
        self.prior_prompt_cache = None
        self.prior_start_time_cache = None
        self.prior_sample_cache_nocond = None
        self.prior_noise_cache_nocond = None
        self.prior_prompt_cache_nocond = None
        self.prior_start_time_cache_nocond = None

        # count prompt types
        self.prompt_type_cnt = {}
        self.prompt_type_cnt_acc = {}
        self.prompt_type_cnt_epochs = 0
        self.prompt_type_cnt_after_norm = {}
        self.used_tokens = {}

        # early stop tracking
        if early_stop is not None:
            self.early_stop_tracker = EarlyStopTracker(**early_stop)
        else:
            self.early_stop_tracker = None
      
        # keep set of updated model keys to save them at on_save_checkpoint time
        self.updated_model_keys = set()

    def configure_optimizers(self):

        assert hasattr(self.optim_config.optimizer, "keywords")
        func_args = self.optim_config.optimizer.keywords
        if "layers" in func_args:
            log.info(f"setting learning rates per layer groups:")
            params_lr = []
            names_lr = []
            for lr, patterns in func_args["layers"].items():
                params, names = get_params_by_regex(
                            self,
                            include=patterns["include"] if "include" in patterns else None,
                            exclude=patterns["exclude"] if "exclude" in patterns else None,
                            output_names=True,
                        )
                for name in names:
                    log.info(f"  {name}: lr={lr}")
                params_lr.append( {"params": params, "lr": lr} )
                names_lr.extend(names)

            # # remaining params with base lr, exclude all params already in params_lr
            params_remaining = []
            names_lr = set(names_lr)
            for name, param in self.named_parameters():
                if param.requires_grad and "metric_collection_" not in name and \
                    name not in names_lr:
                    params_remaining.append(param)
            params_lr.append({ "params": params_remaining })
            log.info(f"  other params: lr={func_args['lr']}")
            del func_args["layers"]
            optimizer = self.optim_config.optimizer( params=params_lr )
        else:
            params = [
                v
                for k, v in self.named_parameters()
                if v.requires_grad and \
                "metric_collection_" not in k
            ]
            optimizer = self.optim_config.optimizer(
                params=params
            )

        scheduler = self._build_scheduler(
            scheduler_config=self.optim_config.scheduler, optimizer=optimizer
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "lr",
                }
            ],
        )

    def on_train_batch_start(self, trainer, pl_module):

        if self.update_layers_schedule:
            # self.update_layers dict is:
            #   global_step_start: update_layers

            # find update_layers config to use for current epoch
            update_layers_key = None
            for global_step_start in self.update_layers:
                if self.global_step>=int(global_step_start):
                    update_layers_key = global_step_start
                else:
                    break
            assert update_layers_key is not None

            # apply new requires_grad config if there was a change in update_layers_key vs. previous epochs
            if (self.last_update_layers_key is None) or \
               ( self.last_update_layers_key is not None and update_layers_key!=self.last_update_layers_key):

                # update requires_grad for this time step according to include and exclude patterns in config
                update_layers = self.update_layers[update_layers_key]
                if isinstance(update_layers, (list, ListConfig)):
                    update_include = update_layers
                    update_exclude = None
                elif isinstance(update_layers, (dict, DictConfig)):
                    update_include = update_layers.include if "include" in update_layers else None
                    update_exclude = update_layers.exclude if "exclude" in update_layers else None
                else:
                    raise ValueError(f"update_layers must be a list or a dict with 'include' and/or 'exclude' lists")
                set_requires_grad(
                        self,
                        include=update_include,
                        exclude=update_exclude,
                        )
                self.last_update_layers_key = update_layers_key

    def on_before_optimizer_step(self, optimizer):

        # update set of updated parameters, to be saved later
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.updated_model_keys.add(name)
                if "diffusion." in name:
                    # add diffusion_ema too
                    self.updated_model_keys.add(name.replace("diffusion.", "diffusion_ema."))

        if self.monitor_gradient:

            clap = self.locate_clap_model(self)
            # monitor embedding gradients
            embeddings = clap.sentence_frontend.embeddings.word_embeddings.weight
            embeddings_grad = embeddings.grad
            if embeddings_grad is None:
                log.info(f"embedding gradient not updated: no gradient available")
            else:
                grad_idxs = torch.where(embeddings_grad.abs().sum(dim=1)>1e-5)[0]
                log.info(f"embedding gradient L1 norm changed for tokens:")
                for idx in grad_idxs:
                    log.info(f"  {clap.tokenizer.decode(idx)}({idx}): {embeddings_grad[idx,:].abs().mean():.7f}")
                # showing gradients for tokens in the following words
                # not some word tokens may include space
                token_ids = set([ t for k in self.prompt_type_cnt_after_norm for t in clap.tokenizer(k)["input_ids"] ])
                log.info(f"embedding gradient L1 norm for tokens:")
                for idx in token_ids:
                    log.info(f"  {clap.tokenizer.decode(idx)}({idx}): grad={embeddings_grad[idx,:].abs().mean():.7f} mag={embeddings[idx,:].abs().mean():.7f}")

            # # show all non None gradients
            # log.info(f"updated gradients in diffusion:")
            # for name, param in self.named_parameters():
            #     if "diffusion." in name and param.grad is not None:
            #         log.info(f"  {name}: grad={param.grad.abs().mean():0.7f} mag={param.abs().mean():0.7f}")

    def locate_clap_model(self, model):
        try:
            return self.locate_cprocessor_model(model).model
        except:
            return None

    def locate_cprocessor_model(self, model):
        try:
            cprocessor_expr = f"model.{self.condition_processor_path}"
            return eval(cprocessor_expr)
        except:
            return None

    def unload_model(self, model, remove=None):

        remove = (
            set([])
            if remove is None
            else (
                remove
                if isinstance(remove, set)
                else set(remove) if isinstance(remove, list) else set([remove])
            )
        )

        for name, module in model.named_modules():
            if any( re.search(s, name) is not None for s in remove ):
                log.info (f"unloading {name}")
                var_name = re.sub(r"\.([0-9]+)", r"[\1]", f"model.{name}")
                expr = f"{var_name} = torch.nn.Identity()"
                exec(expr)

        return model

    def parse_tokens(self, keywords, token_match, out_tokens=None):

        def is_keyword_match (keywords, token_keywords):
        
            def cmp_keywords(k1, k2):
                # k1 can be a regular expression
                if not k1.isalpha():
                    return re.match(k1, k2) is not None
                else:
                    return k2.startswith(k1)

            return any([ cmp_keywords(tk,k) for k in keywords for tk in token_keywords ] )

        # search for stored token keywords as substrings of the given keyword list
        out_tokens = {} if out_tokens is None else out_tokens
        for token_type, tokens in token_match.items():
            for token_id, token_keywords in tokens.items():
                if is_keyword_match( keywords, token_keywords ):
                    if token_type not in out_tokens:
                        out_tokens[token_type] = [token_id]
                    else:
                        out_tokens[token_type].append(token_id)
                    # uncomment if first match only wanted
                    # break

        return out_tokens

    def make_prompts_from_keywords(self, batch):
        """
        creates a batch of DreamBooth prompts based on its keywords/categories.
        it replaces the 'description' field with the prompts.
        """

        if "category" not in batch:
            return batch

        batch_prompts = []
        for n, (category, description) in enumerate(zip(batch["category"], batch["description"])):
            if category is not None:
                # get keyword list from keyword string
                keywords = [ k.strip() for k in category.split(",") if k.strip() != "" ]
                # get dict of token_types and token_ids in this sample
                parsed_tokens = self.parse_tokens ( keywords, self.token_keywords )
            # consider comma-separated phrases
            if description is not None:
                descriptions = [ k.strip() for k in description.split(",") if k.strip() != "" ]
                # match against description phrases
                parsed_tokens = self.parse_tokens ( descriptions, self.token_description_keywords, parsed_tokens )
            # update used tokens and token types
            for k,v in parsed_tokens.items():
                if k not in self.used_tokens:
                    self.used_tokens[k] = set(v)
                else:
                    self.used_tokens[k].update(v)
            # drop token types randomly, backing off from n tokens to a lower number randomly
            # this should make prompt coverage much better
            if len (parsed_tokens)>1:
                n_tokens = random.randint(1, len(parsed_tokens))
                parsed_tokens = dict(random.sample(list(parsed_tokens.items()), n_tokens))

            # make prompt with given tokens
            prompt = self.make_prompt_from_tokens(parsed_tokens)

            # augment prompt if requested
            if self.augment_tokens is not None:
                # check only the augmentation steps of interest
                # get augment indices to check and corresponding token string
                augment_step_idxs = {
                        n: token
                        for token, step_pattern in self.augment_tokens.items()
                        for n, step_name in enumerate(batch["augment_steps"])
                        if step_pattern is not None and re.search(step_pattern, step_name) is not None
                        }

                if isinstance(batch["augment_mask"][n], (list, torch.Tensor)):
                    # if any([ v for i, v in enumerate(batch["augment_mask"][n]) if i in augment_step_idxs ]):
                    augmented = False
                    for i, aug in enumerate(batch["augment_mask"][n]):
                        if i in augment_step_idxs and aug:
                            augment_token = augment_step_idxs[i]
                            prompt = f"{augment_token} {prompt}"
                            augmented = True
                    if not augmented:
                        no_augment_token = [ token for token, step_pattern in self.augment_tokens.items() if step_pattern is None ]
                        if len(no_augment_token)>0:
                            no_augment_token = random.choice(no_augment_token)
                            prompt = f"{no_augment_token} {prompt}"
                elif isinstance(batch["augment_mask"][n], bool):
                    augmented = False
                    if 0 in augment_step_idxs:
                        augment_token = augment_step_idxs[0]
                        prompt = f"{augment_token} {prompt}"
                        augmented = True
                    if not augmented:
                        no_augment_token = [ token for token, step_pattern in self.augment_tokens.items() if step_pattern is None ]
                        if len(no_augment_token)>0:
                            no_augment_token = random.choice(no_augment_token)
                            prompt = f"{no_augment_token} {prompt}"

            # append to batch
            batch_prompts.append(prompt)

        # use prompts as description
        batch["description"] = batch_prompts

        return batch
            
    def make_prompt_from_tokens(self, parsed_tokens):
        """
        creates a prompt from a token dictionary like

        {
          "[MATERIAL]": ["<FS_M_GLASS>", "<FS_M_SNA">"],
          "[SHOE]": ["<FS_S_SNEAKER>"],
        }
        """

        if len(parsed_tokens)>0:
            # get sorted keys to index self.prompts with it
            parsed_tokens_key = ":".join(sorted([ k for k in parsed_tokens.keys() if k in self.allowed_token_types ]))
            if parsed_tokens_key in self.prompts:
                # update used token type counts
                if parsed_tokens_key in self.prompt_type_cnt:
                    self.prompt_type_cnt[parsed_tokens_key] += 1
                else:
                    self.prompt_type_cnt[parsed_tokens_key] = 1

                # get all possible prompts that use the parsed token types
                prompts = self.prompts[parsed_tokens_key]
                # sample a prompt from all possibe prompts
                prompt = random.choice(prompts)
                # replace token_types with the actual tokens ids in this sample
                for token_type, token_id in parsed_tokens.items():
                    chosen_token_id = random.choice(token_id) if len(token_id)>1 else token_id[0]
                    prompt = prompt.replace(token_type, chosen_token_id)

                if self.split_tokens>1:
                    if self.split_token_mode == "sequence":
                        # extend prompt with extended sequence of tokens
                        # <MY_MONSTER> monster growl  =>  <MY_MONSTER_0> <MY_MONSTER_1> ... <MY_MONSTER_N> monster growl
                        chosen_token_extended = " ".join([
                                    chosen_token_id.replace(">", f"_{n}>")
                                    for n in range(self.split_tokens)
                                ])
                    elif self.split_token_mode == "random":
                        split_token_idx = random.randint(0, self.split_tokens-1)
                        chosen_token_extended = chosen_token_id.replace(">", f"_{split_token_idx}>")
                    else:
                        raise ValueError(f"split_token_mode {self.split_token_mode} not supported: aborting")

                    prompt = prompt.replace(chosen_token_id, chosen_token_extended)
            else:
                prompt = self.base_prompt
        else:
            prompt = self.base_prompt

        # update used prompt counts
        if prompt in self.prompt_type_cnt:
            self.prompt_type_cnt[prompt] += 1
        else:
            self.prompt_type_cnt[prompt] = 1

        return prompt

    def make_prompts_prior(self, batch, force_prompt=None, prompts=None):
        """
        creates a batch of prior DreamBooth prompts.
        it replaces the 'description' field with the prompts.
        """

        prompt = self.base_prompt if force_prompt is None else force_prompt
        if prompts is not None:
            batch["description"] = [ prompts[n] for n, b in enumerate(batch["description"]) ]
        else:
            batch["description"] = [ prompt for b in batch["description"] ]

        return batch

    def on_fit_start(self):
        
        super().on_fit_start()

        # generate prior sample cache is uisng prior preservation loss
        if self.prior_lambda > 0.:
            self.prior_sample_cache, \
            self.prior_noise_cache, \
            self.prior_prompt_cache, \
            self.prior_start_time_cache, \
            self.prior_audio_sample_cache = self.generate_prior_sample_cache(
                **self.prior_edm_args,
                force_prompt=self.base_prompt,
            )


    def generate_prior_sample_cache(
        self,
        sampler="heun",
        num_steps=50,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        cfg=1,
        ema=True,
        force_prompt=""):
        """ pre-generate a number of samples for the prior preservation loss evaluation and cache it to disk """

        def make_dummy_batch( x , batch_size):
            if isinstance(x, list):
                if len(x)<batch_size:
                    return [ x[0] for _ in range(batch_size) ]
                return x
            elif isinstance(x, torch.Tensor):
                if len(x)<batch_size:
                    return torch.stack([ x[0] for _ in range(batch_size)])
                return x
            else:
                return x

        if self.prior_datamodule is None:
            sampling_cfg = (
                f"smp{sampler}_"
                f"{'' if self.prior_num_samples is None else f'n{self.prior_num_samples}_' }"
                f"ns{num_steps}_"
                f"sm{sigma_min:.1f}_"
                f"sM{sigma_max:.1f}_"
                f"r{rho:.1f}_"
                f"cfg{cfg:.1f}"
                f"{'_ema' if ema and hasattr(self, 'diffusion_ema') else ''}"
                f"{ '_'+force_prompt.replace(' ','_') if force_prompt!='' else '_nocond' }"
            )
        else:
            prior_dataset = self.prior_datamodule.pool_datasets['train']
            sampling_cfg = (
                f"dm{prior_dataset.__repr__().replace(' ','')}_"
                f"{'' if self.prior_num_samples is None else f'n{self.prior_num_samples}_' }"
                f"{force_prompt.replace(' ','_') if force_prompt!='' else '_nocond' }"
            )

        if self.basename is not None:
            cache_path = self.cache.cache_dir / "dreambooth" / self.basename / sampling_cfg
        else:
            cache_path = self.cache.cache_dir / "dreambooth" / sampling_cfg
       
        if force_prompt != "":
            log.info(f"generating prior samples for prompt {force_prompt} in {cache_path}")
        else:
            log.info(f"generating prior samples for empty prompt in {cache_path}")

        file_path = cache_path / "prior_sample_cache"

        if self.cache.enter(cache_path):
            if self.prior_datamodule is not None:
                prior_datamodule_idxs_kw = []
                prior_datamodule_idxs_clap = []
                if self.prior_datamodule_filter_model is not None:
                    with torch.no_grad():
                        batch = {"captions": [self.base_prompt]}
                        batch = self.prior_datamodule_filter_model.forward_sentence_model(batch, device=self.device)  # type: ignore
                        sentence_features = self.prior_datamodule_filter_model.sentence_head(batch["sentence_features"])  # type: ignore
                        sentence_features = torch.nn.functional.normalize(
                            sentence_features, p=2, dim=1
                        )
                        text_embedding = sentence_features.to("cpu")

                    audio_ids, sfxclap_scores = _faiss_search(
                        self.prior_datamodule_filter_model_state["index_audio"],
                        text_embedding,
                        self.prior_num_samples,
                        subset=None,
                        metadata=self.prior_datamodule_filter_model_state["index_metadata"],
                    )
                    # get datamodule idxs for selected ids on FAISS database
                    # map from id to idx
                    id2idx = { aid: n for n,aid in enumerate(prior_dataset.ids) }
                    prior_datamodule_idxs_clap = [ id2idx[audio_id] for audio_id in audio_ids if audio_id in id2idx ]
                    log.info(f"found {len(prior_datamodule_idxs_clap)} CLAP-matched prior samples in datamodule")
                    for idx in prior_datamodule_idxs_clap:
                        d = prior_dataset.descriptions[idx]
                        log.info(f"  {prior_dataset.descriptions[idx]}")

                if len(prior_datamodule_idxs_clap)<self.prior_num_samples:
                    # get list of idxs in datamodule that match the base prompt
                    keywords = set(re.sub(r"[,./;':]", " ", self.base_prompt).split(" "))

                    def keyword_match(string, keywords):
                        if string is None:
                            return False
                        words = set(re.split("[ ,./;:]", string))
                        return sum([ 1 if kw in words else 0 for kw in keywords]) == len(keywords)

                    prior_datamodule_idxs_kw = [ n for n,(c,d) in enumerate(zip(prior_dataset.categories,prior_dataset.descriptions)) \
                            if keyword_match(d, keywords) or \
                               keyword_match(c, keywords) \
                        ]
                    log.info(f"found {len(prior_datamodule_idxs_kw)} keyword-matched prior samples in datamodule")
                    for idx in prior_datamodule_idxs_kw:
                        d = prior_dataset.descriptions[idx]
                        if keyword_match(d, keywords):
                            log.info(f"  {prior_dataset.descriptions[idx]}")
                            continue
                        c = prior_dataset.categories[idx]
                        if keyword_match(c, keywords):
                            log.info(f"  {prior_dataset.categories[idx]}")
                            continue

                prior_datamodule_idxs = list(set(prior_datamodule_idxs_clap + prior_datamodule_idxs_kw))
                if len(prior_datamodule_idxs)>self.prior_num_samples:
                    prior_datamodule_idxs = prior_datamodule_idxs[:self.prior_num_samples]
                log.info(f"found {len(prior_datamodule_idxs)} keyword-matched or clap-matched prior samples in datamodule")
                random.shuffle(prior_datamodule_idxs)

            if rank == 0:
                t0 = time.time()
                
                pbar = tqdm(
                    desc="Caching prior samples",
                    total=len(self.trainer.datamodule.train_dataset()) if self.prior_num_samples is None else self.prior_num_samples,
                    leave=True,
                    file=sys.stdout,
                )
                stop = False

                batch_size = self.trainer.datamodule.dataloader.batch_size
                n_batches = int( ( self.prior_num_samples + batch_size ) / batch_size )
                sample_size = self.trainer.datamodule.sample_size

                for n, batch in enumerate(self.trainer.datamodule.train_dataloader()):
                    break
                if batch["audio"].size(0)<batch_size:
                    # extend batch
                    batch = { 
                                k: make_dummy_batch(v, batch_size)
                                for k,v in batch.items()
                            }
                batch_src = { k: ( v.to(device=self.device) if isinstance(v, torch.Tensor) else v ) for k,v in batch.items() }

                prior_prompt_cache = []
                prior_start_time_cache = []
                prior_sample_cache = []
                prior_audio_sample_cache = []
                prior_noise_cache = []
                audio_sample_idx = 0
                while not stop:
                    if self.prior_datamodule is None:
                        for n in range(n_batches):
                            batch = copy.deepcopy(batch_src)
                            x = batch["audio"]
                            with torch.no_grad():
                                x = self.normalize_audio(x)
                                x = self.preprocessor(x)
                            # batch["description"] = [ "" for i in range(len(batch["audio"])) ]
                            batch["description"] = [ "" for i in range(batch_size) ]
                            batch = self.make_prompts_prior(batch, force_prompt=force_prompt)
                            description_prior = self.base_prompt
                            # start_time_prior = batch["from_time"]
                            start_time_prior = [ 15.0*random.random() for _ in range(batch_size) ]
                            # start_time_prior = [ 0.0 for _ in range(batch_size) ]
                            # update conditioning
                            # cond = self.get_cond(batch, no_dropout=True)
                            cond = self.get_cond(batch, no_cond=False)
                            # compute loss with generated audio from prior model
                            noise_prior = torch.randn_like(x)
                            x_prior = self.sample(
                                        noise_prior,
                                        sampler=sampler,
                                        cond=cond,
                                        num_steps=num_steps,
                                        sigma_min=sigma_min,
                                        sigma_max=sigma_max,
                                        rho=rho,
                                        cfg=cfg,
                                        ema=ema and hasattr(self, "diffusion_ema"),
                                        # progress_disable=True,
                                    )
                            # generate audio as [B, 1, N]
                            audio = self.preprocessor.inverse(x_prior)

                            # generate audio
                            if self.prior_dump_audio:
                                audio_dir = cache_path / "audio"
                                audio_dir.mkdir(parents=True, exist_ok=True)
                                for audio_sample in audio:
                                    fn = audio_dir / f"sample_{audio_sample_idx}.wav"
                                    torchaudio.save(fn, audio_sample.detach().cpu(), self.trainer.datamodule.sample_rate)
                                    audio_sample_idx += 1
                            x_prior_audio = audio
                            prior_start_time_cache.append( start_time_prior )
                            prior_prompt_cache.append( description_prior )
                            prior_sample_cache.append( x_prior )
                            prior_audio_sample_cache.append( x_prior_audio )
                            prior_noise_cache.append( noise_prior )
                            pbar.update(len(x))
                            if self.prior_num_samples is not None:
                                n_samples = sum ( len(b) for b in prior_sample_cache )
                                if n_samples>=self.prior_num_samples:
                                    stop = True
                                    break
                    else:
                        for idx in prior_datamodule_idxs:
                            sample = prior_dataset[idx]
                            audio = sample["audio"].unsqueeze(0).to(device=self.device)
                            description_prior = [ self.base_prompt ]
                            start_time_prior = [ sample["from_time"] ]

                            with torch.no_grad():
                                x = self.normalize_audio(audio)
                                x_prior = self.preprocessor(x)
                            noise_prior = torch.randn_like(x_prior)

                            # generate audio
                            if self.prior_dump_audio:
                                audio_dir = cache_path / "audio"
                                audio_dir.mkdir(parents=True, exist_ok=True)
                                for audio_sample in audio:
                                    fn = audio_dir / f"sample_{audio_sample_idx}.wav"
                                    torchaudio.save(fn, audio_sample.detach().cpu(), self.trainer.datamodule.sample_rate)
                                    audio_sample_idx += 1
                            x_prior_audio = audio
                            prior_start_time_cache.append( start_time_prior )
                            prior_prompt_cache.append( description_prior )
                            prior_sample_cache.append( x_prior )
                            prior_audio_sample_cache.append( x_prior_audio )
                            prior_noise_cache.append( noise_prior )
                            pbar.update(len(x))
                            if self.prior_num_samples is not None:
                                n_samples = sum ( len(b) for b in prior_sample_cache )
                                if n_samples>=self.prior_num_samples:
                                    stop = True
                                    break


                    if self.prior_num_samples is None:
                        # just do one batch
                        stop = True
                            

                # [N, C, T]
                prior_prompt_cache = [ d for b in prior_prompt_cache for d in b ]
                prior_start_time_cache = [ t for b in prior_start_time_cache for t in b ]
                prior_sample_cache = torch.cat (prior_sample_cache, dim=0).clone().detach().cpu()
                prior_audio_sample_cache = torch.cat (prior_audio_sample_cache, dim=0).clone().detach().cpu()
                prior_noise_cache = torch.cat (prior_noise_cache, dim=0).clone().detach().cpu()
                if self.prior_num_samples is not None:
                    prior_start_time_cache = prior_start_time_cache[:self.prior_num_samples]
                    prior_prompt_cache = prior_prompt_cache[:self.prior_num_samples]
                    prior_sample_cache = prior_sample_cache[:self.prior_num_samples,...]
                    prior_audio_sample_cache = prior_audio_sample_cache[:self.prior_num_samples,...]
                    prior_noise_cache = prior_noise_cache[:self.prior_num_samples,...]

                # store on disk
                cache_path.mkdir(parents=True, exist_ok=True)
                torch.save(prior_prompt_cache, f"{file_path}.d.pt")
                torch.save(prior_start_time_cache, f"{file_path}.st.pt")
                torch.save(prior_sample_cache, f"{file_path}.x.pt")
                torch.save(prior_audio_sample_cache, f"{file_path}.au.pt")
                torch.save(prior_noise_cache, f"{file_path}.n.pt")
                log.info(f"{prior_sample_cache.size(0)} prior samples cached to {file_path} for prompt={force_prompt}")
                self.cache.signal_done(cache_path)
            else:
                self.cache.wait_done(cache_path)
                prior_prompt_cache = torch.load(f"{file_path}.d.pt")
                prior_sample_cache = torch.load(f"{file_path}.x.pt")
                prior_audio_sample_cache = torch.load(f"{file_path}.au.pt")
                prior_noise_cache = torch.load(f"{file_path}.n.pt")
                try:
                    prior_start_time_cache = torch.load(f"{file_path}.st.pt")
                except:
                    prior_start_time_cache = [ 0.0 for _ in range(len(prior_prompt_cache)) ]
        else:
            prior_prompt_cache = torch.load(f"{file_path}.d.pt")
            prior_sample_cache = torch.load(f"{file_path}.x.pt")
            prior_audio_sample_cache = torch.load(f"{file_path}.au.pt")
            prior_noise_cache = torch.load(f"{file_path}.n.pt")
            try:
                prior_start_time_cache = torch.load(f"{file_path}.st.pt")
            except:
                prior_start_time_cache = [ 0.0 for _ in range(len(prior_prompt_cache)) ]
            log.info(f"{prior_sample_cache.size(0)} prior samples loaded from cache {file_path}")

        return prior_sample_cache, prior_noise_cache, prior_prompt_cache, prior_start_time_cache, prior_audio_sample_cache

            
    def training_step(self, batch, batch_idx):

        x = batch["audio"]
        with torch.no_grad():
            x = self.normalize_audio(x)
            # latent variable
            x = self.preprocessor(x)

        # Draw well distributed continuous timesteps
        t = self.sample_t(x=x, t_schedule=self.t_schedule)

        # get inputs
        noise = torch.randn_like(x)

        # compute fine tuning loss
        batch = self.make_prompts_from_keywords(batch)
        cond = self.get_cond(batch, no_cond=False)
        # cond = self.get_cond(batch, no_dropout=True)
        # track prompts and counts after any text normalization
        for prompt in cond["description"]:
            if prompt in self.prompt_type_cnt_after_norm:
                self.prompt_type_cnt_after_norm[prompt] += 1
            else:
                self.prompt_type_cnt_after_norm[prompt] = 1
            if self.current_epoch == 0:
                log.info(f"cond.description: {prompt}")

        mask = torch.ones_like(x[:, 0, :])
        loss_ft, reco_loss_ft = self.loss_fn(x, cond, noise, t, ema=False, mask=mask)

        # compute prior preservation loss for pretrained model
        if self.prior_sample_cache is not None and self.prior_noise_cache is not None:

            # sample indices from prior samples
            batch_size = x.size(0)
            # random prior sample indices
            idxs = torch.randint (self.prior_sample_cache.size(0), (batch_size,) )

            # class-conditioned prior samples
            x_prior = self.prior_sample_cache[idxs,...].to(device=self.device)
            x_prior_audio = self.prior_audio_sample_cache[idxs,...].to(device=self.device)
            noise_prior = self.prior_noise_cache[idxs,...].to(device=self.device)
            prompt_prior = [ self.prior_prompt_cache[i] for i in idxs ]
            start_time_prior = [ self.prior_start_time_cache[i] for i in idxs ]

            for prompt in prompt_prior:
                if self.current_epoch == 0:
                    log.info(f"prior.description: {prompt}")

            # get dream booth prompts
            batch = self.make_prompts_prior(batch, prompts=prompt_prior)
            batch["from_time"] = start_time_prior
            batch["audio"] = x_prior_audio
            # cond_prior = self.get_cond(batch, no_dropout=True)
            # cond_prior = self.get_cond(batch, condition_dropout=1e-4)
            cond_prior = self.get_cond(batch, no_cond=False)
            t_prior = self.sample_t(x=x_prior, t_schedule=self.t_schedule_prior)
            loss_prior, reco_loss_prior = self.loss_fn(x_prior, cond_prior, noise_prior, t_prior, ema=False, mask=mask)
            loss = loss_ft + self.prior_lambda * loss_prior
            reco_loss = reco_loss_ft + self.prior_lambda * reco_loss_prior
        else:
            loss = loss_ft
            reco_loss = reco_loss_ft
            loss_prior = torch.tensor([0.0], dtype=torch.float32).to(device=self.device)
            reco_loss_prior = torch.tensor([0.0], dtype=torch.float32).to(device=self.device)

        with torch.no_grad():
            self.log_everything(
                reco_loss=reco_loss,
                loss=loss,
                reco_loss_ft=reco_loss_ft,
                loss_ft=loss_ft,
                reco_loss_prior=reco_loss_prior,
                loss_prior=loss_prior,
                phase="train",
            )

            # log to progress bar
            self.log("loss", loss.mean().item(), prog_bar=True)
            self.log("loss_ft", loss_ft.mean().item(), prog_bar=True)
            self.log("loss_prior", loss_prior.mean().item(), prog_bar=True)

        if self.early_stop_tracker is not None:
            self.early_stop_tracker.update(loss_ft.mean().item())
            self.log("early_stop_tracker_state", self.early_stop_tracker.state)
            self.log("early_stop_tracker_th", self.early_stop_tracker.state_threshold)
            if self.early_stop_tracker.is_stop():
                # STOP training
                log.info(f"EARLY STOPPING TRAINING")
                self.trainer.save_checkpoint(
                    self.trainer.checkpoint_callback.format_checkpoint_name(
                        {
                            "epoch": self.trainer.current_epoch,
                            "step": self.global_step,
                        }
                    )
                )
                self.trainer.should_stop = True

        return loss

    def on_train_epoch_end(self):
        self.prompt_type_cnt_epochs += 1
        self.prompt_type_cnt_acc = { k: (v if k not in self.prompt_type_cnt_acc else self.prompt_type_cnt_acc[k]+v) for k,v in self.prompt_type_cnt.items() }
        # show used tokens
        # pprint ({k:f'{v/self.prompt_type_cnt_epochs:.2f}' for k,v in self.prompt_type_cnt_acc.items() } )

    def log_everything(self, reco_loss, loss, phase, reco_loss_ft=None, loss_ft=None, reco_loss_prior=None, loss_prior=None):
        """
        reco_loss: unreduced reconstruction loss
        """
        log_dict = {}

        log_dict.update(
            {
                f"{phase}/loss": loss.detach().item(),
            }
        )
        if loss_ft is not None:
            log_dict.update({f"{phase}/loss_ft": loss_ft.detach().item()})
        if loss_prior is not None:
            log_dict.update({f"{phase}/loss_prior": loss_prior.detach().item()})

        self.log_dict(log_dict, sync_dist=phase == "val")
        # del mse_loss_per_band

    def on_save_checkpoint(self, checkpoint):
        """
        Can change checkpoint keys to be saved here
        """
        checkpoint = checkpoint["state_dict"]
        if hasattr(self, "used_tokens") and self.used_tokens is not None:
            checkpoint["used_tokens"] = self.used_tokens
        checkpoint["prompt_type_cnt"] = { k:v/self.prompt_type_cnt_epochs for k,v in self.prompt_type_cnt_acc.items() }
        # don't save metrics if they exist
        is_lora_model = is_lora(self.diffusion)

        for k in list(checkpoint.keys()):
            if k=="used_tokens":
                continue
            if "metric_collection_" in k:
                del checkpoint[k]
                continue
            if is_lora_model and not k in self.updated_model_keys:
                del checkpoint[k]
                continue
        # for k in checkpoint.keys():
        #     log.info(f"saving key {k} to checkpoint")

    def on_load_checkpoint(self, checkpoint):
        """
        Can change checkpoint keys to be saved here
        """
        checkpoint = checkpoint["state_dict"]
        for k in list(checkpoint.keys()):
            if "metric_collection_" in k:
                del checkpoint[k]
                continue
