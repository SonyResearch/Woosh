from typing import List, Dict, Any
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
import hydra
from omegaconf import ListConfig, DictConfig
import numpy as np
from tqdm import tqdm
import gc
from pytorch_lightning.utilities.model_summary import ModelSummary
import time
from sfxfm.utils.dist import rank
from pprint import pprint
from sfxfm.module.model import is_lora, set_requires_grad, set_forward_lora

rank = rank()

# get logger
log = logging.getLogger(__name__)


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

# def embedding_masking_hook(module, grad_input, grad_output) -> torch.Tensor or None:
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

class EDMModuleTextBoost(EDMModule):
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
        split_tokens: int = 1,
        split_token_mode: str = "sequence",
        token_keywords: Dict = {},
        kp_datamodule: Any = None,
        kp_lambda: float = 1.0,
        basename: str = None,
        update_layers: str = None,
        augment_tokens: Dict = None,
        condition_processor_path: str = "condition_processor.condition_processor_dict.our_baseclap",
        monitor_gradient: bool = False,
        cache = None,
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
        self.kp_lambda = kp_lambda
        self.basename = basename
        self.cache = cache
        self.monitor_gradient = monitor_gradient
        self.augment_tokens = augment_tokens

        self.condition_processor = prior_model.condition_processor
        self.condition_processor_path = condition_processor_path


        # prepare prior prompts for TextBoost
        self.kp_datamodule = kp_datamodule
        self.prepare_kp_prompts()

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
                d[token_id] = name_keywords["keywords"] \
                                if isinstance(name_keywords["keywords"], (list, ListConfig)) \
                                else [ k.strip() for k in name_keywords["keywords"].split(",") ]
                self.token_keywords[token_type] = d

        # # determine new tokens to be added for DreamBooth
        # self.new_tokens = [
        #         token_id
        #         for token_type, tokens in self.token_keywords.items()
        #         for token_id, name_keywords in tokens.items()
        #     ]

        # determine new tokens to be added for DreamBooth
        # consider we may update N different embedding tokens 
        # per given token in the config
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
        self.diffusion.eval()
        self.diffusion.requires_grad_(False)
       
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
            self.metric_collection_train.eval()
            self.metric_collection_train.requires_grad_(False)
            self.metric_collection_val.eval()
            self.metric_collection_val.requires_grad_(False)
        
        # count prompt types
        self.prompt_type_cnt = {}
        self.prompt_type_cnt_acc = {}
        self.prompt_type_cnt_epochs = 0
        self.prompt_type_cnt_after_norm = {}
        self.used_tokens = {}
   
    def prepare_kp_prompts(self):

        assert self.basename is not None
        cache_path = self.cache.cache_dir / "textboost" / self.basename
        prompt_fn = cache_path / "tb_prompts.pt"

        if self.cache.enter(cache_path):
            if rank == 0:
                config_datamodule = hydra.compose(
                    config_name=f"datamodule/{self.kp_datamodule}",
                    overrides=[
                         "++sample_rate=48000",
                         "++sample_size=48000",
                        f"++save_dir={cache_path}",
                         "++freeze_cache=true",
                         "++rebuild_cache=false",
                         "++extra_metadata=null",
                         "++filter_min_sample_size=false",
                        f"++cache_dir={self.cache.cache_dir}",
                         "++seed=777",
                         "++mp_shm_strategy=file_descriptor",
                    ],
                ).datamodule
                self.kp_datamodule = hydra.utils.instantiate(config_datamodule)
                self.kp_datamodule.setup()
                self.kp_prompts = None
                if "train" in self.kp_datamodule.pool_datasets:
                    train_dataset = self.kp_datamodule.pool_datasets["train"]
                    self.kp_prompts = train_dataset.get_dataframe()["description"].dropna().tolist()
                assert isinstance(self.kp_prompts, list) and len(self.kp_prompts)>0
                torch.save(self.kp_prompts, prompt_fn)
                log.info(f"cached {len(self.kp_prompts)} prompts to {prompt_fn}")
                self.cache.signal_done(cache_path)
            else:
                self.cache.wait_done(cache_path)
                self.kp_prompts = torch.load(prompt_fn)
        else:
            self.kp_prompts = torch.load(prompt_fn)
            log.info(f"{len(self.kp_prompts)} prompts loaded from cache {prompt_fn}")

        return self.kp_prompts

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

    # def on_train_epoch_start(self, trainer, pl_module):
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

            # show all non None gradients
            log.info(f"updated gradients in clap:")
            for name, param in self.named_parameters():
                if "condition_processor." in name and param.grad is not None:
                    log.info(f"  {name}: grad={param.grad.abs().mean():0.7f} mag={param.abs().mean():0.7f}")

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
                print (f"unloading {name}")
                var_name = re.sub(r"\.([0-9]+)", r"[\1]", f"model.{name}")
                expr = f"{var_name} = torch.nn.Identity()"
                exec(expr)

        return model

    def parse_tokens(self, keywords):

        def is_keyword_match (keywords, token_keywords):
        
            def cmp_keywords(k1, k2):
                # k1 can be a regular expression
                if not k1.isalpha():
                    return re.match(k1, k2) is not None
                else:
                    return k2.startswith(k1)

            return any([ cmp_keywords(tk,k) for k in keywords for tk in token_keywords ] )

        # search for stored token keywords as substrings of the given keyword list
        out_tokens = {}
        for token_type, tokens in self.token_keywords.items():
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
        for n, b in enumerate(batch["category"]):
            # get keyword list from keyword string
            keywords = [ k for k in b.split(",") if k != "" ]
            # get dict of token_types and token_ids in this sample
            parsed_tokens = self.parse_tokens ( keywords )
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

    def make_prompts_prior(self, batch):
        """
        creates a batch of varied prompts from a picked from a dataset
        """

        assert isinstance(self.kp_prompts, list)
        batch_size = len(batch["description"])
        idxs = torch.randint(len(self.kp_prompts), (batch_size,))
        batch["description"] = [ self.kp_prompts[i] for i in idxs ]

        return batch

            
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
        cond = self.get_cond(batch, no_dropout=True)
        # track prompts and counts after any text normalization
        for prompt in cond["description"]:
            if prompt in self.prompt_type_cnt_after_norm:
                self.prompt_type_cnt_after_norm[prompt] += 1
            else:
                self.prompt_type_cnt_after_norm[prompt] = 1
            if self.current_epoch == 0:
                log.info(f"cond.description: {prompt}")

        loss_ft, reco_loss_ft = self.loss_fn(x, cond, noise, t, ema=False)

        # compute knowledge preservation loss
        batch = self.make_prompts_prior(batch)
        clap = self.locate_clap_model(self)
        set_forward_lora(clap, False)
        cond_prior = self.get_cond(batch, no_dropout=True)
        set_forward_lora(clap, True)
        cond_ft = self.get_cond(batch, no_dropout=True)
        # compute Lkp loss as normalized dot products of
        # cond_prior and cond_ft embeddings
        # for cross_attention embeddings
        emb_prior = cond_prior["cross_attn_cond"]
        emb_ft = cond_ft["cross_attn_cond"]
        emb_mask = cond_ft["cross_attn_cond_mask"]
        # compute norm dot products
        prods = emb_prior * emb_ft
        norms_prior = torch.norm(emb_prior, dim=2)
        norms_ft = torch.norm(emb_ft, dim=2)
        norm_cross_prods = torch.sum(prods, dim=2) / (norms_prior * norms_ft)
        # masked mean
        loss_kp = torch.sum(norm_cross_prods * emb_mask, dim=1) / torch.sum(emb_mask, dim=-1)
        # mean over batch dim
        loss_kp = loss_kp.mean()

        loss = loss_ft - self.kp_lambda * loss_kp
        reco_loss = reco_loss_ft

        with torch.no_grad():
            self.log_everything(
                reco_loss=reco_loss,
                loss=loss,
                loss_ft=loss_ft,
                loss_kp=loss_kp,
                phase="train",
            )

            # log to progress bar
            self.log("loss", loss.mean().item(), prog_bar=True)
            self.log("loss_ft", loss_ft.mean().item(), prog_bar=True)
            self.log("loss_kp", loss_kp.mean().item(), prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.prompt_type_cnt_epochs += 1
        self.prompt_type_cnt_acc = { k: (v if k not in self.prompt_type_cnt_acc else self.prompt_type_cnt_acc[k]+v) for k,v in self.prompt_type_cnt.items() }
        # show used tokens
        # pprint ({k:f'{v/self.prompt_type_cnt_epochs:.2f}' for k,v in self.prompt_type_cnt_acc.items() } )

    def log_everything(self, reco_loss, loss, phase, loss_ft=None, loss_kp=None):
        """
        reco_loss: unreduced reconstruction loss
        """
        log_dict = {}

        # # MSE per t-bin
        # mse_loss_per_band = reco_loss
        # loss_per_band = {
        #     f"{phase}/mse_band#{k}": l.item()
        #     for k, l in enumerate(mse_loss_per_band.mean(2).mean(0))
        # }
        # log_dict.update(loss_per_band)

        log_dict.update(
            {
                f"{phase}/loss": loss.detach().item(),
            }
        )
        if loss_ft is not None:
            log_dict.update({f"{phase}/loss_ft": loss_ft.detach().item()})
        if loss_kp is not None:
            log_dict.update({f"{phase}/loss_kp": loss_kp.detach().item()})

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
        for k in list(checkpoint.keys()):
            if "metric_collection_" in k:
                del checkpoint[k]

    def on_load_checkpoint(self, checkpoint):
        """
        Can change checkpoint keys to be saved here
        """
        checkpoint = checkpoint["state_dict"]
        if "used_tokens" in checkpoint:
            self.used_tokens = checkpoint["used_tokens"]
