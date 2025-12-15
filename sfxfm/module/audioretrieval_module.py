"""
Module to train CLAP
"""

import contextlib
import hashlib
import importlib
import logging
import math
import shutil
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import PIL
import PIL.Image
import torch
import torch.utils
import torch.utils.data
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch import nn

from sfxfm.module.model.retrieval.passt import create_passt_model
from sfxfm.module.utils import get_module_by_name
from sfxfm.utils.cache import Cache

from .base import BaseLightningModule

rank = 0

def is_distributed():
    return False
# get logger
log = logging.getLogger(__name__)


def no_op(x):
    return x


remove_punctuation = str.maketrans("", "", string.punctuation)


def default_text_preprocessing(text_list):
    return [text.lower().translate(remove_punctuation).strip() for text in text_list]


def get_text_preprocessing_func(text_preprocessing):
    if text_preprocessing is None:
        text_preprocessing = default_text_preprocessing
    elif text_preprocessing == "no_op":
        text_preprocessing = no_op
    else:
        raise ValueError(f"Unknown text_preprocessing function: {text_preprocessing}")
    return text_preprocessing


def get_audio_frontend_model(audio_config, cache=None) -> Tuple[nn.Module, int]:
    if audio_config.name.startswith("passt"):
        model, output_dim = create_passt_model(audio_config)
        return model, output_dim

    if audio_config.name.startswith("laion"):
        import laion_clap

        assert audio_config.sample_rate == 48000, (
            "Sample rate must be 48kHz for laion-clap models"
        )

        model = laion_clap.CLAP_Module(
            enable_fusion=audio_config["enable_fusion"],
            amodel=audio_config["amodel"],
            device="cpu",
        )
        ckpt_path = CheckpointDownloadCache(cache=cache)(audio_config.checkpoint_path)
        model.load_ckpt(ckpt_path)
        use_proj = audio_config.include_proj

        class LaionClapAudioWrapper(torch.nn.Module):
            def __init__(self, model, include_projection):
                super().__init__()
                self.proj = include_projection
                self.model = model.model.audio_branch
                if include_projection:
                    self.proj = model.model.audio_projection

            def forward(self, x, *args, **kwargs):
                device = x["waveform"].device
                x = self.model(x, mixup_lambda=None, device=device)["embedding"]
                if self.proj:
                    x = self.proj(x)
                return x

        audio_frontend_output_dim = 512 if use_proj else audio_config["amodel_dim"]

        return LaionClapAudioWrapper(model, use_proj), audio_frontend_output_dim
    raise ValueError(f"Unknown audio frontend model: {audio_config.name}")


def get_audio_head_model(
    audio_config, shared_representation_size, audio_output_size
) -> nn.Module:
    if audio_config.adopt_n_layers == -1:
        assert audio_output_size == shared_representation_size
        return nn.Identity()
    layer_sizes = [audio_output_size]
    layer_sizes += [audio_config.adopt_layer_size] * audio_config.adopt_n_layers
    layer_sizes += [shared_representation_size]
    audio_layers = []
    for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
        audio_layers.append(torch.nn.Linear(i, o))
        audio_layers.append(torch.nn.ReLU())

    audio_layers.pop()
    return torch.nn.Sequential(*audio_layers)


def get_sentence_frontend_model(sentence_config):
    from transformers import (
        AutoModel,
        AutoTokenizer,
        BertModel,
        BertTokenizer,
        CLIPTextModel,
        CLIPTokenizer,
        DistilBertModel,
        DistilBertTokenizer,
        GPT2Model,
        GPT2Tokenizer,
        ModernBertModel,
        RobertaModel,
        RobertaTokenizer,
    )

    # Model, tokenizer, embedding_size
    MODELS = {
        "openai/clip-vit-base-patch32": (CLIPTextModel, CLIPTokenizer, 512),
        "prajjwal1/bert-tiny": (BertModel, BertTokenizer, 128),
        "prajjwal1/bert-mini": (BertModel, BertTokenizer, 256),
        "prajjwal1/bert-small": (BertModel, BertTokenizer, 512),
        "prajjwal1/bert-medium": (BertModel, BertTokenizer, 512),
        "gpt2": (GPT2Model, GPT2Tokenizer, 768),
        "distilgpt2": (GPT2Model, GPT2Tokenizer, 768),
        "bert-base-uncased": (BertModel, BertTokenizer, 768),
        "bert-large-uncased": (BertModel, BertTokenizer, 1024),
        "roberta-base": (RobertaModel, RobertaTokenizer, 768),
        "roberta-large": (RobertaModel, RobertaTokenizer, 1024),
        "distilbert-base-uncased": (DistilBertModel, DistilBertTokenizer, 768),
        "distilroberta-base": (RobertaModel, RobertaTokenizer, 768),
        "answerdotai/ModernBERT-base": (ModernBertModel, AutoTokenizer, 768),
        "answerdotai/ModernBERT-large": (ModernBertModel, AutoTokenizer, 1024),
    }

    if sentence_config.model.startswith(
        "sentence-transformers/"
    ) or sentence_config.model in [
        "google/embeddinggemma-300m",
        "Qwen/Qwen3-Embedding-0.6B",
    ]:
        from sentence_transformers import SentenceTransformer

        sentence_embedding_model = SentenceTransformer(sentence_config.model)
        tokenizer = None
        sentence_frontend_output_dim = (
            sentence_embedding_model.get_sentence_embedding_dimension()
        )
    elif sentence_config.model == "chandar-lab/NeoBERT":
        # requires trust_remote_code=True
        sentence_embedding_model = AutoModel.from_pretrained(
            sentence_config.model,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            sentence_config.model,
            trust_remote_code=True,
        )
        sentence_frontend_output_dim = 768

    elif "clip" not in sentence_config.model and not sentence_config.model.startswith(
        "laion"
    ):
        extra_args = {}
        # Removed: transformers v4.51.0 doesn't support multiple loading strategies
        # if sfxfm.utils.loading.lazy_loading_enabled:
        # setting state_dict to empty dict to avoid loading the model weights
        # extra_args["state_dict"] = {}
        # Benno's hack:
        # TODO: issue 1536, fix lazy loading of transformer models
        # extra_args["low_cpu_mem_usage"] = True
        # del extra_args["state_dict"]
        # @TODO: check if other models need these extra args

        if "roberta" in sentence_config.model:
            extra_args = {
                "add_pooling_layer": sentence_config.get("add_pooling_layer", False),
                "hidden_dropout_prob": sentence_config.get("hidden_dropout_prob", 0.2),
                "attention_probs_dropout_prob": sentence_config.get(
                    "attention_probs_dropout_prob", 0.2
                ),
                **extra_args,
            }
        try:
            sentence_embedding_model = MODELS[sentence_config.model][0].from_pretrained(
                sentence_config.model,
                **extra_args,
            )
            # config = AutoConfig.from_pretrained(sentence_config.model)
            # if "roberta" in sentence_config.model:
            #     config.add_pooling_layer = sentence_config.get("add_pooling_layer", False)
            #     config.hidden_dropout_prob = sentence_config.get("hidden_dropout_prob", 0.2)
            #     config.attention_probs_dropout_prob = sentence_config.get("attention_probs_dropout_prob", 0.2)

            # sentence_embedding_model = MODELS[sentence_config.model][0].from_pretrained(
            #     config=config,
            #     pretrained_model_name_or_path=None if sfxfm.utils.loading.lazy_loading else sentence_config.model,
            #     **extra_args,
            # )
            tokenizer = MODELS[sentence_config.model][1].from_pretrained(
                sentence_config.model
            )
            sentence_frontend_output_dim = MODELS[sentence_config.model][2]
        except KeyError:
            sentence_embedding_model = AutoModel.from_pretrained(sentence_config.model)
            tokenizer = AutoTokenizer.from_pretrained(sentence_config.model)
            sentence_frontend_output_dim = sentence_embedding_model.config.hidden_size
    elif sentence_config.model.startswith("laion"):
        import laion_clap

        model = laion_clap.CLAP_Module(
            enable_fusion=self.audio_config["enable_fusion"],
            amodel=self.audio_config["amodel"],
            device="cpu",
        )
        ckpt_path = CheckpointDownloadCache(cache=self.cache)(
            sentence_config.checkpoint_path
        )
        model.load_ckpt(ckpt_path)
        use_proj = sentence_config.include_proj

        class LaionClapTextWrapper(torch.nn.Module):
            def __init__(self, model, include_projection):
                super().__init__()
                self.proj = include_projection
                self.model = model.model.text_branch
                if include_projection:
                    self.proj = model.model.text_projection

            def forward(self, input_ids, attention_mask, **kwargs):
                x = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )["pooler_output"]
                if self.proj:
                    x = self.proj(x)
                    x = {"last_hidden_state": x}
                return x

        sentence_embedding_model = LaionClapTextWrapper(model, use_proj)
        tokenizer = model.tokenize
        sentence_frontend_output_dim = (
            512 if use_proj else 768  # proj dim else roberta-base dim
        )

    else:
        sentence_embedding_model = MODELS[sentence_config.model][0].from_pretrained(
            sentence_config.model
        )
        tokenizer = MODELS[sentence_config.model][1].from_pretrained(
            sentence_config.model
        )
        sentence_frontend_output_dim = MODELS[sentence_config.model][2]

    if sentence_config.get("finetune_n_layers", -1) > 0:
        # freeze all layers except the last n layers
        for param in sentence_embedding_model.parameters():
            param.requires_grad = False
        for param in list(sentence_embedding_model.encoder.layer.parameters())[
            -sentence_config.get("finetune_n_layers", -1) :
        ]:
            param.requires_grad = True
        # unfreeze the pooler layer
        if hasattr(sentence_embedding_model, "pooler"):
            for param in sentence_embedding_model.pooler.parameters():
                param.requires_grad = True

    return (
        sentence_embedding_model,
        tokenizer,
        sentence_frontend_output_dim,
    )


def get_sentence_head_model(
    sentence_config, shared_representation_size, sentence_output_size
) -> nn.Module:
    if sentence_config.adopt_n_layers == -1:
        assert sentence_output_size == shared_representation_size
        return nn.Identity()
    layer_sizes = [sentence_output_size]
    layer_sizes += [sentence_config.adopt_layer_size] * sentence_config.adopt_n_layers
    layer_sizes += [shared_representation_size]
    sentence_layers = []
    for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
        sentence_layers.append(torch.nn.Linear(i, o))
        sentence_layers.append(torch.nn.ReLU())

    sentence_layers.pop()
    return torch.nn.Sequential(*sentence_layers)


class CheckpointDownloadCache:
    def __init__(self, cache):
        self.cache = cache

    def __call__(self, ckpt_url):
        import wget

        model_tag = ckpt_url.split("/")[-1]

        cache_dir = (
            self.cache.cache_dir
            / "download-pretrained-models"
            / "laion_clap"
            / model_tag
        )
        if self.cache.enter(cache_dir):
            if rank == 0:
                log.info(f"Downloading laion_clap weight files from '{ckpt_url}' ...")
                if Path(ckpt_url).is_file():
                    # assume file, copy to cache dir
                    shutil.copyfile(ckpt_url, cache_dir)
                else:
                    # assume url, download to cache dir
                    cache_dir = Path(wget.download(ckpt_url, cache_dir.as_posix()))
                self.cache.signal_done(cache_dir)
            else:
                self.cache.wait_done(cache_dir)

        return cache_dir


class AudioRetrievalModel(BaseLightningModule):
    """
    LightningModule to train an audio retrieval model

    """

    def __init__(
        self,
        optim: DictConfig,
        metric_collection_train,
        metric_collection_val,
        audio: DictConfig,
        sentence: DictConfig,
        shared_representation_size: int,
        normalize: bool = True,
        loss: Optional[nn.Module] = None,
        cache: Optional[Cache] = None,
        multiple_captions_reduce: Optional[str] = "all",
        text_preprocessing: Optional[callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.optim_config = optim
        self.metric_collection_train = metric_collection_train
        self.metric_collection_val = metric_collection_val
        self.audio_config = audio
        self.sentence_config = sentence
        self.shared_representation_size = shared_representation_size
        self.normalize = normalize
        self.cache = cache
        self.multiple_captions_reduce = multiple_captions_reduce

        self.audio_frontend, audio_output_size = get_audio_frontend_model(
            self.audio_config, self.cache
        )
        self.audio_output_size = audio_output_size
        self.audio_head = get_audio_head_model(
            self.audio_config, self.shared_representation_size, audio_output_size
        )

        self.sentence_frontend, self.tokenizer, text_output_size = (
            get_sentence_frontend_model(self.sentence_config)
        )
        self.sentence_head = get_sentence_head_model(
            self.sentence_config, self.shared_representation_size, text_output_size
        )
        self.text_output_size = text_output_size

        self.init_loss(loss)
        self.is_distributed = is_distributed()

        if self.audio_config.frozen:
            log.info("Freezing parameters of audio frontend.")
            self.freeze(self.audio_frontend)
        if self.sentence_config.frozen:
            log.info("Freezing parameters of text frontend.")
            self.freeze(self.sentence_frontend)

        self.text_preprocessing = get_text_preprocessing_func(text_preprocessing)
        # for ONNX export
        self.register_buffer(
            "sample_rate",
            torch.tensor(
                self.audio_config.get("sample_rate", 32000), dtype=torch.int64
            ),
            persistent=False,
        )
        self.register_buffer(
            "eval_max_sec",
            torch.tensor(self.audio_config.get("eval_max_sec", 60), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "segment_length",
            torch.tensor(self.audio_config.get("segment_length", 5), dtype=torch.int64),
            persistent=False,
        )

    def init_loss(self, loss):
        self.loss = loss

    def grad_context_audio(self):
        if self.audio_config.frozen:
            cm = torch.no_grad()
        else:
            cm = contextlib.nullcontext()
        return cm

    def grad_context_sentence(self):
        if self.sentence_config.frozen:
            cm = torch.no_grad()
        else:
            cm = contextlib.nullcontext()
        return cm

    def build_optim_parameters(self):
        if (
            len(self.optim_config.get("wd_blacklist_modules", [])) > 0
            and len(self.optim_config.get("wd_blacklist_names", [])) > 0
        ):
            # create two parameter groups
            params_no_wd = set()
            params_wd = set()

            blacklist_modules = tuple(
                (
                    getattr(
                        importlib.import_module(".".join(m.split(".")[:-1])),
                        m.split(".")[-1],
                    )
                    for m in self.optim_config.get("wd_blacklist_modules")
                )
            )

            def exclude(n, p):
                return (
                    p.ndim < 2
                    or any(ele in n for ele in self.optim_config["wd_blacklist_names"])
                    or (
                        n.endswith("weight")
                        and isinstance(
                            get_module_by_name(self, ".".join(n.split(".")[:-1])),
                            blacklist_modules,
                        )
                    )
                    or (n.endswith("bias"))
                )

            for pn, p in self.named_parameters():
                if exclude(pn, p):
                    params_no_wd.add(pn)
                else:
                    params_wd.add(pn)

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
            log.info(
                f"Separating optimizer parameter group into weight decay (size {len(params_wd)}) \
                    and non weight decay (size {len(params_no_wd)}) groups."
            )

            return [
                {
                    "params": [param_dict[pn] for pn in sorted(list(params_wd))],
                    "weight_decay": self.optim_config.optimizer.keywords[
                        "weight_decay"
                    ],
                },
                {
                    "params": [param_dict[pn] for pn in sorted(list(params_no_wd))],
                    "weight_decay": 0.0,
                },
            ]
        else:
            return self.parameters()

    def configure_optimizers(self):
        optimizer = self.optim_config.optimizer(params=self.build_optim_parameters())

        scheduler = self._build_scheduler(
            self.optim_config.scheduler, optimizer=optimizer
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
            },
        }

    def _build_scheduler(self, scheduler_config, optimizer):
        # All scheduler_configs contain SequentialLR Scheduler-like kwargs
        # milestones: list[int]
        # schedulers: dict[partial_schedulers]
        # note that the dict must be ordered correctly
        schedulers = [
            partial_sch(optimizer)
            for partial_sch in scheduler_config.schedulers.values()
        ]
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=schedulers,
            milestones=scheduler_config["milestones"],
        )

    def freeze(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module):
        module.train()
        for param in module.parameters():
            param.requires_grad = True

    def do_nothing_closure(self):
        return

    def forward_audio_model(self, batch):
        # only compute audio features, if not pre-computed
        if "audio_features" in batch:
            return batch
        if "waveform" in batch:
            # precomputed mel specs, Laion-CLAP forward
            embeddings = self.audio_frontend(batch)
            batch["audio_features"] = embeddings
            return batch

        # freeze audio embedding if required
        if self.audio_config.frozen:
            self.audio_frontend = self.audio_frontend.eval()

        # embed audios
        with self.grad_context_audio():
            # embed the whole audio sequence
            # segment the audio into a sequnece of fixed length segments = segment_length
            segment_length = self.segment_length * self.sample_rate

            longest_audio = (self.sample_rate * batch["audio_length"].max()).to(
                torch.int64
            )
            if longest_audio >= (self.eval_max_sec * self.sample_rate):
                if self.current_epoch == 0:
                    print(
                        f"Warning: Cut the audio max length from {longest_audio} == {longest_audio / self.sample_rate} seconds to a max of {self.eval_max_sec} seconds"
                    )
                longest_audio = torch.tensor(
                    self.eval_max_sec * self.sample_rate,
                    dtype=torch.int64,
                    device=longest_audio.device,
                )

            if segment_length <= 0:
                # no chunking: eval_max_sec if audio is longer, otherwise longest audio in batch
                max_length = longest_audio
                n_segments = 1
            else:
                # chunking: compute number of chunks and length of all chunks combined
                n_segments = (
                    ((longest_audio * 10 / segment_length).round().to(torch.int64) / 10)
                    .ceil()
                    .to(torch.int64)
                )  # ignore re-sampling errors less than 5% of the audio length
                n_segments = torch.max(
                    n_segments,
                    torch.tensor(1, dtype=torch.int64, device=n_segments.device),
                )
                max_length = n_segments * segment_length

            # @TODO maybe this is not good for cudnn benchmarks, variable length tensors
            audio = batch["audio"][:, :max_length]
            # max_audio = torch.zeros(
            #     (audio.size(0), max_length),
            #     device=audio.device,
            #     dtype=audio.dtype,
            # )
            # max_audio[:, : audio.size(-1)] = audio
            # audio = max_audio
            # Differentiable alternative
            pad_len = max_length - audio.size(1)
            if pad_len > 0:
                audio = torch.nn.functional.pad(audio, (0, pad_len))

            # True when zero-padded
            padding_mask = torch.arange(
                audio.shape[1], device=audio.device, dtype=torch.int64
            ).expand(audio.shape) > batch["audio_length_sample"].unsqueeze(1)

            if n_segments > 1:
                # compute embeddings for each chunk, then average chunk embeddings
                split = torch.split(audio, segment_length, -1)
                S = len(split)
                B, L = split[0].shape
                split = torch.concatenate(split)  # (B*S, L)
                padding_split = torch.split(padding_mask, segment_length, -1)
                padding_split = torch.concatenate(padding_split)

                embedding_sequence = torch.stack(
                    torch.split(self.audio_frontend(split, padding_split), B)
                ).permute(1, 0, 2)  # (B*S, L) -> (S, B, L) -> (B, S, L)

                if self.audio_config.aggregate == "mean":
                    used_chunks = (
                        batch["audio_length"].to(dtype=torch.float32)
                        / self.segment_length
                    ).ceil()
                    chunk_mask = torch.arange(
                        S, device=audio.device, dtype=torch.int64
                    ).unsqueeze(0) >= used_chunks.unsqueeze(1)
                    masked_tensor = embedding_sequence.masked_fill(
                        chunk_mask.unsqueeze(2), 0
                    )
                    embeddings = masked_tensor.sum(1) / used_chunks.unsqueeze(1)
                    # TODO: half-precision dtype?
                else:
                    raise ValueError(f"Aggregate {self.audio_config.aggregate}")
            else:
                # True when zero-padded
                padding_mask = torch.arange(audio.shape[1], device=audio.device).expand(
                    audio.shape
                ) > batch["audio_length_sample"].unsqueeze(1)
                embeddings = self.audio_frontend(audio, padding_mask)

            batch["audio_features"] = embeddings

        return batch

    def forward_sentence_model(
        self,
        batch,
        is_tokenized=False,
        return_last_hidden_state=None,
        output_hidden_states=None,
        device=None,
        skip_new_token_normalization=False,
    ):
        """filles the sentence features using the text transformer

        Args:
            batch (dict): contains the keys:
                "captions" -> list[str] of captions
                "audio" -> Tensor of audios

        Raises:
            ValueError: _description_

        Returns:
            dict: the batch dict with the addition of the following keys:
                "input_ids", "attention_mask" from the tokenizer
                "sentence_features": from the text transformer
                "indices", "caption": the flattened input "captions" and their index in the original batch

        """
        if not is_tokenized:
            if skip_new_token_normalization:
                captions = []

                # Marc's stuff
                for i, b in enumerate(batch["captions"]):
                    if isinstance(b, str):
                        b = [b]
                    for caption in b:
                        # remove punctuations
                        remove_punctuation = str.maketrans("", "", string.punctuation)
                        captions.append(
                            " ".join(
                                [
                                    w.lower().translate(remove_punctuation)
                                    if "<" not in w and ">" not in w
                                    else w
                                    for w in caption.split()
                                ]
                            )
                        )
            else:
                captions = self.text_preprocessing(batch["captions"])  # type: ignore

            tokenized = self.tokenizer(
                captions,
                add_special_tokens=True,
                # padding=True,
                padding="max_length",
                truncation=True,  # truncate to longest in batch, otherwise to max_length
                return_tensors="pt",
                max_length=self.sentence_config.max_sentence_tokens,
            )
            device = device if device is not None else batch["audio"].device
            batch["input_ids"] = tokenized["input_ids"].to(device)
            batch["attention_mask"] = tokenized["attention_mask"].to(device)
            batch["caption"] = captions

        # freeze text embeddings if required
        if self.sentence_config.frozen:
            self.sentence_frontend = self.sentence_frontend.eval()

        # embed text
        with self.grad_context_sentence():
            sentence_out = self.sentence_frontend(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            # return hidden_states for the diffusion model
            if output_hidden_states:
                batch["hidden_states"] = sentence_out["hidden_states"]

            token_embeddings = sentence_out["last_hidden_state"]
            if return_last_hidden_state:
                # batch["last_hidden_state"] = token_embeddings[0]
                batch["last_hidden_state"] = token_embeddings
            if self.sentence_config.get("pool_type", "eos") == "eos":
                batch["sentence_features"] = token_embeddings[:, 0, :]
            elif self.sentence_config.pool_type == "default":
                batch["sentence_features"] = token_embeddings
            elif self.sentence_config.pool_type == "pooler":
                batch["sentence_features"] = sentence_out["pooler_output"]
            elif self.sentence_config.pool_type == "attention":
                input_mask_expanded = (
                    batch["attention_mask"]
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                batch["sentence_features"] = torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            else:
                raise NotImplementedError(
                    f"Text output pooling '{self.sentence_config.pool_type}' is not supported."
                )

        return batch

    def forward_sentence_transformers_model(
        self,
        batch,
    ):
        if "caption" not in batch:
            batch["caption"] = [
                c
                for cs in batch["captions"]
                for c in (cs if isinstance(cs, list) else [cs])
            ]
        device = batch["audio"].device
        sentence_features = self.sentence_frontend.encode_query(
            batch["caption"],
            prompt_name=self.sentence_config.get("prompt_name", None),
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device,
            normalize_embeddings=False,
        )
        batch["sentence_features"] = sentence_features
        return batch

    def forward(self, batch):
        if self.sentence_frontend.__class__.__name__ == "SentenceTransformer":
            # very hacky way to detect sentence-transformers model since we decided
            # not to import sentence-transformers at the top level as it may break for
            # older containers.
            batch = self.forward_sentence_transformers_model(batch)
        else:
            batch = self.forward_sentence_model(batch)
        batch = self.forward_audio_model(batch)
        audio_features = self.audio_head(batch["audio_features"])
        sentence_features = self.sentence_head(batch["sentence_features"])

        # in case of multiple captions per audio, repeate audio features
        if len(audio_features) != len(sentence_features):
            audio_features = audio_features[batch["indices"]]
            batch["hash_caption"] = [
                batch["hash_caption"][idx] for idx in batch["indices"]
            ]

        if self.normalize:
            audio_features = torch.nn.functional.normalize(audio_features, p=2, dim=1)
            sentence_features = torch.nn.functional.normalize(
                sentence_features, p=2, dim=1
            )

        return audio_features, sentence_features

    @torch.compiler.disable
    def create_targets(self, hash_captions, device):
        return torch.tensor(hash_captions, device=device, dtype=torch.int64)

    def training_step(self, batch, batch_idx):
        audio_features, sentence_features = self(batch)

        ids = self.create_targets(batch["hash_caption"], audio_features.device)
        loss = self.loss(audio_features, sentence_features, id_hash=ids)

        if self.is_distributed:
            audio_features = self.loss.all_gather(audio_features)
        if self.global_step == 0 and rank == 0:
            self.logger.log_hyperparams({"effective_batch_size": len(audio_features)})

        self.log(
            "train/loss",
            loss.item(),
            batch_size=len(audio_features),
            prog_bar=True,
        )

        if self.loss.logit_bias is not None:
            # siglip
            self.log("train/logit_bias", self.loss.logit_bias.item())
            self.log("train/tau", self.loss.tau.exp().item())
        else:
            if isinstance(self.loss.tau, (nn.ParameterList, Sequence)):
                self.log("train/tau-audio", torch.abs(self.loss.tau[0]).item())
                self.log("train/tau-text", torch.abs(self.loss.tau[1]).item())
            else:
                self.log("train/tau", torch.abs(self.loss.tau).item())

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_features, sentence_features = self(batch)
        ids = self.create_targets(batch["hash_caption"], audio_features.device)

        return {
            "audio_features": audio_features,
            "sentence_features": sentence_features,
            "caption": batch["captions"],
            "id": ids,
            "idx": batch_idx,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        log_dict = {**self.metric_collection_train.compute()}
        self.log_everything(phase="train", log_dict=log_dict, batch_size=1)
        self.metric_collection_train.reset()

    def on_validation_start(self):
        if not hasattr(self, "valid_is_initialized"):
            self.val_names = [name for name in self.trainer.datamodule.val_dataset()]
            if self.metric_collection_val is not None:
                metric_collection_val = self.metric_collection_val
                self.metric_collection_val = torch.nn.ModuleList(
                    [metric_collection_val.clone() for _ in self.val_names]
                )
            self.valid_is_initialized = True

        # if currently in a sanity validation loop, log at epoch -1
        if self.trainer._evaluation_loop.inference_mode:
            self.trainer.fit_loop.epoch_progress.current.completed = -1

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.metric_collection_val[dataloader_idx].update(outputs)

    def on_validation_epoch_end(self):
        for i in range(len(self.metric_collection_val)):
            log_dict = {
                **self.metric_collection_val[i].compute(self.loss, self.val_names[i])
            }
            self.log_everything(
                phase=self.val_names[i], log_dict=log_dict, batch_size=1
            )
            self.metric_collection_val[i].reset()

    def log_everything(self, phase="", log_dict={}, batch_size=None, sync_dist=True):
        """
        reco_loss: unreduced reconstruction loss
        """
        if phase != "" and not phase.endswith("/"):
            phase += "/"

        # dict
        if len(log_dict) > 0:
            # add phase in keys
            log_dict_metrics = {
                f"{phase}{key}": value
                for key, value in log_dict.items()
                if not isinstance(value, PIL.Image.Image)
            }
            # Log dict
            self.log_dict(
                log_dict_metrics,
                sync_dist=sync_dist,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            log_images = {
                f"{phase}{k}": v
                for k, v in log_dict.items()
                if isinstance(v, PIL.Image.Image)
            }
            for k, v in log_images.items():
                if not isinstance(v, list):
                    v = [v]
                k = k.replace("/", "_")
                self.logger.log_image(
                    key=f"plot/{k}",
                    images=v,
                    step=self.current_epoch,
                )

    def on_load_checkpoint(self, checkpoint):
        """Can change checkpoint kcheckpoint['state_dict'].eys to be loaded here.
        We remove the saved metrics if any exist in the checkpoint.
        """
        checkpoint = checkpoint["state_dict"]
        if "tau" in checkpoint.keys():
            checkpoint["loss.tau"] = checkpoint["tau"]
            del checkpoint["tau"]

        # don't load metrics if they exist
        for k in list(checkpoint.keys()):
            if "metric_collection_" in k:
                del checkpoint[k]
        # Put the current metrics
        for k, v in self.state_dict().items():
            if "metric_collection_" in k:
                checkpoint[k] = v

    def on_save_checkpoint(self, checkpoint):
        """Can change checkpoint keys to be saved here"""
        checkpoint = checkpoint["state_dict"]
        # don't save metrics if they exist
        for k in list(checkpoint.keys()):
            if "metric_collection_" in k:
                del checkpoint[k]

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.audio_head, norm_type=2)
        self.log_dict(norms)
