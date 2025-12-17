"""
Module to train CLAP
"""

import logging
import shutil
import string
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

rank = 0

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
