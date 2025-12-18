import logging
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchaudio
from omegaconf import DictConfig
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaModel,
    RobertaTokenizer,
)

from sfxfm.module.model.retrieval.passt import create_passt_model
from sfxfm.utils import loading

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


def get_audio_frontend_model(audio_config) -> Tuple[nn.Module, int]:
    if audio_config.name.startswith("passt"):
        model, output_dim = create_passt_model(audio_config)
        return model, output_dim

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

    # Model, tokenizer, embedding_size
    MODELS = {
        "roberta-base": (RobertaModel, RobertaTokenizer, 768),
        "roberta-large": (RobertaModel, RobertaTokenizer, 1024),
    }
    extra_args = {}

    model_name = sentence_config.model
    if "roberta" in model_name:
        extra_args = {
            "add_pooling_layer": sentence_config.get("add_pooling_layer", False),
            "hidden_dropout_prob": sentence_config.get("hidden_dropout_prob", 0.2),
            "attention_probs_dropout_prob": sentence_config.get(
                "attention_probs_dropout_prob", 0.2
            ),
        }
    try:
        model_cls, tokenizer_cls, sentence_frontend_output_dim = MODELS[model_name]
        tokenizer = tokenizer_cls.from_pretrained(model_name)
        if loading.lazy_loading_enabled:
            config = AutoConfig.from_pretrained(model_name, **extra_args)
            sentence_embedding_model = AutoModel.from_config(config)
            tokenizer = tokenizer_cls.from_pretrained(model_name)
        else:

            sentence_embedding_model = model_cls.from_pretrained(model_name, **extra_args)
    except KeyError:
        sentence_embedding_model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sentence_frontend_output_dim = sentence_embedding_model.config.hidden_size

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


class AudioRetrievalModel(nn.Module):

    def __init__(
        self,
        audio: DictConfig,
        sentence: DictConfig,
        shared_representation_size: int,
        normalize: bool = True,
        text_preprocessing: Optional[callable] = None,
        **kwargs,
    ):
        super().__init__()

        self.audio_config = audio
        self.sentence_config = sentence
        self.shared_representation_size = shared_representation_size
        self.normalize = normalize

        self.audio_frontend, audio_output_size = get_audio_frontend_model(self.audio_config)
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

        self.text_preprocessing = get_text_preprocessing_func(text_preprocessing)
        # # for ONNX export
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
        self.resamplers_cache = {}

    def forward_audio_model(self, batch):
        # embed audios
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

            audio = batch["audio"][:, :max_length]

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
            captions = self.text_preprocessing(batch["captions"])

            tokenized = self.tokenizer(
                captions,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,  # truncate to longest in batch, otherwise to max_length
                return_tensors="pt",
                max_length=self.sentence_config.max_sentence_tokens,
            )
            device = device if device is not None else batch["audio"].device
            batch["input_ids"] = tokenized["input_ids"].to(device)
            batch["attention_mask"] = tokenized["attention_mask"].to(device)
            batch["caption"] = captions

        # embed text
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

    def _load_audio_from_file(self, audio_path: Union[str, Path]) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform[0]  # first channel only
        if sr != self.sample_rate:
            if self.resamplers_cache.get(sr) is None:
                self.resamplers_cache[sr] = torchaudio.transforms.Resample(
                    sr,
                    self.sample_rate,
                    resampling_method="sinc_interp_kaiser",
                )
            waveform = self.resamplers_cache[sr](waveform)
        return waveform

    def _get_audio_embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self.forward_audio_model(batch)
        audio_embeddings = self.audio_head(batch["audio_features"])
        if self.normalize:
            audio_embeddings = torch.nn.functional.normalize(
                audio_embeddings, p=2, dim=1
            )
        return audio_embeddings

    def _batch_audio(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        batch = torch.utils.data.default_collate(
            [
                {k: i[k].squeeze(0) for k in ["audio_length", "audio_length_sample"]}
                for i in inputs
            ]
        )
        batch["audio"] = torch.nn.utils.rnn.pad_sequence(
            [i["audio"].squeeze(0) for i in inputs], batch_first=True
        )

        return batch

    def get_audio_embedding_from_file(
        self,
        x: Union[List, str, Path],
        batched: bool = False,
        device: Union[str, torch.device] = "cpu",
        use_tensor: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, str) or isinstance(x, Path):
            x = [x]

        inputs = []
        for audio_path in x:
            audio = self._load_audio_from_file(audio_path)
            audio = audio.to(device)
            input_dict = {
                "audio_length": torch.tensor(
                    [len(audio) / float(self.sample_rate)], device=device
                ),
                "audio": audio.unsqueeze(0),
                "audio_length_sample": torch.tensor([audio.size(-1)], device=device),
            }
            inputs.append(input_dict)

        if batched:
            batch = self._batch_audio(inputs)
            inputs = [batch]

        audio_embeddings = []
        for b in inputs:
            emb = self._get_audio_embedding(b)
            audio_embeddings.append(emb)
        audio_embeddings = torch.concat(audio_embeddings, dim=0)

        if not use_tensor:
            audio_embeddings = audio_embeddings.detach().cpu().numpy()
        return audio_embeddings

    def get_audio_embedding_from_data(
        self,
        x: Union[List, torch.Tensor, np.ndarray],
        batched: bool = False,
        device: Union[str, torch.device] = "cpu",
        use_tensor: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:

        if (isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)) and x.ndim == 1:
            x = x[None, :]

        inputs = []
        for audio in x:
            audio = torch.tensor(audio, device=device, dtype=torch.float32)
            input_dict = {
                "audio_length": torch.tensor(
                    [len(audio) / float(self.sample_rate)], device=device
                ),
                "audio": audio.unsqueeze(0),
                "audio_length_sample": torch.tensor([audio.size(-1)], device=device),
            }
            inputs.append(input_dict)

        if batched:
            batch = self._batch_audio(inputs)
            inputs = [batch]

        audio_embeddings = []
        for b in inputs:
            emb = self._get_audio_embedding(b)
            audio_embeddings.append(emb)
        audio_embeddings = torch.concat(audio_embeddings, dim=0)

        if not use_tensor:
            audio_embeddings = audio_embeddings.detach().cpu().numpy()
        return audio_embeddings

    def get_text_embedding(
        self,
        x: Union[List, str, Path],
        device: Union[str, torch.device] = "cpu",
        use_tensor: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, str) or isinstance(x, Path):
            x = [x]

        batch = {"captions": x}
        batch = self.forward_sentence_model(batch, device=device)
        text_embeddings = self.sentence_head(batch["sentence_features"])
        if self.normalize:
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)

        if not use_tensor:
            text_embeddings = text_embeddings.detach().cpu().numpy()

        return text_embeddings

    def forward(
        self,
        x: Dict,
        batched: bool = False,
        device: Union[str, torch.device] = "cpu",
        use_tensor: bool = False,
    ):
        result_dict = {}
        if "audio" in x.keys():
            audio = x["audio"]
            if (
                isinstance(audio, str)
                or isinstance(audio, Path)
                or (isinstance(audio, list) and isinstance(audio[0], str))
                or (isinstance(audio, list) and isinstance(audio[0], Path))
            ):
                result_dict["audio"] = self.get_audio_embedding_from_file(
                    audio, batched=batched, device=device, use_tensor=use_tensor
                )
            else:
                result_dict["audio"] = self.get_audio_embedding_from_data(
                    audio, batched=batched, device=device, use_tensor=use_tensor
                )
        if "text" in x.keys():
            text = x["text"]
            result_dict["text"] = self.get_text_embedding(
                text, device=device, use_tensor=use_tensor
            )

        return result_dict
