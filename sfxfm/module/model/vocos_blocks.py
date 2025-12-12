from torch import nn
import torch


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


# class EMANormalization(nn.Module):
#     def __init__(self, dim, alpha=0.99) -> None:
#         super().__init__()
#         # self.gamma = nn.Parameter(torch.zeros(dim), requires_grad=False)
#         # self.beta = nn.Parameter(torch.zeros(dim), requires_grad=False)
#         self.register_buffer("beta", torch.zeros(dim))
#         self.register_buffer("gamma", torch.zeros(dim))
#         self.first_batch = True
#         self.alpha = alpha
#         self.epsilon = 1e-8

#     def forward(self, x):
#         size_to_reduce = [i for i in range(len(x.size()[:-1]))]
#         x_mean = x.mean(size_to_reduce).detach()
#         x_std = x.std(size_to_reduce).detach()
#         self._update_ema(x_mean, x_std)

#         x_mean = x.mean(-1).unsqueeze(-1)
#         x_std = x.std(-1).unsqueeze(-1)
#         return (x - x_mean) / (x_std + self.epsilon) * (
#             x_std.detach() + self.epsilon
#         ) / torch.exp(self.gamma) + (x_mean.detach() - self.beta) / (
#             x_std + self.epsilon
#         )

#     def _update_ema(self, x_mean, x_std):
#         if self.first_batch:
#             self.beta = x_mean
#             self.gamma = x_std.add(self.epsilon).log()
#             self.first_batch = False
#         else:
#             self.beta = self.beta * self.alpha + (1 - self.alpha) * x_mean
#             self.gamma = (
#                 self.gamma * self.alpha
#                 + (1 - self.alpha) * x_std.add(self.epsilon).log()
#             )


class EMANormalization(nn.Module):
    def __init__(self, dim, alpha=0.99) -> None:
        super().__init__()
        # self.gamma = nn.Parameter(torch.zeros(dim), requires_grad=False)
        # self.beta = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.register_buffer("beta", torch.zeros(1, dim, 1))
        self.register_buffer("gamma", torch.zeros(1, dim, 1))
        self.register_buffer("first_batch", torch.BoolTensor([True]))

        self.beta_train = nn.Parameter(torch.zeros(1, dim, 1), requires_grad=True)
        self.gamma_train = nn.Parameter(torch.zeros(1, dim, 1), requires_grad=True)

        self.alpha = alpha
        self.epsilon = 1e-8

    def forward(self, x):
        size_to_reduce = (0, 2)
        x_mean = x.mean(size_to_reduce, keepdim=True).detach()
        x_std = x.std(size_to_reduce, keepdim=True).detach()
        self._update_ema(x_mean, x_std)

        return (x - self.beta) * torch.exp(
            self.gamma_train - self.gamma
        ) + self.beta_train
        # x_mean = x.mean(size_to_reduce, keepdim=True)
        # x_std = x.std(size_to_reduce, keepdim=True)
        # return (x - x_mean) / (x_std + self.epsilon) * (
        #     x_std.detach() + self.epsilon
        # ) / torch.exp(self.gamma) + (x_mean.detach() - self.beta) / (
        #     x_std + self.epsilon
        # )

    def _update_ema(self, x_mean, x_std):
        if self.first_batch.item():
            self.beta = x_mean
            self.gamma = x_std.add(self.epsilon).log()
            self.first_batch[0] = False
        else:
            self.beta = self.beta * self.alpha + (1 - self.alpha) * x_mean
            self.gamma = (
                self.gamma * self.alpha
                + (1 - self.alpha) * x_std.add(self.epsilon).log()
            )


class ContinuousAdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Linear(embedding_dim, embedding_dim)
        self.shift = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        scale = self.scale(modulation)
        shift = self.shift(modulation)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x


class IdentityAdaLayerNorm(nn.Module):
    """
    This is the identity in the first argument
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        return x


class STFTEmbedding(nn.Module):
    def __init__(
        self, n_fft, hop_length, repr_type: str = "complex", stft_normalized=False
    ) -> None:
        super().__init__()
        assert repr_type in ["complex", "magnitude", "gain-shape"]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft_normalized = stft_normalized
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # compute correct number of features out
        self.num_features_out = n_fft // 2 + 1
        self.repr_type = repr_type
        if repr_type == "complex":
            self.num_features_out *= 2
        elif repr_type == "gain-shape":
            self.num_features_out *= 3

        # tested group_norm but was detrimental
        # self.group_norm = nn.GroupNorm(self.num_features_out, self.num_features_out)
        # self.group_norm = EMANormalization(self.num_features_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch_size, channels=1, length) -> (batch_size, num_features_out, length // downsampling_factor)
        """
        batch_size, channels, length = x.size()
        assert channels == 1

        x = x[:, 0]
        y = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
            normalized=self.stft_normalized,
        )

        if self.repr_type == "magnitude":
            # magnitude:
            y = safe_log(y.real**2 + y.imag**2)
        elif self.repr_type == "complex":
            #  real and imag
            y = torch.cat([y.real, y.imag], dim=1)
        elif self.repr_type == "gain-shape":
            #  real and imag
            mag = torch.sqrt(y.real**2 + y.imag**2 + 1e-8)
            y = torch.cat([y.real / mag, y.imag / mag, torch.log(mag)], dim=1)
        else:
            raise NotImplementedError

        # y = self.group_norm(y)

        return y
