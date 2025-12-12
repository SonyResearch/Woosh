import torch
import torchmetrics
from pathlib import Path
import numpy
from sfxfm.utils.dist import rank
import os
import pickle

import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)
rank = rank()


def compute_perplexity(counts):
    probabilities = counts.float() / counts.sum()
    probabilities = probabilities[probabilities != 0]
    entropy = -torch.sum(probabilities * torch.log(probabilities))
    perplexity = torch.exp(entropy)
    return perplexity


class BitrateEfficiency(torchmetrics.Metric):
    def __init__(
        self,
        save_dir,
        plot_figure=False,
    ):
        super().__init__()

        self.codebook_size: int = None  # type: ignore
        self.num_quantizers: int = None  # type: ignore
        self.save_dir = save_dir
        self.plot_figure = plot_figure
        self.add_state(
            "indices",
            default=[],
            dist_reduce_fx=None,
            persistent=False,
        )

    def update(self, preds, batch):
        idx = batch["indices"]  # (b, t, num_quantizers)
        if len(self.indices) == 0:
            self.codebook_size = batch["codebook_size"]
            self.num_quantizers = len(idx)
            self.experiment_name = batch["model_name"]
            self.indices = [
                torch.empty(0, device=preds.device, dtype=torch.int)
                for i in range(self.num_quantizers)
            ]

        for i in range(self.num_quantizers):
            self.indices[i] = torch.concatenate([self.indices[i], idx[i].flatten()])

    def compute(self):
        cb_util = 0
        for i, idx in enumerate(self.indices):
            mask = (idx >= 0) & (idx <= self.codebook_size)
            # Apply the mask to filter the values
            idx = idx[mask]
            counts = torch.zeros(self.codebook_size, device=idx.device).scatter_add_(
                0, idx, torch.ones_like(idx).float()
            )
            cb_util += (counts != 0).sum()

            if rank == 0 and self.plot_figure:
                save_dir = Path(self.save_dir) / self.experiment_name
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.close()

                sorted_counts = numpy.sort(counts.cpu().numpy()) / counts.sum().item()
                plt.hist(
                    list(range(self.codebook_size)),
                    bins=self.codebook_size,
                    weights=sorted_counts,
                )

                plt.xlabel("Codes", fontsize=12)
                plt.ylabel("Probability Density", fontsize=12)
                plt.yticks(rotation=45)
                plt.savefig(str(save_dir) + str(f"codebook_{i}"), dpi=300)
                log.info(f"Codebook utilization {i} saved in {str(save_dir)}")

        return cb_util / (self.codebook_size * self.num_quantizers)


class Perplexity(torchmetrics.Metric):
    def __init__(
        self,
        save_dir,
        plot_figure=False,
    ):
        super().__init__()

        self.save_dir = save_dir
        self.codebook_size: int = None  # type: ignore
        self.num_quantizers: int = None  # type: ignore
        self.plot_figure = plot_figure
        self.add_state(
            "indices",
            default=[],
            dist_reduce_fx=None,
            persistent=False,
        )

    def update(self, preds, batch):
        idx = batch["indices"]  # (b, t, num_quantizers)
        if len(self.indices) == 0:
            self.codebook_size = batch["codebook_size"]
            self.num_quantizers = len(idx)
            self.experiment_name = batch["model_name"]
            self.indices = [
                torch.empty(0, device=preds.device, dtype=torch.int)
                for i in range(self.num_quantizers)
            ]

        for i in range(self.num_quantizers):
            self.indices[i] = torch.concatenate([self.indices[i], idx[i].flatten()])

    def compute(self):
        perplexities = []
        for i, idx in enumerate(self.indices):
            mask = (idx >= 0) & (idx <= self.codebook_size)
            # Apply the mask to filter the values
            idx = idx[mask]
            counts = torch.zeros(self.codebook_size, device=idx.device).scatter_add_(
                0, idx, torch.ones_like(idx).float()
            )
            perplexity = compute_perplexity(counts)
            perplexities.append(perplexity)

        if rank == 0 and self.plot_figure:
            save_dir = Path(self.save_dir)  # / self.experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)

            figure_file = os.path.join(save_dir, ".pkl")
            if os.path.exists(figure_file):
                print("Loading existing figure...")
                with open(figure_file, "rb") as f:
                    fig = pickle.load(f)
                ax = fig.gca()
            else:
                print("Creating a new figure...")
                fig, ax = plt.subplots(figsize=(8, 6))

            plt.close()

            ax.grid(color="black", linestyle="--", linewidth=0.5)
            ax.set_facecolor("white")

            # Plot the data
            ax.plot(
                list(range(1, self.num_quantizers + 1)),
                torch.stack(perplexities).cpu().numpy(),
                marker="o",
                markersize=6,
                label=self.experiment_name,
            )

            # Customize labels and ticks
            ax.set_xlabel("Codebooks", fontsize=22)
            ax.set_xticks(list(range(1, 20, 3)))
            ax.set_ylabel("Perplexity", fontsize=22)
            ax.tick_params(axis="y", labelrotation=45, labelsize=20)
            ax.tick_params(axis="x", labelsize=20)
            ax.yaxis.set_visible(True)
            ax.grid(alpha=0.3)  # Reduce opacity of the grid
            ax.legend(fontsize=16, loc="lower right")

            fig.savefig(os.path.join(save_dir, "perplexity.png"))
            with open(figure_file, "wb") as f:
                pickle.dump(fig, f)
            plt.close(fig)
            # plt.savefig(str(save_dir) + str(f"perplexity.png"), dpi=300)
            log.info(f"Perplexity saved in {str(save_dir)}")

        return torch.stack(perplexities).mean()


class LatentMSE(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "mse", default=torch.tensor(0.0), dist_reduce_fx="sum", persistent=False
        )
        self.add_state(
            "total", default=torch.tensor(0), dist_reduce_fx="sum", persistent=False
        )

    def update(self, preds, batch):
        mse = batch["mse"]
        self.mse += mse.sum()
        self.total += mse.numel()

    def compute(self):
        return self.mse / self.total
