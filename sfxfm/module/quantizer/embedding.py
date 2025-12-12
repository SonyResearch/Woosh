import logging
import os
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus
try:
    from fast_pytorch_kmeans import KMeans as FastKMeans
    import faiss
    # import faiss.contrib.torch_utils 
    # TODO: Disabled for ComfyUI SFX-CLAP demo, as it permanently replaces the index-search function
except:
    pass

log = logging.getLogger(__name__)

available_init_strategies = {
    "scikit": {"++": "k-means++", "uniform_rnd": "random"},
    "fast": {"++": "kmeans++", "uniform_rnd": "random", "gaussian_rnd": "gaussian"},
}
available_custom_strategies = ["++"]
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class Embedding:

    @property
    def codes(self):
        if not hasattr(self, "_codes") or self._codes is None:
            raise ValueError("Model has not been fitted yet.")
        return self._codes

    def fit(self, x, n_clusters):
        """This function must be implemented"""
        raise NotImplementedError

    def reset(self):
        self._codes = None


class KmeansEmbedding(Embedding):
    def __init__(
        self,
        init_strategy: str = "++",
        custom_init: bool = False,
        max_iter: int = 100,
        batch_size: int = 2048,  # type: ignore
        random_state: int = None,  # type: ignore
        **kwargs,
    ):

        self.batch_size = batch_size
        self.init_strategy = init_strategy
        self.custom_init = custom_init

        self.max_iter = max_iter
        self.random_state = random_state
        self.init_centers = None

    def _initialize_centroids(self, x, n_clusters):

        assert self.init_strategy in available_custom_strategies
        if self.init_strategy == "++":
            x_array = x.cpu().numpy()
            self.init_centers = kmeans_plusplus(
                x_array,
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_local_trials=None,  # default is 2 + log(n_clusters) = 12 local trials
            )[0]

    def fit(self, x, n_clusters):
        if self.custom_init:
            self._initialize_centroids(x, n_clusters)
            assert self.init_centers is not None
        self._compute_codes(x, n_clusters)

    def _compute_codes(self, x, n_clusters):
        "This function must be implemented"
        raise NotImplementedError


class ScikitKmeansEmbedding(KmeansEmbedding):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html

    def __init__(
            self,
            n_init="auto",
            init_size: int=None, #type: ignore
            max_no_improvement: int=10,
            reassignment_ratio: float=0.01,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.n_init = n_init
        self.init_size = init_size
        self.max_no_improvement = max_no_improvement
        self.reassignment_ratio = reassignment_ratio

    def _compute_codes(self, x, n_clusters):

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init=(
                self.init_centers
                if self.custom_init
                else available_init_strategies["scikit"][self.init_strategy]
            ),
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            max_no_improvement=self.max_no_improvement,
            reassignment_ratio=self.reassignment_ratio,
            init_size=self.init_size,
            n_init=self.n_init,
            random_state=self.random_state,
        ).fit(x.cpu())

        log.info(f"Number of iterations : {kmeans.n_iter_}")
        log.info(f"Number of mini batches processed: {kmeans.n_steps_}")
        log.info(f"Inertia: {kmeans.inertia_}")

        self._codes = kmeans.cluster_centers_


class FastKmeansEmbedding(KmeansEmbedding):
    # https://github.com/DeMoriarty/fast_pytorch_kmeans

    def __init__(
        self,
        tol: float = 0.0001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tol = tol

    def _compute_codes(self, x, n_clusters):
        kmeans = FastKMeans(
            n_clusters=n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            init_method=available_init_strategies["fast"][self.init_strategy],
            minibatch=self.batch_size,
            mode="euclidean",
        )
        kmeans.fit(x, self.init_centers)
        self._codes = kmeans.centroids  # type: ignore

class FaissKmeansEmbedding(KmeansEmbedding):
    # https://github.com/facebookresearch/faiss/blob/924c24db23b00053fc1c49e67d8787f0a3460ceb/faiss/python/extra_wrappers.py#L443

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.custom_init = True

    def _compute_codes(self, x, n_clusters):
        # https://github.com/facebookresearch/faiss/blob/main/benchs/kmeans_mnist.py

        d = x.size(1)
        k = n_clusters

        gpu = True if x.is_cuda else False
        log.info(f"Faiss using gpu resources: {gpu}")
        kmeans = faiss.Kmeans(
            d=d,  # input dimension
            k=k,
            gpu=True if x.is_cuda else False,
            niter=self.max_iter,
            nredo=1,
            seed=self.random_state,
            frozen_centroids=False,
            max_points_per_centroid=100000000,  #  otherwise the kmeans implementation sub-samples the training set
            min_points_per_centroid=1,
        )

        kmeans.train(x.cpu().numpy().astype("float32"), init_centroids=self.init_centers)
        self._codes = kmeans.centroids.reshape(k, d)
