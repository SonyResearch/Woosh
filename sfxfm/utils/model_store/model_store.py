import os
import sys
import shutil
from typing import Callable
from pathlib import Path
import logging
from tqdm import tqdm
import boto3
from boto3.s3.transfer import TransferConfig
import multiprocessing as mp

log = logging.getLogger()

# Threads are crashing for gaetan on gaia, these env variables control the number of threads or wether to use threads at all

S3_N_PROCs = int(os.environ.get("SFXFM_S3_N_PROCS", 20))
S3_N_THREADS = int(os.environ.get("SFXFM_S3_N_THREADS", 1))


def get_checkpoint_filename(
    ckpt_dir,
    ckpt_name="last",
) -> Path:
    """Return last checkpoint filename"""

    def find_files_in_dir(globex, directory):
        matching_files = list(Path(directory).glob(globex))
        if len(matching_files) > 0:
            return matching_files
        return []

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)

        if not ckpt_dir.exists():
            return None

        if str(ckpt_name) in ("last", "auto"):
            # find last.ckpt in ckpt_dir
            ckpt_last = find_files_in_dir("*last*.ckpt", ckpt_dir)
            if len(ckpt_last) > 0 and Path(ckpt_last[0]).resolve().exists():
                return ckpt_last[0]
            # no last found, use the latest .ckpt
            ckpts = sorted(find_files_in_dir(r"[0-9\-]*.ckpt", ckpt_dir))  # pylint: disable=W1401
            # get last one according to epoch-step sorting
            if len(ckpts) > 0:
                return ckpts[-1]
    return None


def size2str(size):
    for unit in ("b", "Kb", "Mb", "Gb", "Tb"):
        if size < 1024:
            break
        size /= 1024
    return f"{size:.2f}{unit}"


def get_checkpoint_dir(artifacts_dir):
    checkpoint_dir = Path(artifacts_dir) / "checkpoints"
    checkpoint_root_dir = checkpoint_dir
    if checkpoint_dir.is_dir():
        for e in checkpoint_dir.glob("*"):
            if e.is_dir():
                checkpoint_dir = checkpoint_dir / e
                break
    return checkpoint_dir, checkpoint_root_dir


class ModelStore:
    """
    Manages a sort of model zoo for our models, storing code, configs and artifacts.

    Handles local storage and S3 storage.
    """

    def __init__(
        self,
        location: str = "s3://sr-music-sfxfm",
        folder: str = "model-store",
        log_fn: Callable = print,
    ):
        if location is None or (location is not None and len(location) == 0):
            raise ValueError(f"ModelStore location cannot be empty: aborting")
        self.location = Path(location)
        if folder is None or (folder is not None and len(folder) == 0):
            raise ValueError(f"ModelStore folder cannot be empty: aborting")
        self.folder = folder
        self.log_fn = log_fn

    def get_s3_base_path(self):
        """Storage path where models are stored"""
        return self.location / self.folder

    def get_type(self):
        """Get type of storage"""

        if len(self.location.parts) > 0:
            if self.location.parts[0] == "s3:":
                return "s3"
            else:
                return "local"
        else:
            return None

    def get_model_names(self):
        """Get list of experiment names available in model storage"""

        base = self.get_s3_base_path()
        if self.get_type() == "s3":
            s3 = boto3.client("s3")
            bucket_name = base.parts[1]
            prefix = Path(*base.parts[2:])
            objects = s3.list_objects(
                Bucket=bucket_name, Prefix=str(prefix) + "/", Delimiter="/"
            )
            if objects.get("CommonPrefixes") is not None:
                names = [
                    str(Path(o["Prefix"]).name) for o in objects.get("CommonPrefixes")
                ]
                return names
            else:
                return None
        return None

    def download_s3(
        self,
        s3_url,
        download_dir,
        strip_key_levels=0,
        n_procs=S3_N_PROCs,
        n_threads=S3_N_THREADS,
        base_path=None,
    ):
        def download_file_safe(
            bucket_name,
            key,
            filename,
            extra_args,
            callback,
            config,
            sema=None,
        ):
            if sema is not None:
                sema.acquire()

            s3_client = boto3.client("s3")
            s3_client.download_file(
                Bucket=bucket_name,
                Key=key,
                Filename=filename,
                ExtraArgs=extra_args,
                Callback=callback,
                Config=config,
            )

            if sema is not None:
                sema.release()

        s3_client = boto3.client("s3")
        s3_config = TransferConfig(
            multipart_threshold=8 * 1024**2,
            max_concurrency=n_threads,
            multipart_chunksize=8 * 1024**2,
            use_threads=n_threads > 1,
        )

        bucket_name = s3_url.parts[1]
        prefix = Path(*s3_url.parts[2:])

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=str(prefix) + "/")

        max_file_processes = 20
        sema = mp.Semaphore(max_file_processes)
        processes = []

        for page in pages:
            for obj in page["Contents"]:
                file_url = obj["Key"]
                file_url_relative = Path("s3:") / bucket_name / Path(file_url)
                if base_path is not None:
                    file_url_relative = file_url_relative.relative_to(base_path)
                file_tgt = download_dir / Path(
                    *Path(file_url_relative).parts[strip_key_levels:]
                )

                if obj["Size"] == 0:
                    if not file_tgt.is_dir():
                        file_tgt.mkdir(parents=True, exist_ok=True)
                else:
                    if not file_tgt.parent.is_dir():
                        file_tgt.parent.mkdir(parents=True, exist_ok=True)

                    progress_bar = tqdm(
                        total=obj["Size"],
                        desc=f"Downloading {file_url_relative}",
                        leave=False,
                        file=sys.stdout,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        position=0,
                    )

                    p = mp.Process(
                        target=download_file_safe,
                        args=(
                            bucket_name,
                            file_url,
                            str(file_tgt),
                            None,
                            progress_bar.update,
                            s3_config,
                            sema,
                        ),
                    )
                    processes.append(p)

                    p.start()

                    # s3_client.download_file(
                    #     Bucket=bucket_name,
                    #     Key=file_url,
                    #     Filename=str(file_tgt),
                    #     Config=s3_config,
                    #     Callback=progress_bar.update,
                    # )

        for p in processes:
            p.join()

    def upload_s3(
        self,
        upload_dir,
        s3_url,
        strip_key_levels=0,
        base_path=None,
        n_procs=S3_N_PROCs,
        n_threads=S3_N_THREADS,
        exclude=None,
    ):
        def upload_file_safe(
            filename,
            bucket_name,
            key,
            extra_args,
            callback,
            config,
            sema=None,
        ):
            if sema is not None:
                sema.acquire()

            s3_client = boto3.client("s3")
            s3_client.upload_file(
                Filename=Path(filename).resolve(),
                Bucket=bucket_name,
                Key=key,
                ExtraArgs=extra_args,
                Callback=callback,
                Config=config,
            )

            if sema is not None:
                sema.release()

        s3_client = boto3.client("s3")
        s3_config = TransferConfig(
            multipart_threshold=8 * 1024**2,
            max_concurrency=n_threads,
            multipart_chunksize=8 * 1024**2,
            use_threads=n_threads > 1,
        )

        bucket_name = s3_url.parts[1]
        prefix = Path(*s3_url.parts[2:])

        if upload_dir.is_dir():
            files = list(upload_dir.rglob("*"))
            if exclude is not None:
                if isinstance(exclude, str):
                    exclude = list(exclude)
                for exc in exclude:
                    files = [f for f in files if exc not in str(f)]
        else:
            files = [upload_dir]

        max_file_processes = n_procs
        sema = mp.Semaphore(max_file_processes)
        processes = []

        for f in files:
            if f.is_dir():
                continue

            sz_f = f.stat().st_size
            if sz_f == 0:
                continue

            file_url_relative = f
            if base_path is not None:
                file_url_relative = file_url_relative.relative_to(base_path)

            file_url = prefix / Path(*Path(file_url_relative).parts[strip_key_levels:])

            progress_bar = tqdm(
                total=sz_f,
                desc=f"Uploading {file_url_relative}",
                leave=False,
                file=sys.stdout,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                position=0,
            )

            p = mp.Process(
                target=upload_file_safe,
                args=(
                    str(f),
                    bucket_name,
                    str(file_url),
                    None,
                    progress_bar.update,
                    s3_config,
                    sema,
                ),
            )
            processes.append(p)

            p.start()

        for p in processes:
            p.join()

    def delete_s3(self, s3_url, n_procs=S3_N_PROCs):
        def delete_file_safe(
            bucket_name,
            key,
            sema=None,
        ):
            if sema is not None:
                sema.acquire()

            s3_client = boto3.client("s3")
            s3_client.delete_object(
                Bucket=bucket_name,
                Key=key,
            )

            if sema is not None:
                sema.release()

        s3_client = boto3.client("s3")

        bucket_name = s3_url.parts[1]
        prefix = Path(*s3_url.parts[2:])

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=str(prefix) + "/")

        max_file_processes = n_procs
        sema = mp.Semaphore(max_file_processes)
        processes = []

        n_objs = sum([1 for page in pages for obj in page["Contents"]])

        for page in pages:
            for obj in page["Contents"]:
                p = mp.Process(
                    target=delete_file_safe,
                    args=(
                        bucket_name,
                        obj["Key"],
                        sema,
                    ),
                )

                processes.append(p)

                p.start()

        for p in processes:
            p.join()

    def download(
        self,
        model_names: str,
        download_dir: str,
        download_overwrite: bool = None,
        dry_run=False,
    ):
        if isinstance(model_names, str):
            model_names = [model_names]

        download_dir = Path(download_dir)
        downloaded = {}

        base = self.get_s3_base_path()

        if self.get_type() == "s3":
            available_model_names = set(self.get_model_names())

            download_dir.mkdir(parents=True, exist_ok=True)

            for model_name in model_names:
                if model_name not in available_model_names:
                    self.log_fn(
                        f"model {model_name} cannot be found in storage folder {base}: skipping, available models are {available_model_names}"
                    )
                    downloaded[model_name] = None
                    continue
                dest_dir = download_dir / model_name
                self.log_fn(f"downloading {base / model_name} to {dest_dir}")
                if os.path.isdir(dest_dir):
                    if download_overwrite is None:
                        key = input(
                            f"Destination directory {dest_dir} already exists. Do you want to overwrite it? [Yes/No/aBort]: "
                        )
                        if key.lower() == "n":
                            self.log_fn(f"   reusing {dest_dir}")
                            downloaded[model_name] = dest_dir
                            continue
                        elif key.lower() == "a":
                            sys.exit(-1)
                    elif download_overwrite:
                        # remove dest_exp
                        self.log_fn(f"   removing old {dest_dir}")
                        if not dry_run:
                            shutil.rmtree(dest_dir, ignore_errors=True)
                    else:
                        downloaded[model_name] = dest_dir
                        self.log_fn(f"   reusing {dest_dir}")
                        continue

                # actual multi-threaded download
                self.download_s3(
                    base / model_name,
                    dest_dir,
                    strip_key_levels=0,
                    base_path=self.get_base_path(base / model_name),
                )
                downloaded[model_name] = dest_dir

        else:
            # file system copy

            # target dir already exists. make sure user wants to overwrite
            for model_name in model_names:
                self.log_fn(f"{model_name}:")
                src_dir = base / model_name
                dest_dir = download_dir / model_name
                if os.path.isdir(dest_dir):
                    key = input(
                        f"Destination directory {dest_dir} already exists. Do you want to overwrite it? [Yes/No/aBort]: "
                    )
                    if key.lower() == "n":
                        downloaded[model_name] = None
                        return
                    elif key.lower() == "b":
                        sys.exit(-1)
                    # remove dest_exp
                    self.log_fn(f"   removing old {dest_dir}")
                    if not dry_run:
                        shutil.rmtree(dest_dir, ignore_errors=True)

                self.log_fn(f"   copying {src_dir} to {dest_dir}")
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(
                        src_dir, dest_dir, symlinks=True, dirs_exist_ok=True
                    )

                downloaded[model_name] = dest_dir

        if len(model_names) == 1:
            # requested for a single model
            if len(downloaded) == 1:
                return next(iter(downloaded.values()))
            else:
                return None
        else:
            return downloaded

    def get_base_path(self, f, key_dir="project_mfm_sfxfm"):
        """Get path up to project_mfm_sfxfm"""
        out = []
        for p in Path(f).parts:
            out.append(p)
            if key_dir == p:
                break
        return Path(*out)

    def upload_component(
        self,
        model_name,
        component_path,
        dry_run=False,
    ):
        assert component_path is not None
        # base directory, w/o model_name
        base = self.get_s3_base_path()
        # root directory for model_name
        dest_root = base / model_name
        self.log_fn(f"   uploading Component {model_name} to {dest_root}:")
        if self.get_type() == "s3":
            available_names = self.get_model_names()
            available_names = available_names or []
            key = None
            if model_name in available_names:
                key = input(
                    f"   Component {model_name} already exists. do you want to overwrite it? [Yes/No/aBort]: "
                )
                if key.lower() == "b":
                    sys.exit(-1)
                elif key.lower() != "y":
                    return None

            s3_bucket_root = Path(f"s3://{base.parts[1]}")
            s3 = boto3.client("s3")
            bucket_name = base.parts[1]
            model_prefix = Path(*dest_root.parts[2:])
            s3.put_object(Bucket=bucket_name, Body="", Key=str(model_prefix))
            s3_root_prefix = model_prefix

            if key is not None and key.lower() == "y":
                # delete model folder
                self.log_fn(
                    f"      deleting model directory {s3_bucket_root / model_prefix}"
                )
                self.delete_s3(s3_bucket_root / model_prefix)

            self.upload_s3(
                component_path,
                s3_bucket_root / s3_root_prefix,
                base_path=self.get_base_path(component_path, "components"),
                exclude=["__pycache__"],
                strip_key_levels=-1,
            )
        else:
            raise NotImplementedError("Local upload not implemented")

    def upload(
        self,
        model_name,
        artifacts_dir,
        repo_launch_root_dir=None,
        dry_run=False,
    ):
        # base directory, w/o model_name
        base = self.get_s3_base_path()
        # root directory for model_name
        dest_root = base / model_name
        local_code_configs = (
            Path(repo_launch_root_dir) / model_name
            if repo_launch_root_dir is not None
            else None
        )
        local_ckpts, local_ckpts_root = get_checkpoint_dir(artifacts_dir)

        self.log_fn(f"   uploading {model_name} to {dest_root}:")

        if self.get_type() == "s3":
            available_names = self.get_model_names()
            key = None
            if model_name in available_names:
                key = input(
                    f"   model {model_name} already exists. do you want to overwrite it? [Yes/No/aBort]: "
                )
                # @TODO this risky, invalid input -> overwrite
                if key.lower() == "n":
                    return None
                elif key.lower() == "b":
                    sys.exit(-1)

            s3_bucket_root = Path(f"s3://{base.parts[1]}")
            s3 = boto3.client("s3")
            bucket_name = base.parts[1]
            model_prefix = Path(*dest_root.parts[2:])
            s3.put_object(Bucket=bucket_name, Body="", Key=str(model_prefix))
            s3_root_prefix = model_prefix

            if key is not None and key.lower() == "y":
                # delete model folder
                self.log_fn(
                    f"      deleting model directory {s3_bucket_root / model_prefix}"
                )
                self.delete_s3(s3_bucket_root / model_prefix)

            # check to copy code and source configs
            if local_code_configs is not None:
                self.log_fn(
                    f"      uploading code and configs in {local_code_configs} to {s3_bucket_root / s3_root_prefix / 'code'}"
                )
                self.upload_s3(
                    local_code_configs,
                    s3_bucket_root / s3_root_prefix / "code",
                    base_path=self.get_base_path(local_code_configs),
                    exclude=["__pycache__"],
                )

            # check to copy checkpoints
            ckpt_filename = get_checkpoint_filename(local_ckpts)
            if ckpt_filename is not None:
                ckpt_prefix = s3_root_prefix / ckpt_filename.relative_to(artifacts_dir)
                if ckpt_filename is not None:
                    self.log_fn(
                        f"      uploading checkpoint {ckpt_filename} to {s3_bucket_root / ckpt_prefix}"
                    )
                    if not dry_run:
                        self.upload_s3(
                            ckpt_filename,
                            s3_bucket_root / model_prefix,
                            strip_key_levels=2,
                            base_path=self.get_base_path(ckpt_filename),
                        )
                else:
                    self.log_fn(f"      no checkpoints found")

            # check to copy processed configs
            for item in [
                "config_resolved.yaml",
                "config_unresolved.yaml",
                "faiss",
                "onnx",
                "calibration",
            ]:
                item_path = Path(artifacts_dir) / item
                if item_path.exists():
                    # copy to s3
                    s3_prefix = s3_root_prefix
                    self.log_fn(f"      uploading {item_path} to {s3_prefix}")
                    if not dry_run:
                        self.upload_s3(
                            item_path,
                            s3_bucket_root / s3_prefix,
                            strip_key_levels=2,
                            base_path=self.get_base_path(item_path),
                        )

        else:
            # file system copy

            # target dir already exists. make sure user wants to overwrite
            if os.path.isdir(dest_root):
                key = input(
                    f"Destination directory {dest_root} already exists. Do you want to overwrite it? [Yes/No/aBort]: "
                )
                if key.lower() == "n":
                    return
                elif key.lower() == "b":
                    sys.exit(-1)
                # remove dest_exp
                self.log_fn(f"      removing old {dest_root}")
                if not dry_run:
                    shutil.rmtree(dest_root, ignore_errors=True)

            if local_code_configs is not None:
                dest_code_configs = dest_root / "code"
                self.log_fn(
                    f"      copying code and configs in {local_code_configs} to {dest_code_configs}"
                )
            if not dry_run:
                dest_root.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    local_code_configs,
                    dest_code_configs,
                    symlinks=True,
                    dirs_exist_ok=True,
                )

            ckpt_filename = get_checkpoint_filename(local_ckpts)
            dest_ckpts = dest_root / ckpt_filename.relative_to(artifacts_dir).parent
            if ckpt_filename is not None:
                self.log_fn(f"      copying checkpoint {ckpt_filename} to {dest_ckpts}")
                if not dry_run:
                    shutil.rmtree(dest_ckpts, ignore_errors=True)
                    dest_ckpts.mkdir(parents=True, exist_ok=True)
                    shutil.copy(ckpt_filename, dest_ckpts)
            else:
                self.log_fn(f"      no checkpoints found")

            # check to copy processed configs
            for config in ["config_resolved.yaml", "config_unresolved.yaml"]:
                config_file = Path(artifacts_dir) / config
                if config_file.is_file():
                    # copy to s3
                    self.log_fn(
                        f"      copying config file {config_file} to {dest_root}"
                    )
                    if not dry_run:
                        shutil.copy(config_file, dest_root)
