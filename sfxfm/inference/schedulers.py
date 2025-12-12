import math
import torch


def round_sigma(value: float):
    return torch.as_tensor(value)


def get_t_steps(noise_scheduler, device, **kwargs):
    if noise_scheduler is None:
        return None, kwargs
    args_scheduler = {
        "num_steps": kwargs["num_steps"],
        "sigma_min": kwargs["sigma_min"],
        "sigma_max": kwargs["sigma_max"],
        "device": device,
        "add_zero": kwargs.pop("add_zero", True),
    }

    if noise_scheduler == "karras":
        args_scheduler["rho"] = kwargs.pop("rho", 1.0)
        t_steps = karras_scheduler(**args_scheduler)
    elif noise_scheduler == "linear":
        args_scheduler.pop("rho", None)
        args_scheduler.pop("add_zero", None)
        t_steps = linear_scheduler(**args_scheduler)
    elif noise_scheduler == "sigmoid":
        args_scheduler["rho"] = kwargs.pop("rho", 1.0)
        t_steps = sigmoid_scheduler(**args_scheduler)
    elif noise_scheduler == "linear":
        t_steps = linear_scheduler(**args_scheduler)
    elif noise_scheduler == "cosine":
        t_steps = cosine_scheduler(**args_scheduler)
    else:
        raise ValueError(f"Scheduler {noise_scheduler} not found!")

    return t_steps


def linear_scheduler(
    num_steps: int, sigma_min: float, sigma_max: float, device
) -> torch.Tensor:
    t_steps = torch.linspace(
        sigma_max, sigma_min, num_steps, dtype=torch.float64, device=device
    )
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    return t_steps


def karras_scheduler(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device,
    add_zero: float = True,
) -> torch.Tensor:
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    if add_zero:
        t_steps = torch.cat([round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    return t_steps


def sigmoid_scheduler(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device,
    add_zero: float = True,
) -> torch.Tensor:
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = sigma_min + (sigma_max - sigma_min) / (
        1 + torch.exp(-rho * (step_indices - num_steps / 2))
    )
    t_steps = t_steps.flip(0)
    if add_zero:
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    return t_steps


def cosine_scheduler(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    device,
    add_zero: bool = True,
) -> torch.Tensor:
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    cos_values = (torch.cos((step_indices / (num_steps - 1)) * math.pi) + 1) / 2
    t_steps = sigma_min + (sigma_max - sigma_min) * cos_values
    if add_zero:
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    return t_steps
