# from .heun import heun
# from .cfgpp import cfgpp


def get_sampler(sampler_name="heun"):
    if sampler_name == "heun":
        return heun
    elif sampler_name == "cfgpp":
        return cfgpp
    else:
        raise NotImplementedError
