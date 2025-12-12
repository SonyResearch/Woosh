from omegaconf import ListConfig
import re
import torch
from functools import partial

class LoRALayer(torch.nn.Module):

    def __init__(self, in_dim, out_dim, rank, alpha, dropout=0., init_type="zero_a"):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        if init_type=="zero_b":
            self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
            self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        elif init_type=="zero_a":
            self.A = torch.nn.Parameter(torch.zeros(in_dim, rank))
            self.B = torch.nn.Parameter(torch.randn(rank, out_dim) * std_dev)
        self.alpha = alpha
        if dropout > 0.:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.A_factored, self.B_factored = None, None

    def forward(self, x, rank=None):
        if self.dropout is not None:
            x = self.dropout(x)

        if rank is None:
            x = self.alpha * (x @ self.A @ self.B)
        else:
            if self.is_factorized():
                A = self.A_factored[:, :rank]
                B = self.B_factored[:rank, :]
                x = self.alpha * (x @ A @ B)
            else:
                raise ValueError(f"LoRA A and B matrices were not factorized: call factorize() before passing a rank to forward()")

        return x

    def factorize(self, mode="svd"):
        if mode == "svd":
            u, s, v = torch.svd(self.A @ self.B)
            self.A_factored = torch.matmul(u, torch.diag_embed(torch.sqrt(s)))
            self.B_factored = torch.matmul(torch.diag_embed(torch.sqrt(s)), v.mT)
        else:
            self.A_factored = self.A.clone().detach()
            self.B_factored = self.B.clone().detach()

    def is_factorized(self):
        return self.A_factored is not None and self.B_factored is not None



class LinearLoRA(torch.nn.Module):
    """ add LoRA layer to existing Linear layer """

    def __init__(self, linear, alpha, rank=None, strength=None, min_rank=None, max_rank=None, dropout=0., requires_grad=True, init_type="zero_a"):
        super().__init__()
        assert ( rank is not None and strength is None ) or ( rank is None and strength is not None )
        self.linear = linear
        self.linear.requires_grad_(False)
        max_rank = min(self.linear.in_features, self.linear.out_features)
        if rank is not None:
            self.rank = min(rank, max_rank)
            # self.rank = rank
        elif strength is not None:
            assert strength>=0.0 and strength<=1.0
            self.rank = max ( int(max_rank * (1.0 - strength) ), 1 )

        if min_rank is not None:
            self.rank = max (self.rank, min_rank)
        if max_rank is not None:
            self.rank = min (self.rank, max_rank)

        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank=self.rank, alpha=alpha, dropout=dropout, init_type=init_type,
        )
        if requires_grad:
            self.lora.requires_grad_(True)
        else:
            self.lora.requires_grad_(False)

        # forward lora layers by default
        self.forward_lora = True
        # inference rank same as training by default
        self.forward_rank = self.rank

    def forward(self, x):
        if self.forward_lora:
            if self.forward_rank != self.rank:
                if self.lora.is_factorized():
                    return self.linear(x) + self.lora(x, rank=self.forward_rank)
                else:
                    raise ValueError(f"LoRA A and B matrices were not factorized: call factorize() before setting forward_rank")
            else:
                return self.linear(x) + self.lora(x)
        else:
            return self.linear(x)

    def factorize(self, mode="svd"):
        self.lora.factorize(mode)

    def set_forward_rank(self, rank=None):
        if rank is None:
            self.forward_rank = self.rank
        else:
            self.forward_rank = rank if rank<=self.rank and rank>0 else self.rank



class LoRA(torch.nn.Module):
    """ Wrapper class transforming the input model into a LoRA model """

    def __new__(cls, model, rank=None, strength=None, min_rank=None, max_rank=None, alpha=1.0, dropout=0., include=None,  exclude=None, init_type="zero_a"):
        """ 
        Convert Linear layers to Linear+LoRA layers. Only LoRA layer weights will have requires_grad=True

        - include is a list of layer keywords where LoRa is allowed to be applied
        - exclude is a list of layer keywords where LoRa should not be applied

        e.g.

           include:
             - diffusion
           exclude:
             - to_kqv

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

        # set all base parameters to not trainable  
        for param in model.parameters():
            param.requires_grad = False

        print (f"LoRA: Linear -> LinearLoRA, setting requires_grad for A and B LoRA matrices")
        for name, module in model.named_modules():
            include_ = any( re.search( s, name) is not None for s in include ) or len(include)==0
            exclude_ = any( re.search( s, name) is not None for s in exclude )
            if include_ and not exclude_:
                if isinstance(module, torch.nn.Linear):
                    var_name = re.sub(r"\.([0-9]+)", r"[\1]", f"model.{name}")
                    expr = f"{var_name} = LinearLoRA({var_name}, rank={rank}, strength={strength}, min_rank={min_rank}, max_rank={max_rank}, alpha={alpha}, dropout={dropout}, init_type='{init_type}', requires_grad=True)"
                    exec(expr)
                    used_rank = eval(f"{var_name}.rank")
                    print (f"  {name}, rank {used_rank}")

        return model

def is_lora(model):

    for name, module in model.named_modules():
        if isinstance(module, LinearLoRA):
            return True
    return False


def set_requires_grad(model, include=None, exclude=None):
    """ 
    Sets requires_grad for module names matching or not matching the given layer names
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

    for param in model.parameters():
        param.requires_grad = False

    print (f"setting trainable parameters:")
    for name, module in model.named_modules():
        include_ = any( re.search( s, name) is not None for s in include ) or len(include)==0
        exclude_ = any( re.search( s, name) is not None for s in exclude )
        if include_ and not exclude_:
            print (f"  {name}")
            module.requires_grad_(True)

    return model

def factorize_lora(model, include=None, exclude=None, mode="svd"):
    """ 
    Prepare LoRA layers to use an arbitrary rank. This may imply factorizing them using SVD or other.
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

    for name, module in model.named_modules():
        include_ = any( re.search( s, name) is not None for s in include ) or len(include)==0
        exclude_ = any( re.search( s, name) is not None for s in exclude )
        if include_ and not exclude_ and isinstance(module, LinearLoRA):
            module.factorize(mode)

    return model

def set_forward_lora(model, forward_lora, include=None, exclude=None, rank=None):
    """ 
    Sets forward_lora flag on LinearLoRA layers for module names matching or not matching the given layer names
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

    for name, module in model.named_modules():
        include_ = any( re.search( s, name) is not None for s in include ) or len(include)==0
        exclude_ = any( re.search( s, name) is not None for s in exclude )
        if include_ and not exclude_ and isinstance(module, LinearLoRA):
            module.forward_lora = forward_lora
            if rank is not None:
                module.set_forward_rank(rank)

    return model

def is_key_in_pattern(key, include=None, exclude=None, rank=None):
    """
    True if key is included in the list of include/exclude regex patterns
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

    include_ = any( re.search( s, key) is not None for s in include ) or len(include)==0
    exclude_ = any( re.search( s, key) is not None for s in exclude )

    if include_ and not exclude_:
        return True
    else:
        return False


def get_ranks_lora(model):
    """ 
    Sets forward_lora flag on LinearLoRA layers for module names matching or not matching the given layer names
    """
    
    if not is_lora(model):
        return None

    ranks = {}
    for name, module in model.named_modules():
        if isinstance(module, LinearLoRA):
            ranks[name] = module.rank

    return ranks


def get_max_rank_lora(model):
    """ 
    Get maximum loira rank across the whole model
    """
    ranks =  get_ranks_lora(model)
    return max(ranks.values()) if ranks is not None else None

def get_min_rank_lora(model):
    """ 
    Get maximum loira rank across the whole model
    """
    ranks =  get_ranks_lora(model)
    return min(ranks.values()) if ranks is not None else None
