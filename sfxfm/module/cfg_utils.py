from scipy.stats import beta
class BetaFunc:
    def __init__(self, alpha: float=3, beta_param: float=3, x_min: float=0, x_max: float=1, normalized: bool=True):
        """
        Initializes beta pdf function. Alpha and beta are the parameters. x_min, x_max is the support. normalized is enabled if outputs should be between 0 and 1. 
        """
        self.alpha = alpha
        self.beta_param = beta_param
        self.x_min = x_min
        self.x_max = x_max
        self.normalized = normalized

    def compute(self, x: float) -> float:
        """
        Compute the Beta distribution function value at a given point x,
        for given alpha, beta, and custom interval [x_min, x_max].

        Parameters:
        x (float): The point at which to compute the Beta distribution.

        Returns:
        float: The Beta distribution value at point x. If normalized, the value is normalized such that min is 0 max is 1
        """
        if self.alpha < 2 or self.beta_param < 2:
            raise ValueError("Alpha and beta should be greater or equal than two")
        if x < self.x_min or x > self.x_max:
            return 0.

        x_transformed = (x - self.x_min) / (self.x_max - self.x_min)

        beta_value = beta.pdf(x_transformed, self.alpha, self.beta_param).item()
        max_y = 1
        if self.normalized:
            max_x = (self.alpha - 1) / (self.alpha + self.beta_param - 2)
            max_y = beta.pdf(max_x, self.alpha, self.beta_param).item()
        scale_factor = 1/max_y
        beta_value *= scale_factor

        return beta_value