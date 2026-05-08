import numpy as np
from scipy.optimize import minimize


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def nll(self, temperature, logits, labels):
        scaled = logits / temperature

        probs = 1 / (1 + np.exp(-scaled))

        eps = 1e-8

        loss = -np.mean(
            labels * np.log(probs + eps)
            + (1 - labels) * np.log(1 - probs + eps)
        )

        return loss

    def fit(self, logits, labels):
        result = minimize(
            lambda t: self.nll(t, logits, labels),
            x0=[1.0],
            bounds=[(0.05, 10.0)]
        )

        self.temperature = result.x[0]

    def transform(self, logits):
        return logits / self.temperature