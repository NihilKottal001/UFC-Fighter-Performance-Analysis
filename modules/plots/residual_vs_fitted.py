from typing import Any
import matplotlib.pyplot as plt


class ResidualVSFitted:
    def __init__(
            self,
            y: Any,
            y_hat: Any,
    ) -> None:
        self.y = y
        self.y_hat = y_hat

    def plot(self):
        residuals = self.y - self.y_hat
        plt.scatter(self.y_hat, residuals, color='blue', alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')
        plt.show()
