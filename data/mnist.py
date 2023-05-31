import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def normalize(x: np.array) -> np.array:
    max_val = np.max(x)
    return x / max_val

class MNIST:
    """
    N: number of data\n
    D: dimension of input data\n
    """
    def __init__(self):
        self.dataset = load_digits()
        self.X = normalize(self.dataset.data)  # [1797(N), 64(D)]
        self.Y = self.dataset.target  # [1797(N)]

        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(self.X, self.Y, test_size=0.33, random_state=42)
    
    def get_nclasses(self):
        return 10
