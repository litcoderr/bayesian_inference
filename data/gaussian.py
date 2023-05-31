import numpy as np
from sklearn.model_selection import train_test_split


class GaussianDataset:
    def __init__(self, N, M, D=2):
        """
        N: number of data point
        M: number of classes
        D: input space dimension
        """
        self.M = M

        # sample mu of each class
        self.mu = np.random.uniform(low=0, high=1, size=(M, D))  # [M, D]

        self.Y = np.random.randint(low=0, high=M, size=(N)) # [N]
        self.X = np.stack([np.random.normal(self.mu[c], 0.07) for c in self.Y], axis=0) # [N, D]

        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(self.X, self.Y, test_size=0.33, random_state=42)
    
    def get_nclasses(self):
        return self.M
