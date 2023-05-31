from data.mnist import MNIST
from model import softmax

if __name__ == "__main__":
    mnist = MNIST()
    N, D = mnist.X_train.shape
    M = 10

    # sampling
    W, b = softmax.sample_theta(D, M)

    # predition
    result = softmax.model(mnist.X_train, W, b)
    print(result.shape)
    print(result)