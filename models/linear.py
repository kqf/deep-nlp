import torch


class LinearRegression:

    def __init__(self, alpha=0.01, n_ephoch=1000):
        self.n_ephoch = n_ephoch
        self.alpha = alpha

    def fit(self, X, y):
        X = torch.as_tensor(X).float()
        y = torch.as_tensor(y).float()

        self.w = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

        for i in range(self.n_ephoch):
            loss = ((y - self.predict(X)) ** 2).mean()
            loss.backward()

            self.w.data -= self.alpha * self.w.grad
            self.b.data -= self.alpha * self.b.grad

            self.w.grad.zero_()
            self.b.grad.zero_()

        return self

    def predict(self, X):
        return X * self.w + self.b
