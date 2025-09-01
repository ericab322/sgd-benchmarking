import numpy as np

class RegressionModel:
    def __init__(self, X, y):
        """
        Linear regression model with squared error loss.

        Args:
            X: Input data of shape (m, d).
            y: Target values of shape (m,) or (m, 1).
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(np.ravel(y), dtype=np.float64)
        m = X.shape[0]
        Xb = np.hstack([np.ones((m, 1), dtype=np.float64), X])
        
        self.X = np.ascontiguousarray(Xb, dtype=np.float64)
        self.y = y
        self.m, self.n = self.X.shape

        self.w_star = np.linalg.pinv(self.X) @ self.y
        
        self.A = (self.X.T @ self.X) / self.m       
        self.b = (self.X.T @ self.y) / self.m           
        self.c = 0.5 * (self.y @ self.y) / self.m


    def initialize_weights(self):
        """
        Initializes weights to zero.

        Returns:
            Initial weight vector of shape (n,).
        """
        return np.zeros(self.n)

    def f_i(self, w, i):
        """
        Loss on a single sample.

        Args:
            w: Weight vector.
            i: Index of sample.

        Returns:
            Squared error loss for sample i.
        """
        r = self.X[i] @ w - self.y[i]
        return 0.5 * r ** 2

    def grad_f_i(self, w, i):
        """
        Gradient of loss at sample i.

        Args:
            w: Weight vector.
            i: Index of sample.

        Returns:
            Gradient vector of shape (n,).
        """
        x = self.X[i]
        return (x @ w - self.y[i]) * x

    def F(self, w):
        """
        Full objective (average loss over all samples).

        Args:
            w: Weight vector.

        Returns:
            Scalar average loss.
        """
        return 0.5 * (w @ (self.A @ w)) - (self.b @ w) + self.c

    def grad_F(self, w):
        """
        Gradient of the full objective.

        Args:
            w: Weight vector.

        Returns:
            Gradient vector of shape (n,).
        """
        return self.A @ w - self.b

    def stochastic_grad(self, w, i):
        """
        Stochastic gradient for a single sample.

        Args:
            w: Weight vector.
            X_sample: Sample input of shape (1, n).
            y_sample: Sample target of shape (1,).

        Returns:
            Gradient vector of shape (n,).
        """
        x = self.X[i]
        y = self.y[i]
        return (x @ w - y) * x

    def mini_batch_grad(self, w, X_batch, y_batch):
        """
        Mini-batch gradient over a batch.

        Args:
            w: Weight vector.
            batch_size: Number of samples.
            X_batch: Input batch of shape (batch_size, n).
            y_batch: Target batch of shape (batch_size,).

        Returns:
            Average gradient over the batch.
        """
        b = X_batch.shape[0] 
        err = X_batch @ w - y_batch
        return (X_batch.T @ err) / b

    def dist_to_opt(self, w):
        """
        Distance to the optimal solution.

        Args:
            w: Weight vector.

        Returns:
            Euclidean distance to w_star.
        """
        return np.linalg.norm(w - self.w_star)
