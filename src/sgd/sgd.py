import sys
sys.path.append("..")

import numpy as np
from src.utils.parameter_estimator import ParameterEstimator
from src.utils.stepsize_strategy import FixedStepsize, DiminishingStepsize, HalvingStepsize

class SGD:
    def __init__(self, model, num_epochs=10, batch_size=1, noise=0.01,
                 stepsize_type='fixed', use_empirical_noise=False):
        self.model = model
        self.X = np.ascontiguousarray(model.X, dtype=np.float64)
        self.y = np.ascontiguousarray(model.y, dtype=np.float64)
        self.n = self.X.shape[0]
        self.num_epochs = num_epochs
        self.batch_size = int(batch_size)
        self.noise = float(noise)
        self.F_star = model.F(model.w_star)

        # --- explicit knob instead of dataset_name ---
        estimator = ParameterEstimator(self.X, self.y, model, noise=self.noise)
        params = estimator.estimate_parameters(use_empirical_noise=use_empirical_noise)
        self.params = params

        if stepsize_type == 'fixed':
            self.strategy = FixedStepsize(params)
        elif stepsize_type == 'diminishing':
            self.strategy = DiminishingStepsize(params)
        else:
            self.strategy = HalvingStepsize(params, F_star=self.F_star)
        self.stepsize_type = stepsize_type

    def optimize(self):
        w = self.model.initialize_weights()

        obj_history = [self.model.F(w)]
        grad_norm_history = [np.linalg.norm(self.model.grad_F(w))**2]
        dist_to_opt_history = [np.linalg.norm(w - self.model.w_star)**2]

        iteration = 0
        steps_per_epoch = int(np.ceil(self.n / self.batch_size))  

        for _ in range(self.num_epochs):
            for _ in range(steps_per_epoch):
                if self.batch_size == 1:
                    i = int(np.random.randint(0, self.n))
                    X_batch = self.X[i:i+1]
                    y_batch = self.y[i:i+1]
                else:
                    batch_indices = np.random.randint(0, self.n, size=self.batch_size)
                    X_batch = self.X[batch_indices]
                    y_batch = self.y[batch_indices]

                if self.stepsize_type == 'halving':
                    self.strategy.update(self.model.F(w), iteration)
                alpha_k = self.strategy.get(iteration)

                
                g_k = self.model.mini_batch_grad(w, X_batch, y_batch)
                w -= alpha_k * g_k

                if iteration % 100 == 0 or iteration == 0:
                    obj_history.append(self.model.F(w))
                    grad_norm_history.append(np.linalg.norm(self.model.grad_F(w))**2)
                    dist_to_opt_history.append(np.linalg.norm(w - self.model.w_star)**2)

                iteration += 1

        return w, np.array(obj_history), np.array(grad_norm_history), np.array(dist_to_opt_history)
