import unittest
import numpy as np
from tools.opt_parent import build


class Parameter:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)


class TestSGDOptimiser(unittest.TestCase):
    def test_sgd_step_updates_params(self):
        params = [Parameter([1.0, 2.0]), Parameter([0.5])]
        opt = build("SGD", params, lr=0.1, weight_decay=0.0)

        # Fake some grads
        params[0].grad = np.array([0.2, -0.1])  # dL/dp
        params[1].grad = np.array([0.05])

        # Expected update: p = p - lr * grad
        expected0 = np.array([1.0, 2.0]) - 0.1 * np.array([0.2, -0.1])
        expected1 = np.array([0.5]) - 0.1 * np.array([0.05])

        # Apply step
        opt.step()

        # Assert close (float safety)
        np.testing.assert_allclose(params[0].data, expected0, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(params[1].data, expected1, rtol=1e-7, atol=1e-7)

    def test_zero_grad_resets_gradients(self):
        params = [Parameter([1.0, 2.0])]
        opt = build("SGD", params, lr=0.1)

        params[0].grad = np.array([0.3, -0.2])
        opt.zero_grad()

        np.testing.assert_array_equal(params[0].grad, np.array([0.0, 0.0]))


if __name__ == '__main__':
    unittest.main()
