import numpy as np

from models.linear import LinearRegression


def test_linear_regression():
    w_orig, b_orig = 2.6, -0.4

    X = np.random.rand(100) * 10. - 5.
    y_orig = w_orig * X + b_orig

    model = LinearRegression().fit(X, y_orig)

    assert np.isclose(model.w.data, w_orig)
    assert np.isclose(model.b.data, b_orig)
