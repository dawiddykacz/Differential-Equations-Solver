import tensorflow

zero_constant = tensorflow.constant(0.0, dtype=tensorflow.float64)


def calculate_new_wang(mean, max, eps=zero_constant, old_value=zero_constant):
    return tensorflow.cond(not tensorflow.equal(mean + eps, zero_constant),
                           lambda: max / (mean + eps),
                           lambda: old_value)


def calculate_wang(mean, max, eps, alpha, old_value):
    w = calculate_new_wang(mean, max, eps, old_value)
    return (1 - alpha) * old_value + alpha * w
