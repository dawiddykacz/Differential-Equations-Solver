import tensorflow


def calculate_wang(mean, max, eps, alpha, old_value):
    w = tensorflow.cond(mean + eps != 0,
                        lambda: max / (mean + eps),
                        lambda: old_value)
    return (1 - alpha) * old_value + alpha * w
