import tensorflow

from equations.ai.secondDegree.examples.WangHelper import calculate_new_wang
from objects.functions.loss.LossFunction import LossFunction


@tensorflow.function
def calculate_grads(current_loss, loss_pde, loss_conditions, loss_conditions_data, grads_data, grads_pde, grads_bc):
    grad_pde_max_bc = tensorflow.constant(0, dtype=tensorflow.float64)
    grad_pde_max_data = tensorflow.constant(0, dtype=tensorflow.float64)
    grad_bc_max = tensorflow.constant(0, dtype=tensorflow.float64)
    grad_data_max = tensorflow.constant(0, dtype=tensorflow.float64)
    grad_pde_mean_bc = tensorflow.constant(0, dtype=tensorflow.float64)
    grad_pde_mean_data = tensorflow.constant(0, dtype=tensorflow.float64)
    grad_data_mean = tensorflow.constant(0, dtype=tensorflow.float64)
    grad_bc_mean = tensorflow.constant(0, dtype=tensorflow.float64)

    max_wang_bc = tensorflow.constant(0, dtype=tensorflow.float64)
    max_wang_data = tensorflow.constant(0, dtype=tensorflow.float64)

    for g_data, g_pde, g_bc in zip(grads_data, grads_pde, grads_bc):
        if g_pde is None:
            continue

        grad_pde_list = [tensorflow.reshape(g_pde, [-1])]
        mean_pde = LossFunction.mean_abs_grads(grad_pde_list)
        max_pde = LossFunction.max_abs_grads(grad_pde_list)

        if g_bc is not None:
            grad_bc_list = [tensorflow.reshape(g_bc, [-1])]
            mean_bc = LossFunction.mean_abs_grads(grad_bc_list)
            max_bc = LossFunction.max_abs_grads(grad_bc_list)

            wang_bc = calculate_new_wang(
                max=max_pde,
                mean=mean_bc
            )

            if wang_bc > max_wang_bc:
                max_wang_bc = wang_bc
                grad_pde_max_bc = max_pde
                grad_pde_mean_bc = mean_pde
                grad_bc_max = max_bc
                grad_bc_mean = mean_bc

        if g_data is not None:
            grad_data_list = [tensorflow.reshape(g_data, [-1])]
            mean_data = LossFunction.mean_abs_grads(grad_data_list)
            max_data = LossFunction.max_abs_grads(grad_data_list)

            wang_data = calculate_new_wang(
                max=max_pde,
                mean=mean_data
            )

            if wang_data > max_wang_data:
                max_wang_data = wang_data
                grad_pde_max_data = max_pde
                grad_pde_mean_data = mean_pde
                grad_data_max = max_data
                grad_data_mean = mean_data

    return {
        'loss': current_loss,
        'loss_pde': loss_pde,
        'loss_conditions': loss_conditions,
        'loss_conditions_data': loss_conditions_data,
        'grad_pde_max_bc': grad_pde_max_bc,
        'grad_pde_max_data': grad_pde_max_data,
        'grad_bc_max': grad_bc_max,
        'grad_data_max': grad_data_max,
        'grad_pde_mean_bc': grad_pde_mean_bc,
        'grad_pde_mean_data': grad_pde_mean_data,
        'grad_data_mean': grad_data_mean,
        'grad_bc_mean': grad_bc_mean,
    }
