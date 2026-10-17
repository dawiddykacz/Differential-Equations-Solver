import tensorflow

from objects.functions.loss.LossFunction import LossFunction


class WangWeightCalculator:

    @staticmethod
    def calculate_new_weight(alpha, old_weight, max_grad, mean_grad):
        return tensorflow.cond(
            tensorflow.equal(mean_grad, 0.0),
            lambda: old_weight,
            lambda: (1.0 - alpha) * old_weight + alpha * (max_grad / mean_grad)
        )

    def calculate(self, loss, kernels, tape):
        loss_pde = tensorflow.convert_to_tensor(loss['loss_pde'], dtype=tensorflow.float64)
        conditions = tensorflow.convert_to_tensor(loss['conditions'], dtype=tensorflow.float64)
        conditions_data = tensorflow.convert_to_tensor(loss['conditions_data'], dtype=tensorflow.float64)

        grad_data = tape.gradient(conditions_data, kernels)
        grad_pde = tape.gradient(loss_pde, kernels)
        grad_bc = tape.gradient(conditions, kernels)

        pde_max_list, bc_max_list, data_max_list = [], [], []
        pde_mean_list, bc_mean_list, data_mean_list = [], [], []
        wang_bc_list, wang_data_list = [], []

        save_data = {
            "grad_pde_max": dict(),
            "grad_pde_mean": dict(),
            "grad_bc_max": dict(),
            "grad_bc_mean": dict(),
            "grad_data_max": dict(),
            "grad_data_mean": dict(),
        }
        for i in range(len(kernels)):
            pde_max = LossFunction.max_abs_grads(grad_pde[i])
            bc_max = LossFunction.max_abs_grads(grad_bc[i])
            data_max = LossFunction.max_abs_grads(grad_data[i])

            pde_mean = LossFunction.mean_abs_grads(grad_pde[i])
            bc_mean = LossFunction.mean_abs_grads(grad_bc[i])
            data_mean = LossFunction.mean_abs_grads(grad_data[i])

            pde_max_list.append(pde_max)
            bc_max_list.append(bc_max)
            data_max_list.append(data_max)
            pde_mean_list.append(pde_mean)
            bc_mean_list.append(bc_mean)
            data_mean_list.append(data_mean)

            wang_bc = tensorflow.math.divide_no_nan(pde_max, bc_mean)
            wang_data = tensorflow.math.divide_no_nan(pde_max, data_mean)

            wang_bc_list.append(wang_bc)
            wang_data_list.append(wang_data)
            save_data['grad_pde_max'][kernels[i].path] = pde_max
            save_data['grad_pde_mean'][kernels[i].path] = pde_mean
            save_data['grad_bc_max'][kernels[i].path] = bc_max
            save_data['grad_bc_mean'][kernels[i].path] = bc_mean
            save_data['grad_data_max'][kernels[i].path] = data_max
            save_data['grad_data_mean'][kernels[i].path] = data_mean

        t_pde_max = tensorflow.stack(pde_max_list)
        t_bc_max = tensorflow.stack(bc_max_list)
        t_data_max = tensorflow.stack(data_max_list)
        t_pde_mean = tensorflow.stack(pde_mean_list)
        t_bc_mean = tensorflow.stack(bc_mean_list)
        t_data_mean = tensorflow.stack(data_mean_list)

        t_wang_bc = tensorflow.stack(wang_bc_list)
        t_wang_data = tensorflow.stack(wang_data_list)

        max_index_bc = tensorflow.argmax(t_wang_bc)
        max_index_data = tensorflow.argmax(t_wang_data)

        return {
            'grad_pde_max_bc': t_pde_max[max_index_bc],
            'grad_pde_max_data': t_pde_max[max_index_data],
            'grad_bc_max': t_bc_max[max_index_bc],
            'grad_data_max': t_data_max[max_index_data],
            'grad_pde_mean_bc': t_pde_mean[max_index_bc],
            'grad_pde_mean_data': t_pde_mean[max_index_data],
            'grad_data_mean': t_data_mean[max_index_data],
            'grad_bc_mean': t_bc_mean[max_index_bc],
            'data': save_data
        }
