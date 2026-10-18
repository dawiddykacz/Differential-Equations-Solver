import tensorflow

from objects.functions.loss.LossFunction import LossFunction


def parse_layer_path_to_name(layer_path):
    for part in layer_path.split('/'):
        if part.startswith("hidden_") or part.startswith("out") or part.startswith("dense_wang"):
            return part
    return layer_path


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

        grad_pde = [g if g is not None else tensorflow.zeros_like(k) for g, k in zip(grad_pde, kernels)]
        grad_bc = [g if g is not None else tensorflow.zeros_like(k) for g, k in zip(grad_bc, kernels)]
        grad_data = [g if g is not None else tensorflow.zeros_like(k) for g, k in zip(grad_data, kernels)]

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

            layer_name = parse_layer_path_to_name(kernels[i].path)
            save_data['grad_pde_max'][layer_name] = pde_max
            save_data['grad_pde_mean'][layer_name] = pde_mean
            save_data['grad_bc_max'][layer_name] = bc_max
            save_data['grad_bc_mean'][layer_name] = bc_mean
            save_data['grad_data_max'][layer_name] = data_max
            save_data['grad_data_mean'][layer_name] = data_mean

        max_pde = tensorflow.reduce_max(tensorflow.stack([tensorflow.reduce_max(tensorflow.abs(g)) for g in grad_pde]))
        max_bc = tensorflow.reduce_max(tensorflow.stack([tensorflow.reduce_max(tensorflow.abs(g)) for g in grad_bc]))
        max_data = tensorflow.reduce_max(
            tensorflow.stack([tensorflow.reduce_max(tensorflow.abs(g)) for g in grad_data]))

        mean_pde = tensorflow.reduce_mean(
            tensorflow.stack([tensorflow.reduce_mean(tensorflow.abs(g)) for g in grad_pde]))
        mean_bc = tensorflow.reduce_mean(tensorflow.stack([tensorflow.reduce_mean(tensorflow.abs(g)) for g in grad_bc]))
        mean_data = tensorflow.reduce_mean(
            tensorflow.stack([tensorflow.reduce_mean(tensorflow.abs(g)) for g in grad_data]))

        d = {
            'grad_pde_max': max_pde,
            'grad_bc_max': max_bc,
            'grad_data_max': max_data,
            'grad_pde_mean': mean_pde,
            'grad_data_mean': mean_data,
            'grad_bc_mean': mean_bc
        }
        for key in save_data.keys():
            for key2 in save_data[key].keys():
                d[f'{key}-{key2}'] = save_data[key][key2]
        return d
