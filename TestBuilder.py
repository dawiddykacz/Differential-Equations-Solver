from tasks.ai.secondDegree.SecondDegreeImport import *


def get_is_calculating_wang_grads_by_weight(weight):
    if weight == 1:
        return [False]
    return [True, False]


def test_1(task_repository):
    d_min = 0.0001
    range_weight = 10 ** 4
    alpha = 0.1

    task_repository.add_task(ExampleFirst2ProblemLossTask(weight_pde=d_min,
                                                          weight_conditions=1))
    task_repository.add_task(ExampleSimpleFirst2ProblemLossTask())
    task_repository.add_task(ExampleSecond2ProblemLossTask(weight_pde=d_min,
                                                           weight_conditions=1))
    task_repository.add_task(ExampleSimpleSecond2ProblemLossTask())
    for is_calculating_wang_grads in [False, True]:
        task_repository.add_task(
            ExampleFirst2ProblemLossWithWeightTask(alpha=alpha,
                                                   weight_pde=d_min,
                                                   weight_conditions=1,
                                                   is_calculating_wang_grads=is_calculating_wang_grads))
        task_repository.add_task(
            ExampleSecond2ProblemLossWithWeightTask(alpha=alpha,
                                                    weight_pde=d_min,
                                                    weight_conditions=1,
                                                    is_calculating_wang_grads=is_calculating_wang_grads))
    for weight in [1, range_weight]:
        task_repository.add_task(ExampleFirst2ProblemLossTask(
            weight_pde=1,
            weight_conditions=weight))
        for is_calculating_wang_grads in get_is_calculating_wang_grads_by_weight(weight):
            task_repository.add_task(
                ExampleFirst2ProblemLossWithWeightTask(alpha=alpha,
                                                       weight_pde=1,
                                                       weight_conditions=weight,
                                                       is_calculating_wang_grads=
                                                       is_calculating_wang_grads))
        task_repository.add_task(ExampleSecond2ProblemLossTask(
            weight_pde=1,
            weight_conditions=weight))
        for is_calculating_wang_grads in get_is_calculating_wang_grads_by_weight(weight):
            task_repository.add_task(
                ExampleSecond2ProblemLossWithWeightTask(alpha=alpha,
                                                        weight_pde=1,
                                                        weight_conditions=weight,
                                                        is_calculating_wang_grads=
                                                        is_calculating_wang_grads))


def test_2(task_repository):
    d_min = 0.0001
    range_weight = 10 ** 4
    alpha = 0.1

    for with_noise in [False, True]:
        task_repository.add_task(ExampleFirst2ProblemLossTaskWithTrainableVariable(weight_data=1,
                                                                                   weight_pde=d_min,
                                                                                   weight_conditions=1,
                                                                                   with_noise=with_noise))
        task_repository.add_task(ExampleSimpleFirst2ProblemLossTaskWithTrainableVariable(weight_pde=d_min,
                                                                                         weight_data=1,
                                                                                         with_noise=with_noise))
        task_repository.add_task(ExampleSecond2ProblemLossTaskWithTrainableVariable(weight_data=1,
                                                                                    weight_pde=d_min,
                                                                                    weight_conditions=1,
                                                                                    with_noise=with_noise))
        task_repository.add_task(ExampleSimpleSecond2ProblemLossTaskWithTrainableVariable(weight_pde=d_min,
                                                                                          weight_data=1,
                                                                                          with_noise=with_noise))
        for is_calculating_wang_grads in [False, True]:
            task_repository.add_task(
                ExampleFirst2ProblemLossWithWeightTaskWithTrainableVariable(alpha=alpha,
                                                                            weight_pde=d_min,
                                                                            weight_conditions=1,
                                                                            weight_data=1, with_noise=with_noise,
                                                                            is_calculating_wang_grads=
                                                                            is_calculating_wang_grads))
            task_repository.add_task(
                ExampleSimpleFirst2ProblemLossWithWeightTaskWithTrainableVariable(alpha=alpha,
                                                                                  weight_pde=d_min,
                                                                                  weight_data=1,
                                                                                  with_noise=with_noise,
                                                                                  is_calculating_wang_grads=
                                                                                  is_calculating_wang_grads))
            task_repository.add_task(
                ExampleSecond2ProblemLossWithWeightTaskWithTrainableVariable(alpha=alpha,
                                                                             weight_pde=d_min,
                                                                             weight_conditions=1,
                                                                             weight_data=1, with_noise=with_noise,
                                                                             is_calculating_wang_grads=
                                                                             is_calculating_wang_grads))
            task_repository.add_task(
                ExampleSimpleSecond2ProblemLossWithWeightTaskWithTrainableVariable(alpha=alpha,
                                                                                   weight_pde=d_min,
                                                                                   weight_data=1,
                                                                                   with_noise=with_noise,
                                                                                   is_calculating_wang_grads=
                                                                                   is_calculating_wang_grads))
        for weight in [1, range_weight]:
            task_repository.add_task(ExampleFirst2ProblemLossTaskWithTrainableVariable(weight_data=weight,
                                                                                       weight_pde=1,
                                                                                       weight_conditions=weight,
                                                                                       with_noise=with_noise))
            task_repository.add_task(ExampleSimpleFirst2ProblemLossTaskWithTrainableVariable(weight_pde=1,
                                                                                             weight_data=weight,
                                                                                             with_noise=with_noise))
            task_repository.add_task(ExampleSecond2ProblemLossTaskWithTrainableVariable(weight_data=weight,
                                                                                        weight_pde=1,
                                                                                        weight_conditions=weight,
                                                                                        with_noise=with_noise))
            task_repository.add_task(ExampleSimpleSecond2ProblemLossTaskWithTrainableVariable(weight_pde=1,
                                                                                              weight_data=weight,
                                                                                              with_noise=with_noise))
            for is_calculating_wang_grads in get_is_calculating_wang_grads_by_weight(weight):
                task_repository.add_task(
                    ExampleFirst2ProblemLossWithWeightTaskWithTrainableVariable(alpha=alpha,
                                                                                weight_pde=1,
                                                                                weight_conditions=weight,
                                                                                weight_data=weight,
                                                                                with_noise=with_noise,
                                                                                is_calculating_wang_grads=
                                                                                is_calculating_wang_grads))
                task_repository.add_task(
                    ExampleSimpleFirst2ProblemLossWithWeightTaskWithTrainableVariable(alpha=alpha,
                                                                                      weight_pde=1,
                                                                                      weight_data=weight,
                                                                                      with_noise=with_noise,
                                                                                      is_calculating_wang_grads=
                                                                                      is_calculating_wang_grads))
                task_repository.add_task(
                    ExampleSecond2ProblemLossWithWeightTaskWithTrainableVariable(alpha=alpha,
                                                                                 weight_pde=1,
                                                                                 weight_conditions=weight,
                                                                                 weight_data=weight,
                                                                                 with_noise=with_noise,
                                                                                 is_calculating_wang_grads=
                                                                                 is_calculating_wang_grads))
                task_repository.add_task(
                    ExampleSimpleSecond2ProblemLossWithWeightTaskWithTrainableVariable(alpha=alpha,
                                                                                       weight_pde=1,
                                                                                       weight_data=weight,
                                                                                       with_noise=with_noise,
                                                                                       is_calculating_wang_grads=
                                                                                       is_calculating_wang_grads))
