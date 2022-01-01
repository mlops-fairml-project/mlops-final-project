from typing import Tuple
import numpy as np
from sklearn.metrics import get_scorer
from fairlearn.metrics import MetricFrame, selection_rate, count


def get_metrics(CF, A, y_true, y_pred,
                control_feature: str, sensitive_feature: str, clf_metric_name: str) -> Tuple[MetricFrame, float]:
    """
      Get metrics relevant for a model trained with sensitive feature optionaly over a control feature.

      params:
        - CF: control feature in the dataset to measure fairness over.
        - A: sensitive feature in the dataset to measure its fairness.
        - y_true: ground truth y targets.
        - y_pred: predicted y targets.
        - control_feature: name of control feature in the dataset to measure fairness over.
        - sensitive_feature: name of sensitive feature in the dataset to measure its fairness.
        - clf_metric_name: classifier metric such as accuracy, f1.

      returns:
        - metric_frame: metric frame(`clf_metric_name`, `selection_rate`, `count`) resulted from the model.
        - fairness_metric: fairness metric like `demographic parity difference` resulted from the model.
    """
    fairlearn_metrics = {clf_metric_name: get_scorer(clf_metric_name)._score_func,
                         "selection_rate": selection_rate,
                         "count": count,
                         }

    metric_frame = MetricFrame(metrics=fairlearn_metrics, y_true=y_true,
                               y_pred=y_pred, sensitive_features=A, control_features=CF)

    diff = metric_frame.difference()
    demographic_parity_difference = diff['selection_rate']
    mean_demographic_parity_difference = np.mean(demographic_parity_difference)

    print(
        f'control feature: {control_feature}\nsensitive feature: {sensitive_feature}\n')
    print(metric_frame.overall)
    print(metric_frame.by_group)
    print(f'demographic parity difference:\n{demographic_parity_difference}')
    print(
        f'mean demographic parity difference over {control_feature}: {mean_demographic_parity_difference}')
    print(
        f"mean {clf_metric_name} over {control_feature}: {np.mean(metric_frame.overall[clf_metric_name])}")
    print('****************')

    return metric_frame, mean_demographic_parity_difference
