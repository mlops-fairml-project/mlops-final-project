from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fairml.preprocessing import reduce_groups
from fairml.evaluation import get_metrics


def sensitive_train_XGB(data_X: pd.DataFrame, data_y: pd.Series,
                        control_feature: str, sensitive_feature: str, clf_metric_name: str) -> Dict[str, Any]:
    """
      Train model with sensitive feature optionaly over a control feature.

      params:
        - data_X: X features for the model.
        - data_y: y targets for the model.
        - control_feature: name of control feature in the dataset to measure fairness over.
        - sensitive_feature: name of sensitive feature in the dataset to measure its fairness.
        - clf_metric_name: classifier metric such as accuracy, f1.

      returns:
        - sensitive_feature: the given sensitive feature.
        - classifier: the trained xgboost classifier model.
        - data: the data was given to the model: X, y, control, sensitive.
        - metric_frame: metric frame resulted from the model(`get_metrics`).
        - fairness_metric: fairness metric like demographic parity difference resulted from the model(`get_metrics`).
    """

    # control feature + bucketize if required
    CF = reduce_groups(data_X[control_feature]) if control_feature else None
    # sensitive feature + bucketize if required
    A = reduce_groups(data_X[sensitive_feature])
    # remove sensitive feature from X
    data_X = data_X.drop([sensitive_feature], axis=1)

    # one hot encoding for categorical features
    data_X = pd.get_dummies(data_X)

    # split train-test datasets
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        data_X, data_y, A, test_size=0.3, random_state=2, stratify=data_y)
    CF_train = None
    CF_test = None
    if control_feature:
        CF_train, CF_test = train_test_split(
            CF, test_size=0.3, random_state=2, stratify=data_y)

    # Create an XGB classifier and train it on the training set.
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)

    metric_frame, demographic_parity_difference = get_metrics(
        CF_test, A_test, y_test, clf.predict(X_test), control_feature, sensitive_feature, clf_metric_name)

    return {"sensitive_feature": sensitive_feature,
            "classifier": clf,
            "data": (X_train, X_test, y_train, y_test, CF_train, CF_test, A_train, A_test),
            "metric_frame": metric_frame,
            "fairness_metric": demographic_parity_difference,
            }


def unfairness_recognition(data_X: pd.DataFrame, data_y: pd.Series,
                           control_feature: str, optional_sensitive_features: List[str], clf_metric_name: str,
                           num_of_mitigated_features: int = 1) -> List[Dict[str, Any]]:
    """
      Recognize unfairness in given list of sesitive features.
      The unfairness of each model trained with different sensitive feature is defined by pre-defined fairness metric.

      params:
        - data_X: X features for the model
        - data_y: y targets for the model
        - control_feature: name of control feature in the dataset to measure fairness over.
        - optional_sensitive_features: list of potential sensitive features in the dataset.
        - clf_metric_name: classifier metric such as accuracy, f1.
        - num_of_mitigated_features: number of sensitive features to return.

        returns:
          The most `num_of_mitigated_features` unfair models.
    """
    sensitive_features_trained_result = [sensitive_train_XGB(
        data_X, data_y, control_feature, feature, clf_metric_name) for feature in optional_sensitive_features]
    sorted_sensitive_features_trained_result = sorted(
        sensitive_features_trained_result, key=lambda r: -r["fairness_metric"])

    return sorted_sensitive_features_trained_result[0:num_of_mitigated_features]
