from fairml.evaluation import get_metrics
from fairml.detection import unfairness_recognition
from fairml.mitigation import exponentiated_gradient, grid_search


def full_auto_pipeline(dataset, optional_sensitive_features,
                       control_feature=None,
                       clf_metric_name='accuracy',
                       num_of_mitigated_features=1):
    """
        Full automated pipeline function.
            - detection
            - mitigation
    """
    data_y = dataset.pop('y')
    data_X = dataset

    print("Check how biased is each sensitive feature:")
    most_biased_result = unfairness_recognition(
        data_X, data_y, control_feature, optional_sensitive_features, clf_metric_name, num_of_mitigated_features)[0]

    sensitive_feature = most_biased_result["sensitive_feature"]
    XGB_classifier = most_biased_result["classifier"]
    X_train, X_test, y_train, y_test, CF_train, CF_test, A_train, A_test = most_biased_result[
        "data"]
    unmitigated_metric_frame = most_biased_result["metric_frame"]
    unmitigated_fairness_metric = most_biased_result["fairness_metric"]

    # "fairness_metric": demographic parity difference
    print("The most biased feature is: ", sensitive_feature)
    print("======================================\n\n\n")

    metric_frame, _ = get_metrics(CF_test, A_test, y_test, XGB_classifier.predict(
        X_test), control_feature, sensitive_feature, clf_metric_name)

    metric_frame.by_group.plot.bar(
        subplots=True, layout=[4, 1], legend=False, figsize=[12, 8],
        title=f'Baseline model sensitive feature: {sensitive_feature} f1_score and selection rate by group')

    # Exponentiated Gradient mitigator
    print("Exponentiated Gradient mitigator:")
    exponentiated_gradient(XGB_classifier, X_train, X_test, y_train, y_test,
                           CF_train, CF_test, A_train, A_test,
                           control_feature, sensitive_feature, clf_metric_name)
    print("======================================\n\n\n")

    # Grid Search mitigator
    print("Grid Search mitigator:")
    grid_search(XGB_classifier, X_train, X_test, y_train, y_test,
                CF_train, CF_test, A_train, A_test,
                control_feature, sensitive_feature, clf_metric_name,
                unmitigated_metric_frame, unmitigated_fairness_metric)
