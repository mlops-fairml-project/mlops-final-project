import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fairlearn.reductions import DemographicParity, ErrorRate, ExponentiatedGradient, GridSearch

from fairml.evaluation import get_metrics


def exponentiated_gradient(clf, X_train, X_test, y_train, y_test,
                           CF_train, CF_test, A_train, A_test,
                           control_feature, sensitive_feature, clf_metric_name):
    """
        Exponentiated Gradient mitigation.
    """
    constraint = DemographicParity()
    mitigator = ExponentiatedGradient(clf, constraint)
    mitigator.fit(X_train, y_train, sensitive_features=A_train,
                  control_features=CF_train)
    metric_frame, _ = get_metrics(CF_test, A_test, y_test, mitigator.predict(
        X_test), control_feature, sensitive_feature, clf_metric_name)

    metric_frame.by_group.plot.bar(
        subplots=True, layout=[4, 1], legend=False, figsize=[12, 8],
        title=f'Exponentiated Gradient sensitive feature: {sensitive_feature} {clf_metric_name} and selection rate by group')


def grid_search(clf, X_train, X_test, y_train, y_test,
                CF_train, CF_test, A_train, A_test,
                control_feature, sensitive_feature, clf_metric_name,
                unmitigated_metric_frame, unmitigated_fairness_metric):
    """
        Grid Search mitigation.
    """
    sweep = GridSearch(clf,
                       constraints=DemographicParity(),
                       grid_size=71)
    sweep.fit(X_train, y_train, sensitive_features=A_train,
              control_features=CF_train)

    predictors = sweep.predictors_

    errors, disparities = [], []
    for m in predictors:
        def classifier(X): return m.predict(X)

        error = ErrorRate()
        error.load_data(X_train, y_train,
                        sensitive_features=A_train, control_features=CF_train)
        disparity = DemographicParity()
        disparity.load_data(
            X_train, y_train, sensitive_features=A_train, control_features=CF_train)
        errors.append(error.gamma(classifier)[0])
        disparities.append(disparity.gamma(classifier).max())

    all_results = pd.DataFrame(
        {"predictor": predictors, "error": errors, "disparity": disparities})

    non_dominated = []
    for row in all_results.itertuples():
        errors_for_lower_or_eq_disparity = all_results["error"][all_results["disparity"] <= row.disparity]
        if row.error <= errors_for_lower_or_eq_disparity.min():
            non_dominated.append(row.predictor)

    predictions = {"unmitigated": clf.predict(X_test)}
    metric_frames = {"unmitigated": unmitigated_metric_frame}
    fairness_metrics = {"unmitigated": unmitigated_fairness_metric}
    min = 1
    for i, model in enumerate(non_dominated):
        key = "dominant_model_{0}".format(i)
        predictions[key] = model.predict(X_test)
        metric_frames[key], fairness_metrics[key] = get_metrics(
            CF_test, A_test, y_test, predictions[key], control_feature, sensitive_feature, clf_metric_name)

        if fairness_metrics[key] < min:
            min = fairness_metrics[key]
            selected_model = model

    print("Selected Model:")
    metric_frame, _ = get_metrics(CF_test, A_test, y_test, selected_model.predict(
        X_test), control_feature, sensitive_feature, clf_metric_name)

    metric_frame.by_group.plot.bar(
        subplots=True, layout=[4, 1], legend=False, figsize=[12, 8],
        title=f'Grid Search sensitive feature: {sensitive_feature} {clf_metric_name} and selection rate by group')

    x = [np.mean(metric_frame.overall[clf_metric_name])
         for metric_frame in metric_frames.values()]
    y = list(fairness_metrics.values())
    keys = list(metric_frames.keys())
    plt.figure()
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(keys[i], (x[i] + 0.0003, y[i]))
    plt.xlabel(clf_metric_name)
    plt.ylabel("selection rate difference")
    plt.show()
