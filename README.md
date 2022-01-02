# MLOps Final Project
Fully automated ML pipeline using [xgboost](https://xgboost.readthedocs.io/en/stable/) and [fairlearn](https://fairlearn.org/).

# What is about?
Improve model fairness compared to the baseline.
# Datasets
- [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

# Baselines
- [Bank Marketing](https://www.kaggle.com/kevalm/xgboost-implementation-on-bank-marketing-dataset)
- [German Credit](https://www.kaggle.com/hendraherviawan/predicting-german-credit-default)

# Running
- Install dependencies: `pip install -r requirements.txt`
- Pipeline : `python main.py <dataset>`
    - `dataset`: 
        - bank-marketing
        - german-credit

# Configuration
`config.yml`

For each dataset one can configure its:
- `optional_sensitive_features`: list of potentially sensitive features.
- `control_feature`: [control feature](https://fairlearn.org/v0.7.0/user_guide/assessment.html#control-features-for-grouped-metrics) to measure fairness over.
- `clf_metric_name`: the name of the metric for the classifier model, e.g `accuracy`, `f1`.
