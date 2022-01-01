import sys
import yaml

from fairml.datasets import get_dataset
from fairml.pipeline import full_auto_pipeline


def main() -> int:
    _, dataset_name = sys.argv
    dataset = get_dataset(dataset_name)

    # exit if there is no such dataset
    if dataset is None:
        return 1

    # load configuration
    config = yaml.safe_load(open("config.yml"))
    dataset_config = config['datasets'][dataset_name]
    for k, v in config['default'].items():
        if k not in dataset_config:
            dataset_config[k] = v

    # execute full automated pipeline
    full_auto_pipeline(dataset, **dataset_config)

    return 0


if __name__ == '__main__':
    exit(main())
