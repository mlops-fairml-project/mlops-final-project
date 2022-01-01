import os
from collections import defaultdict
from typing import Optional
import pandas as pd
import numpy as np


DATA = './data'


def _bank_marketing() -> pd.DataFrame:
    """
        Banking Dataset
    """

    # load data
    banking_data = pd.read_csv(os.path.join(
        DATA, 'bank-full.csv'), delimiter=";", header='infer')

    # change the classification to binary values
    banking_data.y.replace(('yes', 'no'), (1, 0), inplace=True)

    return banking_data


def _german() -> pd.DataFrame:
    """
        German Data
    """

    # load data
    german_data = pd.read_csv(os.path.join(
        DATA, 'german_credit_data.csv'), delimiter=",", header='infer')

    # jobs feature
    german_data = german_data[german_data.Job != 0]
    jobs = np.unique(german_data.Job)
    german_data.Job.replace(jobs, jobs.astype(str), inplace=True)

    # load classification
    names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']
    full_german_data = pd.read_csv(os.path.join(
        DATA, 'german.data'), names=names, delimiter=' ')
    german_data['y'] = full_german_data['classification']

    # change the classification to binary values
    german_data.y.replace((2, 1), (1, 0), inplace=True)

    return german_data


_dataset_funcs = defaultdict(lambda : (lambda : None))
_dataset_funcs.update({
    'bank-marketing': _bank_marketing,
    'german-credit': _german
})


def get_dataset(dataset_name: str) -> Optional[pd.DataFrame]:
    """
        Get Dataset.

        params:
            - dataset: `bank-marketing` or `german-credit`.
        
        returns:
            Corresponding dataset.
    """    
    return _dataset_funcs[dataset_name]()
