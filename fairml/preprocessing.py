import pandas as pd
import numpy as np


def reduce_groups(A: pd.Series, num_of_groups: int = 2) -> pd.Series:
    """
      Bucketize numeric feature into `num_of_groups` buckets.

      params:
        - A: a feature to bucketize.
        - num_of_groups: number of buckets to divide feature to.

      returns:
        A bucketized feature. 
    """
    uniq = len(np.unique(A))
    # bucketize if it's a numeric feature with too many values
    if (uniq > num_of_groups) and (not isinstance(A[0], str)):
        interval = int((100/(2*num_of_groups)))
        A_percentile = [np.percentile(A, interval*g)
                        for g in range(2*num_of_groups + 1)]
        for i in range(num_of_groups):
            low_per = A_percentile[2*i]
            mid_per = A_percentile[2*i+1]
            high_per = A_percentile[2*i+2]
            mask = ((A > low_per) & (A <= high_per)
                    ) if i > 0 else (A <= high_per)
            A.loc[mask] = mid_per

    return A
