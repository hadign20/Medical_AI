import numpy as np
import matplotlib as plt
import os
import SimpleITK as sitk
import seaborn as sns
import pandas as pd
from pingouin import intraclass_corr



def dice_coefficient(seg1, seg2):
    intersection = (np.logical_and(seg1, seg2)).sum()
    union = seg1.sum() + seg2.sum()
    return 2 * intersection / union


def calculate_icc3(data):
    """
    Calculate ICC3 (two-way mixed, consistency) for the given data.

    Parameters:
        data (pd.DataFrame): A DataFrame with columns ['subject', 'rater', 'value'] where:
                             - 'subject' is the identifier for the cases.
                             - 'rater' is the identifier for the readers.
                             - 'value' is the measurement value for each reader and case.

    Returns:
        float: The ICC3 value.
    """
    assert all([v in data.columns for v in ['subject', 'rater', 'value']]), "Data does not have the required columns"

    #data['subject'] = pd.to_numeric(data['subject'], errors='coerce')
    data['value'] = pd.to_numeric(data['value'], errors='coerce')

    icc_results = intraclass_corr(data=data, targets='subject', raters='rater', ratings='value')
    icc3_value = icc_results.loc[icc_results['Type'] == 'ICC3', 'ICC'].values[0]

    return icc3_value


def calculate_icc3_for_features(features_data):
    """
    Calculate ICC3 for each feature.

    Parameters:
        features_data (dict): A dictionary where keys are feature names and values are DataFrames
                              with columns ['subject', 'rater', 'value'].

    Returns:
        pd.DataFrame: A DataFrame with 'Features' and 'ICC3' columns.
    """
    icc3_values = []
    for feature, data in features_data.items():
        icc3_value = calculate_icc3(data)
        icc3_values.append({'Feature': feature, 'ICC3': icc3_value})

    return pd.DataFrame(icc3_values)

def prepare_data_for_icc(df, feature):
    icc_data = df[['Case_ID', 'Rater', feature]].rename(columns={'Case_ID': 'subject', 'Rater': 'rater', feature: 'value'})
    #icc_data['subject'] = icc_data['subject'].str.extract('(\d+)').astype(int)
    icc_data['value'] = pd.to_numeric(icc_data['value'], errors='coerce')
    return icc_data