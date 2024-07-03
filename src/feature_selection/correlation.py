import numpy as np
import pandas as pd

def calculate_correlation_matrix(df):
    corr_matrix = df.corr().abs()
    return corr_matrix

def select_highly_correlated_features(corr_matrix, threshold=0.9):
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

def remove_collinear_features(df, threshold):
    """
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold.
    :param x: features dataframe
    :param threshold: features with correlations greater than this value are removed
    :return: dataframe that contains only the non-highly-collinear features
    """
    case = df.iloc[:, 0]
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    corr_matrix = x.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper_tri.columns if any (upper_tri[column] > threshold)]
    x_dropped = x.drop(columns = to_drop)

    out_df = pd.concat([case, x_dropped, y], axis=1)
    return out_df