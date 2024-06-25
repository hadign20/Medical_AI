import pandas as pd

def calculate_correlation_matrix(df):
    corr_matrix = df.corr().abs()
    return corr_matrix

def select_highly_correlated_features(corr_matrix, threshold=0.9):
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop
