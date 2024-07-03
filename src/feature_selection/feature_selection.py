import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, fisher_exact, wilcoxon, chi2_contingency, mannwhitneyu, ranksums
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mrmr import mrmr_classif
from sklearn.model_selection import KFold
from collections import defaultdict
from typing import List, Optional

def calculate_p_values(df: pd.DataFrame,
                       outcome_column: str,
                       categorical_columns: List[str] = [],
                       exclude_columns: List[str] = [],
                       test_numeric: Optional[str] = 'wilcox',
                       test_categorical: Optional[str] = 'fisher') -> pd.DataFrame:
    """
    Calculate p-values for each feature in the dataframe compared to the outcome variable.

    :param df: DataFrame containing features and the outcome variable.
    :param outcome_column: The name of the outcome column.
    :param categorical_columns: List of names of categorical feature columns.
    :param exclude_columns: List of columns to exclude from the analysis.
    :param test_numeric: Statistical test to use for numeric features ('ttest' or 'wilcox').
    :param test_categorical: Statistical test to use for categorical features ('fisher' or 'chi2').
    :return: DataFrame with features and their corresponding p-values.
    """
    p_values = {}

    for column in df.columns:
        if column in exclude_columns or column == outcome_column: continue

        if column in categorical_columns:
            if test_categorical == 'fisher':
                try:
                    contingency_table = pd.crosstab(df[column], df[outcome_column])
                    _, p_value = fisher_exact(contingency_table)
                except ValueError:
                    p_value = np.nan
            elif test_categorical == 'chi2':
                try:
                    contingency_table = pd.crosstab(df[column], df[outcome_column])
                    _, p_value = chi2_contingency(contingency_table)
                except ValueError:
                    p_value = np.nan
            else:
                raise ValueError("Invalid test for categorical features. Choose 'fisher' or 'chi2'.")

        elif pd.api.types.is_numeric_dtype(df[column]):
            if test_numeric == 'ttest':
                _, p_value = ttest_ind(df[column], df[outcome_column])
            elif test_numeric == 'wilcox':
                try:
                    group_0 = df[df[outcome_column] == 0][column]
                    group_1 = df[df[outcome_column] == 1][column]
                    _, p_value = mannwhitneyu(group_0, group_1)
                    #_, p_value = wilcoxon(df[column], df[outcome_column])
                except ValueError:
                    p_value = np.nan
            else:
                raise ValueError("Invalid test for numeric features. Choose 'ttest' or 'wilcoxon'.")
        else:
            raise ValueError(f"Column {column} not specified in categorical or numerical columns list.")

        p_values[column] = p_value

    p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P_Value'])
    p_values_df = p_values_df.sort_values(by=['P_Value'], ascending=True)
    return p_values_df


def calculate_auc_values(df: pd.DataFrame,
                       outcome_column: str,
                       categorical_columns: List[str] = [],
                       exclude_columns: List[str] = [] ) -> pd.DataFrame:
    """
    Calculate auc-values for each feature in the dataframe compared to the outcome variable.

    :param df: DataFrame containing features and the outcome variable.
    :param outcome_column: The name of the outcome column.
    :param categorical_columns: List of names of categorical feature columns.
    :param exclude_columns: List of columns to exclude from the analysis.
    :return: DataFrame with features and their corresponding auc-values.
    """
    auc_values = {}
    scaler = MinMaxScaler()
    #df[outcome_column] = pd.factorize(df[outcome_column])[0]
    for column in df.columns:
        if column == outcome_column or column in exclude_columns: continue

        if column in categorical_columns:
            df[column] = pd.factorize(df[column])[0]

        feature_values = df[column].values.reshape(-1, 1)
        normalized_feature_values = scaler.fit_transform(feature_values).flatten()

        if len(df[outcome_column].unique()) == 2:
            try:
                fpr, tpr, _ = roc_curve(df[outcome_column], normalized_feature_values)
                roc_auc = auc(fpr, tpr)
                auc_values[column] = roc_auc
            except ValueError:
                auc_values[column] = np.nan
        else:
            auc_values[column] = np.nan

    auc_values_df = pd.DataFrame(list(auc_values.items()), columns=['Feature', 'AUC'])
    auc_values_df = auc_values_df.sort_values(by=['AUC'], ascending=False)
    return auc_values_df





def MRMR_feature_count(df: pd.DataFrame,
                           outcome_column: str,
                           categorical_columns: List[str] = [],
                           exclude_columns: List[str] = [],
                           num_features: Optional[int] = 10,
                           CV_folds: Optional[int] = 20) -> pd.DataFrame:
    """
    Select best features defined by cross validation on MRMR method.

    :param df: DataFrame containing features and the outcome variable.
    :param outcome_column: The name of the outcome column.
    :param exclude_columns: List of columns to exclude from the analysis.
    :return: DataFrame with MRMR-selected features.
    """
    x = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
    y = df[outcome_column]

    kf = KFold(n_splits=CV_folds)
    selected_feature_count = defaultdict(int)

    for train_index, val_index in kf.split(x):
        x_train_fold, x_val_fold = x.iloc[train_index], x.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        selected_features_fold = mrmr_classif(X = x_train_fold, y = y_train_fold, K = num_features)
        for feature in selected_features_fold:
            selected_feature_count[feature] += 1

    mrmr_count_df = pd.DataFrame(list(selected_feature_count.items()), columns=['Feature', 'MRMR_Count'])
    mrmr_count_df = mrmr_count_df.sort_values(by=['MRMR_Count'], ascending=False)
    return mrmr_count_df






def save_feature_analysis(p_values_df: pd.DataFrame,
                          auc_values_df: pd.DataFrame,
                          mrmr_count_df: pd.DataFrame,
                          results_dir: str):
    """
    Save the resutls of feature analysis to the output dir.

    :param p_values_df: DF of feature p-values.
    :param auc_values_df: DF of feature AUC values.
    :param mrmr_count_df: DF of selected features by MRMR.
    :param results_dir: Path to reulsts directory.
    """
    analysis_df = p_values_df.merge(auc_values_df, on='Feature').merge(mrmr_count_df, on='Feature')
    analysis_df = analysis_df.sort_values(by=['AUC', 'P_Value', 'MRMR_Count'], ascending=[False, True, False])

    output_dir = os.path.join(results_dir, "feature_analysis")
    os.makedirs(output_dir, exist_ok=True)
    analysis_df.to_excel(os.path.join(output_dir, 'feature_analysis.xlsx'), index=False)


