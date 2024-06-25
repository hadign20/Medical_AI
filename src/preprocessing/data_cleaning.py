import pandas as pd
from sklearn.impute import SimpleImputer

def clean_clinical_data(df):
    imputer = SimpleImputer(strategy='mean')
    df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_cleaned

def normalize_radiomics_features(df):
    return (df - df.mean()) / df.std()
