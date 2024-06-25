import os
import pandas as pd
from src.preprocessing.data_loader import load_nifti, load_dicom
from src.preprocessing.data_cleaning import clean_clinical_data, normalize_radiomics_features
from src.feature_extraction.radiomics_features import extract_radiomics_features
#from src.feature_extraction.deep_features import extract_deep_features
from src.feature_selection.correlation import calculate_correlation_matrix, select_highly_correlated_features
from src.model.train_test_split import split_data
from src.model.train import train_model, evaluate_model
from src.visualization.auc_plot import plot_auc_with_ci


def main():
    # Load and preprocess data
    image, affine = load_nifti('data/raw/image.nii')
    mask, _ = load_nifti('data/raw/mask.nii')
    clinical_data = pd.read_csv('data/raw/clinical.csv')




if __name__ == "__main__":
    main()


