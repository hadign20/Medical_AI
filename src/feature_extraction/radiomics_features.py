import os
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

def extract_radiomics_features(image_path, mask_path, params_path=None):
    """
    Extract radiomics features from the given image and mask using PyRadiomics.

    :param image_path: Path to the image file
    :param mask_path: Path to the mask file
    :param params_path: Path to the YAML parameters file
    :return: Dictionary containing extracted features
    """
    if params_path:
        extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()

    features = extractor.execute(image_path, mask_path)
    return features

# Example usage:
# features = extract_radiomics_features('path/to/image.nii', 'path/to/mask.nii', 'path/to/params.yaml')
