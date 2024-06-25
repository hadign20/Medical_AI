import radiomics
from radiomics import featureextractor

def extract_radiomics_features(image_path, mask_path, params_path):
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    features = extractor.execute(image_path, mask_path)
    return features
