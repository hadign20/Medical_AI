import os
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
import SimpleITK as sitk
import nrrd


def load_nifti(file_path):
    """Load NIFTI file."""
    img = nib.load(file_path)
    return img.get_fdata(), img.affine


def load_mha(file_path):
    """Load MHA file."""
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image), image.GetOrigin(), image.GetSpacing()


def load_nrrd(file_path):
    """Load NRRD file."""
    data, header = nrrd.read(file_path)
    return data, header


def load_dicom(directory):
    """Load DICOM series from a directory."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image), image.GetOrigin(), image.GetSpacing()


def load_excel(file_path, sheet_name=0):
    """Load Excel file."""
    return pd.read_excel(file_path, sheet_name=sheet_name)


def load_csv(file_path):
    """Load CSV file."""
    return pd.read_csv(file_path)


def resample_image(image, reference_image):
    """Function to resample images to the same shape"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(image)



def load_excel_sheets(file_path, sheets):
    data = {}
    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet)
        data[sheet] = df
    return data


def load_data(file_path, file_type=None):
    """General data loading function."""
    if file_type is None:
        file_type = os.path.splitext(file_path)[1].lower()

    if file_type in ['.nii', '.nii.gz']:
        return load_nifti(file_path)
    elif file_type == '.mha':
        return load_mha(file_path)
    elif file_type == '.nrrd':
        return load_nrrd(file_path)
    elif file_type == '.dcm':
        return load_dicom(file_path)
    elif file_type == '.xlsx':
        return load_excel(file_path)
    elif file_type == '.csv':
        return load_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")







# Example usage:
# data, header = load_data('path/to/file.nii', file_type='.nii')
# data, header = load_data('path/to/file.mha')
# data, header = load_data('path/to/dicom/folder', file_type='.dcm')



