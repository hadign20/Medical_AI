import os
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
import SimpleITK as sitk

def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata(), img.affine

def load_dicom(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image), image.GetOrigin(), image.GetSpacing()
