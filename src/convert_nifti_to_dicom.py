
from pathlib import Path
import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# Input and output
INPUT_FILE = Path("data/spleen_2 (1).nii")
OUTPUT_DIR = Path("data/orthanc_test_dicom")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load NIfTI volume
image = nib.load(str(INPUT_FILE))
volume = image.get_fdata()

# Create unique identifiers for this test series
study_uid = generate_uid()
series_uid = generate_uid()
frame_uid = generate_uid()

# Use a single axial slice for the DICOM upload test
slice_data = volume[:, :, volume.shape[2] // 2]
slice_data = np.nan_to_num(slice_data)

# Convert pixel values to signed 16-bit integers
pixel_data = np.clip(slice_data, -32768, 32767).astype(np.int16)

# DICOM metadata
file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
file_meta.MediaStorageSOPInstanceUID = generate_uid()
file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
file_meta.ImplementationClassUID = generate_uid()

filename = OUTPUT_DIR / "test_spleen_slice.dcm"

ds = FileDataset(
    str(filename),
    {},
    file_meta=file_meta,
    preamble=b"\0" * 128
)

ds.PatientName = "TEST^PATIENT"
ds.PatientID = "MEDVISION_TEST_001"
ds.PatientBirthDate = ""
ds.PatientSex = "O"
ds.Modality = "CT"
ds.StudyInstanceUID = study_uid
ds.SeriesInstanceUID = series_uid
ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
ds.FrameOfReferenceUID = frame_uid
ds.StudyDate = "20260101"
ds.SeriesNumber = 1
ds.InstanceNumber = 1
ds.ImageType = ["DERIVED", "SECONDARY"]
ds.ConversionType = "WSD"
ds.Rows, ds.Columns = pixel_data.shape
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.BitsAllocated = 16
ds.BitsStored = 16
ds.HighBit = 15
ds.PixelRepresentation = 1
ds.RescaleIntercept = 0
ds.RescaleSlope = 1
ds.PixelData = pixel_data.tobytes()
ds.is_little_endian = True
ds.is_implicit_VR = False

ds.save_as(str(filename), enforce_file_format=True)

# Verify output
check = pydicom.dcmread(str(filename))
print("DICOM file created successfully!")
print("File:", filename.resolve())
print("Image dimensions:", check.Rows, "x", check.Columns)
print("Modality:", check.Modality)
