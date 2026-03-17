from pathlib import Path

# ==================================================
#                  PATH DEFINITIONS
# ==================================================
PATH_TO_ROOT = Path(__file__).parent.parent.parent # Project root directory
PATH_TO_DATA = PATH_TO_ROOT/ 'data' # Data Root Directory
PATH_TO_RAW_DATA = PATH_TO_DATA / 'raw' # Directory for Raw Data
PATH_TO_ANNOTATED_DATA = PATH_TO_DATA / 'annotated_data' # Directory for Annotated Data
PATH_TO_JSON_DATA = PATH_TO_DATA / 'json' # Segment and recording metadata for all subjects

# Unprocessed TUAR Data
PATH_TO_TUH_EEG_ARTIFACT_DATASET = PATH_TO_RAW_DATA / 'tuh_eeg_artifact/01_tcp_ar'
PATH_TO_TUAR_EDF_FILES = PATH_TO_TUH_EEG_ARTIFACT_DATASET / 'edf'
PATH_TO_TAUR_ANNOTATION_FILES = PATH_TO_TUH_EEG_ARTIFACT_DATASET / 'annotations'

# Processed TUAR Data
PATH_TO_ANNOTATED_TUAR_EDF = PATH_TO_ANNOTATED_DATA / 'edf_format' # Path to EDF format TUAR data with embedded annotations
PATH_TO_ANNOTATED_TUAR_FIF = PATH_TO_ANNOTATED_DATA / 'fif_format' # Path to FIF format TUAR data with embedded annotations

# Custom Metadata
PATH_TO_TUAR_JSON = PATH_TO_JSON_DATA / 'tuh_eeg_artifact/01_tcp_ar'
PATH_TO_TUAR_01_TCP_AR_CUSTOM_MONTAGE = PATH_TO_TUH_EEG_ARTIFACT_DATASET / '01_tcp_ar_montage.txt'

