"""
config.py

Global configuration variables, paths, and hyperparameters for pipeline execution.
"""
import os
from pathlib import Path
import torch

# Base Paths (default to Modal volume path, overridable via environment variable)
BASE_PATH_STR = os.getenv("RADIOLOGY_BASE_PATH", "/mnt/radiology-data/archive")
BASE_PATH = Path(BASE_PATH_STR)

# Artifacts & Images
ARTIFACTS_DIR = BASE_PATH / "modal_artifacts"
IMAGES_DIR = BASE_PATH / "images" / "images_normalized"

# Data Files
REPORTS_FILE = BASE_PATH / "indiana_reports.csv"
PROJECTIONS_FILE = BASE_PATH / "indiana_projections.csv"
CHEXPERT_FILE = BASE_PATH / "chexpert_labels.csv"

# Batch Sizes & Workers
NUM_IMAGE_WORKERS = max(2, min(8, os.cpu_count() or 2))
OPENCLIP_BATCH_SIZE = 128
TEXT_BATCH_SIZE = 128
RAD_DINO_BATCH_SIZE = 32
RADGRAPH_BATCH_SIZE = 64

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def configure_torch_for_gpu():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

# Models
BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMED_ROBERTA_MODEL = "allenai/biomed_roberta_base"
RADDINO_MODEL = "microsoft/rad-dino"
RADGRAPH_MODEL = "radgraph-xl"

# Retrieval & Thresholds
RETRIEVAL_K = 10
CLIP_THRESHOLD     = 0.85   # cosine threshold for CLIP-only blind detection
MOF_THRESHOLD      = 0.70   # equivalent threshold for MoF distances ((clip+dino)/2)
CHEXPERT_THRESHOLD = 0.70   # inner-product threshold for CheXpert label vectors
DINO_THRESHOLD     = 0.60

# Retrieval mode: 'baseline' (CLIP-only) | 'mof' (CLIP+DINO) | 'chexpert'
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "mof")

_results_dir_map = {
    "baseline": "modal_artifacts",
    "mof":      "modal_artifacts_mof",
    "chexpert": "modal_artifacts_chexpert",
}
# Results directory is mode-specific; ARTIFACTS_DIR holds shared embedding caches
RESULTS_DIR = BASE_PATH / _results_dir_map.get(RETRIEVAL_MODE, "modal_artifacts_mof")
