
*Note*================================================

The dataset was compressed and uploaded separately due to its large size, and the Material Images dataset was split into three parts for upload. Please maintain the dataset according to the image Datasets-structureFigure.png

======================================================



numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0


Code_Material/
├── Code/
│   ├── TheoremsValidation.py          # Main validation script (all 4 experiments)
│   ├── ReadFinanceDataset.py          # Financial transactions dataset loader
│   ├── ReadIoTDataset.py              # IIoT time-series dataset loader
│   ├── ReadADImageDataset.py          # Alzheimer's Disease MRI dataset loader
│   └── ReadMaterialImageDataset.py    # Material images dataset loader
│
├── requirements.txt                    # Python dependencies
│
├── README.md                          # Execution instructions
│   ├── Environment Setup
│   ├── Dataset Preparation
│   ├── Execution Commands
│   └── Results Directory Structure
│
└── Results/                           # Output results (optional)
    ├── Experiment1_results/           # Cross-Modal Universality (Theorem 1)
    │   ├── Experiment1_Cross_Modal_Universality.tiff
    │   └── summary.txt
    ├── Experiment2_results/           # Projection Invariance & Parameter Rigidity (Theorem 3)
    │   ├── Experiment2_Projection_Invariance.tiff
    │   ├── Experiment2_Parameter_Rigidity.tiff
    │   └── summary.txt
    ├── Experiment3_results/           # Stability Boundary under Non-Gaussianity (Theorem 2)
    │   ├── Experiment3_Stability_Boundary.tiff
    │   ├── Experiment3_Moment_Decomposition.tiff
    │   └── summary.txt
    └── Experiment4_results/           # Geometric Interpretability of Confidence (Theorem 4)
        ├── Experiment4_Geometric_Interpretability.tiff
        └── summary.txt