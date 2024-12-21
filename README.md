# Melanoma Segmentation Using Pseudo Annotations

A deep learning approach for melanoma segmentation using pseudo annotations and semi-supervised learning. This project is implemented as a Kaggle notebook utilizing the ISIC 2017 dataset.

## Overview

![Project Pipeline](docs/pipeline.png)

This notebook implements a semi-supervised learning pipeline for melanoma segmentation with the following key components:
- DullRazor preprocessing for hair removal
- U-Net architecture for image segmentation
- Pseudo-label generation for unlabeled data
- Comprehensive evaluation metrics

## Dataset

The project uses the ISIC 2017 dataset, which should be structured in your Kaggle environment as follows:

```
../input/isic-2017/
    ├── ISIC-2017_Training_Data/
    │   └── ISIC-2017_Training_Data/
    └── ISIC-2017_Training_Part1_GroundTruth/
        └── ISIC-2017_Training_Part1_GroundTruth/
```

## Implementation Details

The notebook contains:
1. Data preprocessing using DullRazor
2. U-Net model implementation
3. Semi-supervised training pipeline
4. Pseudo-label generation and validation
5. Metrics calculation and visualization

## Usage

1. Upload the notebook to Kaggle
2. Add the ISIC 2017 dataset to your notebook
3. Run all cells sequentially

## Requirements

The notebook utilizes standard Kaggle GPU environment with:
- PyTorch
- OpenCV
- scikit-learn
- PIL
- matplotlib
- tqdm

## License

[MIT License](LICENSE)


## Contact

For questions or support, please open an issue in the repository.
