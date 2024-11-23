# Melanoma Detection with Pseudo-Annotations

A deep learning project for melanoma detection using U-Net architecture with pseudo-labeling technique for semi-supervised learning. This implementation uses PyTorch and is designed to work with both labeled and unlabeled melanoma images.

## Project Overview


```
Input Image → U-Net → Predicted Mask
                         ↓
               Compare with Ground Truth
                         ↓
                  Calculate Metrics
```


This project implements a semi-supervised learning approach for melanoma detection using:
- U-Net architecture for image segmentation
- Pseudo-labeling technique for leveraging unlabeled data
- Confidence-based label generation
- Gradient accumulation for handling larger batch sizes
- Automated model checkpointing

## Directory Structure

```
melanoma-pseudo-annotation/
├── data/
│   ├── labeled_images/     # Labeled melanoma images
│   ├── labeled_masks/      # Ground truth segmentation masks
│   └── unlabeled_images/   # Unlabeled melanoma images
├── checkpoints/            # Saved model checkpoints
├── results/
│   ├── predictions/        # Model predictions
│   └── visualizations/     # Performance visualizations
├── enhanced.py            # Main implementation file
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Requirements

```
numpy
torch
torchvision
scikit-learn
pillow
matplotlib
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Organize your dataset following the directory structure:
   - Place labeled melanoma images in `data/labeled_images/`
   - Place corresponding segmentation masks in `data/labeled_masks/`
   - Place unlabeled images in `data/unlabeled_images/`

2. Supported image formats:
   - Images: PNG format
   - Masks: Binary PNG format (0 for background, 255 for melanoma)
   - Recommended image size: 256x256 pixels

## Model Architecture

The implementation uses a U-Net architecture with:
- Encoder: 4 encoding blocks with double convolution
- Bottleneck layer
- Decoder: 4 decoding blocks with skip connections
- Input channels: 3 (RGB)
- Output channels: 1 (binary mask)
- Batch normalization and ReLU activation

## Training Process

The training process includes:

1. Initial training on labeled data
2. Pseudo-label generation:
   - Model predicts on unlabeled data
   - Confidence threshold filtering (≥ 0.7)
   - High-confidence predictions become pseudo-labels
3. Additional training with pseudo-labeled data
4. Model evaluation and checkpoint saving

### Hyperparameters

- Learning rate: 1e-4
- Batch size: 16
- Number of epochs: 20
- Gradient accumulation steps: 2
- Confidence threshold: 0.7
- Test split ratio: 0.2

## Usage

1. Prepare your dataset as described above

2. Run the training script:
```bash
python3 enhanced.py
```

3. Monitor the training process:
   - Batch-wise loss updates
   - Epoch average loss
   - Number of confident pseudo-labels generated
   - Model checkpoint saves
   - Confusion matrix visualization

## Model Outputs

The training process generates:
1. Trained model checkpoints in `checkpoints/`
2. Prediction images in `results/predictions/`
   - Validation set predictions
   - Pseudo-label visualizations
3. Performance visualizations in `results/visualizations/`
   - Confusion matrices
   - Other metrics plots

## Performance Metrics

The model's performance is evaluated using:
- Binary Cross-Entropy Loss
- Confusion Matrix
- Prediction visualizations

## File Descriptions

- `enhanced.py`: Main implementation file containing:
  - Dataset class for handling labeled and pseudo-labeled data
  - U-Net model implementation
  - Training and evaluation loops
  - Pseudo-label generation logic
  - Utility functions for saving results

## Additional Features

1. Automatic directory creation
2. GPU support with automatic detection
3. Memory optimization through gradient accumulation
4. Comprehensive logging
5. Visualization tools
6. Model checkpointing

## Best Practices

1. Data Preparation:
   - Ensure images are properly preprocessed
   - Verify mask-image pairs match
   - Use consistent naming conventions

2. Training:
   - Monitor GPU memory usage
   - Check pseudo-label quality periodically
   - Save intermediate results

3. Evaluation:
   - Review generated pseudo-labels
   - Analyze confusion matrices
   - Inspect prediction visualizations

## Troubleshooting

Common issues and solutions:

1. CUDA out of memory:
   - Reduce batch size
   - Increase gradient accumulation steps
   - Reduce image size

2. Poor pseudo-label quality:
   - Increase confidence threshold
   - Improve initial training
   - Check data normalization

3. Slow training:
   - Enable GPU support
   - Optimize data loading
   - Adjust batch size

## Contributing

Feel free to contribute to this project by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Sharing your experience and results

## License

This project is available under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{melanoma_pseudo_annotation,
  title = {Melanoma Detection with Pseudo-Labeling},
  year = {2024},
  description = {A semi-supervised learning approach for melanoma detection using U-Net and pseudo-labeling}
}
```

## Contact

For questions or support, please open an issue in the repository.
