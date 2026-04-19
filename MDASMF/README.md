# HSI Deep Clustering (Frequency-Spatial Transformer)

This repository contains the official implementation for Hyperspectral Image (HSI) clustering using a dual-branch Frequency-Spatial Transformer and Deep Embedded Clustering (DEC).

## Structure
* `dataset.py`: Handles HSI data loading, PCA dimensionality reduction, and spatial patch extraction.
* `models.py`: Contains the core neural network architectures including `FreqAttention`, `MDFM`, `FastFourierAdjustmentBlock`, `AutoEncoder`, and `DEC`.
* `utils.py`: Includes clustering metrics (ACC, NMI, ARI, etc.), matching algorithms, and visualization tools.
* `main.py`: The main entry point for model pretraining, DEC fine-tuning, and evaluation.

## Requirements
* Python 3.8+
* PyTorch 1.10+
* NumPy, SciPy, Scikit-learn, Matplotlib
* Spectral, thop

## Usage
To run the clustering model on the Indian Pines dataset:

```bash
python main.py --name indian --patch_size 5 --pca_dim 30 --pretrain_epochs 200 --train_epochs 200