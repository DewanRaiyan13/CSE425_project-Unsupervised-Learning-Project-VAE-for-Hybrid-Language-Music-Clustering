# CSE425 Neural Network Project: Variational Autoencoders (VAE)

## Overview
This project implements and compares Variational Autoencoders (VAE) for audio feature extraction and clustering. The project explores VAE performance across three difficulty levels using music spectrograms and lyric embeddings as input data.

## Project Structure
```
project/
├── data/
│   ├── dataset.csv                          # Main dataset metadata
│   ├── processed_spectrograms.npy           # Processed audio spectrograms
│   ├── processed_ids.npy                    # Sample IDs
│   └── lyrics_embeddings.npy                # Text embeddings from lyrics
├── notebooks/
│   ├── VAE_Easy_Task.ipynb                  # Easy difficulty task
│   ├── VAE_Medium_Task.ipynb                # Medium difficulty task
│   └── VAE_hard_Task.ipynb                  # Hard difficulty task
├── results/
│   ├── eask_task/                           # Easy task results
│   │   ├── models/                          # Trained VAE models
│   │   ├── clustering_metrics.csv
│   │   └── latent features
│   ├── medium_task/                         # Medium task results
│   │   └── comprehensive_metrics.csv
│   └── hard_task/                           # Hard task results
│       └── comprehensive_comparison.csv
├── requirements.txt
└── README.md
```

## Project Tasks

### Easy Task
- Basic VAE implementation with PCA dimensionality reduction
- Clustering evaluation on latent representations
- Baseline for model performance

### Medium Task
- Convolutional VAE architecture
- Hybrid models combining different architectures
- Enhanced feature extraction from spectrograms

### Hard Task
- Advanced VAE architectures
- Comprehensive comparison across models
- Optimization and evaluation metrics

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone or download the project repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Navigate to the project directory:
```bash
cd project
```

### Running the Notebooks

Launch Jupyter and open the desired notebook:

```bash
jupyter notebook
```

Then select one of the task notebooks:
- `notebooks/VAE_Easy_Task.ipynb` - Start here for basic VAE implementation
- `notebooks/VAE_Medium_Task.ipynb` - For intermediate architectures
- `notebooks/VAE_hard_Task.ipynb` - For advanced implementations

## Data Description

### Input Data
- **Spectrograms**: Processed audio features extracted from music tracks
- **Lyrics Embeddings**: Text embeddings created from song lyrics
- **Sample IDs**: Unique identifiers linking samples across datasets

### Dataset
- **dataset.csv**: Contains metadata about the music samples and their associated features

## Results

All results are saved in the `results/` directory organized by task difficulty:
- Trained model weights (.h5 files)
- Latent space representations (.npy files)
- Clustering performance metrics (.csv files)

## Key Metrics

The project evaluates models using:
- Reconstruction loss
- KL divergence
- Clustering quality (Silhouette score, Davies-Bouldin index)
- Latent space properties

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities and clustering
- **Matplotlib**: Visualization
- **SciPy**: Scientific computing

## Project Requirements

See `requirements.txt` for complete list of dependencies with versions.

## Notes

- All notebooks use GPU acceleration when available
- Results may vary slightly due to random initialization; set seeds for reproducibility
- Ensure sufficient disk space for storing large .npy files and trained models

## Author

Md. Raiyan Uddin

## License

Academic Project
