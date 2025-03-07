# Molecular Clustering with FAISS Multi-GPU Implementation

## Overview
This code contains a high-performance implementation for clustering a large library ~ 10 million molecules based on molecular fingerprints. This code leverages FAISS (Facebook AI Similarity Search, for more info about FAISS (https://ai.meta.com/tools/faiss/) distributed across multiple GPUs (I used 4 * V100 on DRAC clusters) to enable efficient similarity searching and clustering of molecular structures.

## Requirements
- FAISS library
- CUDA environment
- GPUs (I used 4 NVIDIA V100 32G)
- RDKit 

## Applications
This code supports various cheminformatics applications:
- Clustering of compound libraries
- Virtual screening
- Chemical space exploration
- Diversity analysis
