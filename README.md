# Multi-Level Evaluation Reveals Heterogeneous Bias in Pathology Foundation Models

This framework evaluates the generalization capability of Pathology Foundation Models (PFMs) by quantifying multi-level biases within Whole Slide Image (WSI) datasets. It provides a systematic approach to measure bias across three hierarchical levels: Cohort, Slide, and Patch.

Key Features
Feature Extraction: Extracts patch-level embeddings using state-of-the-art PFMs. This framework is optimized for features generated via TRIDENT.

Data Preparation: Facilitates feature organization for downstream analysis. It supports custom filtering, such as including only specific patches after stain normalization or other bias-mitigation preprocessing.

Cohort-level Evaluation: Quantifies global dataset bias using Uniform Manifold Approximation and Projection (UMAP) for visualization and Normalized Mutual Information (NMI) to measure the alignment between feature clusters and metadata (e.g., subtype, center, race, scanner).

Prototype Construction: Builds representative Bias Prototypes (e.g., for specific Centers, Races, or Scanners) using a prototype-based method. It calculates the embedding distance between individual patches and these prototypes for every slide.

Slide-level Evaluation: Employs the Wasserstein Distance to evaluate the distributional shift of bias at the slide level, offering a robust metric for inter-slide variability.

Patch-level Evaluation: Pinpoints localized bias by analyzing the distance of each patch to the defined prototypes, revealing how specific tissue regions are affected by non-biological factors.

Visualization: Supports Bias Heatmap Overlay, allowing users to visualize the intensity and spatial distribution of patch-level bias directly onto the original WSI.
