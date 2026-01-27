# Multi-Level Evaluation Reveals Heterogeneous Bias in Pathology Foundation Models

This framework evaluates the generalization capability of **Pathology Foundation Models (PFMs)** by quantifying multi-level biases within Whole Slide Image (WSI) datasets, spanning from cohort and slide to patch levels.

---

## ## Key Features

* **Feature Extraction**: Extracts embedding features of each WSI using PFMs via [TRIDENT](https://github.com/mahmoodlab/TRIDENT).
* **Feature Preparation**: Prepares features for analysis. If stain normalization or other mitigation data is used, the framework ensures only relevant patches from the whole dataset are processed.
* **Cohort-level Evaluation**: Performed using Uniform Manifold Approximation and Projection (UMAP) visualization and Normalized Mutual Information (NMI) calculation to quantify dataset-wide bias.
* **Prototype Construction**: Bias-level prototypes (e.g., Center, Race, Scanner) are constructed using a prototype-based method. The distance between each patch and its corresponding prototype is calculated per slide.
* **Slide-level Evaluation**: Uses **Wasserstein distance** to evaluate bias distribution at the slide level.
* **Patch-level Evaluation**: Quantifies localized bias based on the distance between individual patches and the defined center prototypes.
* **Visualization**: Supports spatial overlay of patch-level bias directly onto the Whole Slide Image for intuitive analysis.
