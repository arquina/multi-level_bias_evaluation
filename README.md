# Multi-Level Evaluation Reveals Heterogeneous Bias in Pathology Foundation Models

This framework evaluates the generalization capability of **Pathology Foundation Models (PFMs)** by quantifying multi-level biases within Whole Slide Image (WSI) datasets, spanning from cohort and slide to patch levels.

---

## Key Features

* **Feature Extraction**: Extracts embedding features of each WSI using PFMs via [TRIDENT](https://github.com/mahmoodlab/TRIDENT).
* **Feature Preparation**: Prepares features for analysis. If stain normalization or other mitigation data is used, the framework ensures only relevant patches from the whole dataset are processed.
* **Cohort-level Evaluation**: Performed using Uniform Manifold Approximation and Projection (UMAP) visualization and Normalized Mutual Information (NMI) calculation to quantify dataset-wide bias.
* **Prototype Construction**: Bias-level prototypes (e.g., Center, Race, Scanner) are constructed using a prototype-based method. The distance between each patch and its corresponding prototype is calculated per slide.
* **Slide-level Evaluation**: Uses **Wasserstein distance** to evaluate bias distribution at the slide level.
* **Patch-level Evaluation**: Quantifies localized bias based on the distance between individual patches and the defined center prototypes.
* **Visualization**: Supports spatial overlay of patch-level bias directly onto the Whole Slide Image for intuitive analysis.

## How to Run?
1.  **Data preparation:** Metadata (Column : Patient	sample	database	subtype	subtype_name	center	race	scanner	svs_file_path	feature_file_name), feature extracted using [TRIDENT](https://github.com/mahmoodlab/TRIDENT).
```bash
python extract_feature.py --savedir <Path to save directory> --featuredir <Path to trident root> --metadata <metadata csv file> --pfm_list <PFM list>
```
3.  **Step 1: Feature preparation** : Our pipeline prepare features for downstream analysis.

