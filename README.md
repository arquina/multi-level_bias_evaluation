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
2.  **Step 1: Feature preparation** : Our pipeline prepare features for downstream analysis.
```bash
python 1.extract_feature.py --savedir <Path to save directory> --featuredir <Path to trident root> --metadata <metadata csv file> --pfm_list <add PFM with space>
```
3. **Step 2: Cohort-level analysis** : Cohort-level analysis is performed with prepared data using UMAP and NMI analysis
```bash
python 2.cohort_level_analysis.py --feature_rootdir <save directory of step 1> --metadata <metadata csv file> --savedir <cohort_level analysis result dir>  --target_database <add database for comparison with space> --target_center <add center for comparison with space> --pfm_list <add PFM with space> --target_data <Name of the directory>
```
4. **Step 3: Prototype construction** : Make a category specific prototype and calculate the distance between patch and prototype for downstream analysis
```bash
python 3.prototype_construction.py --feature_rootdir <save directory of step 1> --metadata <metadata csv file> --savedir <prototye vector save directory>  --target_database <add database for comparison with space> --target_center <add center for comparison with space> --pfm_list <add PFM with space> 
```

5. **Step 4: Slide-level analysis** : Calculate Slide-wise distribution difference(SDD) and plot for comparison
```bash
python 4.slide_level_analysis.py --root_dir <save directory of step 3> --metadata <metadata csv file> --savedir <Slide-level analysis result> --pfm_list <add PFM with space> --target_column <select the category to compare (e.g. center, race, scanner)
```

6. **Step 5: Patch-level analysis** : Calculate Patch-wise distribution difference(PDD) and plot for comparison
```bash
python 5.patch_level_analysis.py --root_dir <save directory of step 3> --metadata <metadata csv file> --savedir <Slide-level analysis result> --pfm_list <add PFM with space> --target_column <select the category to compare (e.g. center, race, scanner)
```

You can also use additional pipeline for mitigation (6.compare_mitigation.py) and visualziation (7.overlay_visualization.ipynb)

