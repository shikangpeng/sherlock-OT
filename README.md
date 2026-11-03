# sherlock-OT

Figure and analysis-pipeline code accompanying 

**Peng, S., Levine, B., & Buchsbaum, B. R. (2026). Naturalistic recall preserves temporal specificity despite cortical feature reorganization. bioRxiv.**

Rationale of the analysis choice can be found in the article.

If you use or adapt this optimal transport alignment approach in this code for your analysis, please cite the article.


## Repository structure

### `figure_code/`

Notebooks used to generate the manuscript figures:

- `figure2.ipynb` - validation of the optimal-transport alignment, including scene correspondence and recalled-versus-not-recalled scene analyses.
- `figure3.ipynb` - cortical maps of encoding and recall accuracy and their relationship across cortical networks.
- `figure4.ipynb` - within-scene recall accuracy, network summaries, and sensory-to-association comparisons.
- `figure5_ab.ipynb` - encoding and recall accuracy along the principal cortical gradient.
- `figure5_d.ipynb` - comparison of low-level and semantic feature separation between encoding and recall.

### `pipeline_code/`

Example codes covering the main stages of analyses:

- `encode_training.py` - trains PLS encoding models and selects the number of components using cross-validation.
- `permutation.py` - constructs block-permutation null distributions 
- `recall_gridsearch.py` - alignment code with grid search for each participant.
- `grid_loso.py` - selects FGW hyperparameters using leave-one-subject-out validation.
- `recall_within_scene_rankacc.py` - evaluates OT-based recall predictions while restricting rank comparisons to TRs from the same scene.
- `aggregate_within_scene_rankacc.py` - aggregates participant-level results, performs group Wilcoxon tests, and applies FDR correction across parcels.

### `data/`

Derived data used by the figure notebooks, including summary tables, atlas files, cortical surface files, and selected example transport plans. Raw neuroimaging data are not included.

## License

The analysis code is released under the MIT License. Please consult the original sources and licenses for redistributed atlas and template files.

## Citation

Please cite the associated Peng et al. (2026) publication when using this code or the accompanying derived data.