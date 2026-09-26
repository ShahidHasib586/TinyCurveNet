# Repository cleanup

The cleanup retains training/inference/evaluation code, original split manifests, unique checkpoint states, configuration and dependency files, tests, and paper results. Unique best/last/stage/export checkpoints are retained because their equivalence to the paper-selected weights has not been established. Original manifests are retained even when filenames are aliases, because scripts and checkpoint metadata reference them. Local absolute paths still need remapping.

Removed files remain recoverable from Git history at commit `e07dda79bb3cba7e2bd04cf9f82bb36c0be4e48f`; history has not been rewritten.

## Removed files

- `Outputs/readme.md`
- `Outputs/tinynet_output.png`
- `Resources/resources.md`
- `checkpoints/compressed/readme.md`
- `checkpoints/testTinycurvenet.pt`
- `scripts/slurm_stage3_continue_291740.out`
- `src/__pycache__/dataset.cpython-39.pyc`
- `src/__pycache__/metrics.cpython-39.pyc`
- `src/__pycache__/model.cpython-39.pyc`
- `src/__pycache__/vis.cpython-39.pyc`

Removed categories: generated caches, empty/placeholder documentation, redundant visual assets, deployment/demo utilities outside the paper reproduction workflow, and unused alternative/copied model code. Training SLURM scripts and configuration examples are retained for provenance.

`checkpoints/testTinycurvenet.pt` was byte-identical to retained `checkpoints/compressed/tinycurvenet.pt`. The removed SLURM output recorded only a failed launch; no training metrics were lost.
