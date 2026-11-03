# DTU Scan65 sparse-view reconstruction

This is a standalone implementation. It does not import code or weights from other local reconstruction projects. Inputs are read-only under `../DTU/scan65`; generated files live under this directory's `outputs/`.

## Create a fresh conda environment

```bash
cd DTU/scan65_reconstruction
conda env create --prefix ./.conda -f environment.yml
conda activate ./.conda
```

The environment installs VGGT from its public upstream repository and downloads its public pretrained checkpoint on the first inference run. The input data stays untouched.

## Run in validated stages

```bash
python scan65.py prepare
python scan65.py infer
python scan65.py points
python scan65.py mesh
python scan65.py evaluate
```

Run `python scan65.py --help` for path, model input size, confidence, depth and TSDF settings. `prepare` inspects image/camera mapping, emits a five-view contact sheet and camera distribution, and must be inspected before `infer`. The `infer` stage saves per-view depth and confidence arrays and visualization, plus predicted intrinsics/extrinsics. `points` unprojects confidence-filtered predictions into the VGGT coordinate frame. `mesh` filters points, integrates confidence-masked depth with Open3D scalable TSDF, extracts a surface, cleans it, and writes topology statistics. `evaluate` records which evaluation outputs are possible from the local data and marks unavailable ground-truth or appearance metrics as not run.

## Dataset conventions and limitations

The local dataset contains images `000000.png`–`000048.png`, `cameras.npz`, and masks `000.png`–`048.png`. The requested five view IDs map to both image stems and camera archive indices. Camera inspection decomposes `world_mat_i @ scale_mat_i`; the camera archive is used for dataset inspection only. VGGT predicted cameras drive reconstruction. VGGT uses square resized inputs and exports intrinsics at that resolution.

Open3D's scalable TSDF class does not serialize its internal sparse volume. `phase5_mesh/tsdf_volume.npz` therefore records the exact fusion settings, camera matrices, and views needed to reproduce fusion from the preserved phase2 predictions; it is not a voxel-by-voxel dump. Sparse five-view depth may have scale or pose inconsistencies. Inspect the point clouds and mesh before using them downstream. Ground-truth geometry is not present in this Scan65 folder, so geometry metrics are not fabricated. Novel-view appearance evaluation is outside this geometry deliverable and needs an appearance model.

## Outputs

`outputs/phase1` contains dataset mapping, selected views and camera distributions. `phase2_vggt` preserves model outputs. `phase3_points` holds per-view and fused PLY clouds; `phase4_clean_points` contains filtered normals. `phase5_mesh` stores the initial TSDF mesh and fusion manifest. `phase6_clean_mesh` contains PLY/OBJ and mesh statistics.
