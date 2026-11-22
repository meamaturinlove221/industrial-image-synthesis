# Industrial Image Synthesis

Standalone source for the synthetic 2D texture and layout task generator (module version **3.0.0**). This is not the full ManiTwin 2.1.2 workbench or the optional Blender-based 3D extension.

The generator creates matched A/B image pairs from procedural shapes, textures, and backgrounds. Members of a pair share scene settings; the selected texture or layout relationship distinguishes them. The code provides previews, group metadata, reproducible plans, resumable generation, and a pixel audit. It does not train or evaluate a classifier.

## Install and run

Use Python 3.10–3.13. The desktop interface also needs Tkinter (included with the standard Windows Python installer when Tcl/Tk support is enabled). Pillow and NumPy are declared in the package metadata.

From the repository root, install into an activated virtual environment using `python -m pip install .`. On Windows the launchers also recognize a local `.venv`. Run `启动纹理差异任务集.cmd` on Windows, or `python ui/texture_task_app.py` in a desktop session with Tkinter.

Start with a preview before generating a dataset:

```text
python tools/texture_task_preview.py --output-dir ./output/preview --recipe layout
python tools/texture_task_generate.py --output-root ./output --dataset-name demo --recipe layout --train-groups 4 --val-groups 4 --workers 1
python tools/texture_task_audit.py --dataset-root ./output/demo
```

The full default is 3,200 training groups and 640 validation groups (16 images per group). Generation writes locally below the selected output directory. The audit defaults to pixel checks; `--quick` checks structure only.

## Release scope

This source-only release omits the main workbench, 3D relation-task extension, asset catalog and assets, real industrial samples and sample-derived configuration, experiment reports and screenshots, models, and generated outputs. Those capabilities and artifacts are not included or validated by this release. No external data is needed for the synthetic 2D generator.

No license is included. Public visibility alone does not grant permission to redistribute or reuse this source.
