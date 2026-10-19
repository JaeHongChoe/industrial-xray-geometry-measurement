# V7 HV Model Export

Trained on the full HV dataset for local inference/export.

- Stage 1 YOLO weight: `yolo_best.pt`
- Stage 2 triangle pose weight: `tip_model.pth`
- Source model code copy: `run_coarse_fine_v7.py`
- Visualization generator: `make_v7_export_visualizations.py`
- Training data: `vhole_v7_data_hv`
- Images: 49
- V-holes: 98

## Visualizations

Generated result images are included in `visualizations/`.

- `visualizations/full_overlay_jpg/`: compressed full-image overlays
- `visualizations/zoom_review/`: zoomed V-hole panels for easier inspection
- `visualizations/summary_sheets/final_model_zoom_contact_sheet.jpg`: compact overview sheet
- `visualizations/three_way_review_sheets/`: HV / RawPreGray / combined comparison sheets
- `visualizations/final_model_visualization_results.json`: prediction/error values

Final exported model on the HV training set:

- Images within 5 px: 49/49
- Tips: 98
- Mean error: 1.22 px
- Median error: 1.04 px
- Max error: 3.45 px

Use the same constants in `run_coarse_fine_v7.py` when loading the weights.
