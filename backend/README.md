# Backend: FastAPI Inference Service

FastAPI backend for EO+SAR building footprint extraction and height estimation.

The service loads two pretrained models:
- WinningFusionUNet (basic early-fusion baseline)
- FusionHeightNet (advanced multi-level cross-fusion architecture)

It accepts TIFF + CSV uploads, performs inference and evaluation, and returns metrics plus visualizations for the frontend.

## What This Service Does

- Loads model weights from local .pth files
- Parses SpaceNet-style CSV polygons and heights into GT masks
- Runs inference on CPU
- Computes:
  - segmentation metrics: IoU, F1, precision, recall
  - regression metrics: MAE, RMSE, R2
  - instance-level building matching + ranking
- Returns contour and heatmap images as base64 PNG

## Required Files in This Folder

- app.py
- model_architecture.py
- basic_model.pth
- fusion_height_net_best.pth
- requirements.txt

If model files are missing, startup will fail.

## Python Setup

## Windows PowerShell

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If torch install fails or is very slow:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## macOS/Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Service

From backend/ directory:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Expected startup log:
- Loading AI Models onto CPU...
- Both Models Ready!

## API Endpoints

Base URL: http://localhost:8000

## POST /predict/basic

Runs the basic early-fusion baseline.

Form-data fields:
- image_id: string (required)
- optical_file: file (required)
- sar_file: file (required)
- csv_file: file (required)

## POST /predict/advanced

Runs FusionHeightNet advanced cross-attention model.

Form-data fields are identical to /predict/basic.

## Response Payload

Both routes return:

```json
{
  "status": "success",
  "data_summary": {
    "buildings_detected": 0,
    "max_height_detected_m": 0
  },
  "metrics": {
    "regression_heights": {
      "mae": 0,
      "rmse": 0,
      "r2_score": 0
    },
    "segmentation_footprints": {
      "iou": 0,
      "f1_score": 0,
      "precision": 0,
      "recall": 0
    }
  },
  "buildings_data": [
    {
      "predicted_height_m": 0,
      "actual_height_m": 0,
      "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
      "area_pixels": 0,
      "rank": 1
    }
  ],
  "visualizations": {
    "image_contours_b64": "data:image/png;base64,...",
    "image_heatmap_b64": "data:image/png;base64,..."
  }
}
```

## Input Expectations

## CSV

Required columns:
- ImageId
- PolygonWKT_Pix
- Mean_Building_Height

Rows are filtered by ImageId == image_id.

## TIFF Processing

- TIFFs are read using tifffile
- Data is resized to 512x512 for model input/output alignment
- Optical channels are normalized to [0, 1]
- SAR channels are normalized by max value

## Notes on Advanced Route

- Uses separate EO and SAR encoders
- Performs multi-level cross-fusion via cross-attention
- Height branch output is mapped with an exponential transform using ADV_MAX_HEIGHT

## Development Tips

- Open API docs at: http://localhost:8000/docs
- CORS currently allows all origins for local development
- For production, restrict allow_origins and remove wildcard settings

## Common Issues

## RuntimeError loading checkpoint

Cause:
- incorrect model file
- architecture mismatch

Fix:
- verify checkpoint corresponds to the matching architecture in model_architecture.py

## Uvicorn cannot import app

Cause:
- wrong working directory

Fix:
- run command from backend directory where app.py is located

## 400 error on prediction

Cause examples:
- malformed CSV
- missing required form field
- invalid TIFF
- image_id mismatch with CSV ImageId

Fix:
- validate files and ensure image_id exactly matches CSV content
