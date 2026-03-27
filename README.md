## Cosmix-2026: FusionHeightNet EO+SAR Building Intelligence

End-to-end demo for building footprint extraction and height estimation from multi-source remote sensing data.

This documentation is based on the current local filesystem state of this workspace (including local-only/untracked files), not git commit history.

The app contains:
- A FastAPI backend with two models:
	- Early-fusion baseline (single-stream, 7-channel input)
	- FusionHeightNet (multi-level cross-fusion + multi-task joint learning + semantic refinement)
- A React + Vite frontend to upload EO image, SAR image, and CSV labels, then visualize:
	- segmentation footprints
	- height heatmap
	- per-building predicted vs actual heights
	- comparative metrics across both models

## System Architecture

- Frontend (default: http://localhost:5173)
	- Uploads files and image id via multipart/form-data
	- Calls backend endpoints:
		- POST /predict/basic
		- POST /predict/advanced
- Backend (default: http://localhost:8000)
	- Loads pretrained weights from local .pth files on startup
	- Runs inference on CPU
	- Parses SpaceNet-style CSV labels to generate GT masks
	- Computes regression + segmentation metrics
	- Returns JSON payload including base64 visualizations

## Repository Layout

- backend/
	- app.py: FastAPI server, preprocessing, inference, metrics, response packaging
	- model_architecture.py: WinningFusionUNet and FusionHeightNet model definitions
	- basic_model.pth: baseline model checkpoint
	- fusion_height_net_best.pth: advanced model checkpoint
	- requirements.txt: Python dependencies
	- README.md: backend-specific setup and API reference
	- .gitignore
- client/
	- src/App.jsx: upload flow, model selection, comparison view, metric rendering
	- src/App.css, src/index.css: UI styles
	- src/main.jsx: React entrypoint
	- src/assets/: static UI assets (hero.png, react.svg, vite.svg)
	- public/: favicon.svg, icons.svg
	- package.json: Node scripts and dependencies
	- package-lock.json
	- vite.config.js, eslint.config.js, index.html
	- README.md: frontend-specific setup and usage
	- .gitignore
- notebooks (.ipynb): experimentation/training artifacts

Local generated/dependency folders commonly present but not required to commit:
- backend/venv/
- backend/__pycache__/
- client/node_modules/
- client/dist/

## Prerequisites

## 1. Required Tools

- Python 3.10 or 3.11 recommended
- Node.js 20.19+ (or 22.12+) and npm
- Git

## 2. Data Inputs

For each inference request, you need:
- image_id string matching ImageId in CSV
- optical_file: EO TIFF image (.tif/.tiff)
- sar_file: SAR TIFF image (.tif/.tiff)
- csv_file: CSV containing at least:
	- ImageId
	- PolygonWKT_Pix
	- Mean_Building_Height

## 3. Model Weights

Ensure these files exist in backend/:
- basic_model.pth
- fusion_height_net_best.pth

Without them, backend startup will fail.

## Local Setup (Complete Walkthrough)

## Step 1: Clone and Enter Project

PowerShell:

```powershell
git clone <your-repo-url>
cd Cosmix-2026
```

Bash:

```bash
git clone <your-repo-url>
cd Cosmix-2026
```

## Step 2: Backend Setup (FastAPI)

### Windows PowerShell

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If torch install is slow/fails, install CPU wheel explicitly, then re-run requirements:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### macOS/Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Start Backend

From backend/:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Expected startup logs include model load messages:
- Loading AI Models onto CPU...
- Both Models Ready!

## Step 3: Frontend Setup (React + Vite)

Open a second terminal at project root:

```bash
cd client
npm install
```

Optional environment variable file (client/.env):

```env
VITE_API_BASE_URL=http://localhost:8000
```

If omitted, frontend defaults to http://localhost:8000.

Start frontend:

```bash
npm run dev
```

Open the printed local URL (usually http://localhost:5173).

## Step 4: Run Inference from UI

In the web app:
- Select Model Route:
	- Basic Early Fusion (/predict/basic)
	- Advanced Cross-Attention (/predict/advanced)
	- Compare Both Models
- Enter Image ID exactly matching CSV ImageId.
- Upload optical TIFF, SAR TIFF, and ground-truth CSV.
- Submit.

The UI returns:
- Summary cards: MAE, RMSE, R2, Delta1, Delta2, detected buildings
- Footprint extraction bars: IoU, F1, precision, recall
- Height comparison bars for top buildings
- Table with predicted vs actual heights
- EO contour overlay + semantic-refined heatmap

## API Reference

Base URL: http://localhost:8000

## POST /predict/basic

Baseline early-fusion path using concatenated EO+SAR channels.

Form fields:
- image_id (string, required)
- optical_file (file, required)
- sar_file (file, required)
- csv_file (file, required)

## POST /predict/advanced

FusionHeightNet route using dual encoders + cross-attention fusion.

Form fields are identical to /predict/basic.

## Response Shape (Both Endpoints)

```json
{
	"status": "success",
	"data_summary": {
		"buildings_detected": 12,
		"max_height_detected_m": 38.42
	},
	"metrics": {
		"regression_heights": {
			"mae": 3.21,
			"rmse": 4.9,
			"r2_score": 0.77
		},
		"segmentation_footprints": {
			"iou": 0.62,
			"f1_score": 0.76,
			"precision": 0.79,
			"recall": 0.73
		}
	},
	"buildings_data": [
		{
			"predicted_height_m": 29.4,
			"actual_height_m": 31.1,
			"bbox": {"x": 100, "y": 120, "w": 42, "h": 30},
			"area_pixels": 905,
			"rank": 1
		}
	],
	"visualizations": {
		"image_contours_b64": "data:image/png;base64,...",
		"image_heatmap_b64": "data:image/png;base64,..."
	}
}
```

## Verify Backend Quickly (Optional)

Swagger UI:
- http://localhost:8000/docs

Health check style quick test:
- If /docs opens and model startup logs appear, service is running.

## Troubleshooting

## 1. Backend crashes at startup

Possible causes:
- Missing .pth files in backend/
- Incompatible torch/python versions
- Running command from wrong directory

Fix:
- Confirm files exist in backend/
- Use Python 3.10/3.11
- Run uvicorn from backend directory

## 2. CORS or connection errors in frontend

Symptoms:
- Browser shows network error
- UI shows unable to connect to backend

Fix:
- Ensure backend is running at port 8000
- Set client/.env with correct VITE_API_BASE_URL
- Restart npm run dev after .env changes

## 3. Empty predictions or poor metrics

Potential reasons:
- image_id does not match CSV ImageId
- wrong optical/SAR pair
- malformed CSV columns

Fix:
- Validate exact ImageId text
- Verify CSV has required columns
- Use aligned optical and SAR scenes for same tile

## 4. TIFF parsing issues

The backend expects TIFF arrays and internally resizes to 512x512. Corrupted TIFFs or unusual channel layouts may fail.

## Development Notes

- Backend currently runs on CPU by default.
- CORS is open to all origins for local development.
- Compare mode in UI calls both endpoints in parallel.
- Frontend computes extra comparison metrics (Delta1, Delta2, within-25% ratio) from returned building rows.

## Useful Commands

From backend/:

```bash
uvicorn app:app --reload
```

From client/:

```bash
npm run dev
npm run build
npm run preview
```

## Detailed Folder Docs

- See backend/README.md for backend-only setup and API details.
- See client/README.md for frontend-only setup and environment configuration.

