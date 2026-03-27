# Client: FusionHeightNet UI (React + Vite)

Frontend for EO+SAR building intelligence visualization.

It lets you:
- choose model route (basic, advanced, compare)
- upload optical TIFF, SAR TIFF, and CSV label file
- submit one inference request or compare both models
- inspect metrics, per-building heights, and image outputs

## Tech Stack

- React 19
- Vite 8
- Native fetch API for backend communication
- Plain CSS (no UI framework dependency)

## Local Setup

## 1. Install Dependencies

From project root:

```bash
cd client
npm install
```

## 2. Configure Backend URL (Optional)

Create client/.env:

```env
VITE_API_BASE_URL=http://localhost:8000
```

If this file is not present, the app defaults to http://localhost:8000.

## 3. Run Development Server

```bash
npm run dev
```

Open the URL shown in terminal (usually http://localhost:5173).

Important: backend must be running before submitting files.

## Available Scripts

- npm run dev: start local dev server
- npm run build: production build
- npm run preview: preview built output locally
- npm run lint: run eslint checks

## UI Workflow

## 1. Select Model Route

- Basic Early Fusion (/predict/basic)
- Advanced Cross-Attention (/predict/advanced)
- Compare Both Models (calls both endpoints)

## 2. Provide Inputs

- Image ID must exactly match CSV ImageId
- Optical TIFF
- SAR TIFF
- Ground-truth CSV

## 3. Submit

- Single mode: one endpoint call
- Compare mode: both endpoints called in parallel

## 4. Review Outputs

- summary cards: status, detected buildings, MAE, RMSE, R2, Delta1, Delta2
- footprint extraction chart: IoU, F1, precision, recall
- top building height bars: predicted vs actual
- ranked table for top buildings
- EO contour visualization + semantic-refined heatmap

## Backend Contract

Expected backend endpoints:
- POST /predict/basic
- POST /predict/advanced

Expected response sections:
- data_summary
- metrics.regression_heights
- metrics.segmentation_footprints
- buildings_data
- visualizations.image_contours_b64
- visualizations.image_heatmap_b64

## Troubleshooting

## Cannot connect to backend

- verify backend is running on port 8000
- check VITE_API_BASE_URL in client/.env
- restart npm run dev after changing .env

## Submit button disabled

Button stays disabled until all are provided:
- non-empty image id
- optical file
- SAR file
- CSV file

## Error response shown in UI

The client displays backend error text directly when request fails.
Common causes:
- invalid image_id
- unsupported/corrupt TIFF
- CSV missing required columns

## Build for Deployment

```bash
npm run build
```

Generated static files are placed in client/dist.
