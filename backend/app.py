from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import io
import base64
import tifffile
import pandas as pd
from shapely import wkt
import scipy.ndimage as ndimage

# Import your model architectures
from model_architecture import WinningFusionUNet, FusionHeightNet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Constants
BASIC_MAX_HEIGHT = 50.0  
ADV_MAX_HEIGHT = 100.0   

print("Loading AI Models onto CPU...")
device = torch.device('cpu')

# 1. Load Basic Model
basic_model = WinningFusionUNet().to(device)
basic_checkpoint = torch.load("basic_model.pth", map_location=device) # UPDATE FILENAME
basic_model.load_state_dict(basic_checkpoint['model_state_dict'] if 'model_state_dict' in basic_checkpoint else basic_checkpoint)
basic_model.eval()

# 2. Load Advanced Model
advanced_model = FusionHeightNet().to(device)
adv_checkpoint = torch.load("fusion_height_net_best.pth", map_location=device) # UPDATE FILENAME
advanced_model.load_state_dict(adv_checkpoint['model_state_dict'] if 'model_state_dict' in adv_checkpoint else adv_checkpoint)
advanced_model.eval()

print("Both Models Ready!")


# ==========================================
# DATA PARSING & EVALUATION ENGINE
# ==========================================
def generate_gt_masks(csv_bytes, image_id):
    """Parses the SpaceNet6 CSV to create the exact Ground Truth masks for evaluation."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    group = df[df['ImageId'] == image_id]
    
    # Draw at native 900x900 resolution first
    mask_footprint = np.zeros((900, 900), dtype=np.float32)
    mask_height = np.zeros((900, 900), dtype=np.float32)
    
    for _, row in group.iterrows():
        poly_wkt = row['PolygonWKT_Pix']
        height = row['Mean_Building_Height']
        
        if pd.isna(poly_wkt) or poly_wkt.strip() == 'POLYGON EMPTY': 
            continue
        height = 0.0 if pd.isna(height) else max(0.0, float(height))
        
        try:
            poly = wkt.loads(poly_wkt)
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask_footprint, [coords], 1.0)
            cv2.fillPoly(mask_height, [coords], height)
        except Exception:
            pass
            
    # Resize to 512x512 safely to match model output
    mask_footprint = cv2.resize(mask_footprint, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask_height = cv2.resize(mask_height, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    return mask_footprint, mask_height

def calculate_real_metrics(pred_prob, pred_height, gt_mask, gt_height, threshold=0.5):
    """Evaluates the predictions against the actual Ground Truth data."""
    pred_bin = (pred_prob > threshold).astype(np.uint8)
    gt_bin = gt_mask.astype(np.uint8)

    # 1. Pixel-Level Segmentation Metrics
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = intersection / (union + 1e-8)
    
    precision = intersection / (pred_bin.sum() + 1e-8)
    recall = intersection / (gt_bin.sum() + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # 2. Pixel-Level Regression Metrics (Evaluated ONLY where real buildings exist)
    valid_pixels = gt_bin > 0
    if valid_pixels.sum() > 0:
        mae = np.mean(np.abs(pred_height[valid_pixels] - gt_height[valid_pixels]))
        rmse = np.sqrt(np.mean((pred_height[valid_pixels] - gt_height[valid_pixels])**2))
        
        ss_res = np.sum((gt_height[valid_pixels] - pred_height[valid_pixels])**2)
        ss_tot = np.sum((gt_height[valid_pixels] - np.mean(gt_height[valid_pixels]))**2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    else:
        mae, rmse, r2_score = 0.0, 0.0, 0.0

    # 3. Instance-Level Matching (Connecting predicted buildings to real ones)
    pred_labeled, pred_num = ndimage.label(pred_bin)
    gt_labeled, _ = ndimage.label(gt_bin)

    response_buildings = []
    
    for i in range(1, pred_num + 1):
        pred_pixels = (pred_labeled == i)
        
        # Skip tiny noise blips
        if pred_pixels.sum() < 10: 
            continue
            
        pred_h = float(np.mean(pred_height[pred_pixels]))
        
        # Find overlapping GT building
        overlaps = gt_labeled[pred_pixels]
        overlaps = overlaps[overlaps > 0]
        
        actual_h = 0.0
        if len(overlaps) > 0:
            match_id = np.bincount(overlaps).argmax()
            actual_h = float(np.mean(gt_height[gt_labeled == match_id]))
            
        # Get Bounding Box for Frontend UI
        y_coords, x_coords = np.where(pred_pixels)
        x, y = int(x_coords.min()), int(y_coords.min())
        w, h = int(x_coords.max() - x), int(y_coords.max() - y)
        
        response_buildings.append({
            "predicted_height_m": round(pred_h, 2),
            "actual_height_m": round(actual_h, 2),
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "area_pixels": int(pred_pixels.sum())
        })
        
    # Sort and rank buildings by height
    response_buildings = sorted(response_buildings, key=lambda b: b['predicted_height_m'], reverse=True)
    for rank, b in enumerate(response_buildings):
        b['rank'] = rank + 1
        
    metrics = {
        "regression_heights": {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2), "r2_score": round(float(r2_score), 2)},
        "segmentation_footprints": {
            "iou": round(float(iou), 2), "f1_score": round(float(f1_score), 2),
            "precision": round(float(precision), 2), "recall": round(float(recall), 2)
        }
    }
    
    return response_buildings, metrics, pred_bin

def generate_frontend_payload(opt_array, pred_prob_np, pred_height_np, gt_mask_np, gt_height_np):
    """Packages the REAL data, visuals, and metrics into a clean JSON response."""
    response_buildings, metrics, final_binary_mask = calculate_real_metrics(pred_prob_np, pred_height_np, gt_mask_np, gt_height_np)

    # Generate Visuals
    opt_display = (opt_array[..., :3] * 255).astype(np.uint8).copy()
    contours, _ = cv2.findContours(final_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(opt_display, contours, -1, (0, 255, 0), 2) 
    
    _, buffer_opt = cv2.imencode('.png', cv2.cvtColor(opt_display, cv2.COLOR_RGB2BGR))
    opt_b64 = base64.b64encode(buffer_opt).decode('utf-8')

    max_h = np.max(pred_height_np) if np.max(pred_height_np) > 0 else 1
    height_normalized = (pred_height_np / max_h * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(height_normalized, cv2.COLORMAP_PLASMA)
    heatmap[final_binary_mask == 0] = [0, 0, 0] 
    
    _, buffer_hm = cv2.imencode('.png', heatmap)
    hm_b64 = base64.b64encode(buffer_hm).decode('utf-8')

    return {
        "status": "success",
        "data_summary": {
            "buildings_detected": len(response_buildings),
            "max_height_detected_m": round(float(np.max(pred_height_np)), 2) if len(response_buildings) > 0 else 0
        },
        "metrics": metrics,
        "buildings_data": response_buildings[:15], 
        "visualizations": {
            "image_contours_b64": f"data:image/png;base64,{opt_b64}",
            "image_heatmap_b64": f"data:image/png;base64,{hm_b64}"
        }
    }


# ==========================================
# ROUTE 1: BASIC EARLY FUSION MODEL
# ==========================================
@app.post("/predict/basic")
async def predict_basic(
    image_id: str = Form(...),
    optical_file: UploadFile = File(...), 
    sar_file: UploadFile = File(...),
    csv_file: UploadFile = File(...)
):
    try:
        opt_bytes, sar_bytes, csv_bytes = await optical_file.read(), await sar_file.read(), await csv_file.read()

        # Parse Ground Truth
        gt_mask_np, gt_height_np = generate_gt_masks(csv_bytes, image_id)

        # Load and reshape TIFFs
        opt_array = tifffile.imread(io.BytesIO(opt_bytes))
        if len(opt_array.shape) == 3 and opt_array.shape[0] < opt_array.shape[2]: opt_array = np.transpose(opt_array, (1, 2, 0))
        opt_array = cv2.resize(opt_array, (512, 512)).astype(np.float32) / (65535.0 if opt_array.max() > 255 else 255.0)
        
        sar_array = tifffile.imread(io.BytesIO(sar_bytes))
        if len(sar_array.shape) == 3 and sar_array.shape[0] < sar_array.shape[2]: sar_array = np.transpose(sar_array, (1, 2, 0))
        sar_array = cv2.resize(sar_array, (512, 512)).astype(np.float32) / (sar_array.max() if sar_array.max() > 0 else 1.0)
        
        combined_input = np.concatenate([opt_array, sar_array], axis=-1)
        input_tensor = torch.tensor(combined_input).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_footprint, pred_height = basic_model(input_tensor)
        
        pred_prob_np = pred_footprint.squeeze().cpu().numpy()
        pred_height_np = (pred_height.squeeze().cpu().numpy()) * BASIC_MAX_HEIGHT

        return generate_frontend_payload(opt_array, pred_prob_np, pred_height_np, gt_mask_np, gt_height_np)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================
# ROUTE 2: ADVANCED CROSS-ATTENTION MODEL
# ==========================================
@app.post("/predict/advanced")
async def predict_advanced(
    image_id: str = Form(...),
    optical_file: UploadFile = File(...), 
    sar_file: UploadFile = File(...),
    csv_file: UploadFile = File(...)
):
    try:
        opt_bytes, sar_bytes, csv_bytes = await optical_file.read(), await sar_file.read(), await csv_file.read()

        # Parse Ground Truth
        gt_mask_np, gt_height_np = generate_gt_masks(csv_bytes, image_id)

        opt_array = tifffile.imread(io.BytesIO(opt_bytes))
        if len(opt_array.shape) == 3 and opt_array.shape[0] < opt_array.shape[2]: opt_array = np.transpose(opt_array, (1, 2, 0))
        opt_array = cv2.resize(opt_array, (512, 512))
        opt_array_3ch = opt_array[..., :3].astype(np.float32) / (65535.0 if opt_array.max() > 255 else 255.0)
        
        sar_array = tifffile.imread(io.BytesIO(sar_bytes))
        if len(sar_array.shape) == 3 and sar_array.shape[0] < sar_array.shape[2]: sar_array = np.transpose(sar_array, (1, 2, 0))
        sar_array = cv2.resize(sar_array, (512, 512))
        
        if len(sar_array.shape) == 2:
            sar_array_3ch = np.stack((sar_array,)*3, axis=-1)
        else:
            sar_array_3ch = sar_array[..., :3]
        sar_array_3ch = sar_array_3ch.astype(np.float32) / (sar_array_3ch.max() if sar_array_3ch.max() > 0 else 1.0)

        opt_tensor = torch.tensor(opt_array_3ch).permute(2, 0, 1).unsqueeze(0).to(device)
        sar_tensor = torch.tensor(sar_array_3ch).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_foot, pred_height = advanced_model(opt_tensor, sar_tensor)
            
        pred_prob_np = F.softmax(pred_foot, dim=1)[0, 1, :, :].cpu().numpy()
        pred_norm = pred_height[0, 0, :, :].cpu().numpy()
        pred_height_np = np.expm1(pred_norm * np.log1p(ADV_MAX_HEIGHT))

        return generate_frontend_payload(opt_array_3ch, pred_prob_np, pred_height_np, gt_mask_np, gt_height_np)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))