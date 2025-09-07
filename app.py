import os
import io
import base64
import hashlib
import json
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HEATMAP_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmaps')
app.config['RESULTS_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load the YOLOv8 classification model and detection model
# You can change these to other sizes like 'yolov8s-cls.pt' or 'yolov8s.pt'
model = YOLO('yolov8n-cls.pt')
detect_model = YOLO('yolov8n.pt')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def result_path(result_id: str) -> str:
    return os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")


def _find_last_conv2d(module: nn.Module) -> nn.Module:
    """Find the last nn.Conv2d layer in a model for Grad-CAM target."""
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def generate_gradcam(
    image_path: str,
    class_idx: int | None = None,
    algorithm: str = 'gradcam',
    colormap_name: str = 'jet',
    overlay_alpha: int = 110,
) -> Image.Image:
    """Generate Grad-CAM heatmap overlay for a classification model.

    Returns a PIL Image (RGBA) with heatmap blended on top of the original.
    """
    # Make sure gradients are enabled for Grad-CAM
    torch.set_grad_enabled(True)
    model.model.eval()
    for p in model.model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    # Choose the last Conv2d layer dynamically
    target_layer = _find_last_conv2d(model.model)
    if target_layer is None:
        raise RuntimeError('No Conv2d layer found for Grad-CAM')

    activations = None
    gradients = None

    def forward_hook(_, __, output):
        nonlocal activations
        # Clone to avoid holding a view that might be modified in-place elsewhere
        activations = output.clone().detach()

    def backward_hook(_, grad_input, grad_output):
        nonlocal gradients
        # Clone to avoid view + inplace issues from autograd hooks
        gradients = grad_output[0].clone().detach()

    # Register hooks
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Load image
        pil = Image.open(image_path).convert('RGB')

        # Ultralytics' classify models accept PIL path directly, but we need tensors for hooks/grad
        # We'll follow standard preprocessing: resize shorter side to 224 with center-crop
        pil_resized = pil.copy()
        pil_resized.thumbnail((256, 256))
        # Create square by padding/cropping center to 224
        pil_resized = pil_resized.resize((224, 224))
        img = np.array(pil_resized).astype(np.float32) / 255.0
        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1x3x224x224

        device = next(model.model.parameters()).device
        img_t = img_t.to(device)
        # We need gradients w.r.t. feature maps and possibly input
        img_t.requires_grad_(True)

        # Forward pass with gradients explicitly enabled
        with torch.enable_grad():
            logits = model.model(img_t)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        # Select class
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()

        # Backward to get gradients
        model.model.zero_grad(set_to_none=True)
        # Avoid creating higher-order graphs and don't retain the graph since we won't reuse it
        score.backward(retain_graph=False, create_graph=False)

        if activations is None or gradients is None:
            raise RuntimeError('Failed to capture activations/gradients for Grad-CAM')

        # Compute CAM according to selected algorithm
        if algorithm.lower() in ['gradcam', 'cam', 'gc']:
            # Grad-CAM: weights = GAP over gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # BxCx1x1
            cam = F.relu(torch.sum(weights * activations, dim=1, keepdim=False))  # BxHxW
        elif algorithm.lower() in ['gradcam++', 'gradcampp', 'gcpp']:
            # Grad-CAM++ implementation
            grads = gradients
            grads2 = grads * grads
            grads3 = grads2 * grads
            sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
            eps = 1e-6
            denom = (2.0 * grads2 + activations * torch.sum(grads3, dim=(2, 3), keepdim=True) + eps)
            alphas = grads2 / denom
            # Positive gradients only
            positive_grads = F.relu(grads)
            weights = torch.sum(alphas * positive_grads, dim=(2, 3), keepdim=True)  # BxCx1x1
            cam = F.relu(torch.sum(weights * activations, dim=1, keepdim=False))  # BxHxW
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        cam = cam[0]

        # Normalize cam to [0,1] (out-of-place to avoid in-place on potential views)
        cam_min = cam.min()
        cam = cam - cam_min
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Resize cam to original image size
        cam_np = cam.cpu().numpy()
        cam_img = Image.fromarray(np.uint8(cam_np * 255)).resize(pil.size, resample=Image.BILINEAR)

        # Apply selectable colormap and blend
        import matplotlib.cm as cm
        # validate colormap
        valid_maps = {'jet','magma','viridis','plasma','inferno','turbo','cividis','cubehelix'}
        if colormap_name not in valid_maps:
            colormap_name = 'jet'
        colormap = cm.get_cmap(colormap_name)
        colored = (colormap(np.array(cam_img) / 255.0) * 255).astype(np.uint8)  # RGBA
        heatmap = Image.fromarray(colored).convert('RGBA')

        overlay = Image.new('RGBA', pil.size)
        overlay = Image.alpha_composite(overlay, pil.convert('RGBA'))
        # adjust alpha of heatmap
        alpha = int(max(0, min(255, overlay_alpha)))
        heatmap.putalpha(alpha)
        blended = Image.alpha_composite(overlay, heatmap)
        return blended
    finally:
        # Remove hooks
        fh.remove()
        bh.remove()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # compute hash and adopt as stable id
        file_hash = sha256_of_file(filepath)
        result_id = file_hash[:32]
        # rename file to include hash if not already to avoid collisions
        ext = os.path.splitext(filename)[1]
        stored_name = f"{result_id}{ext}"
        stored_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
        if not os.path.exists(stored_path):
            try:
                os.replace(filepath, stored_path)
            except Exception:
                stored_path = filepath  # fallback keep original
        else:
            # remove the just-saved duplicate if different
            try:
                if filepath != stored_path:
                    os.remove(filepath)
            except Exception:
                pass
        file_url = f"/{stored_path}"
        
        try:
            # If cached, return cached
            rp = result_path(result_id)
            if os.path.exists(rp):
                with open(rp, 'r') as jf:
                    cached = json.load(jf)
                return jsonify({
                    'status': 'cached',
                    **cached
                })

            # Run inference for classification
            results = model.predict(source=stored_path, verbose=False)

            # Process classification results: get top-5 classes with confidences
            r = results[0]
            names = model.names
            predictions = []
            if hasattr(r, 'probs') and r.probs is not None:
                top5_idxs = (r.probs.top5 if hasattr(r.probs, 'top5') else [])
                top5_confs = (r.probs.top5conf if hasattr(r.probs, 'top5conf') else [])
                for idx, conf in zip(top5_idxs, top5_confs):
                    predictions.append({
                        'class': names[int(idx)],
                        'class_index': int(idx),
                        'confidence': round(float(conf) * 100, 2)
                    })

            payload = {
                'id': result_id,
                'filename': stored_name,
                'file_url': file_url,
                'predictions': predictions,
                'created_at': int(time.time()),
                'model': 'yolov8n-cls.pt'
            }
            # persist to disk
            with open(result_path(result_id), 'w') as jf:
                json.dump(payload, jf)

            return jsonify({'status': 'success', **payload})

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/explain', methods=['POST'])
def explain():
    """Generate and return Grad-CAM heatmap for a given uploaded image filename and class index."""
    data = request.get_json(silent=True) or {}
    filename = data.get('filename')
    class_index = data.get('class_index')
    algorithm = data.get('algorithm', 'gradcam')
    colormap = data.get('colormap', 'jet')
    overlay_alpha = int(data.get('overlay_alpha', 110))

    if not filename:
        return jsonify({'error': 'filename is required'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'error': 'file not found'}), 404

    try:
        heat_img = generate_gradcam(
            filepath,
            class_idx=class_index,
            algorithm=algorithm,
            colormap_name=colormap,
            overlay_alpha=overlay_alpha,
        )
        # Save and also return base64 for convenience
        heat_name = os.path.splitext(os.path.basename(filename))[0]
        out_name = f"{heat_name}_cam_{class_index if class_index is not None else 'top1'}.png"
        out_path = os.path.join(app.config['HEATMAP_FOLDER'], out_name)
        heat_img.save(out_path)

        # Encode base64
        buf = io.BytesIO()
        heat_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'filename': filename,
            'heatmap_url': f"/{out_path}",
            'heatmap_base64': f"data:image/png;base64,{b64}",
            'algorithm': algorithm,
            'colormap': colormap
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    """Run YOLOv8 object detection and return bounding boxes.

    Response format:
    {
      status: 'success',
      filename: <stored filename>,
      predictions: [
        { class: str, class_index: int, confidence: float(0-100), box: [x1,y1,x2,y2] }, ...
      ]
    }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Run detection
            results = detect_model.predict(source=filepath, verbose=False)
            names = detect_model.names
            predictions = []
            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()  # absolute coordinates
                    predictions.append({
                        'class': names[cls],
                        'class_index': cls,
                        'confidence': round(conf * 100, 2),
                        'box': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                    })

            return jsonify({
                'status': 'success',
                'filename': filename,
                'predictions': predictions
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/result/<result_id>', methods=['GET'])
def get_result(result_id: str):
    """Return a previously saved prediction payload by ID."""
    rp = result_path(secure_filename(result_id))
    if not os.path.exists(rp):
        return jsonify({'error': 'result not found'}), 404
    try:
        with open(rp, 'r') as jf:
            data = json.load(jf)
        return jsonify({'status': 'success', **data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
