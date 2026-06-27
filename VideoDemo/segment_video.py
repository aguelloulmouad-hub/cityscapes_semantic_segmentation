"""
Semantic Segmentation Video Demo
=================================
Uses DeepLabV3+ (ResNet50) trained on Cityscapes to perform semantic segmentation
on a sequence of frames and produces a side-by-side video:
  - Left:  original frame
  - Right: segmentation overlay
Output: videodemo.mp4 at 60 FPS, 1280×720

Usage:
    python segment_video.py                 (uses defaults)
    python segment_video.py --gpu           (force GPU)
    python segment_video.py --cpu           (force CPU)

Requirements:
    pip install torch torchvision segmentation-models-pytorch opencv-python numpy
"""

import os
import sys
import glob
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import segmentation_models_pytorch as smp

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# Cityscapes colour palette (20 classes: 0=background/void + 19 semantic)
# Index 0 is treated as "unlabeled / void"
# ──────────────────────────────────────────────────────────────────────────────
CITYSCAPES_COLORS = np.array([
    [  0,   0,   0],   # 0  unlabeled / void
    [128,  64, 128],   # 1  road
    [244,  35, 232],   # 2  sidewalk
    [ 70,  70,  70],   # 3  building
    [102, 102, 156],   # 4  wall
    [190, 153, 153],   # 5  fence
    [153, 153, 153],   # 6  pole
    [250, 170,  30],   # 7  traffic light
    [220, 220,   0],   # 8  traffic sign
    [107, 142,  35],   # 9  vegetation
    [152, 251, 152],   # 10 terrain
    [ 70, 130, 180],   # 11 sky
    [220,  20,  60],   # 12 person
    [255,   0,   0],   # 13 rider
    [  0,   0, 142],   # 14 car
    [  0,   0,  70],   # 15 truck
    [  0,  60, 100],   # 16 bus
    [  0,  80, 100],   # 17 train
    [  0,   0, 230],   # 18 motorcycle
    [119,  11,  32],   # 19 bicycle
], dtype=np.uint8)


def decode_segmentation(pred: np.ndarray) -> np.ndarray:
    """Convert a HxW class-index map to an HxW×3 RGB colour image."""
    return CITYSCAPES_COLORS[pred]


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing (ImageNet normalization)
# ──────────────────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.ToTensor(),                     # HWC uint8 → CHW float [0,1]
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DeepLabV3+ Segmentation Video Demo")
    parser.add_argument("--model", type=str,
                        default=os.path.join(SCRIPT_DIR, "best_deeplabv3plus_resnet50.pth"),
                        help="Path to the .pth checkpoint")
    parser.add_argument("--input-dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "stuttgart_00"),
                        help="Directory containing frame images")
    parser.add_argument("--output", type=str,
                        default=os.path.join(SCRIPT_DIR, "videodemo.mp4"),
                        help="Output video path")
    parser.add_argument("--fps", type=int, default=60,
                        help="Output video frame rate")
    parser.add_argument("--width", type=int, default=1280,
                        help="Output video width")
    parser.add_argument("--height", type=int, default=720,
                        help="Output video height")
    parser.add_argument("--num-classes", type=int, default=20,
                        help="Number of segmentation classes (20 for Cityscapes with void)")
    parser.add_argument("--gpu", action="store_true", help="Force GPU")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Blend alpha for segmentation overlay (0=colour only, 1=original only)")
    parser.add_argument("--infer-width", type=int, default=256,
                        help="Width to resize to BEFORE inference (smaller = faster)")
    parser.add_argument("--infer-height", type=int, default=256,
                        help="Height to resize to BEFORE inference (smaller = faster)")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.cpu:
        device = torch.device("cpu")
    elif args.gpu:
        assert torch.cuda.is_available(), "CUDA not available"
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # ── Load model (segmentation_models_pytorch) ─────────────────────────
    print(f"[*] Loading model from {args.model} ...")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,          # we load our own weights
        classes=args.num_classes,
        activation=None,               # raw logits
    )

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("[+] Model loaded successfully.")

    # ── Collect frame paths ───────────────────────────────────────────────
    frame_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jpeg")))
    if not frame_paths:
        print(f"[!] No images found in {args.input_dir}")
        print("    Please add the 'stuttgart_00' folder with frame images.")
        sys.exit(1)

    total_frames = len(frame_paths)
    print(f"[*] Found {total_frames} frames in {args.input_dir}")

    # ── Video writer setup ────────────────────────────────────────────────
    out_w, out_h = args.width, args.height      # 1280 × 720
    half_w = out_w // 2                         # 640 each side

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        print("[!] Failed to create VideoWriter. Check your OpenCV build.")
        sys.exit(1)

    infer_w, infer_h = args.infer_width, args.infer_height
    print(f"[*] Inference resolution: {infer_w}x{infer_h}  (original images will be resized before model)")
    print(f"[*] Writing video to {args.output}  ({out_w}x{out_h} @ {args.fps} FPS)")
    print(f"[*] Processing frames ...")

    t0 = time.time()

    with torch.no_grad():
        for idx, fpath in enumerate(frame_paths):
            # ── Read frame ────────────────────────────────────────────────
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                print(f"  [!] Could not read {fpath}, skipping.")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # ── Resize for inference (MUCH faster on CPU) ─────────────────
            img_small = cv2.resize(img_rgb, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
            input_tensor = preprocess(img_small).unsqueeze(0).to(device)  # 1×3×infer_h×infer_w

            # ── Inference ─────────────────────────────────────────────────
            output = model(input_tensor)           # 1×C×infer_h×infer_w
            pred = output.argmax(dim=1).squeeze(0) # infer_h×infer_w
            pred_np = pred.cpu().numpy().astype(np.uint8)

            # ── Upscale prediction to display size ────────────────────────
            pred_up = cv2.resize(pred_np, (half_w, out_h), interpolation=cv2.INTER_NEAREST)
            seg_rgb = decode_segmentation(pred_up)  # out_h × half_w × 3 RGB

            # ── Resize original to display size ───────────────────────────
            left = cv2.resize(img_rgb, (half_w, out_h), interpolation=cv2.INTER_LINEAR)

            # ── Blend segmentation overlay with resized original ──────────
            blended_rgb = cv2.addWeighted(left, args.alpha, seg_rgb, 1 - args.alpha, 0)

            # ── Right side is the blended result ──────────────────────────
            right = blended_rgb

            # ── Concatenate side-by-side ──────────────────────────────────
            combined_rgb = np.concatenate([left, right], axis=1)  # H × W × 3  RGB

            # ── Add labels ────────────────────────────────────────────────
            combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)

            # Semi-transparent label backgrounds
            overlay = combined_bgr.copy()
            cv2.rectangle(overlay, (10, 10), (200, 50), (0, 0, 0), -1)
            cv2.rectangle(overlay, (half_w + 10, 10), (half_w + 200, 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, combined_bgr, 0.5, 0, combined_bgr)

            cv2.putText(combined_bgr, "Original", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_bgr, "Segmentation", (half_w + 20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # Frame counter
            frame_text = f"Frame {idx + 1}/{total_frames}"
            cv2.putText(combined_bgr, frame_text, (out_w - 250, out_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # ── Write frame ───────────────────────────────────────────────
            writer.write(combined_bgr)

            # Progress
            if (idx + 1) % 50 == 0 or (idx + 1) == total_frames:
                elapsed = time.time() - t0
                fps_actual = (idx + 1) / elapsed
                eta = (total_frames - idx - 1) / fps_actual if fps_actual > 0 else 0
                print(f"  [{idx + 1:>4d}/{total_frames}]  "
                      f"{fps_actual:.1f} frames/s  "
                      f"ETA: {eta:.0f}s")

    writer.release()
    elapsed_total = time.time() - t0
    print(f"\n[+] Done!  {total_frames} frames processed in {elapsed_total:.1f}s")
    print(f"[+] Video saved to: {args.output}")


if __name__ == "__main__":
    main()
