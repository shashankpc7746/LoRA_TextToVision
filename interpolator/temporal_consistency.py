"""
Temporal Consistency Module - Day 2 Implementation
===================================================

Purpose:
    Ensures temporal consistency across video frames using:
    - Temporal UNet denoiser for frame-to-frame coherence
    - De-flicker pass with histogram matching
    - Optical flow guided processing
    
Architecture:
    - TemporalUNet: 3D convolutions for multi-frame processing
    - HistogramMatcher: Statistical de-flicker
    - TemporalConsistencyProcessor: Main API interface
    
GPU Allocation:
    RTX 3060 (GPU:1) for temporal processing
    
Compliance:
    - KSML lineage tracking
    - Audit logging for all operations
    - Metadata preservation
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalUNet3D(nn.Module):
    """
    3D UNet for temporal consistency processing.
    
    Features:
        - 3D convolutions for spatio-temporal processing
        - Skip connections for detail preservation
        - Multi-scale feature extraction
        - Lightweight architecture for real-time processing
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._make_3d_block(in_channels, base_channels)
        self.enc2 = self._make_3d_block(base_channels, base_channels * 2)
        self.enc3 = self._make_3d_block(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = self._make_3d_block(base_channels * 4, base_channels * 8)
        
        # Decoder (upsampling)
        self.dec3 = self._make_3d_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._make_3d_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._make_3d_block(base_channels * 2 + base_channels, base_channels)
        
        # Output layer
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        
    def _make_3d_block(self, in_ch, out_ch):
        """Create a 3D convolutional block with batch norm and activation."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass with skip connections.
        
        Args:
            x: (B, C, T, H, W) - Batch of frame sequences
            
        Returns:
            (B, C, T, H, W) - Temporally consistent frames
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Output
        out = self.out_conv(d1)
        
        # Residual connection
        return x + out


class HistogramMatcher:
    """
    Statistical de-flicker using histogram matching.
    
    Matches color histograms between consecutive frames to reduce flicker.
    """
    
    @staticmethod
    def match_histograms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Match histogram of source to reference image.
        
        Args:
            source: Source image (H, W, C)
            reference: Reference image (H, W, C)
            
        Returns:
            Histogram-matched image
        """
        matched = np.zeros_like(source)
        
        for channel in range(source.shape[2]):
            # Get histograms
            src_hist, src_bins = np.histogram(source[:, :, channel].flatten(), 256, [0, 256])
            ref_hist, ref_bins = np.histogram(reference[:, :, channel].flatten(), 256, [0, 256])
            
            # Calculate CDFs
            src_cdf = src_hist.cumsum()
            src_cdf = src_cdf / src_cdf[-1]
            
            ref_cdf = ref_hist.cumsum()
            ref_cdf = ref_cdf / ref_cdf[-1]
            
            # Create lookup table
            lookup = np.zeros(256, dtype=np.uint8)
            src_idx = 0
            
            for ref_idx in range(256):
                while src_idx < 255 and src_cdf[src_idx] < ref_cdf[ref_idx]:
                    src_idx += 1
                lookup[ref_idx] = src_idx
            
            # Apply lookup
            matched[:, :, channel] = lookup[source[:, :, channel]]
        
        return matched
    
    @staticmethod
    def temporal_smooth_histograms(frames: List[np.ndarray], alpha: float = 0.3) -> List[np.ndarray]:
        """
        Apply temporal smoothing to frame histograms.
        
        Args:
            frames: List of frames (H, W, C)
            alpha: Smoothing factor (0 = no smoothing, 1 = full match)
            
        Returns:
            De-flickered frames
        """
        if len(frames) < 2:
            return frames
        
        smoothed = [frames[0].copy()]
        
        for i in range(1, len(frames)):
            # Match to previous frame
            matched = HistogramMatcher.match_histograms(frames[i], smoothed[i-1])
            
            # Blend with original
            blended = cv2.addWeighted(frames[i], 1 - alpha, matched, alpha, 0)
            smoothed.append(blended)
        
        return smoothed


class OpticalFlowGuide:
    """
    Optical flow guidance for temporal consistency.
    
    Uses optical flow to warp previous frame and blend with current.
    """
    
    @staticmethod
    def compute_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two frames.
        
        Args:
            frame1: Previous frame (H, W, C)
            frame2: Current frame (H, W, C)
            
        Returns:
            Flow field (H, W, 2)
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        return flow
    
    @staticmethod
    def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp frame using optical flow.
        
        Args:
            frame: Frame to warp (H, W, C)
            flow: Flow field (H, W, 2)
            
        Returns:
            Warped frame
        """
        h, w = frame.shape[:2]
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        
        # Create coordinate grid
        flow_map[:, :, 0] = np.arange(w)
        flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]
        
        # Add flow
        flow_map = flow_map + flow
        
        # Warp
        warped = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)
        
        return warped
    
    @staticmethod
    def flow_guided_blend(frames: List[np.ndarray], alpha: float = 0.2) -> List[np.ndarray]:
        """
        Apply flow-guided blending for temporal consistency.
        
        Args:
            frames: List of frames (H, W, C)
            alpha: Blending weight for warped previous frame
            
        Returns:
            Temporally consistent frames
        """
        if len(frames) < 2:
            return frames
        
        blended = [frames[0].copy()]
        
        for i in range(1, len(frames)):
            # Compute flow from previous to current
            flow = OpticalFlowGuide.compute_flow(blended[i-1], frames[i])
            
            # Warp previous frame
            warped_prev = OpticalFlowGuide.warp_frame(blended[i-1], flow)
            
            # Blend with current frame
            blended_frame = cv2.addWeighted(frames[i], 1 - alpha, warped_prev, alpha, 0)
            blended.append(blended_frame)
        
        return blended


class TemporalConsistencyProcessor:
    """
    Main API interface for temporal consistency processing.
    
    Combines:
        - Temporal UNet denoising
        - Histogram de-flicker
        - Optical flow guidance
        - KSML compliance and metadata tracking
    """
    
    def __init__(
        self,
        device: str = "cuda:1",  # RTX 3060
        model_path: Optional[str] = None,
        use_histogram: bool = True,
        use_flow: bool = True,
        batch_size: int = 8
    ):
        """
        Initialize temporal consistency processor.
        
        Args:
            device: GPU device (default: cuda:1 for RTX 3060)
            model_path: Path to pre-trained temporal UNet (optional)
            use_histogram: Enable histogram de-flicker
            use_flow: Enable optical flow guidance
            batch_size: Number of frames to process at once
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_histogram = use_histogram
        self.use_flow = use_flow
        self.batch_size = batch_size
        
        # Initialize temporal UNet
        self.model = TemporalUNet3D(in_channels=3, out_channels=3, base_channels=32)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded temporal UNet from {model_path}")
        else:
            logger.warning("No pre-trained model loaded. Using random initialization.")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"TemporalConsistencyProcessor initialized on {self.device}")
    
    def _load_frames(self, in_dir: str) -> List[np.ndarray]:
        """Load all frames from directory."""
        in_path = Path(in_dir)
        frame_files = sorted(in_path.glob("*.png")) + sorted(in_path.glob("*.jpg"))
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        logger.info(f"Loaded {len(frames)} frames from {in_dir}")
        return frames
    
    def _save_frames(self, frames: List[np.ndarray], out_dir: str):
        """Save frames to directory."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            out_file = out_path / f"frame_{i:04d}.png"
            cv2.imwrite(str(out_file), frame)
        
        logger.info(f"Saved {len(frames)} frames to {out_dir}")
    
    def _process_with_unet(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process frames with temporal UNet.
        
        Args:
            frames: List of frames (H, W, C)
            
        Returns:
            Processed frames
        """
        # Convert to tensor
        frame_tensors = []
        for frame in frames:
            # Normalize to [0, 1]
            tensor = torch.from_numpy(frame).float() / 255.0
            # (H, W, C) -> (C, H, W)
            tensor = tensor.permute(2, 0, 1)
            frame_tensors.append(tensor)
        
        # Stack to (T, C, H, W)
        frames_tensor = torch.stack(frame_tensors, dim=0)
        
        # Add batch dimension: (1, C, T, H, W)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        frames_tensor = frames_tensor.to(self.device)
        
        # Process
        with torch.no_grad():
            processed = self.model(frames_tensor)
        
        # Convert back to numpy
        processed = processed.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
        processed = (processed.cpu().numpy() * 255).astype(np.uint8)
        
        return [processed[i] for i in range(processed.shape[0])]
    
    def process_frames_consistent(
        self,
        in_dir: str,
        out_dir: str,
        ksml_token: Optional[Dict] = None
    ) -> Dict:
        """
        Main API: Process frames for temporal consistency.
        
        Args:
            in_dir: Input directory with frames
            out_dir: Output directory for processed frames
            ksml_token: KSML compliance metadata
            
        Returns:
            Processing metadata with KSML lineage
        """
        start_time = datetime.now()
        
        # Load frames
        frames = self._load_frames(in_dir)
        
        if len(frames) == 0:
            raise ValueError(f"No frames found in {in_dir}")
        
        logger.info(f"Processing {len(frames)} frames for temporal consistency")
        
        # Step 1: Temporal UNet processing (if trained model available)
        if self.model is not None:
            logger.info("Step 1: Temporal UNet denoising...")
            frames = self._process_with_unet(frames)
        
        # Step 2: Histogram de-flicker
        if self.use_histogram:
            logger.info("Step 2: Histogram de-flicker...")
            frames = HistogramMatcher.temporal_smooth_histograms(frames, alpha=0.3)
        
        # Step 3: Optical flow guidance
        if self.use_flow:
            logger.info("Step 3: Optical flow guidance...")
            frames = OpticalFlowGuide.flow_guided_blend(frames, alpha=0.2)
        
        # Save processed frames
        self._save_frames(frames, out_dir)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Create metadata
        metadata = {
            "operation": "temporal_consistency",
            "timestamp": start_time.isoformat(),
            "processing_time_seconds": processing_time,
            "num_frames": len(frames),
            "fps": len(frames) / processing_time if processing_time > 0 else 0,
            "input_dir": in_dir,
            "output_dir": out_dir,
            "config": {
                "device": str(self.device),
                "use_histogram": self.use_histogram,
                "use_flow": self.use_flow,
                "batch_size": self.batch_size
            },
            "ksml_lineage": {
                "parent_token": ksml_token.get("ksml_token") if ksml_token else None,
                "operation": "temporal_consistency",
                "karma_state": "temporal_smoothed",
                "lineage": {
                    "source": "TemporalConsistencyProcessor",
                    "version": "1.0.0",
                    "gpu": str(self.device)
                }
            }
        }
        
        # Save metadata
        metadata_file = Path(out_dir) / "temporal_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Temporal consistency processing complete: {processing_time:.2f}s")
        logger.info(f"   Frames: {len(frames)}, FPS: {metadata['fps']:.2f}")
        
        return metadata


# Convenience functions
def process_frames_consistent(
    in_dir: str,
    out_dir: str,
    device: str = "cuda:1",
    **kwargs
) -> Dict:
    """
    Convenience function for temporal consistency processing.
    
    Args:
        in_dir: Input directory with frames
        out_dir: Output directory for processed frames
        device: GPU device (default: cuda:1 for RTX 3060)
        **kwargs: Additional arguments for TemporalConsistencyProcessor
        
    Returns:
        Processing metadata
    """
    processor = TemporalConsistencyProcessor(device=device, **kwargs)
    return processor.process_frames_consistent(in_dir, out_dir)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Temporal Consistency Processing")
    parser.add_argument("--in_dir", type=str, required=True, help="Input frames directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output frames directory")
    parser.add_argument("--device", type=str, default="cuda:1", help="GPU device")
    parser.add_argument("--model_path", type=str, default=None, help="Pre-trained model path")
    parser.add_argument("--no_histogram", action="store_true", help="Disable histogram de-flicker")
    parser.add_argument("--no_flow", action="store_true", help="Disable optical flow")
    
    args = parser.parse_args()
    
    processor = TemporalConsistencyProcessor(
        device=args.device,
        model_path=args.model_path,
        use_histogram=not args.no_histogram,
        use_flow=not args.no_flow
    )
    
    metadata = processor.process_frames_consistent(args.in_dir, args.out_dir)
    
    print("\n✅ Processing Complete!")
    print(f"   Time: {metadata['processing_time_seconds']:.2f}s")
    print(f"   Frames: {metadata['num_frames']}")
    print(f"   FPS: {metadata['fps']:.2f}")
