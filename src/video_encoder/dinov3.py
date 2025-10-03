from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from src.env import env


class DINOv3VideoEncoder:
    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        device: str | None = None,
    ):
        """
        Initialize DINOv3 video embedder.

        Args:
            model_name: DINOv3 model name from Hugging Face (e.g., "facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
            device: Device to run the model on. If None, automatically selects cuda if available.
        """
        self.device = device or (0 if torch.cuda.is_available() else -1)
        self.feature_extractor = pipeline(
            model=model_name,
            task="image-feature-extraction",
            device=self.device,
            token=env.HF_TOKEN,
        )

    def extract_frames(self, video_path: Path) -> list[np.ndarray]:
        """Extract all frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames

    def get_embedding(self, video_path: Path) -> np.ndarray:
        """
        Extract frame-level features from video and return the averaged embedding.

        Args:
            video_path: Path to the video file

        Returns:
            Averaged embedding as numpy array of shape (768,)
        """
        frames = self.extract_frames(video_path)

        # Convert all frames to PIL Images
        pil_images = [Image.fromarray(frame) for frame in frames]

        # Extract features for all frames at once
        features = self.feature_extractor(pil_images)

        # Convert to numpy array: shape will be (num_frames, 1, num_patches, hidden_dim)
        # Average across all frames, batch, and patches to get single embedding vector
        features_array = np.array(features)
        avg_embedding = np.mean(features_array, axis=(0, 1, 2))

        return avg_embedding


def main():
    """Example usage of DINOv3VideoEncoder."""
    # Initialize encoder
    encoder = DINOv3VideoEncoder()

    # Extract embedding from video
    video_path = Path(
        "/home/ogata-katsuya/Study/VideoSum/Code/few-shot-tsum/data/videos/v__4wEUsTft44.mp4"
    )
    embedding = encoder.get_embedding(video_path)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")


if __name__ == "__main__":
    main()
