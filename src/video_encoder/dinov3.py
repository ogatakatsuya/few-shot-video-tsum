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
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str | None = None,
        max_frames: int = 32,
    ):
        """
        Initialize DINOv3 video embedder.

        Args:
            model_name: DINOv3 model name from Hugging Face
            device: Device to run the model on. If None, automatically selects cuda if available.
            max_frames: Maximum number of frames to sample from each video
        """
        self.device = device or (0 if torch.cuda.is_available() else -1)
        self.max_frames = max_frames

        self.feature_extractor = pipeline(
            model=model_name,
            task="image-feature-extraction",
            device=self.device,
            token=env.HF_TOKEN,
        )

    def extract_sampled_frames(self, video_path: Path) -> list[np.ndarray]:
        """
        Extract only the sampled frames from video without loading all frames into memory.

        Args:
            video_path: Path to the video file

        Returns:
            List of sampled frames
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate sampling indices
        if total_frames <= self.max_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)

        frames = []
        indices_set = set(indices)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Only keep frames at sampling indices
            if i in indices_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        print(f"Sampled {len(frames)} frames from {total_frames} total frames")
        return frames

    def get_embedding(self, video_path: Path) -> np.ndarray:
        """
        Extract frame-level features from video and return the averaged embedding.

        Args:
            video_path: Path to the video file

        Returns:
            Averaged embedding as numpy array
        """
        # Extract only sampled frames to reduce memory usage
        sampled_frames = self.extract_sampled_frames(video_path)

        # Convert sampled frames to PIL Images
        pil_images = [Image.fromarray(frame) for frame in sampled_frames]

        # Extract features for sampled frames
        features = self.feature_extractor(pil_images)

        # Convert to numpy array and average
        features_array = np.array(features)
        avg_embedding = np.mean(features_array, axis=(0, 1, 2))

        return avg_embedding

    def get_embeddings_batch(self, video_paths: list[Path]) -> dict[str, np.ndarray]:
        """
        Extract embeddings from multiple videos in batch for better GPU utilization.

        Args:
            video_paths: List of paths to video files

        Returns:
            Dictionary mapping video_id to embedding
        """
        print(f"Processing {len(video_paths)} videos in batch...")

        # Extract and sample frames from all videos
        all_frames = []
        video_frame_counts = []
        video_ids = []

        for video_path in video_paths:
            video_id = video_path.stem
            sampled_frames = self.extract_sampled_frames(video_path)

            all_frames.extend(sampled_frames)
            video_frame_counts.append(len(sampled_frames))
            video_ids.append(video_id)

        if not all_frames:
            return {}

        print(
            f"Total sampled frames: {len(all_frames)} (from {len(video_paths)} videos)"
        )

        # Convert all sampled frames to PIL Images
        pil_images = [Image.fromarray(frame) for frame in all_frames]

        # Extract features for all sampled frames at once
        print("Extracting features from all sampled frames...")
        features = self.feature_extractor(pil_images)
        features_array = np.array(features)

        # Split features back by video and compute average embeddings
        embeddings = {}
        frame_idx = 0

        for i, (video_id, frame_count) in enumerate(zip(video_ids, video_frame_counts)):
            # Get features for this video's frames
            video_features = features_array[frame_idx : frame_idx + frame_count]

            # Average across frames, batch, and patches
            avg_embedding = np.mean(video_features, axis=(0, 1, 2))
            embeddings[video_id] = avg_embedding

            frame_idx += frame_count
            print(f"  ✓ Generated embedding for {video_id} ({frame_count} frames)")

        return embeddings


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
