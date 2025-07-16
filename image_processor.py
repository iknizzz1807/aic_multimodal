import os
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class ImageProcessor:
    """Handles basic image processing and CLIP embeddings"""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        """
        Initialize image processor with CLIP model

        Args:
            model_id: Hugging Face model identifier
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading CLIP model: {model_id}")
        print(f"Device: {self.device}")

        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        print("✅ Image processor initialized")

    def text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to normalized embedding vector

        Args:
            text: Input text description

        Returns:
            Normalized embedding vector
        """
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Normalize for cosine similarity
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def image_to_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Convert image to normalized embedding vector

        Args:
            image_path: Path to image file

        Returns:
            Normalized embedding vector or None if failed
        """
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Normalize for cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    @staticmethod
    def image_to_base64(
        image_path: str, max_size: tuple = (800, 600), quality: int = 85
    ) -> Optional[str]:
        """
        Convert image file to base64 string with optimization

        Args:
            image_path: Path to image file
            max_size: Maximum dimensions (width, height)
            quality: JPEG quality (1-100)

        Returns:
            Base64-encoded image string or None if failed
        """
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None

            with Image.open(image_path) as img:
                # Resize image if too large
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Save to bytes
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=quality, optimize=True)
                img_data = buffered.getvalue()

                # Skip if too large (1MB limit)
                if len(img_data) > 1000000:
                    print(f"Image too large, skipping: {image_path}")
                    return None

                # Encode to base64
                img_base64 = base64.b64encode(img_data).decode("utf-8")
                return f"data:image/jpeg;base64,{img_base64}"

        except Exception as e:
            print(f"Error converting image {image_path}: {e}")
            return None

    @staticmethod
    def get_image_files(
        directory: str, extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    ) -> List[str]:
        """
        Get all image files from directory

        Args:
            directory: Directory to search
            extensions: Tuple of valid image extensions

        Returns:
            List of image file paths
        """
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []

        image_files = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(extensions):
                image_files.append(os.path.join(directory, filename))

        return sorted(image_files)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "embedding_dim": self.model.config.projection_dim,
        }
