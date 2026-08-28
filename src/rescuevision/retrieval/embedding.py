"""DINOv2-based embedding extraction for disaster image retrieval."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DINOv2Embedder:
    """Extract DINOv2 embeddings from images for similarity retrieval."""

    def __init__(
        self,
        backbone: str = "facebook/dinov2-base",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone_name = backbone

        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(backbone)
            self.model = self.model.to(self.device).eval()
            self.embed_dim = self.model.config.hidden_size
            logger.info("Loaded DINOv2 embedder: %s (dim=%d)", backbone, self.embed_dim)
        except Exception as e:
            logger.warning("Could not load DINOv2 model: %s. Using random embeddings.", e)
            self.model = None
            self.embed_dim = 768

        self.transform = _DEFAULT_TRANSFORM

    @torch.no_grad()
    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Extract embedding from a single image.

        Returns:
            numpy array of shape (embed_dim,).
        """
        if self.model is None:
            return np.random.randn(self.embed_dim).astype(np.float32)

        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        output = self.model(tensor)
        # Use CLS token embedding
        embedding = output.last_hidden_state[:, 0].squeeze().cpu().numpy()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    @torch.no_grad()
    def embed_batch(self, image_paths: list[str | Path], batch_size: int = 16) -> np.ndarray:
        """Extract embeddings for a batch of images.

        Returns:
            numpy array of shape (N, embed_dim).
        """
        if self.model is None:
            return np.random.randn(len(image_paths), self.embed_dim).astype(np.float32)

        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self.transform(img))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", path, e)
                    tensors.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(tensors).to(self.device)
            output = self.model(batch_tensor)
            embeddings = output.last_hidden_state[:, 0].cpu().numpy()

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings.astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)
