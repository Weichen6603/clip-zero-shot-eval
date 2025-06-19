### clip_classifier.py
# CLIP zero-shot classifier implementation
###

import torch
import clip
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm


class CLIPZeroShotClassifier:
    """CLIP-based zero-shot classifier.

    This class handles the creation of text embeddings for classes
    and performs zero-shot classification using CLIP.
    """

    def __init__(self,
                 model_name: str = "ViT-B/32",
                 device: str = "cuda",
                 use_ensemble: bool = True):
        """
        Args:
            model_name: CLIP model variant to use
            device: Device to run the model on
            use_ensemble: Whether to use ensemble of templates for text encoding
        """
        self.device = device
        self.model_name = model_name
        self.use_ensemble = use_ensemble

        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        # Cache for text features
        self._text_features_cache = {}

    def encode_text_classes(self,
                            classes: List[str],
                            templates: List[str]) -> torch.Tensor:
        """Encode text descriptions for all classes.

        Args:
            classes: List of class names
            templates: List of prompt templates

        Returns:
            Tensor of shape (num_classes, embed_dim) containing normalized text features
        """
        # Create cache key
        cache_key = (tuple(classes), tuple(templates))
        if cache_key in self._text_features_cache:
            return self._text_features_cache[cache_key]

        # Encode text for each class
        text_features = []

        with torch.no_grad():
            for class_name in tqdm(classes, desc="Encoding class descriptions"):
                if self.use_ensemble:
                    # Create multiple prompts for each class
                    texts = [template.format(class_name) for template in templates]
                    tokens = clip.tokenize(texts, truncate=True).to(self.device)

                    # Encode all templates
                    embeddings = self.model.encode_text(tokens)
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

                    # Average embeddings from all templates
                    class_embedding = embeddings.mean(dim=0)
                    class_embedding = class_embedding / class_embedding.norm()
                else:
                    # Use only the first template
                    text = templates[0].format(class_name)
                    tokens = clip.tokenize([text], truncate=True).to(self.device)

                    class_embedding = self.model.encode_text(tokens)[0]
                    class_embedding = class_embedding / class_embedding.norm()

                text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=0)

        # Cache the results
        self._text_features_cache[cache_key] = text_features

        return text_features

    def classify_images(self,
                        images: torch.Tensor,
                        text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify images using pre-computed text features.

        Args:
            images: Batch of images (already preprocessed)
            text_features: Pre-computed text features for all classes

        Returns:
            predictions: Predicted class indices
            similarities: Similarity scores for all classes
        """
        with torch.no_grad():
            # Encode images
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Compute similarities
            similarities = image_features @ text_features.T

            # Get predictions
            predictions = similarities.argmax(dim=1)

        return predictions, similarities

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'use_ensemble': self.use_ensemble,
            'vision_width': self.model.visual.input_resolution,
            'vision_layers': len(self.model.visual.transformer.resblocks)
            if hasattr(self.model.visual, 'transformer') else 'N/A',
            'embed_dim': self.model.visual.output_dim,
        }