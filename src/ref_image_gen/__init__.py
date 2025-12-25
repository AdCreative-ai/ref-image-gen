"""Reference-based image generation using Google Vertex AI Imagen."""

from ref_image_gen.base import (
    BaseGenerator,
    AspectRatio,
    ModelType,
    Resolution,
    CategoryType,
    ReferenceSet,
    GenerationConfig,
    GenerationResult,
)
from ref_image_gen.object_generator import ObjectGenerator
from ref_image_gen.style_generator import StyleGenerator
from ref_image_gen.face_generator import FaceGenerator
from ref_image_gen.multi_generator import MultiRefGenerator
from ref_image_gen.async_utils import (
    GenerationRequest,
    BatchResult,
    batch_generate_async,
    generate_variations_async,
)

__version__ = "0.1.0"

__all__ = [
    # Generators
    "BaseGenerator",
    "ObjectGenerator",
    "StyleGenerator",
    "FaceGenerator",
    "MultiRefGenerator",
    # Enums and Config
    "AspectRatio",
    "ModelType",
    "Resolution",
    "CategoryType",
    "ReferenceSet",
    "GenerationConfig",
    "GenerationResult",
    # Async utilities
    "GenerationRequest",
    "BatchResult",
    "batch_generate_async",
    "generate_variations_async",
]
