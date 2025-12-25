"""Base generator class using Google GenAI client."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Optional, Union

from google import genai
from google.genai import types
from google.oauth2 import service_account
import base64

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Type alias for image input
ImageInput = Union[bytes, "Image.Image"]


class AspectRatio(str, Enum):
    """Supported aspect ratios for image generation."""

    ULTRA_WIDE_21_9 = "21:9"
    LANDSCAPE_16_9 = "16:9"
    LANDSCAPE_3_2 = "3:2"
    LANDSCAPE_4_3 = "4:3"
    LANDSCAPE_5_4 = "5:4"
    SQUARE_1_1 = "1:1"
    PORTRAIT_4_5 = "4:5"
    PORTRAIT_3_4 = "3:4"
    PORTRAIT_2_3 = "2:3"
    PORTRAIT_9_16 = "9:16"


class ModelType(str, Enum):
    """Available model types."""

    NANO = "nano"
    PRO = "pro"


class Resolution(str, Enum):
    """Resolution options for Pro model."""

    RES_1K = "1K"
    RES_2K = "2K"
    RES_4K = "4K"


class CategoryType(str, Enum):
    """Reference category types for multi-reference generation."""

    FACE = "face"
    OBJECT = "object"
    STYLE = "style"


@dataclass
class ReferenceSet:
    """A set of reference images for a specific category.

    Attributes:
        category: The type of reference (face, object, or style).
        images: List of reference images as PIL Images or bytes.
    """

    category: CategoryType
    images: list[ImageInput]

    def __post_init__(self):
        if not self.images:
            raise ValueError("At least one image is required per reference set")
        if len(self.images) > 4:
            raise ValueError("Maximum 4 images per reference set")


@dataclass
class GenerationConfig:
    """Configuration for image generation."""

    num_images: int = 1
    aspect_ratio: AspectRatio = AspectRatio.SQUARE_1_1
    model: ModelType = ModelType.NANO
    resolution: Optional[Resolution] = None  # Only for Pro model
    output_mime_type: str = "image/png"


@dataclass
class GenerationResult:
    """Result of image generation.

    Attributes:
        images: List of generated images as raw bytes.
        mime_type: MIME type of the images (e.g., "image/png").
        prompt_used: The full prompt that was sent to the model.
        metadata: Additional metadata about the generation.
    """

    images: list[bytes]
    mime_type: str
    prompt_used: str
    metadata: dict = field(default_factory=dict)

    def to_base64(self) -> list[str]:
        """Convert images to base64-encoded strings.

        Useful for API responses where you need to send images as JSON.

        Returns:
            List of base64-encoded image strings.
        """
        return [base64.b64encode(img).decode("utf-8") for img in self.images]

    def to_data_uris(self) -> list[str]:
        """Convert images to data URIs.

        Useful for embedding directly in HTML/CSS.

        Returns:
            List of data URI strings (e.g., "data:image/png;base64,...").
        """
        return [f"data:{self.mime_type};base64,{b64}" for b64 in self.to_base64()]

    def to_pil(self) -> list["PIL.Image.Image"]:
        """Convert images to PIL Image objects.

        Useful for image processing, manipulation, or display in notebooks/Gradio.

        Returns:
            List of PIL Image objects.

        Raises:
            ImportError: If Pillow is not installed.
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for PIL conversion. Install with: pip install pillow")

        from io import BytesIO

        return [Image.open(BytesIO(img_bytes)) for img_bytes in self.images]

    def save_images(self, output_dir: str, prefix: str = "generated") -> list[str]:
        """Save generated images to disk.

        Args:
            output_dir: Directory to save images to.
            prefix: Filename prefix for saved images.

        Returns:
            List of saved file paths.
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        extension = "png" if self.mime_type == "image/png" else "jpg"
        saved_paths = []

        for i, image_bytes in enumerate(self.images):
            filename = f"{prefix}_{i}.{extension}"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            saved_paths.append(filepath)

        return saved_paths


class BaseGenerator(ABC):
    """Base class for reference-based image generation using Google GenAI."""

    # Model IDs for Nano and Pro
    MODEL_ID_NANO = "gemini-2.5-flash-image"
    MODEL_ID_PRO = "gemini-3-pro-image-preview"

    def __init__(
        self,
        credentials_path: str,
        location: str = "global",
        project_id: Optional[str] = None,
    ):
        """Initialize the generator.

        Args:
            credentials_path: Path to service account JSON file.
            location: Google Cloud region.
            project_id: Google Cloud project ID. If not provided, extracted from credentials file.
        """
        import json

        # Load credentials and extract project_id from service account file
        with open(credentials_path, "r") as f:
            creds_data = json.load(f)

        self.project_id = project_id or creds_data.get("project_id")
        if not self.project_id:
            raise ValueError("project_id not found in credentials file and not provided")

        self.location = location

        # Load credentials from service account file
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        # Initialize GenAI client with Vertex AI
        self._client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=location,
            credentials=credentials,
        )

    def _get_model_id(self, model: ModelType) -> str:
        """Get the model ID for the specified model type."""
        return self.MODEL_ID_PRO if model == ModelType.PRO else self.MODEL_ID_NANO

    def _convert_image_to_bytes(self, image: ImageInput) -> bytes:
        """Convert an image (PIL or bytes) to bytes.

        Args:
            image: PIL Image or bytes.

        Returns:
            Image as bytes.
        """
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            # Convert PIL Image to bytes
            buffer = BytesIO()
            # Save as PNG for lossless quality
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        elif isinstance(image, bytes):
            return image
        else:
            raise TypeError(f"Expected PIL Image or bytes, got {type(image)}")

    @property
    @abstractmethod
    def category(self) -> str:
        """Return the category name (object, style, or face)."""
        pass

    @abstractmethod
    def _build_prompt(self, user_prompt: str, reference_count: int) -> str:
        """Build the optimized prompt for this category.

        Args:
            user_prompt: The user's original prompt.
            reference_count: Number of reference images provided.

        Returns:
            Optimized prompt string.
        """
        pass

    async def generate_async(
        self,
        prompt: str,
        reference_images: list[ImageInput],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate images asynchronously based on reference images and prompt.

        This is the async version of generate() for concurrent request handling.

        Args:
            prompt: Text description of the desired output.
            reference_images: List of reference images as PIL Images or bytes (4-10 recommended).
            config: Generation configuration options.

        Returns:
            GenerationResult containing generated images.
        """
        return await self._generate_internal_async(prompt, reference_images, config)

    def generate(
        self,
        prompt: str,
        reference_images: list[ImageInput],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate images based on reference images and prompt.

        Args:
            prompt: Text description of the desired output.
            reference_images: List of reference images as PIL Images or bytes (4-10 recommended).
            config: Generation configuration options.

        Returns:
            GenerationResult containing generated images.

        Raises:
            ValueError: If reference_images is empty or exceeds limits.
            TypeError: If images are not PIL Images or bytes.
        """
        return self._generate_internal_sync(prompt, reference_images, config)

    def _generate_internal_sync(
        self,
        prompt: str,
        reference_images: list[ImageInput],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Internal synchronous generation implementation."""
        contents, generation_config, optimized_prompt, config = self._prepare_generation(
            prompt, reference_images, config
        )

        model_id = self._get_model_id(config.model)
        response = self._client.models.generate_content(
            model=model_id,
            contents=contents,
            config=generation_config,
        )

        return self._process_response(response, config, optimized_prompt, len(reference_images))

    async def _generate_internal_async(
        self,
        prompt: str,
        reference_images: list[ImageInput],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Internal asynchronous generation implementation."""
        contents, generation_config, optimized_prompt, config = self._prepare_generation(
            prompt, reference_images, config
        )

        model_id = self._get_model_id(config.model)
        response = await self._client.aio.models.generate_content(
            model=model_id,
            contents=contents,
            config=generation_config,
        )

        return self._process_response(response, config, optimized_prompt, len(reference_images))

    def _prepare_generation(
        self,
        prompt: str,
        reference_images: list[ImageInput],
        config: Optional[GenerationConfig],
    ) -> tuple[list, types.GenerateContentConfig, str, GenerationConfig]:
        """Prepare generation request (shared by sync and async methods)."""
        if not reference_images:
            raise ValueError("At least one reference image is required")

        if len(reference_images) > 4:
            raise ValueError("Maximum 4 reference images supported")

        config = config or GenerationConfig()

        # Validate resolution is only set for Pro model
        if config.resolution and config.model != ModelType.PRO:
            raise ValueError("Resolution option is only available for Pro model")

        # Build optimized prompt for this category
        optimized_prompt = self._build_prompt(prompt, len(reference_images))

        # Build content parts: reference images + prompt
        contents = []

        # Add reference images as parts
        for img in reference_images:
            image_bytes = self._convert_image_to_bytes(img)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            contents.append(image_part)

        # Add the prompt
        contents.append(optimized_prompt)

        # Configure safety settings using proper enum types
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
        ]

        # Create generation config (matching working pattern)
        cfg = types.GenerateContentConfig(
            response_modalities=[types.Modality.IMAGE, types.Modality.TEXT],
            safety_settings=safety_settings,
        )

        # Build image config with aspect ratio and optional resolution (Pro only)
        img_config_kwargs = {}

        if config.aspect_ratio:
            img_config_kwargs["aspect_ratio"] = config.aspect_ratio.value

        # Set image_size for Pro model (1K, 2K, 4K)
        if config.resolution and config.model == ModelType.PRO:
            img_config_kwargs["image_size"] = config.resolution.value

        # NOTE: number_of_images is NOT supported by Gemini image models
        # (gemini-2.5-flash-image, gemini-3-pro-image-preview)
        # Multiple images are generated via concurrent API calls instead

        if img_config_kwargs:
            cfg.image_config = types.ImageConfig(**img_config_kwargs)

        return contents, cfg, optimized_prompt, config

    def _process_response(
        self,
        response,
        config: GenerationConfig,
        optimized_prompt: str,
        reference_count: int,
    ) -> GenerationResult:
        """Process API response and extract images (shared by sync and async methods)."""
        images = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data:
                    images.append(part.inline_data.data)

        if not images:
            raise RuntimeError("No images were generated by the model")

        return GenerationResult(
            images=images,
            mime_type=config.output_mime_type,
            prompt_used=optimized_prompt,
            metadata={
                "category": self.category,
                "reference_count": reference_count,
                "config": {
                    "num_images": config.num_images,
                    "aspect_ratio": config.aspect_ratio.value,
                    "model": config.model.value,
                    "resolution": config.resolution.value if config.resolution else None,
                },
            },
        )

    @abstractmethod
    def _get_reference_type(self) -> str:
        """Get the reference type for this category.

        Returns:
            Reference type string for Imagen API.
        """
        pass
