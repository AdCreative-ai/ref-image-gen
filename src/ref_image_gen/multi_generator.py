"""Multi-reference generator for combining multiple categories."""

from collections import Counter
from io import BytesIO
from typing import Optional

from google import genai
from google.genai import types
from google.oauth2 import service_account

from .base import (
    CategoryType,
    GenerationConfig,
    GenerationResult,
    ImageInput,
    ModelType,
    ReferenceSet,
    PIL_AVAILABLE,
)

if PIL_AVAILABLE:
    from PIL import Image


class MultiRefGenerator:
    """Generator for multi-reference image generation.

    Supports combining up to 3 reference sets of any category combination:
    face, object, style, or multiples of the same category.
    """

    MODEL_ID_NANO = "gemini-2.5-flash-image"
    MODEL_ID_PRO = "gemini-3-pro-image-preview"

    def __init__(
        self,
        credentials_path: str,
        location: str = "global",
        project_id: Optional[str] = None,
    ):
        """Initialize the multi-reference generator.

        Args:
            credentials_path: Path to service account JSON file.
            location: Google Cloud region.
            project_id: Google Cloud project ID. If not provided, extracted from credentials.
        """
        import json

        with open(credentials_path, "r") as f:
            creds_data = json.load(f)

        self.project_id = project_id or creds_data.get("project_id")
        if not self.project_id:
            raise ValueError("project_id not found in credentials file and not provided")

        self.location = location

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

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
        """Convert an image (PIL or bytes) to bytes."""
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        elif isinstance(image, bytes):
            return image
        else:
            raise TypeError(f"Expected PIL Image or bytes, got {type(image)}")

    def _get_combination_key(self, reference_sets: list[ReferenceSet]) -> str:
        """Get a normalized key representing the combination of categories.

        Returns a sorted string like 'face+object', 'face+face+style', etc.
        """
        categories = sorted([rs.category.value for rs in reference_sets])
        return "+".join(categories)

    def _build_prompt(
        self,
        user_prompt: str,
        reference_sets: list[ReferenceSet],
    ) -> str:
        """Build the optimized prompt based on the combination of categories.

        Args:
            user_prompt: The user's original prompt.
            reference_sets: List of reference sets with their categories.

        Returns:
            Optimized prompt string.
        """
        # Count categories
        category_counts = Counter(rs.category for rs in reference_sets)
        num_faces = category_counts.get(CategoryType.FACE, 0)
        num_objects = category_counts.get(CategoryType.OBJECT, 0)
        num_styles = category_counts.get(CategoryType.STYLE, 0)

        # Calculate image positions for each reference set
        image_positions = []
        current_pos = 1
        for rs in reference_sets:
            start = current_pos
            end = current_pos + len(rs.images) - 1
            image_positions.append((start, end))
            current_pos = end + 1

        # Build prompt based on combination
        prompt_parts = []

        # Add reference instructions based on combination
        prompt_parts.append(self._get_combination_prompt(
            reference_sets, image_positions, num_faces, num_objects, num_styles
        ))

        # Add user's prompt
        prompt_parts.append(f"\nScene description: {user_prompt}")

        # Add quality instructions
        prompt_parts.append(
            "\nEnsure high quality, coherent composition, and natural integration of all referenced elements."
        )

        return "\n".join(prompt_parts)

    def _get_combination_prompt(
        self,
        reference_sets: list[ReferenceSet],
        image_positions: list[tuple[int, int]],
        num_faces: int,
        num_objects: int,
        num_styles: int,
    ) -> str:
        """Generate the specific prompt instructions based on the combination."""

        total_sets = len(reference_sets)

        # Helper to describe image range
        def img_range(idx: int) -> str:
            start, end = image_positions[idx]
            if start == end:
                return f"reference image {start}"
            return f"reference images {start}-{end}"

        # Helper to get sets by category
        def get_sets_by_category(cat: CategoryType) -> list[tuple[int, ReferenceSet]]:
            return [(i, rs) for i, rs in enumerate(reference_sets) if rs.category == cat]

        face_sets = get_sets_by_category(CategoryType.FACE)
        object_sets = get_sets_by_category(CategoryType.OBJECT)
        style_sets = get_sets_by_category(CategoryType.STYLE)

        # Single category cases
        if total_sets == 1:
            rs = reference_sets[0]
            if rs.category == CategoryType.FACE:
                return (
                    f"Generate an image featuring the person shown in {img_range(0)}. "
                    f"Maintain ONLY their facial identity and likeness. "
                    f"Do NOT copy the pose, clothing, background, or setting from the reference - "
                    f"use the scene description provided below instead."
                )
            elif rs.category == CategoryType.OBJECT:
                return (
                    f"Generate an image featuring the product/object shown in {img_range(0)}. "
                    f"Preserve the product's EXACT appearance: same shape, colors, design, branding, and all details. "
                    f"The product must look identical to the reference."
                )
            else:  # STYLE
                return (
                    f"Generate an image applying ONLY the artistic style from {img_range(0)}. "
                    f"Extract the style's color palette, brushwork, textures, and artistic techniques. "
                    f"Do NOT copy the content, subjects, or composition from the style reference - "
                    f"create new content based on the scene description below."
                )

        # Two category cases
        if total_sets == 2:
            return self._get_two_set_prompt(
                reference_sets, image_positions, face_sets, object_sets, style_sets,
                num_faces, num_objects, num_styles, img_range
            )

        raise ValueError(f"Unsupported number of reference sets: {total_sets}")

    def _get_two_set_prompt(
        self,
        reference_sets: list[ReferenceSet],
        image_positions: list[tuple[int, int]],
        face_sets: list,
        object_sets: list,
        style_sets: list,
        num_faces: int,
        num_objects: int,
        num_styles: int,
        img_range,
    ) -> str:
        """Generate prompt for two reference set combinations."""

        # Face + Face
        if num_faces == 2:
            return (
                f"Generate an image featuring both people: "
                f"the person from {img_range(face_sets[0][0])} and "
                f"the person from {img_range(face_sets[1][0])}. "
                f"Maintain ONLY each person's facial identity and likeness. "
                f"Do NOT copy poses, clothing, or backgrounds from references - "
                f"place them in the scene described below."
            )

        # Object + Object
        if num_objects == 2:
            return (
                f"Generate an image featuring both products/objects: "
                f"the item from {img_range(object_sets[0][0])} and "
                f"the item from {img_range(object_sets[1][0])}. "
                f"Preserve each product's EXACT appearance: same shapes, colors, designs, branding, and all details. "
                f"Both products must look identical to their references."
            )

        # Style + Style
        if num_styles == 2:
            return (
                f"Generate an image blending ONLY the artistic characteristics from two styles: "
                f"the style from {img_range(style_sets[0][0])} and "
                f"the style from {img_range(style_sets[1][0])}. "
                f"Fuse their color palettes, brushwork, textures, and techniques into a hybrid aesthetic. "
                f"Do NOT copy the content, subjects, or composition from either style reference - "
                f"create new content based on the scene description below."
            )

        # Face + Object
        if num_faces == 1 and num_objects == 1:
            face_idx = face_sets[0][0]
            obj_idx = object_sets[0][0]
            return (
                f"Generate an image of the person from {img_range(face_idx)} "
                f"with the product/object from {img_range(obj_idx)}. "
                f"For the person: maintain ONLY their facial identity and likeness, NOT their pose, clothing, or background. "
                f"For the product: preserve its EXACT appearance - same shape, colors, design, branding, and all details. "
                f"Place them in the scene described below."
            )

        # Face + Style
        if num_faces == 1 and num_styles == 1:
            face_idx = face_sets[0][0]
            style_idx = style_sets[0][0]
            return (
                f"Generate an image of the person from {img_range(face_idx)} "
                f"applying ONLY the artistic style from {img_range(style_idx)}. "
                f"For the person: maintain their facial identity and likeness. "
                f"For the style: apply ONLY the color palette, brushwork, textures, and artistic techniques - "
                f"do NOT copy subjects or composition from the style reference. "
                f"Create the scene based on the description below."
            )

        # Object + Style
        if num_objects == 1 and num_styles == 1:
            obj_idx = object_sets[0][0]
            style_idx = style_sets[0][0]
            return (
                f"Generate an image of the product/object from {img_range(obj_idx)} "
                f"applying ONLY the artistic style from {img_range(style_idx)}. "
                f"For the product: preserve its EXACT shape, design, branding, and recognizable details. "
                f"For the style: apply ONLY the color palette, brushwork, and artistic techniques - "
                f"do NOT copy subjects or composition from the style reference. "
                f"Create the scene based on the description below."
            )

        raise ValueError("Unhandled two-set combination")

    async def generate_async(
        self,
        prompt: str,
        reference_sets: list[ReferenceSet],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate images asynchronously based on multiple reference sets.

        This is the async version of generate() for concurrent request handling.

        Args:
            prompt: Text description of the desired output scene.
            reference_sets: List of ReferenceSet objects (max 2).
            config: Generation configuration options.

        Returns:
            GenerationResult containing generated images.
        """
        return await self._generate_internal_async(prompt, reference_sets, config)

    def generate(
        self,
        prompt: str,
        reference_sets: list[ReferenceSet],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate images based on multiple reference sets.

        Args:
            prompt: Text description of the desired output scene.
            reference_sets: List of ReferenceSet objects (max 2).
            config: Generation configuration options.

        Returns:
            GenerationResult containing generated images.

        Raises:
            ValueError: If reference_sets is empty, exceeds 2, or invalid.
        """
        return self._generate_internal_sync(prompt, reference_sets, config)

    def _generate_internal_sync(
        self,
        prompt: str,
        reference_sets: list[ReferenceSet],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Internal synchronous generation implementation."""
        contents, generation_config, optimized_prompt, config = self._prepare_generation(
            prompt, reference_sets, config
        )

        model_id = self._get_model_id(config.model)
        response = self._client.models.generate_content(
            model=model_id,
            contents=contents,
            config=generation_config,
        )

        return self._process_response(response, config, optimized_prompt, reference_sets)

    async def _generate_internal_async(
        self,
        prompt: str,
        reference_sets: list[ReferenceSet],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Internal asynchronous generation implementation."""
        contents, generation_config, optimized_prompt, config = self._prepare_generation(
            prompt, reference_sets, config
        )

        model_id = self._get_model_id(config.model)
        response = await self._client.aio.models.generate_content(
            model=model_id,
            contents=contents,
            config=generation_config,
        )

        return self._process_response(response, config, optimized_prompt, reference_sets)

    def _prepare_generation(
        self,
        prompt: str,
        reference_sets: list[ReferenceSet],
        config: Optional[GenerationConfig],
    ) -> tuple[list, types.GenerateContentConfig, str, GenerationConfig]:
        """Prepare generation request (shared by sync and async methods)."""
        if not reference_sets:
            raise ValueError("At least one reference set is required")

        if len(reference_sets) > 2:
            raise ValueError("Maximum 2 reference sets supported")

        config = config or GenerationConfig()

        # Validate resolution is only set for Pro model
        if config.resolution and config.model != ModelType.PRO:
            raise ValueError("Resolution option is only available for Pro model")

        # Build optimized prompt
        optimized_prompt = self._build_prompt(prompt, reference_sets)

        # Build content parts: all reference images + prompt
        contents = []

        for rs in reference_sets:
            for img in rs.images:
                image_bytes = self._convert_image_to_bytes(img)
                image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                contents.append(image_part)

        contents.append(optimized_prompt)

        # Configure safety settings
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

        # Create generation config
        cfg = types.GenerateContentConfig(
            response_modalities=[types.Modality.IMAGE, types.Modality.TEXT],
            safety_settings=safety_settings,
        )

        # Build image config
        img_config_kwargs = {}

        if config.aspect_ratio:
            img_config_kwargs["aspect_ratio"] = config.aspect_ratio.value

        if config.resolution and config.model == ModelType.PRO:
            img_config_kwargs["image_size"] = config.resolution.value

        # NOTE: number_of_images is NOT supported by Gemini image models
        # Multiple images are generated via concurrent API calls instead

        if img_config_kwargs:
            cfg.image_config = types.ImageConfig(**img_config_kwargs)

        return contents, cfg, optimized_prompt, config

    def _process_response(
        self,
        response,
        config: GenerationConfig,
        optimized_prompt: str,
        reference_sets: list[ReferenceSet],
    ) -> GenerationResult:
        """Process API response and extract images (shared by sync and async methods)."""
        images = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data:
                    images.append(part.inline_data.data)

        if not images:
            raise RuntimeError("No images were generated by the model")

        # Build metadata
        combination_key = self._get_combination_key(reference_sets)
        total_ref_images = sum(len(rs.images) for rs in reference_sets)

        return GenerationResult(
            images=images,
            mime_type=config.output_mime_type,
            prompt_used=optimized_prompt,
            metadata={
                "combination": combination_key,
                "reference_sets": [
                    {"category": rs.category.value, "image_count": len(rs.images)}
                    for rs in reference_sets
                ],
                "total_reference_images": total_ref_images,
                "config": {
                    "num_images": config.num_images,
                    "aspect_ratio": config.aspect_ratio.value,
                    "model": config.model.value,
                    "resolution": config.resolution.value if config.resolution else None,
                },
            },
        )
