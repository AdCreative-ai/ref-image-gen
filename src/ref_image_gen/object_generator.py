"""Object-based image generation using reference images."""

from ref_image_gen.base import BaseGenerator


class ObjectGenerator(BaseGenerator):
    """Generator for object/product-based image generation.

    Use this generator when you want to generate images of specific objects,
    products, or items that maintain their physical characteristics and identity
    across generated images.

    Example use cases:
        - Product photography with different backgrounds
        - Object placement in various scenes
        - Product variations and configurations
    """

    @property
    def category(self) -> str:
        return "object"

    def _get_reference_type(self) -> str:
        """Get reference type for object generation."""
        return "SUBJECT_REFERENCE"

    def _build_prompt(self, user_prompt: str, reference_count: int) -> str:
        """Build optimized prompt for object generation.

        Enhances the user prompt with object-specific guidance to ensure
        the generated images maintain the subject's physical characteristics.

        Args:
            user_prompt: The user's original prompt.
            reference_count: Number of reference images provided.

        Returns:
            Optimized prompt for object generation.
        """
        # Object-specific prompt template
        # Focuses on maintaining physical attributes, shape, texture, and details
        prompt_parts = [
            f"Generate an image featuring the exact object shown in the reference images.",
            f"Maintain the precise shape, color, texture, and all physical details of the subject.",
            user_prompt,
            "Ensure high fidelity to the reference object's appearance and characteristics.",
            "Photorealistic quality with accurate lighting and shadows.",
        ]

        return " ".join(prompt_parts)
