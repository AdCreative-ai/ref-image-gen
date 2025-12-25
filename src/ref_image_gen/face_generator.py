"""Face-based image generation using reference images."""

from ref_image_gen.base import BaseGenerator


class FaceGenerator(BaseGenerator):
    """Generator for face/portrait-based image generation.

    Use this generator when you want to generate images featuring a specific
    person's face while maintaining their identity across different contexts,
    poses, and scenarios.

    Example use cases:
        - Generate portraits in different styles or settings
        - Create profile pictures with various backgrounds
        - Generate person in different scenarios while preserving identity
        - Create avatar variations

    Note:
        Ensure you have appropriate consent and rights to use reference
        images of individuals for image generation.
    """

    @property
    def category(self) -> str:
        return "face"

    def _get_reference_type(self) -> str:
        """Get reference type for face generation."""
        return "SUBJECT_REFERENCE"

    def _build_prompt(self, user_prompt: str, reference_count: int) -> str:
        """Build optimized prompt for face generation.

        Enhances the user prompt with face-specific guidance to ensure
        the generated images maintain the person's identity and facial features.

        Args:
            user_prompt: The user's original prompt.
            reference_count: Number of reference images provided.

        Returns:
            Optimized prompt for face generation.
        """
        # Face-specific prompt template
        # Focuses on identity preservation: facial features, structure, expression range
        prompt_parts = [
            "Generate a portrait image featuring the exact person shown in the reference images.",
            "Preserve the precise facial features, face shape, skin tone, and distinguishing characteristics.",
            "Maintain identity consistency with natural facial proportions.",
            user_prompt,
            "High quality portrait with natural skin texture and realistic lighting on the face.",
            "Ensure the person is clearly recognizable as the same individual from references.",
        ]

        return " ".join(prompt_parts)
