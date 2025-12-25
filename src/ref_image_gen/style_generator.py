"""Style-based image generation using reference images."""

from ref_image_gen.base import BaseGenerator


class StyleGenerator(BaseGenerator):
    """Generator for style-based image generation.

    Use this generator when you want to apply a specific artistic style
    from reference images to new content. The generator extracts and
    applies visual style characteristics like color palette, brushwork,
    artistic technique, and overall aesthetic.

    Example use cases:
        - Apply an artist's style to new subjects
        - Create images matching a brand's visual style
        - Generate content in a specific artistic movement
        - Transfer illustration or photography styles
    """

    @property
    def category(self) -> str:
        return "style"

    def _get_reference_type(self) -> str:
        """Get reference type for style generation."""
        return "STYLE_REFERENCE"

    def _build_prompt(self, user_prompt: str, reference_count: int) -> str:
        """Build optimized prompt for style generation.

        Enhances the user prompt with style-specific guidance to ensure
        the generated images capture the artistic style from references.

        Args:
            user_prompt: The user's original prompt.
            reference_count: Number of reference images provided.

        Returns:
            Optimized prompt for style generation.
        """
        # Style-specific prompt template
        # Focuses on artistic elements: colors, textures, techniques, mood
        prompt_parts = [
            "Generate an image that captures the exact artistic style shown in the reference images.",
            "Match the color palette, brushwork, texture, lighting style, and overall aesthetic.",
            user_prompt,
            "Maintain consistent artistic technique and visual mood throughout.",
            "Apply the style naturally while preserving the subject's recognizability.",
        ]

        return " ".join(prompt_parts)
