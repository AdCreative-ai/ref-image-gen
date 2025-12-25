"""Async utilities for parallel image generation."""

import asyncio
from dataclasses import dataclass
from typing import Optional, Union

from .base import (
    BaseGenerator,
    GenerationConfig,
    GenerationResult,
    ImageInput,
    ReferenceSet,
)
from .multi_generator import MultiRefGenerator


@dataclass
class GenerationRequest:
    """A single generation request for batch processing.

    Attributes:
        prompt: Text description of the desired output.
        reference_images: For single-category generators (ObjectGenerator, etc.).
        reference_sets: For MultiRefGenerator.
        config: Generation configuration options.
        request_id: Optional identifier for tracking this request.
    """

    prompt: str
    reference_images: Optional[list[ImageInput]] = None
    reference_sets: Optional[list[ReferenceSet]] = None
    config: Optional[GenerationConfig] = None
    request_id: Optional[str] = None


@dataclass
class BatchResult:
    """Result of a batch generation operation.

    Attributes:
        results: List of successful GenerationResult objects.
        errors: List of (request_id, error) tuples for failed requests.
        total_requests: Total number of requests processed.
        successful: Number of successful generations.
        failed: Number of failed generations.
    """

    results: list[tuple[Optional[str], GenerationResult]]
    errors: list[tuple[Optional[str], Exception]]

    @property
    def total_requests(self) -> int:
        return len(self.results) + len(self.errors)

    @property
    def successful(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return len(self.errors)


async def batch_generate_async(
    generator: Union[BaseGenerator, MultiRefGenerator],
    requests: list[GenerationRequest],
    max_concurrent: int = 5,
) -> BatchResult:
    """Run multiple generation requests in parallel.

    Args:
        generator: The generator instance to use (ObjectGenerator, StyleGenerator,
                   FaceGenerator, or MultiRefGenerator).
        requests: List of GenerationRequest objects to process.
        max_concurrent: Maximum number of concurrent requests (default 5).
                       Helps avoid rate limiting.

    Returns:
        BatchResult containing successful results and any errors.

    Example:
        ```python
        import asyncio
        from ref_image_gen import MultiRefGenerator, GenerationConfig, ReferenceSet, CategoryType
        from ref_image_gen.async_utils import batch_generate_async, GenerationRequest

        async def main():
            generator = MultiRefGenerator(credentials_path="service-account.json")

            requests = [
                GenerationRequest(
                    prompt="Person on a beach",
                    reference_sets=[ReferenceSet(category=CategoryType.FACE, images=[face_img])],
                    request_id="beach_scene"
                ),
                GenerationRequest(
                    prompt="Person in a city",
                    reference_sets=[ReferenceSet(category=CategoryType.FACE, images=[face_img])],
                    request_id="city_scene"
                ),
            ]

            batch_result = await batch_generate_async(generator, requests)

            for request_id, result in batch_result.results:
                print(f"{request_id}: Generated {len(result.images)} images")

            for request_id, error in batch_result.errors:
                print(f"{request_id}: Failed - {error}")

        asyncio.run(main())
        ```
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_request(request: GenerationRequest, max_retries: int = 2) -> tuple[Optional[str], Union[GenerationResult, Exception]]:
        async with semaphore:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    if isinstance(generator, MultiRefGenerator):
                        if not request.reference_sets:
                            raise ValueError("reference_sets required for MultiRefGenerator")
                        result = await generator.generate_async(
                            prompt=request.prompt,
                            reference_sets=request.reference_sets,
                            config=request.config,
                        )
                    else:
                        if not request.reference_images:
                            raise ValueError("reference_images required for single-category generators")
                        result = await generator.generate_async(
                            prompt=request.prompt,
                            reference_images=request.reference_images,
                            config=request.config,
                        )
                    return (request.request_id, result)
                except Exception as e:
                    last_error = e
                    # Retry on transient errors like "Event loop is closed"
                    if attempt < max_retries and ("loop" in str(e).lower() or "closed" in str(e).lower()):
                        await asyncio.sleep(0.5 * (attempt + 1))  # Backoff
                        continue
                    break
            return (request.request_id, last_error)

    # Run all requests concurrently
    tasks = [process_request(req) for req in requests]
    outcomes = await asyncio.gather(*tasks)

    # Separate successes and failures
    results = []
    errors = []
    for request_id, outcome in outcomes:
        if isinstance(outcome, Exception):
            errors.append((request_id, outcome))
        else:
            results.append((request_id, outcome))

    return BatchResult(results=results, errors=errors)


async def generate_variations_async(
    generator: Union[BaseGenerator, MultiRefGenerator],
    prompts: list[str],
    reference_images: Optional[list[ImageInput]] = None,
    reference_sets: Optional[list[ReferenceSet]] = None,
    config: Optional[GenerationConfig] = None,
    max_concurrent: int = 5,
) -> BatchResult:
    """Generate multiple variations with different prompts but same references.

    Useful for generating the same subject/product in different scenes.

    Args:
        generator: The generator instance to use.
        prompts: List of different prompts to generate.
        reference_images: Reference images (for single-category generators).
        reference_sets: Reference sets (for MultiRefGenerator).
        config: Shared generation configuration.
        max_concurrent: Maximum concurrent requests.

    Returns:
        BatchResult containing all generated variations.

    Example:
        ```python
        prompts = [
            "Product on white background",
            "Product on wooden table",
            "Product being held by person",
            "Product in outdoor setting",
        ]

        result = await generate_variations_async(
            generator=object_generator,
            prompts=prompts,
            reference_images=product_images,
        )
        ```
    """
    requests = [
        GenerationRequest(
            prompt=prompt,
            reference_images=reference_images,
            reference_sets=reference_sets,
            config=config,
            request_id=f"variation_{i}",
        )
        for i, prompt in enumerate(prompts)
    ]

    return await batch_generate_async(generator, requests, max_concurrent)
