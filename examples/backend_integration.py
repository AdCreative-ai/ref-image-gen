"""Example showing how to integrate ref-image-gen with your existing backend.

This example demonstrates how to replace your Flux/fal.ai fine-tuning workflow
with reference-based generation using Vertex AI Imagen.

Includes both synchronous and asynchronous usage patterns.
"""

import asyncio
from typing import Literal

from ref_image_gen import (
    ObjectGenerator,
    StyleGenerator,
    FaceGenerator,
    MultiRefGenerator,
    GenerationConfig,
    AspectRatio,
    ModelType,
    CategoryType,
    ReferenceSet,
    GenerationRequest,
    batch_generate_async,
    generate_variations_async,
)


# Type alias for categories
CategoryTypeLiteral = Literal["object", "style", "face"]


class ImageGenerationService:
    """Service class for handling image generation requests from your backend.

    This replaces your previous Flux fine-tuning + inference workflow:
    - Old: Upload images -> Train LoRA -> Wait -> Inference
    - New: Upload images -> Direct generation with references (no training!)

    Supports both synchronous and asynchronous generation.
    """

    def __init__(self, credentials_path: str, location: str = "global"):
        """Initialize the service with GCP credentials.

        Args:
            credentials_path: Path to service account JSON file.
            location: Vertex AI location.
        """
        self.credentials_path = credentials_path
        self.location = location

        # Initialize single-category generators
        self._generators: dict[str, ObjectGenerator | StyleGenerator | FaceGenerator] = {
            "object": ObjectGenerator(credentials_path, location),
            "style": StyleGenerator(credentials_path, location),
            "face": FaceGenerator(credentials_path, location),
        }

        # Initialize multi-reference generator
        self._multi_generator = MultiRefGenerator(credentials_path, location)

    def generate(
        self,
        category: CategoryTypeLiteral,
        prompt: str,
        reference_images: list[bytes],
        num_images: int = 1,
        aspect_ratio: str = "1:1",
        model: str = "nano",
    ) -> dict:
        """Generate images using reference-based approach (synchronous).

        Args:
            category: Type of generation ("object", "style", or "face").
            prompt: Text description of desired output.
            reference_images: List of images as bytes (max 4).
            num_images: Number of images to generate (1-4).
            aspect_ratio: Output aspect ratio (e.g., "1:1", "16:9").
            model: Model to use ("nano" or "pro").

        Returns:
            Dictionary containing generated images and metadata.
        """
        generator = self._generators.get(category)
        if not generator:
            raise ValueError(f"Unknown category: {category}. Use 'object', 'style', or 'face'.")

        config = self._build_config(num_images, aspect_ratio, model)

        result = generator.generate(
            prompt=prompt,
            reference_images=reference_images,
            config=config,
        )

        return self._format_result(result)

    async def generate_async(
        self,
        category: CategoryTypeLiteral,
        prompt: str,
        reference_images: list[bytes],
        num_images: int = 1,
        aspect_ratio: str = "1:1",
        model: str = "nano",
    ) -> dict:
        """Generate images using reference-based approach (asynchronous).

        Non-blocking version for use in async web frameworks.

        Args:
            category: Type of generation ("object", "style", or "face").
            prompt: Text description of desired output.
            reference_images: List of images as bytes (max 4).
            num_images: Number of images to generate (1-4).
            aspect_ratio: Output aspect ratio (e.g., "1:1", "16:9").
            model: Model to use ("nano" or "pro").

        Returns:
            Dictionary containing generated images and metadata.
        """
        generator = self._generators.get(category)
        if not generator:
            raise ValueError(f"Unknown category: {category}. Use 'object', 'style', or 'face'.")

        config = self._build_config(num_images, aspect_ratio, model)

        result = await generator.generate_async(
            prompt=prompt,
            reference_images=reference_images,
            config=config,
        )

        return self._format_result(result)

    def generate_multi(
        self,
        prompt: str,
        reference_sets: list[dict],
        num_images: int = 1,
        aspect_ratio: str = "1:1",
        model: str = "nano",
    ) -> dict:
        """Generate images with multiple reference sets (synchronous).

        Args:
            prompt: Text description of desired output scene.
            reference_sets: List of dicts with 'category' and 'images' keys.
                Example: [{"category": "face", "images": [bytes1, bytes2]},
                          {"category": "object", "images": [bytes3]}]
            num_images: Number of images to generate (1-4).
            aspect_ratio: Output aspect ratio.
            model: Model to use ("nano" or "pro").

        Returns:
            Dictionary containing generated images and metadata.
        """
        ref_sets = self._build_reference_sets(reference_sets)
        config = self._build_config(num_images, aspect_ratio, model)

        result = self._multi_generator.generate(
            prompt=prompt,
            reference_sets=ref_sets,
            config=config,
        )

        return self._format_result(result)

    async def generate_multi_async(
        self,
        prompt: str,
        reference_sets: list[dict],
        num_images: int = 1,
        aspect_ratio: str = "1:1",
        model: str = "nano",
    ) -> dict:
        """Generate images with multiple reference sets (asynchronous).

        Args:
            prompt: Text description of desired output scene.
            reference_sets: List of dicts with 'category' and 'images' keys.
            num_images: Number of images to generate (1-4).
            aspect_ratio: Output aspect ratio.
            model: Model to use ("nano" or "pro").

        Returns:
            Dictionary containing generated images and metadata.
        """
        ref_sets = self._build_reference_sets(reference_sets)
        config = self._build_config(num_images, aspect_ratio, model)

        result = await self._multi_generator.generate_async(
            prompt=prompt,
            reference_sets=ref_sets,
            config=config,
        )

        return self._format_result(result)

    async def batch_generate_async(
        self,
        requests: list[dict],
        max_concurrent: int = 5,
    ) -> dict:
        """Generate multiple images in parallel.

        Args:
            requests: List of generation request dicts. Each dict should have:
                - prompt: Text description
                - category: "object", "style", or "face" (for single-category)
                - reference_images: List of image bytes (for single-category)
                - OR reference_sets: List of {category, images} dicts (for multi)
                - request_id: Optional identifier
            max_concurrent: Maximum concurrent API calls.

        Returns:
            Dictionary with 'results' and 'errors' lists.
        """
        gen_requests = []
        use_multi = any("reference_sets" in req for req in requests)

        for i, req in enumerate(requests):
            if "reference_sets" in req:
                ref_sets = self._build_reference_sets(req["reference_sets"])
                gen_requests.append(GenerationRequest(
                    prompt=req["prompt"],
                    reference_sets=ref_sets,
                    config=self._build_config(
                        req.get("num_images", 1),
                        req.get("aspect_ratio", "1:1"),
                        req.get("model", "nano"),
                    ),
                    request_id=req.get("request_id", f"request_{i}"),
                ))
            else:
                gen_requests.append(GenerationRequest(
                    prompt=req["prompt"],
                    reference_images=req["reference_images"],
                    config=self._build_config(
                        req.get("num_images", 1),
                        req.get("aspect_ratio", "1:1"),
                        req.get("model", "nano"),
                    ),
                    request_id=req.get("request_id", f"request_{i}"),
                ))

        generator = self._multi_generator if use_multi else self._generators.get(requests[0].get("category", "object"))
        batch_result = await batch_generate_async(generator, gen_requests, max_concurrent)

        return {
            "results": [
                {"request_id": req_id, **self._format_result(result)}
                for req_id, result in batch_result.results
            ],
            "errors": [
                {"request_id": req_id, "error": str(error)}
                for req_id, error in batch_result.errors
            ],
            "total": batch_result.total_requests,
            "successful": batch_result.successful,
            "failed": batch_result.failed,
        }

    def _build_config(self, num_images: int, aspect_ratio: str, model: str) -> GenerationConfig:
        """Build GenerationConfig from simple parameters."""
        ratio_map = {
            "21:9": AspectRatio.ULTRA_WIDE_21_9,
            "16:9": AspectRatio.LANDSCAPE_16_9,
            "3:2": AspectRatio.LANDSCAPE_3_2,
            "4:3": AspectRatio.LANDSCAPE_4_3,
            "5:4": AspectRatio.LANDSCAPE_5_4,
            "1:1": AspectRatio.SQUARE_1_1,
            "4:5": AspectRatio.PORTRAIT_4_5,
            "3:4": AspectRatio.PORTRAIT_3_4,
            "2:3": AspectRatio.PORTRAIT_2_3,
            "9:16": AspectRatio.PORTRAIT_9_16,
        }

        return GenerationConfig(
            num_images=num_images,
            aspect_ratio=ratio_map.get(aspect_ratio, AspectRatio.SQUARE_1_1),
            model=ModelType.PRO if model.lower() == "pro" else ModelType.NANO,
        )

    def _build_reference_sets(self, reference_sets: list[dict]) -> list[ReferenceSet]:
        """Build ReferenceSet objects from simple dicts."""
        return [
            ReferenceSet(
                category=CategoryType(rs["category"].lower()),
                images=rs["images"],
            )
            for rs in reference_sets
        ]

    def _format_result(self, result) -> dict:
        """Format GenerationResult to dictionary."""
        return {
            "success": True,
            "images": result.images,
            "images_base64": result.to_base64(),
            "mime_type": result.mime_type,
            "prompt_used": result.prompt_used,
            "metadata": result.metadata,
        }


# Example FastAPI integration with async support
FASTAPI_EXAMPLE = '''
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from backend_integration import ImageGenerationService

app = FastAPI()
service = ImageGenerationService(
    credentials_path="service-account.json",
    location="global"
)


class GenerateRequest(BaseModel):
    category: str  # "object", "style", or "face"
    prompt: str
    reference_images: list[bytes]
    num_images: int = 1
    aspect_ratio: str = "1:1"
    model: str = "nano"


class MultiGenerateRequest(BaseModel):
    prompt: str
    reference_sets: list[dict]  # [{"category": "face", "images": [...]}, ...]
    num_images: int = 1
    aspect_ratio: str = "1:1"
    model: str = "nano"


class BatchRequest(BaseModel):
    requests: list[dict]
    max_concurrent: int = 5


@app.post("/generate")
async def generate_single(request: GenerateRequest):
    """Single-category generation (async)."""
    try:
        result = await service.generate_async(
            category=request.category,
            prompt=request.prompt,
            reference_images=request.reference_images,
            num_images=request.num_images,
            aspect_ratio=request.aspect_ratio,
            model=request.model,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/multi")
async def generate_multi(request: MultiGenerateRequest):
    """Multi-reference generation (async)."""
    try:
        result = await service.generate_multi_async(
            prompt=request.prompt,
            reference_sets=request.reference_sets,
            num_images=request.num_images,
            aspect_ratio=request.aspect_ratio,
            model=request.model,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/batch")
async def generate_batch(request: BatchRequest):
    """Batch generation - multiple requests in parallel."""
    try:
        result = await service.batch_generate_async(
            requests=request.requests,
            max_concurrent=request.max_concurrent,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''


# Example async usage
async def example_parallel_generation():
    """Example showing parallel generation of multiple prompts."""
    service = ImageGenerationService(
        credentials_path="service-account.json",
        location="global",
    )

    # Load some reference images (in practice, these would come from your storage)
    face_images = [b"face_image_bytes"]  # Replace with actual image bytes
    product_images = [b"product_image_bytes"]

    # Generate multiple scenes with the same person in parallel
    prompts = [
        "Person standing on a beach at sunset",
        "Person in a modern office setting",
        "Person walking in a park",
        "Person at a coffee shop",
    ]

    requests = [
        {
            "prompt": prompt,
            "reference_sets": [{"category": "face", "images": face_images}],
            "request_id": f"scene_{i}",
        }
        for i, prompt in enumerate(prompts)
    ]

    # Run all 4 generations in parallel (respecting max_concurrent)
    result = await service.batch_generate_async(requests, max_concurrent=3)

    print(f"Generated {result['successful']} images successfully")
    print(f"Failed: {result['failed']}")

    for item in result["results"]:
        print(f"  {item['request_id']}: {len(item['images'])} images")


if __name__ == "__main__":
    # Run the async example
    asyncio.run(example_parallel_generation())
