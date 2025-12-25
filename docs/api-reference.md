# API Reference

Technical documentation for the Reference-Based Image Generation API.

---

## Table of Contents

1. [Constraints](#constraints)
2. [Generation Modes](#generation-modes)
3. [Generation Types](#generation-types)
4. [Multi-Reference Combinations](#multi-reference-combinations)
5. [Input Schema](#input-schema)
6. [Output Schema](#output-schema)
7. [Python Usage Examples](#python-usage-examples)
8. [Async & Batch Processing](#async--batch-processing)
9. [Errors](#errors)
10. [Rate Limits](#rate-limits)

---

## Constraints

### Reference Images

| Constraint | Value | Description |
|------------|-------|-------------|
| Min per category | 1 | At least 1 reference image required |
| Max per category | 4 | Maximum 4 reference images per category |
| Max reference sets | 2 | Maximum 2 different categories combined |
| Supported formats | PNG, JPEG, WebP | Standard image formats |
| Max file size | 20MB | Per image |

### Aspect Ratios

| Value | Name | Use Case |
|-------|------|----------|
| `21:9` | Ultra Wide | Cinematic banners |
| `16:9` | Landscape | Video thumbnails, headers |
| `3:2` | Landscape | Photography standard |
| `4:3` | Landscape | Presentations |
| `5:4` | Landscape | Print media |
| `1:1` | Square | Social media posts |
| `4:5` | Portrait | Instagram posts |
| `3:4` | Portrait | Mobile screens |
| `2:3` | Portrait | Photography standard |
| `9:16` | Portrait | Stories, reels |

### Models

| Model | ID | Resolution Support | Description |
|-------|----|--------------------|-------------|
| Nano | `nano` | 1K only | Fast, cost-effective |
| Pro | `pro` | 1K, 2K, 4K | Higher quality, more detail |

### Resolution (Pro Model Only)

| Value | Approximate Size |
|-------|------------------|
| `1K` | ~1024px |
| `2K` | ~2048px |
| `4K` | ~4096px |

---

## Generation Modes

The API supports two generation modes:

### Text-to-Image (Default)

Generate new images from scratch using a text prompt and reference images.

```
Reference Images + Prompt → Generated Image
```

**Requirements:**
- Prompt is **required**
- 1-4 reference images per category
- Up to 2 reference sets

**Example:**
```json
{
  "prompt": "Product on a marble table with soft lighting",
  "reference_sets": [{"category": "object", "images": [...]}]
}
```

---

### Image-to-Image

Transform an existing source image using your fine-tuning references.

```
Source Image + Reference Images + Prompt (optional) → Transformed Image
```

**Requirements:**
- Source image is **required**
- Prompt is **optional** (auto-generated if empty)
- 1-4 reference images per category
- Up to 2 reference sets

**Behavior:**
- The source image provides the composition, layout, and scene
- Reference images apply the fine-tuning (style, object identity, or face)
- Optional prompt adds additional guidance

**Auto-Generated Prompts:**

When no prompt is provided, the system generates context-aware prompts:

| Fine-Tuning | Auto-Generated Prompt |
|-------------|----------------------|
| Style | "Transform this image using the artistic style from the reference" |
| Face | "Transform this image, preserving the facial identity from the reference" |
| Object | "Transform this image, incorporating the exact object from the reference" |
| Face + Style | "Transform this image with the person's face in the artistic style" |
| Face + Object | "Transform this image with the person holding/using the product" |
| Object + Style | "Transform this image with the product in the artistic style" |
| Face + Face | "Transform this image with both people's identities preserved" |
| Object + Object | "Transform this image with both products incorporated" |
| Style + Style | "Transform this image blending both artistic styles" |

**Example:**
```json
{
  "prompt": "",
  "reference_sets": [{"category": "style", "images": [...]}],
  "config": {
    "source_image": "<source_image_bytes>"
  }
}
```

---

## Generation Types

### Object Generation

Generates images preserving the **exact appearance** of a product/object.

**Use Cases:**
- Product photography
- E-commerce catalogs
- Marketing materials

**Behavior:**
- Preserves exact shape, colors, design, branding, labels
- Places object in new scenes/contexts
- Maintains product identity across variations

---

### Style Generation

Generates images applying **artistic characteristics only** from reference images.

**Use Cases:**
- Brand-consistent content
- Artistic transformations
- Style-matched marketing

**Behavior:**
- Extracts color palette, brushwork, textures, lighting techniques
- Does NOT copy subjects, objects, or composition from reference
- Applies artistic essence to new content described in prompt

---

### Face Generation

Generates images preserving **facial identity and likeness** of a person.

**Use Cases:**
- Person in new scenes
- Avatar generation
- Marketing with consistent identity

**Behavior:**
- Maintains facial features, identity, likeness
- Does NOT copy pose, clothing, background, or expression
- Places person in new contexts as described in prompt

---

## Multi-Reference Combinations

Combine up to 2 reference sets for complex generations.

### Valid Combinations (9 total)

| Combination | Use Case | Example Prompt |
|-------------|----------|----------------|
| **Object** | Product shots | "Product on marble surface with soft lighting" |
| **Style** | Artistic content | "Mountain landscape" (in reference style) |
| **Face** | Person scenes | "Person walking on beach at sunset" |
| **Face + Face** | Two people together | "Two people having coffee at a cafe" |
| **Face + Object** | Person with product | "Person holding the product in a studio" |
| **Face + Style** | Stylized portrait | "Person in a garden" (in artistic style) |
| **Object + Object** | Multiple products | "Both products arranged on a table" |
| **Object + Style** | Stylized product | "Product floating in space" (in artistic style) |
| **Style + Style** | Blended styles | "City street scene" (blending both styles) |

---

## Input Schema

### Single Category Generation

```json
{
  "category": "object | style | face",
  "prompt": "string (required for text-to-image, optional for image-to-image)",
  "reference_images": [
    "bytes | base64 | PIL.Image (1-4 images, required)"
  ],
  "config": {
    "aspect_ratio": "1:1 | 16:9 | 9:16 | ... (optional, default: 1:1)",
    "model": "nano | pro (optional, default: nano)",
    "resolution": "1K | 2K | 4K (optional, pro model only)",
    "source_image": "bytes | base64 | PIL.Image (optional, enables image-to-image)"
  }
}
```

### Multi-Reference Generation

```json
{
  "prompt": "string (required for text-to-image, optional for image-to-image)",
  "reference_sets": [
    {
      "category": "object | style | face",
      "images": ["bytes | base64 | PIL.Image (1-4 images)"]
    },
    {
      "category": "object | style | face",
      "images": ["bytes | base64 | PIL.Image (1-4 images)"]
    }
  ],
  "config": {
    "aspect_ratio": "1:1 | 16:9 | 9:16 | ... (optional, default: 1:1)",
    "model": "nano | pro (optional, default: nano)",
    "resolution": "1K | 2K | 4K (optional, pro model only)",
    "source_image": "bytes | base64 | PIL.Image (optional, enables image-to-image)"
  }
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `category` | string | Yes | One of: `object`, `style`, `face` |
| `prompt` | string | Conditional | Required for text-to-image, optional for image-to-image |
| `reference_images` | array | Yes | 1-4 reference images |
| `reference_sets` | array | Yes* | 1-2 reference sets (*for multi-reference) |
| `config.aspect_ratio` | string | No | Output aspect ratio (default: `1:1`) |
| `config.model` | string | No | Model to use (default: `nano`) |
| `config.resolution` | string | No | Output resolution (Pro model only) |
| `config.source_image` | bytes/base64/PIL | No | Source image for image-to-image mode |

---

## Output Schema

### Response Structure

```json
{
  "images": ["bytes (raw image data)"],
  "mime_type": "image/png",
  "prompt_used": "string - The optimized prompt sent to the model",
  "metadata": {
    "category": "object | style | face",
    "combination": "face+object | style+style | ... (multi-ref only)",
    "mode": "text-to-image | image-to-image",
    "reference_count": 4,
    "reference_sets": [
      {"category": "face", "image_count": 2},
      {"category": "object", "image_count": 2}
    ],
    "config": {
      "aspect_ratio": "1:1",
      "model": "nano",
      "resolution": null
    }
  }
}
```

### Output Formats

The API returns raw bytes. Available conversion methods:

| Method | Output | Use Case |
|--------|--------|----------|
| `images` | `list[bytes]` | Direct storage, streaming |
| `to_base64()` | `list[str]` | JSON APIs, web transport |
| `to_pil()` | `list[PIL.Image]` | Image processing, manipulation |
| `save_images(dir, prefix)` | Files on disk | Local storage |

### Example: Accessing Output

```python
result = generator.generate(prompt="...", reference_images=[...])

# Raw bytes (for storage/streaming)
raw_bytes = result.images[0]

# Base64 (for JSON response)
base64_strings = result.to_base64()

# PIL Image (for processing)
pil_images = result.to_pil()

# Save to disk
result.save_images(output_dir="./outputs", prefix="generated")
```

---

## Python Usage Examples

### Single Category (Text-to-Image)

```python
from ref_image_gen import ObjectGenerator, GenerationConfig, AspectRatio
from PIL import Image

generator = ObjectGenerator(credentials_path="service-account.json")

result = generator.generate(
    prompt="Product on marble surface with soft lighting",
    reference_images=[
        Image.open("product1.jpg"),
        Image.open("product2.jpg"),
    ],
    config=GenerationConfig(
        aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    ),
)

# Get base64 for JSON API response
response = {
    "images": result.to_base64(),
    "mime_type": result.mime_type,
    "metadata": result.metadata,
}
```

### Multi-Reference (Text-to-Image)

```python
from ref_image_gen import MultiRefGenerator, GenerationConfig, ReferenceSet, CategoryType
from PIL import Image

generator = MultiRefGenerator(credentials_path="service-account.json")

result = generator.generate(
    prompt="Person holding the product in a modern kitchen",
    reference_sets=[
        ReferenceSet(
            category=CategoryType.FACE,
            images=[Image.open("face1.jpg"), Image.open("face2.jpg")],
        ),
        ReferenceSet(
            category=CategoryType.OBJECT,
            images=[Image.open("product1.jpg")],
        ),
    ],
)
```

### Image-to-Image Transformation

```python
from ref_image_gen import MultiRefGenerator, GenerationConfig, ReferenceSet, CategoryType
from PIL import Image

generator = MultiRefGenerator(credentials_path="service-account.json")

# Transform a photo using style references
result = generator.generate(
    prompt="",  # Auto-generated: "Transform this image using the artistic style..."
    reference_sets=[
        ReferenceSet(
            category=CategoryType.STYLE,
            images=[Image.open("style1.jpg"), Image.open("style2.jpg")],
        ),
    ],
    config=GenerationConfig(
        source_image=Image.open("photo_to_transform.jpg"),
    ),
)
```

### Image-to-Image with Custom Prompt

```python
result = generator.generate(
    prompt="Apply the artistic style with emphasis on warm colors",  # Custom guidance
    reference_sets=[
        ReferenceSet(category=CategoryType.STYLE, images=[style_img]),
    ],
    config=GenerationConfig(
        source_image=source_img,
    ),
)
```

---

## Async & Batch Processing

### Single Async Generation

Non-blocking for web server integration:

```python
import asyncio
from ref_image_gen import MultiRefGenerator, ReferenceSet, CategoryType

async def generate_image(prompt, face_img):
    generator = MultiRefGenerator(credentials_path="service-account.json")

    result = await generator.generate_async(
        prompt=prompt,
        reference_sets=[
            ReferenceSet(category=CategoryType.FACE, images=[face_img]),
        ],
    )
    return result.to_base64()

# In async context (FastAPI, aiohttp, etc.)
images = await generate_image("Person on beach", face_img)
```

### Batch Parallel Generation

Generate multiple images concurrently:

```python
import asyncio
from ref_image_gen import MultiRefGenerator, ReferenceSet, CategoryType
from ref_image_gen.async_utils import batch_generate_async, GenerationRequest

async def generate_batch():
    generator = MultiRefGenerator(credentials_path="service-account.json")

    requests = [
        GenerationRequest(
            prompt="Person on a beach",
            reference_sets=[ReferenceSet(category=CategoryType.FACE, images=[face_img])],
            request_id="beach",
        ),
        GenerationRequest(
            prompt="Person in a city",
            reference_sets=[ReferenceSet(category=CategoryType.FACE, images=[face_img])],
            request_id="city",
        ),
        GenerationRequest(
            prompt="Person in a forest",
            reference_sets=[ReferenceSet(category=CategoryType.FACE, images=[face_img])],
            request_id="forest",
        ),
    ]

    batch_result = await batch_generate_async(
        generator=generator,
        requests=requests,
        max_concurrent=3,
    )

    return {
        "successful": batch_result.successful,
        "failed": batch_result.failed,
        "results": {
            req_id: result.to_base64()
            for req_id, result in batch_result.results
        },
        "errors": {
            req_id: str(error)
            for req_id, error in batch_result.errors
        },
    }
```

### Performance Comparison

| Mode | 4 Images | Use Case |
|------|----------|----------|
| Sequential | ~40s | Simple scripts |
| Async (single) | ~40s | Non-blocking web servers |
| Batch Parallel | ~12s | Bulk generation |

---

## Errors

### Error Types

| Error | Cause | Resolution |
|-------|-------|------------|
| `ValueError: At least one reference image is required` | Empty reference images | Provide 1-4 reference images |
| `ValueError: Maximum 4 reference images supported` | Too many images per category | Reduce to 4 or fewer images |
| `ValueError: Maximum 2 reference sets supported` | Too many categories combined | Use at most 2 reference sets |
| `ValueError: At least one image is required per reference set` | Empty reference set | Each set needs 1+ images |
| `ValueError: Resolution option is only available for Pro model` | Resolution set with Nano model | Use Pro model or remove resolution |
| `RuntimeError: No images were generated by the model` | Model failed to generate | Retry or adjust prompt |
| `AuthenticationError` | Invalid credentials | Check service account JSON |
| `PermissionError` | API not enabled | Enable Vertex AI API in GCP |

### HTTP Status Code Mapping (for REST APIs)

| Error Type | Suggested HTTP Status |
|------------|----------------------|
| `ValueError` | 400 Bad Request |
| `AuthenticationError` | 401 Unauthorized |
| `PermissionError` | 403 Forbidden |
| `RuntimeError` | 500 Internal Server Error |
| Rate limit exceeded | 429 Too Many Requests |

---

## Rate Limits

### Google Vertex AI Limits

| Limit Type | Value | Scope |
|------------|-------|-------|
| Requests per minute | 60 | Per project |
| Requests per day | 1,000 | Per project (free tier) |
| Concurrent requests | 5 | Recommended max |
| Request timeout | 60s | Per request |

### Recommendations

1. **Implement exponential backoff** for retries
2. **Use batch parallel** with `max_concurrent=3-5` to avoid rate limits
3. **Cache results** when possible to reduce API calls
4. **Monitor usage** in GCP Console

### Batch Processing

For multiple generations, use batch parallel mode:

```python
# Sequential: ~40s for 4 images (10s each)
# Parallel:   ~12s for 4 images (concurrent)

batch_result = await batch_generate_async(
    generator=generator,
    requests=requests,
    max_concurrent=3  # Respect rate limits
)

print(f"Successful: {batch_result.successful}")
print(f"Failed: {batch_result.failed}")
```

---

## Quick Reference

### Minimum Viable Request

```json
{
  "category": "object",
  "prompt": "Product on white background",
  "reference_images": ["<image_bytes>"]
}
```

### Full Request Example (Text-to-Image)

```json
{
  "prompt": "Person holding the product in a modern kitchen",
  "reference_sets": [
    {
      "category": "face",
      "images": ["<face_image_1>", "<face_image_2>"]
    },
    {
      "category": "object",
      "images": ["<product_image_1>", "<product_image_2>"]
    }
  ],
  "config": {
    "aspect_ratio": "16:9",
    "model": "pro",
    "resolution": "2K"
  }
}
```

### Image-to-Image Example

```json
{
  "prompt": "",
  "reference_sets": [
    {
      "category": "style",
      "images": ["<style_image_1>", "<style_image_2>"]
    }
  ],
  "config": {
    "source_image": "<photo_to_transform>",
    "aspect_ratio": "1:1",
    "model": "nano"
  }
}
```
