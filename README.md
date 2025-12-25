# ref-image-gen

Reference-based image generation using Google Gemini. Replace expensive fine-tuning with instant, reference-based generation.

## Installation

```bash
pip install ref-image-gen
```

Or install from source:

```bash
pip install -e .
```

## Prerequisites

1. Google Cloud project with Vertex AI API enabled
2. Service account JSON file with appropriate permissions

## Quick Start

```python
from ref_image_gen import ObjectGenerator, GenerationConfig, AspectRatio
from PIL import Image

# Initialize generator with service account
generator = ObjectGenerator(
    credentials_path="service-account.json",
)

# Load reference images
reference_images = [
    Image.open("product_1.jpg"),
    Image.open("product_2.jpg"),
    Image.open("product_3.jpg"),
    Image.open("product_4.jpg"),
]

# Generate images
result = generator.generate(
    prompt="Product on a white background with soft studio lighting",
    reference_images=reference_images,
    config=GenerationConfig(
        num_images=2,
        aspect_ratio=AspectRatio.SQUARE_1_1,
    ),
)

# Access generated images
pil_images = result.to_pil()

# Or get base64 for API responses
base64_images = result.to_base64()
```

## Generators

### ObjectGenerator

For generating images of specific objects/products while maintaining their physical characteristics.

```python
from ref_image_gen import ObjectGenerator
from PIL import Image

generator = ObjectGenerator(credentials_path="service-account.json")

images = [Image.open("watch1.jpg"), Image.open("watch2.jpg")]
result = generator.generate(
    prompt="The watch displayed on a marble surface",
    reference_images=images,
)
```

### StyleGenerator

For applying artistic styles from reference images to new content.

```python
from ref_image_gen import StyleGenerator
from PIL import Image

generator = StyleGenerator(credentials_path="service-account.json")

style_refs = [Image.open("style1.jpg"), Image.open("style2.jpg")]
result = generator.generate(
    prompt="A mountain landscape at sunset",
    reference_images=style_refs,
)
```

### FaceGenerator

For generating portraits while preserving a person's identity.

```python
from ref_image_gen import FaceGenerator
from PIL import Image

generator = FaceGenerator(credentials_path="service-account.json")

face_refs = [Image.open("face1.jpg"), Image.open("face2.jpg")]
result = generator.generate(
    prompt="Professional headshot in an office setting",
    reference_images=face_refs,
)
```

## Configuration

```python
from ref_image_gen import GenerationConfig, AspectRatio, ModelType, Resolution

config = GenerationConfig(
    num_images=4,                             # Number of images to generate
    aspect_ratio=AspectRatio.LANDSCAPE_16_9,  # Output aspect ratio
    model=ModelType.NANO,                     # Model: NANO or PRO
    resolution=Resolution.RES_2K,             # Resolution (Pro only): 1K, 2K, 4K
    output_mime_type="image/png",             # Output format
)
```

### Supported Aspect Ratios

- `ULTRA_WIDE_21_9` - 21:9
- `LANDSCAPE_16_9` - 16:9
- `LANDSCAPE_3_2` - 3:2
- `LANDSCAPE_4_3` - 4:3
- `LANDSCAPE_5_4` - 5:4
- `SQUARE_1_1` - 1:1
- `PORTRAIT_4_5` - 4:5
- `PORTRAIT_3_4` - 3:4
- `PORTRAIT_2_3` - 2:3
- `PORTRAIT_9_16` - 9:16

### Models

- `ModelType.NANO` - Fast, efficient model (default)
- `ModelType.PRO` - Higher quality, supports resolution selection

## Working with Results

```python
result = generator.generate(...)

# PIL Images (for display or processing)
pil_images = result.to_pil()

# Raw image bytes
for img_bytes in result.images:
    with open("output.png", "wb") as f:
        f.write(img_bytes)

# Base64 encoded (for JSON APIs)
base64_strings = result.to_base64()

# Data URIs (for HTML embedding)
data_uris = result.to_data_uris()

# Save to disk
saved_paths = result.save_images(output_dir="./output", prefix="generated")

# Access metadata
print(result.prompt_used)
print(result.metadata)
```

## Input Formats

Reference images can be provided as:

- **PIL Images**: `Image.open("file.jpg")`
- **Bytes**: Raw image bytes

```python
# Using PIL Images
from PIL import Image
images = [Image.open(f) for f in ["img1.jpg", "img2.jpg"]]

# Using bytes
with open("image.jpg", "rb") as f:
    image_bytes = f.read()
images = [image_bytes]

result = generator.generate(prompt="...", reference_images=images)
```

## Demo

Run the Gradio demo to test the package:

```bash
# Install demo dependencies
pip install -e ".[demo]"

# Place your service-account.json in the project root
# Run the demo
python demo/app.py
```

Open http://localhost:7860 in your browser.

## Comparison with Fine-Tuning

| Aspect | Fine-Tuning (Flux/LoRA) | Reference-Based (This Package) |
|--------|-------------------------|-------------------------------|
| Setup Time | 15-60 minutes training | Instant |
| Cost | Training + inference | Inference only |
| Flexibility | Fixed to trained concept | Any reference images |
| Quality | High after training | High immediately |
| Storage | Model weights per user | Just reference images |

## License

MIT
