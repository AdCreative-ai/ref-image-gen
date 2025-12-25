"""Gradio demo for ref-image-gen package with multi-reference support.

Run with: python demo/app.py
Or: gradio demo/app.py
"""

import asyncio
import os
import time
import gradio as gr
from PIL import Image

# Add src to path for local development
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ref_image_gen import (
    MultiRefGenerator,
    GenerationConfig,
    AspectRatio,
    ModelType,
    Resolution,
    CategoryType,
    ReferenceSet,
    GenerationRequest,
    batch_generate_async,
)


# Service account file path - look in project root by default
CREDENTIALS_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(__file__), "..", "service-account.json")
)
LOCATION = os.environ.get("GCP_LOCATION", "global")

# Lazy initialization of generator
_generator = None


def get_generator():
    """Get or create the multi-reference generator."""
    global _generator
    if not os.path.exists(CREDENTIALS_PATH):
        raise ValueError(f"Credentials file not found: {CREDENTIALS_PATH}")

    if _generator is None:
        _generator = MultiRefGenerator(
            credentials_path=CREDENTIALS_PATH,
            location=LOCATION,
        )
    return _generator


def load_pil_images(files) -> list[Image.Image]:
    """Load files as PIL Images."""
    if not files:
        return []
    images = []
    for f in files:
        if isinstance(f, str):
            images.append(Image.open(f))
        elif hasattr(f, 'name'):
            images.append(Image.open(f.name))
        else:
            images.append(Image.open(f))
    return images


def build_reference_sets(cat1, files1, cat2, files2):
    """Build reference sets from UI inputs."""
    reference_sets = []

    if files1:
        images1 = load_pil_images(files1)
        reference_sets.append(ReferenceSet(
            category=CategoryType(cat1.lower()),
            images=images1,
        ))

    if files2:
        images2 = load_pil_images(files2)
        reference_sets.append(ReferenceSet(
            category=CategoryType(cat2.lower()),
            images=images2,
        ))

    return reference_sets


def build_config(aspect_ratio, model_type, resolution, source_image=None):
    """Build generation config from UI inputs."""
    ratio_map = {
        "21:9 (Ultra Wide)": AspectRatio.ULTRA_WIDE_21_9,
        "16:9 (Landscape)": AspectRatio.LANDSCAPE_16_9,
        "3:2 (Landscape)": AspectRatio.LANDSCAPE_3_2,
        "4:3 (Landscape)": AspectRatio.LANDSCAPE_4_3,
        "5:4 (Landscape)": AspectRatio.LANDSCAPE_5_4,
        "1:1 (Square)": AspectRatio.SQUARE_1_1,
        "4:5 (Portrait)": AspectRatio.PORTRAIT_4_5,
        "3:4 (Portrait)": AspectRatio.PORTRAIT_3_4,
        "2:3 (Portrait)": AspectRatio.PORTRAIT_2_3,
        "9:16 (Portrait)": AspectRatio.PORTRAIT_9_16,
    }

    model = ModelType.PRO if "pro" in model_type.lower() else ModelType.NANO

    res_map = {
        "1K": Resolution.RES_1K,
        "2K": Resolution.RES_2K,
        "4K": Resolution.RES_4K,
    }
    res = res_map.get(resolution) if model == ModelType.PRO else None

    # Load source image if provided
    source_img = None
    if source_image:
        source_img = Image.open(source_image)

    return GenerationConfig(
        aspect_ratio=ratio_map.get(aspect_ratio, AspectRatio.SQUARE_1_1),
        model=model,
        resolution=res,
        source_image=source_img,
    )


async def generate_images(
    prompt: str,
    # Reference Set 1
    cat1: str,
    files1: list,
    # Reference Set 2
    cat2: str,
    files2: list,
    # Source image for image-to-image mode
    source_image,
    # Generation settings
    aspect_ratio: str,
    model_type: str,
    resolution: str,
    use_mock: bool,
    use_async: bool,
):
    """Generate images using multi-reference generator (sync or async)."""
    reference_sets = build_reference_sets(cat1, files1, cat2, files2)

    if not reference_sets:
        raise gr.Error("Please upload at least one reference set")

    # Determine mode
    has_source = source_image is not None
    mode_type = "image-to-image" if has_source else "text-to-image"

    # Prompt is required for text-to-image, optional for image-to-image
    if not has_source and not prompt:
        raise gr.Error("Please enter a prompt for text-to-image mode")

    # Use empty string if no prompt provided (for image-to-image)
    prompt = prompt or ""

    config = build_config(aspect_ratio, model_type, resolution, source_image)

    if use_mock:
        return create_mock_output(reference_sets, prompt, config), f"Mock mode ({mode_type})"

    generator = get_generator()
    start_time = time.time()

    if use_async:
        # Run async generation (non-blocking)
        result = await generator.generate_async(
            prompt=prompt,
            reference_sets=reference_sets,
            config=config,
        )
        mode = "Async"
    else:
        # Run sync generation in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generator.generate(
                prompt=prompt,
                reference_sets=reference_sets,
                config=config,
            )
        )
        mode = "Sync"

    elapsed = time.time() - start_time
    timing_info = f"{mode} {mode_type}: {elapsed:.2f}s"

    return result.to_pil(), timing_info


async def generate_batch_async(
    prompts_text: str,
    # Reference Set 1
    cat1: str,
    files1: list,
    # Reference Set 2
    cat2: str,
    files2: list,
    # Generation settings
    aspect_ratio: str,
    model_type: str,
    resolution: str,
    max_concurrent: int,
    use_mock: bool,
):
    """Generate multiple images in parallel with different prompts."""
    reference_sets = build_reference_sets(cat1, files1, cat2, files2)

    if not reference_sets:
        raise gr.Error("Please upload at least one reference set")

    # Parse prompts (one per line)
    prompts = [p.strip() for p in prompts_text.strip().split("\n") if p.strip()]

    if not prompts:
        raise gr.Error("Please enter at least one prompt (one per line)")

    config = build_config(aspect_ratio, model_type, resolution)

    if use_mock:
        # Generate mock for each prompt
        all_images = []
        for prompt in prompts:
            mock_imgs = create_mock_output(reference_sets, prompt, config)
            all_images.extend(mock_imgs)
        return all_images, f"Mock mode - {len(prompts)} prompts"

    generator = get_generator()

    # Build batch requests
    requests = [
        GenerationRequest(
            prompt=prompt,
            reference_sets=reference_sets,
            config=config,
            request_id=f"prompt_{i}",
        )
        for i, prompt in enumerate(prompts)
    ]

    start_time = time.time()

    # Run batch generation
    batch_result = await batch_generate_async(
        generator=generator,
        requests=requests,
        max_concurrent=int(max_concurrent),
    )

    elapsed = time.time() - start_time

    # Collect all images
    all_images = []
    for _, result in batch_result.results:
        all_images.extend(result.to_pil())

    timing_info = (
        f"Batch parallel: {elapsed:.2f}s total\n"
        f"Prompts: {len(prompts)}, Successful: {batch_result.successful}, Failed: {batch_result.failed}\n"
        f"Avg per prompt: {elapsed/len(prompts):.2f}s"
    )

    if batch_result.errors:
        timing_info += f"\nErrors: {[str(e) for _, e in batch_result.errors]}"

    return all_images, timing_info


def create_mock_output(
    reference_sets: list[ReferenceSet],
    prompt: str,
    config: GenerationConfig
) -> list[Image.Image]:
    """Create mock output for testing without GCP credentials."""
    from PIL import ImageDraw

    # Determine output size based on aspect ratio
    base_size = 512
    ratio_sizes = {
        AspectRatio.ULTRA_WIDE_21_9: (int(base_size * 21 / 9), base_size),
        AspectRatio.LANDSCAPE_16_9: (int(base_size * 16 / 9), base_size),
        AspectRatio.LANDSCAPE_3_2: (int(base_size * 3 / 2), base_size),
        AspectRatio.LANDSCAPE_4_3: (int(base_size * 4 / 3), base_size),
        AspectRatio.LANDSCAPE_5_4: (int(base_size * 5 / 4), base_size),
        AspectRatio.SQUARE_1_1: (base_size, base_size),
        AspectRatio.PORTRAIT_4_5: (base_size, int(base_size * 5 / 4)),
        AspectRatio.PORTRAIT_3_4: (base_size, int(base_size * 4 / 3)),
        AspectRatio.PORTRAIT_2_3: (base_size, int(base_size * 3 / 2)),
        AspectRatio.PORTRAIT_9_16: (base_size, int(base_size * 16 / 9)),
    }
    size = ratio_sizes.get(config.aspect_ratio, (base_size, base_size))

    # Build combination string
    combo = " + ".join([rs.category.value.title() for rs in reference_sets])
    total_refs = sum(len(rs.images) for rs in reference_sets)

    mock_images = []
    img = Image.new("RGB", size, color=(40, 40, 50))
    draw = ImageDraw.Draw(img)

    text_lines = [
        "MOCK OUTPUT",
        f"Combination: {combo}",
        f"Total refs: {total_refs}",
        f"Ratio: {config.aspect_ratio.value}",
        f"Model: {config.model.value}",
    ]
    if config.resolution:
        text_lines.append(f"Resolution: {config.resolution.value}")

    text_lines.extend([
        "",
        "Prompt:",
        prompt[:40] + "..." if len(prompt) > 40 else prompt,
    ])

    y = 20
    for line in text_lines:
        draw.text((20, y), line, fill=(200, 200, 200))
        y += 22

    draw.rectangle([0, 0, size[0]-1, size[1]-1], outline=(100, 100, 100), width=2)
    mock_images.append(img)

    return mock_images


def update_resolution_visibility(model_type: str):
    """Show/hide resolution dropdown based on model selection."""
    return gr.update(visible=("pro" in model_type.lower()))


def update_preview(files):
    """Update preview gallery when files are uploaded."""
    if not files:
        return []
    return load_pil_images(files)


# Build the Gradio interface
with gr.Blocks(title="Multi-Reference Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Multi-Reference Image Generator

    Generate images by combining up to **2 reference sets** from categories: **Face**, **Object**, and **Style**.

    Examples: Face + Object, Style + Style, Object + Style, etc.
    """)

    with gr.Row():
        # Left column - Reference Sets
        with gr.Column(scale=2):
            gr.Markdown("### Reference Sets (up to 2)")

            # Reference Set 1
            with gr.Group():
                gr.Markdown("**Reference Set 1** (required)")
                with gr.Row():
                    cat1 = gr.Dropdown(
                        choices=["Face", "Object", "Style"],
                        value="Object",
                        label="Category",
                        scale=1,
                    )
                    files1 = gr.File(
                        label="Images (max 4)",
                        file_count="multiple",
                        file_types=["image"],
                        scale=3,
                    )
                preview1 = gr.Gallery(label="Preview", columns=5, height=100)

            # Reference Set 2
            with gr.Group():
                gr.Markdown("**Reference Set 2** (optional)")
                with gr.Row():
                    cat2 = gr.Dropdown(
                        choices=["Face", "Object", "Style"],
                        value="Face",
                        label="Category",
                        scale=1,
                    )
                    files2 = gr.File(
                        label="Images (max 4)",
                        file_count="multiple",
                        file_types=["image"],
                        scale=3,
                    )
                preview2 = gr.Gallery(label="Preview", columns=5, height=100)

            # Generation settings (shared)
            with gr.Group():
                gr.Markdown("### Generation Settings")

                aspect_ratio = gr.Dropdown(
                    choices=[
                        "21:9 (Ultra Wide)",
                        "16:9 (Landscape)",
                        "3:2 (Landscape)",
                        "4:3 (Landscape)",
                        "5:4 (Landscape)",
                        "1:1 (Square)",
                        "4:5 (Portrait)",
                        "3:4 (Portrait)",
                        "2:3 (Portrait)",
                        "9:16 (Portrait)",
                    ],
                    value="1:1 (Square)",
                    label="Aspect Ratio",
                )

                with gr.Row():
                    model_type = gr.Radio(
                        choices=["Nano", "Pro"],
                        value="Nano",
                        label="Model",
                        info="Pro supports 2K/4K resolution",
                    )

                    resolution = gr.Dropdown(
                        choices=["1K", "2K", "4K"],
                        value="1K",
                        label="Resolution (Pro only)",
                        visible=False,
                    )

                # Mock mode toggle
                use_mock = gr.Checkbox(
                    label="Mock Mode (test UI without GCP)",
                    value=not os.path.exists(CREDENTIALS_PATH),
                )

        # Right column - Generation Modes
        with gr.Column(scale=2):
            with gr.Tabs():
                # Tab 1: Single Generation (Sync vs Async)
                with gr.TabItem("Single Generation"):
                    gr.Markdown("""
                    **Text-to-Image**: Enter a prompt to generate from scratch.
                    **Image-to-Image**: Upload a source image to transform it using your fine-tuning.
                    """)

                    # Source image for image-to-image mode
                    with gr.Group():
                        source_image = gr.Image(
                            label="Source Image (optional - for image-to-image)",
                            type="filepath",
                            height=150,
                        )

                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Required for text-to-image. Optional for image-to-image (adds extra guidance)...",
                        lines=3,
                    )

                    use_async = gr.Checkbox(
                        label="Use Async Mode",
                        value=False,
                        info="Test async vs sync generation (timing shown below)",
                    )

                    generate_btn = gr.Button("Generate", variant="primary", size="lg")

                    timing_info = gr.Textbox(
                        label="Timing Info",
                        interactive=False,
                        lines=1,
                    )

                    output_gallery = gr.Gallery(
                        label="Generated Images",
                        columns=2,
                        height=350,
                    )

                # Tab 2: Batch Parallel Generation
                with gr.TabItem("Batch Parallel"):
                    gr.Markdown("""
                    **Generate multiple scenes in parallel** - Enter one prompt per line.
                    All prompts will use the same reference images but run concurrently.
                    """)

                    batch_prompts = gr.Textbox(
                        label="Prompts (one per line)",
                        placeholder="Person on a beach at sunset\nPerson in a modern office\nPerson walking in a park\nPerson at a coffee shop",
                        lines=6,
                    )

                    max_concurrent = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Max Concurrent Requests",
                        info="Higher = faster but may hit rate limits",
                    )

                    batch_btn = gr.Button("Generate Batch (Parallel)", variant="primary", size="lg")

                    batch_timing = gr.Textbox(
                        label="Batch Timing Info",
                        interactive=False,
                        lines=3,
                    )

                    batch_gallery = gr.Gallery(
                        label="All Generated Images",
                        columns=3,
                        height=350,
                    )

    # Event handlers for preview updates
    files1.change(fn=update_preview, inputs=[files1], outputs=[preview1])
    files2.change(fn=update_preview, inputs=[files2], outputs=[preview2])

    # Show/hide resolution based on model selection
    model_type.change(
        fn=update_resolution_visibility,
        inputs=[model_type],
        outputs=[resolution],
    )

    # Single generation button click
    generate_btn.click(
        fn=generate_images,
        inputs=[
            prompt,
            cat1, files1,
            cat2, files2,
            source_image,
            aspect_ratio,
            model_type,
            resolution,
            use_mock,
            use_async,
        ],
        outputs=[output_gallery, timing_info],
    )

    # Batch generation button click
    batch_btn.click(
        fn=generate_batch_async,
        inputs=[
            batch_prompts,
            cat1, files1,
            cat2, files2,
            aspect_ratio,
            model_type,
            resolution,
            max_concurrent,
            use_mock,
        ],
        outputs=[batch_gallery, batch_timing],
    )

    # Example combinations
    gr.Markdown("""
    ### Example Combinations

    | Combination | Use Case |
    |-------------|----------|
    | **Object** | Product photography |
    | **Face** | Person in a new scene |
    | **Style** | New content in artistic style |
    | **Face + Object** | Person holding/using a product |
    | **Face + Face** | Two people together |
    | **Face + Style** | Person in artistic style |
    | **Object + Object** | Two products together |
    | **Object + Style** | Product in artistic style |
    | **Style + Style** | Blend two artistic styles |

    ### Async vs Batch Comparison
    - **Single Async**: Non-blocking, good for web servers, but still one request at a time
    - **Batch Parallel**: Multiple requests run simultaneously, much faster for generating variations

    ### Tips
    - Use up to 4 reference images per category
    - Be specific in your scene description
    - Pro model supports higher resolutions (2K, 4K)
    - For batch: try 3-4 concurrent requests to avoid rate limiting
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
