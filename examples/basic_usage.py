"""Basic usage examples for ref-image-gen package."""

from ref_image_gen import (
    ObjectGenerator,
    StyleGenerator,
    FaceGenerator,
    GenerationConfig,
    AspectRatio,
)


def object_generation_example():
    """Example: Generate product images with different backgrounds."""

    # Initialize the object generator
    generator = ObjectGenerator(
        project_id="your-gcp-project-id",
        location="us-central1",
    )

    # GCS URIs to reference images of the product (provided by your backend)
    reference_images = [
        "gs://your-bucket/products/shoe_front.jpg",
        "gs://your-bucket/products/shoe_side.jpg",
        "gs://your-bucket/products/shoe_back.jpg",
        "gs://your-bucket/products/shoe_detail.jpg",
    ]

    # Configure generation settings
    config = GenerationConfig(
        num_images=4,
        aspect_ratio=AspectRatio.SQUARE_1_1,
        guidance_scale=7.5,
    )

    # Generate images
    result = generator.generate(
        prompt="The shoe on a white marble surface with soft studio lighting",
        reference_image_uris=reference_images,
        config=config,
    )

    # Save generated images
    saved_paths = result.save_images(output_dir="./output", prefix="product")
    print(f"Generated {len(saved_paths)} images: {saved_paths}")
    print(f"Prompt used: {result.prompt_used}")


def style_generation_example():
    """Example: Apply artistic style to new content."""

    generator = StyleGenerator(
        project_id="your-gcp-project-id",
        location="us-central1",
    )

    # Reference images showing the desired artistic style
    style_references = [
        "gs://your-bucket/styles/watercolor_1.jpg",
        "gs://your-bucket/styles/watercolor_2.jpg",
        "gs://your-bucket/styles/watercolor_3.jpg",
        "gs://your-bucket/styles/watercolor_4.jpg",
        "gs://your-bucket/styles/watercolor_5.jpg",
    ]

    config = GenerationConfig(
        num_images=2,
        aspect_ratio=AspectRatio.LANDSCAPE_16_9,
        guidance_scale=8.0,
    )

    result = generator.generate(
        prompt="A serene mountain landscape at sunset",
        reference_image_uris=style_references,
        config=config,
    )

    saved_paths = result.save_images(output_dir="./output", prefix="styled")
    print(f"Generated styled images: {saved_paths}")


def face_generation_example():
    """Example: Generate portraits maintaining person identity."""

    generator = FaceGenerator(
        project_id="your-gcp-project-id",
        location="us-central1",
    )

    # Reference images of the person
    face_references = [
        "gs://your-bucket/faces/person_front.jpg",
        "gs://your-bucket/faces/person_side.jpg",
        "gs://your-bucket/faces/person_smile.jpg",
        "gs://your-bucket/faces/person_casual.jpg",
    ]

    config = GenerationConfig(
        num_images=1,
        aspect_ratio=AspectRatio.PORTRAIT_3_4,
        guidance_scale=7.0,
    )

    result = generator.generate(
        prompt="Professional headshot in a modern office setting, wearing business attire",
        reference_image_uris=face_references,
        config=config,
    )

    saved_paths = result.save_images(output_dir="./output", prefix="portrait")
    print(f"Generated portrait: {saved_paths}")


def batch_generation_example():
    """Example: Generate multiple variations with different prompts."""

    generator = ObjectGenerator(
        project_id="your-gcp-project-id",
        location="us-central1",
    )

    reference_images = [
        "gs://your-bucket/products/watch_1.jpg",
        "gs://your-bucket/products/watch_2.jpg",
        "gs://your-bucket/products/watch_3.jpg",
        "gs://your-bucket/products/watch_4.jpg",
    ]

    # Different prompts for various marketing scenarios
    prompts = [
        "Elegant watch on a black velvet surface with dramatic lighting",
        "Watch being worn on a wrist during outdoor adventure",
        "Watch displayed in luxury jewelry store setting",
        "Minimalist flat lay with the watch and leather accessories",
    ]

    # Different aspect ratios for different platforms
    aspect_ratios = [
        AspectRatio.SQUARE_1_1,      # Instagram post
        AspectRatio.LANDSCAPE_16_9,   # Website banner
        AspectRatio.PORTRAIT_9_16,    # Instagram story
        AspectRatio.LANDSCAPE_4_3,    # Product page
    ]

    for i, (prompt, ratio) in enumerate(zip(prompts, aspect_ratios)):
        config = GenerationConfig(
            num_images=1,
            aspect_ratio=ratio,
        )

        result = generator.generate(
            prompt=prompt,
            reference_image_uris=reference_images,
            config=config,
        )

        result.save_images(output_dir="./output", prefix=f"watch_scene_{i}")
        print(f"Generated: {prompt[:50]}... ({ratio.value})")


if __name__ == "__main__":
    # Run the example you want to test
    # object_generation_example()
    # style_generation_example()
    # face_generation_example()
    # batch_generation_example()
    print("Uncomment the example you want to run!")
