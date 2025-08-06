from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import torch

# Step 1: Generate Image (Stable Diffusion)
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,  # Use float32 for CPU
).to("cpu")  # Changed from cuda to cpu

image = sd_pipe("A king in a vibrant Indian palace dancing with seven fish, colorful and funny").images[0]
image = image.resize((256, 256))  # Lower resolution
image.save("generated_image.png")

# Step 2: Generate Video (Stable Video Diffusion)
svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float32,  # Use float32 for CPU
    variant="fp16"
).to("cpu")  # Changed from cuda to cpu

frames = svd_pipe(
    image,
    num_frames=8,  # Fewer frames
    decode_chunk_size=2,  # Smaller chunks
).frames

export_to_video(frames, "output_video.mp4", fps=8)