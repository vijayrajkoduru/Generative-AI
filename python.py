from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()  # Optional but recommended

prompt = "A futuristic cityscape at sunset"
video_frames = pipe(prompt, num_frames=24, height=320, width=576).frames
video_frames[0].save("output.gif")