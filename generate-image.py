from diffusers import DiffusionPipeline
import torch
from datetime import datetime

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
# from diffusers import EDMDPMSolverMultistepScheduler
# pipe.scheduler = EDMDPMSolverMultistepScheduler()

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
prompt = "An AI-driven cybersecurity solution in action, protecting a computer system from potential threats and attacks."
image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]

# Create the filename string to save it as a .jpg file that starts with the date and time
# from the datetime module

filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"

# Save the image
image.save(filename)
