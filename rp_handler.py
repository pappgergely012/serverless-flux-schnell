import runpod
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16) # can replace schnell with dev
# to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()


@torch.inference_mode()
def generate_image(job):
    print("Start generate image")

    job_input = job["input"]
    prompt = job_input["prompt"]

    image = pipe(
        prompt=prompt,
        guidance_scale=0.,
        height=768,
        width=1360,
        num_inference_steps=4,
        max_sequence_length=256,
    ).images[0]
   
    results = {
        "image": image
    }

    return results

runpod.serverless.start({"handler": generate_image})
