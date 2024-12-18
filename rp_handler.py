import numpy as np
import random
import runpod
import torch
from diffusers import DiffusionPipeline

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=dtype).to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

def run_inference(
    prompt, 
    seed=42, 
    randomize_seed=False, 
    width=1024, 
    height=1024, 
    num_inference_steps=4, 
    num_samples=3, 
):
    images = []
    seeds_used = []

    for i in range(num_samples):
        current_seed = seed
        if randomize_seed:
            current_seed = random.randint(0, MAX_SEED)
        generator = torch.Generator().manual_seed(current_seed)
        
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=0.0
        ).images[0]

        images.append(image)
        seeds_used.append(current_seed)
    
    # Return a gallery of images and a string listing the used seeds
    return images, ", ".join(str(s) for s in seeds_used)


def handler(event):
    
     json = run_inference(event["input"])

     return json

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})