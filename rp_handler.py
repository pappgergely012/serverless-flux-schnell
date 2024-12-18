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
    seed, 
    randomize_seed, 
    width, 
    height, 
    num_inference_steps, 
    num_samples, 
):
    print(f"Processed prompt is: {prompt}")
    
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
    
    input_data = event["input"]
    prompt = input_data["prompt"]
    seed = input_data.get("seed", 42)
    randomize_seed = input_data.get("randomize_seed", False)
    width = input_data.get("width", 1024)
    height = input_data.get("height", 1024)
    num_inference_steps = input_data.get("num_inference_steps", 4)
    num_samples = input_data.get("num_samples", 1)

    # Call run_inference with destructured arguments
    json = run_inference(
        prompt=prompt,
        seed=seed,
        randomize_seed=randomize_seed,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        num_samples=num_samples
    )

    return json

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})