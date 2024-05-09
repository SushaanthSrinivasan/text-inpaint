import time
start_time = time.time()
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

prompt = "a litter of puppies"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    variant="fp16",
    # torch_dtype=torch.float16,
    torch_dtype=torch.float32,
    use_safetensors=True,
    safety_checker = None,
)
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing("max")

image_pil = Image.open("./images/park.png")
mask_image = Image.open("./images/mask.png")

image = pipe(prompt=prompt, image=image_pil, mask_image= mask_image).images[0]
image.save("./images/output.png")

print("--- %s seconds ---" % (time.time() - start_time))
# Approx time: 2.4 mins