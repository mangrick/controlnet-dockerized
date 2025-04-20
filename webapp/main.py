import sys
import torch
import random
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pytorch_lightning import seed_everything
from typing import Optional
sys.path.append("ControlNet")
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from utils import image_from_base64, image_to_base64


class ImageGenerationRequest(BaseModel):
    """
    Image generation parameter configuration for ControlNet
    """
    prompt: str
    image: Optional[str] = None
    n_samples: int = 1
    low_threshold: int = 50
    high_threshold: int = 100
    image_resolution: int = 512
    seed: int = 1
    save_memory: bool = True
    quality: str = "good quality"
    n_prompt: str = ("animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, "
                     "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")  # Negative prompt
    strength: float = 1.0
    scale: float = 9.0
    eta: float = 0.0
    ddim_steps: int = 10


app = FastAPI()

# Load model and its weights (use CPU for now as no GPU available)
model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./ControlNet/models/control_sd15_canny.pth', location='cpu'))
ddim_sampler = DDIMSampler(model)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate/")
async def generate(config: ImageGenerationRequest):
    """
    Endpoint for issuing the image generation request.
    :param config: Image generation configuration.
    :return: Dictionary containing a list of base64 encoded images in its "result" key.
    """
    if config.image is not None:
        # Convert image as a numpy array from base64 encoded string
        in_image = image_from_base64(config.image)

        # Image pre-processing (Inverting black/white, rescaling to [0, 1], and converting to torch tensor)
        in_image = 255 - in_image
        in_image = in_image / 255.0
        H, W = in_image.shape
        in_image = np.tile(in_image, (1, 3, 1, 1))  # Shape: b c h w
        in_image = torch.from_numpy(in_image).float()

        # Make results reproducible by setting the seed (if not provided, use a random seed)
        seed_everything(random.randint(0, 65535) if config.seed is None else config.seed)

        # Control properties for the spatial localization
        cond = {
            "c_concat": [in_image],
            "c_crossattn": [model.get_learned_conditioning([f"{config.prompt}, {config.quality}"] * config.n_samples)]
        }
    else:
        cond = None

    # Control properties for the unconditional conditioning when generating images without specific guidance
    un_cond = {
        "c_concat": None if config.image is None else [in_image],
        "c_crossattn": [model.get_learned_conditioning([config.n_prompt] * config.n_samples)]
    }

    # Downsampling for Stable Diffusion's latent space in its autoencoder
    shape = (4, H // 8, W // 8)

    scaling = [config.strength * (0.825 ** float(12 - i)) for i in range(13)] if in_image is None else ([config.strength] * 13)
    model.control_scales = scaling

    # Generate images
    samples, intermediates = ddim_sampler.sample(
        S=config.ddim_steps,
        batch_size=config.n_samples,
        shape=shape,
        conditioning=cond,
        verbose=False,
        eta=config.eta,
        unconditional_guidance_scale=config.scale,
        unconditional_conditioning=un_cond
    )

    x_samples = model.decode_first_stage(samples)
    x_samples = x_samples.numpy()  # In my case already on cpu
    x_samples = np.transpose(x_samples, (0, 2, 3, 1))
    results = np.array_split(x_samples, config.n_samples, axis=0)

    # Rescaling to convert image representation to [0, 255] pixel values
    results = [(gen_image.squeeze() * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for gen_image in results]

    # Pack resulting images into a JSON string
    serialized_results = []
    for out_image in results:
        img_data = image_to_base64(out_image)
        serialized_results.append(img_data)

    return {"result": serialized_results}
