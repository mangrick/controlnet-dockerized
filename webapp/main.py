import io
import base64
import logging
import torch
import random
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything
from typing import Optional


class ImageGenerationRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    n_samples: int = 1
    low_threshold: int = 50
    high_threshold: int = 100
    image_resolution: int = 512
    seed: int = 1
    save_memory: bool = True
    quality: str = "good quality"
    n_prompt: str = ""
    strength: float = 1.0
    scale: float = 9.0
    eta: float = 0.0
    ddim_steps: int = 10


app = FastAPI()

# Load model
model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./ControlNet/models/control_sd15_canny.pth', location='cpu'))
model = model.cpu()
ddim_sampler = DDIMSampler(model)


@app.get("/")
async def index():
    return {"Hello": "World"}


@app.post("/generate/")
async def generate(config: ImageGenerationRequest):
    # Check if a conditioning image is provided
    if config.image is not None:
        img_data = config.image.encode("utf-8")
        buffer = io.BytesIO(base64.b64decode(img_data))
        in_image = np.load(buffer, allow_pickle=False)

        # Image between 0 and 1
        in_image = in_image / 255.0
        H, W = in_image.shape
        in_image = np.tile(in_image, (1, 3, 1, 1))  # Shape: b c h w

        # in_image = current_app.to_tensor(in_image).float()
        in_image = torch.from_numpy(in_image).float()

        seed_everything(random.randint(0, 65535) if config.seed is None else config.seed)

        cond = {
            "c_concat": [in_image],
            "c_crossattn": [model.get_learned_conditioning([f"{config.prompt}, {config.quality}"] * config.n_samples)]
        }

    un_cond = {
        "c_concat": None if config.image is None else [in_image],
        "c_crossattn": [model.get_learned_conditioning([config.n_prompt] * config.n_samples)]
    }

    shape = (4, H // 8, W // 8)

    scaling = [config.strength * (0.825 ** float(12 - i)) for i in range(13)] if in_image is None else ([config.strength] * 13)
    model.control_scales = scaling
    samples, intermediates = synthesize_images(
        S=config.ddim_steps,
        batch_size=config.n_samples,
        shape=shape,
        conditioning=cond if config.image is not None else None,
        verbose=False,
        eta=config.eta,
        unconditional_guidance_scale=config.scale,
        unconditional_conditioning=un_cond
    )

    x_samples = model.decode_first_stage(samples)
    x_samples = x_samples.numpy()  # In my case already on cpu
    x_samples = np.transpose(x_samples, (0, 2, 3, 1))
    results = np.array_split(x_samples, config.n_samples, axis=0)
    results = [(gen_image.squeeze() * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for gen_image in results]
    print("Done!!!")

    return {"status": 200}
