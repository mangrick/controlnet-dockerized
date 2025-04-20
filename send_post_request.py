import argparse
import requests
import base64
import imageio
import io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


class ImageGenerationRequest:
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


# r = requests.post("http://localhost:80/generate", json={"width": "100", "height": "200"})
# print(r.status_code)
# print(r.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Issue a post request to synthesize images based on a prompt and an "
                                     "optional conditional image.")
    parser.add_argument("prompt", help="Prompt for diffusion model to synthesize image.")
    parser.add_argument("-i", "--image", help="Optional image path for spatial control of image synthesis.")
    parser.add_argument("-n", "--n_samples", default="1", help="Number of images to generate.")
    parser.add_argument("--low_threshold", default="50", help="Low threshold.")
    parser.add_argument("--high_threshold", default="100", help="High threshold.")
    parser.add_argument("-r", "--resolution", default="512", help="Image resolution.")
    parser.add_argument("-s", "--seed", default="1", help="Seed for random number generator.")
    parser.add_argument("-m", "--memory", default="True", help="Save memory.")
    parser.add_argument("-q", "--quality", default="good quality", help="Quality of image.")

    parser.add_argument("--n_prompt", default="", help="Prompt for unconditional conditioning.")
    parser.add_argument("--strength", default="1.0", help="Control parameter for the scales.")
    parser.add_argument("--scale", default="9.0", help="Control parameter for the unconditional guidance.")
    parser.add_argument("--eta", default="0.0", help="ETA parameter.")
    parser.add_argument("--ddim_steps", default="10", help="Number of steps for DDIM sampling.")

    args = parser.parse_args()

    # Prepare a POST request
    param_data = vars(args)

    # Encode image in base64
    if args.image is not None:
        img = imageio.imread(Path(args.image).as_posix())
        buffer = io.BytesIO()
        np.save(buffer, img, allow_pickle=False)
        img_data = base64.b64encode(buffer.getvalue())
        param_data["image"] = img_data.decode("utf-8")

    # Make a POST request
    r = requests.post("http://localhost:80/generate", json=param_data)

    # Return error code if not successful
    exit(0 if r.status_code == 200 else 1)
