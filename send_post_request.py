import argparse
import requests
import imageio
import matplotlib.pyplot as plt
from webapp.image_utils import image_from_base64, image_to_base64
from pathlib import Path


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
        param_data["image"] = image_to_base64(img)

    # Make a POST request
    r = requests.post("http://localhost:80/generate", json=param_data)
    if r.status_code == 200:
        # Plot resulting images
        response = r.json()
        for img in response["result"]:
            img = image_from_base64(img)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            plt.show()

    # Return error code if not successful
    exit(0 if r.status_code == 200 else 1)
