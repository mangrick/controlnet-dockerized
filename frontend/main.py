import gradio as gr
import imageio
import requests
from image_utils import image_to_base64, image_from_base64


def send_synthesis_request(prompt, in_image, low_threshold, high_threshold, resolution):
    """
    Use the prompt, input image, and the 3 sliders to send a POST request to the backend.
    """
    param_data = {"prompt": prompt, "low_threshold": low_threshold, "high_threshold": high_threshold, "resolution": resolution}

    img = imageio.imread(in_image)
    param_data["image"] = image_to_base64(img)  # The temporary file seems to contain 3 channels even though orig only has 1

    # Make a POST request
    r = requests.post("http://controlnet-backend-1:5557/generate", json=param_data)
    if r.status_code != 200:
        return None

    response = r.json()
    image = response["result"][0]
    image = image_from_base64(image)
    return image


# Build GUI with gradio
with gr.Blocks() as app:
    gr.Markdown("# ControlNet Image Synthesis")

    with gr.Row():
        # Control parameters for image generation
        with gr.Column():
            input_image = gr.Image(label="Control image", type="filepath", height=256, image_mode="L")
            prompt = gr.Textbox(label="Prompt", placeholder="Please provide a prompt!")
            low_threshold = gr.Slider(label="Low Threshold", minimum=1, maximum=255, value=100, step=1)
            high_threshold = gr.Slider(label="High Threshold", minimum=1, maximum=255, value=200, step=1)
            resolution = gr.Slider(label="Resolution", minimum=256, maximum=1024, value=512, step=64)
            generate_btn = gr.Button("Generate Image")

        # ControlNet Output Column
        with gr.Column():
            output_image = gr.Image(label="ControlNet result", height=256)

    # Prepare input and output variables
    inputs = [prompt, input_image, low_threshold, high_threshold, resolution]
    output = output_image

    # Also add a button for processing
    generate_btn.click(fn=send_synthesis_request, inputs=inputs, outputs=output)


# Launch frontend on port 80
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=80)