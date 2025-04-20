import io
import base64
import numpy as np


def image_from_base64(image: str) -> np.ndarray:
    """
    Convert a base64 encoded image into a numpy array.
    """
    img_data = image.encode("utf-8")
    buffer = io.BytesIO(base64.b64decode(img_data))
    return np.load(buffer, allow_pickle=False)


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert an image represented as a NumPy array to a Base64-encoded string.
    """
    buffer = io.BytesIO()
    np.save(buffer, image, allow_pickle=False)
    img_data = base64.b64encode(buffer.getvalue())
    return img_data.decode("utf-8")
