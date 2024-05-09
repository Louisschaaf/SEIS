import numpy as np
import asyncio
from PIL import Image
import io
import base64

def setup_camera(robot):
    camera = robot.getDevice("camera rgb")
    camera.enable(int(robot.getBasicTimeStep()))
    return camera

def capture_image(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    image = camera.getImage()
    if image:
        # Convert to an array and reshape. Assuming BGRA input from Webots.
        np_image = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
        # Convert BGRA to RGB and ensure memory continuity
        np_image = np.ascontiguousarray(np_image[:, :, :3])
        return np_image
    return None

async def send_images(camera, websocket):
    image_width = camera.getWidth()
    image_height = camera.getHeight()
    image_size = (image_width, image_height)
    while True:
        image = capture_image(camera)
        if image is not None:
            try:
                # Ensure image data is correctly formatted
                image = Image.frombytes("RGB", image_size, image)
                with io.BytesIO() as image_bytes:
                    image.save(image_bytes, format="JPEG")
                    encoded_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
                    await websocket.send(encoded_image)
            except Exception as e:
                print(f"Failed to process or send image: {e}")
                # Optionally, add reconnect or retry logic here
        await asyncio.sleep(0.1)  # Adjust delay to match desired frame rate
