import numpy as np

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
