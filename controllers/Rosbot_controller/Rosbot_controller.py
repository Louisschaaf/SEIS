from controller import Robot
from robot_package.motors import setup_motors
from robot_package.camera import setup_camera, capture_image
import websockets
import asyncio
import functools
from PIL import Image
import io
import base64

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

async def control_robot(websocket, path, robot, motors, camera):
    print("WebSocket session started.")
    image_task = asyncio.create_task(send_images(camera, websocket))  # Start sending images

    try:
        async for message in websocket:
            print(f"Received command: {message}")
            if message == 'forward':
                velocity = 5.0
            elif message == 'backward':
                velocity = -5.0
            elif message == 'right':
                velocity = 5.0
                motors['front_right'].setVelocity(-velocity)
                motors['rear_right'].setVelocity(-velocity)
                continue
            elif message == 'left':
                velocity = 5.0
                motors['front_left'].setVelocity(-velocity)
                motors['rear_left'].setVelocity(-velocity)
                continue
            elif message == 'stop':
                velocity = 0.0
            else:
                continue

            for motor in motors.values():
                motor.setVelocity(velocity)
    finally:
        image_task.cancel()  # Ensure image task is cancelled

async def run_websocket_server(robot, motors, camera):
    async with websockets.serve(functools.partial(control_robot, robot=robot, motors=motors, camera=camera), "0.0.0.0", 8765):
        await asyncio.Future()  # This will run forever unless cancelled

def main(robot):
    timestep = int(robot.getBasicTimeStep())
    motors = setup_motors(robot)
    camera = setup_camera(robot)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_task = loop.create_task(run_websocket_server(robot, motors, camera))

    try:
        while robot.step(timestep) != -1:
            loop.run_until_complete(asyncio.sleep(0))  # Run a single event loop iteration
    finally:
        server_task.cancel()
        loop.run_until_complete(server_task)
        loop.close()

if __name__ == "__main__":
    my_robot = Robot()
    main(my_robot)
