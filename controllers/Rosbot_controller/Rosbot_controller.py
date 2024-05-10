from controller import Robot
from robot_package.motors import setup_motors
from robot_package.camera import setup_camera, capture_image, send_images
from robot_package.sensors import setup_accelerometer, read_accelerometer, send_accelerometer_data, send_velocity_data
import websockets
import asyncio
import functools

async def control_robot(websocket, path, robot, motors, camera, accelerometer, position_sensors, prev_positions, timestep):
    print("WebSocket session started.")
    image_task = asyncio.create_task(send_images(camera, websocket))  # Start sending images
    accelerometer_task = asyncio.create_task(send_accelerometer_data(accelerometer, websocket))  # Start sending accelerometer data
    velocity_task = asyncio.create_task(send_velocity_data(position_sensors, websocket, prev_positions, timestep))  # Start sending velocity data
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
        accelerometer_task.cancel()
        velocity_task.cancel()

async def run_websocket_server(robot, motors, camera, accelerometer, position_sensors, prev_positions, timestep):
    async with websockets.serve(functools.partial(control_robot, robot=robot, motors=motors, camera=camera, accelerometer=accelerometer, position_sensors=position_sensors, prev_positions=prev_positions, timestep=timestep), "0.0.0.0", 8765):
        await asyncio.Future()  # This will run forever unless cancelled

def main(robot):
    timestep = int(robot.getBasicTimeStep())
    motors, position_sensors, prev_positions = setup_motors(robot)
    camera = setup_camera(robot)
    accelerometer = setup_accelerometer(robot)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_task = loop.create_task(run_websocket_server(robot, motors, camera, accelerometer, position_sensors, prev_positions, timestep))

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
