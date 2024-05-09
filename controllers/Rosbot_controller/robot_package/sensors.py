import asyncio
from time import time
robot = None
CALIBRATION_SAMPLES = 100

def setup_accelerometer(robot):
    robot = robot
    accelerometer = robot.getDevice("imu accelerometer")
    accelerometer.enable(int(robot.getBasicTimeStep()))
    print("Accelerometer enabled")
    return accelerometer

def read_accelerometer(accelerometer):
    return accelerometer.getValues()

async def send_accelerometer_data(accelerometer, websocket):
    while True:
        accelerometer_data = read_accelerometer(accelerometer)
        try:
            await websocket.send("accelerometer: "+str(accelerometer_data))
        except Exception as e:
            print(f"Failed to send accelerometer data: {e}")
            # Optionally, add reconnect or retry logic here
        await asyncio.sleep(0.1)  # Adjust delay to match desired update rate

def update_velocity(accelerometer_data):
    global velocity, last_timestamp
    if last_timestamp is None:
        last_timestamp = time()

    current_timestamp = time()
    delta_time = current_timestamp - last_timestamp

    # Update velocity by integrating acceleration over time
    velocity["x"] += accelerometer_data[0] * delta_time
    velocity["y"] += accelerometer_data[1] * delta_time
    velocity["z"] += accelerometer_data[2] * delta_time

    last_timestamp = current_timestamp

async def send_velocity_data(accelerometer, websocket):
    while True:
        accelerometer_data = read_accelerometer(accelerometer)
        print(f"accelerometer_data: {accelerometer_data}")
        update_velocity(accelerometer_data)
        print(f"velocity: {velocity}")
        message = f"velocity: {velocity}"
        try:
            await websocket.send(message)
        except Exception as e:
            print(f"Failed to send velocity data: {e}")
        await asyncio.sleep(0.1)