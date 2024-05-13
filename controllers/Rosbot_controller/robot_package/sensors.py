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

async def send_velocity_data(position_sensors, websocket, prev_positions, timestep):
    delta_t = timestep / 1000.0  # Convert to seconds

    while True:
        # Get the current positions
        current_positions = {name: sensor.getValue() for name, sensor in position_sensors.items()}

        # Calculate the velocities for each wheel
        velocities = {
            name: calculate_velocity(current_positions[name], prev_positions[name], delta_t)
            for name in position_sensors
        }

        # Update previous positions
        prev_positions.update(current_positions)

        # Send the velocity data via WebSocket
        await websocket.send(f"Wheel Velocities: {velocities}")

        await asyncio.sleep(delta_t)

def setup_lidar(robot):
    lidar = robot.getDevice("lidar")
    lidar.enable(int(robot.getBasicTimeStep()))
    lidar.enablePointCloud()
    print("Lidar enabled")
    return lidar

async def send_lidar_data(lidar, websocket, timestep):
    while True:
        try:
            point_cloud = lidar.getPointCloud()
            if point_cloud:  # Ensure there is data to process
                point_cloud_data = [{"x": point.x, "y": point.y, "z": point.z} for point in point_cloud]
                await websocket.send(f"Lidar Point Cloud: {point_cloud_data}")
            else:
                print("No lidar data available.")
        except Exception as e:
            print(f"Failed to send lidar data: {e}")
            # Optionally, add reconnect or retry logic here
            break  # or handle reconnection

        await asyncio.sleep(timestep / 1000.0)  # Wait for the next set of data



def setup_distance_sensors(robot):
    sensors = {
        'front left': robot.getDevice('front left distance sensor'),
        'front right': robot.getDevice('front right distance sensor'),
        'rear left': robot.getDevice('rear left distance sensor'),
        'rear right': robot.getDevice('rear right distance sensor'),
    }
    for sensor in sensors.values():
        sensor.enable(int(robot.getBasicTimeStep()))
    print("Distance sensors enabled")
    return sensors

def read_distance_sensors(sensors):
    return {key: sensor.getValue() for key, sensor in sensors.items()}

def calculate_velocity(current_position, previous_position, delta_t):
    return (current_position - previous_position) / delta_t