import asyncio
from time import time
robot = None
CALIBRATION_SAMPLES = 100
from robot_package.motors import calculate_velocity


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
        # Get the point cloud data
        point_cloud = lidar.getPointCloud()

        # Prepare the point cloud data for JSON
        point_cloud_data = [{"x": point.x, "y": point.y, "z": point.z} for point in point_cloud]

        # Send the point cloud data via WebSocket
        await websocket.send(f"Lidar Point Cloud: {point_cloud_data}")

        await asyncio.sleep(timestep / 1000.0) 