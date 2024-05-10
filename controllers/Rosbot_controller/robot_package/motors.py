from robot_package.sensors import read_distance_sensors, setup_distance_sensors
import asyncio

BRAKING_DISTANCE = 0.3  # Distance at which the robot will stop automatically

def setup_motors(robot):
    motors = {
        'front_left': robot.getDevice("front left wheel motor"),
        'front_right': robot.getDevice("front right wheel motor"),
        'rear_left': robot.getDevice("rear left wheel motor"),
        'rear_right': robot.getDevice("rear right wheel motor")
    }

    position_sensors = {
        'front_left': robot.getDevice("front left wheel motor sensor"),
        'front_right': robot.getDevice("front right wheel motor sensor"),
        'rear_left': robot.getDevice("rear left wheel motor sensor"),
        'rear_right': robot.getDevice("rear right wheel motor sensor")
    }

    # Set each motor to velocity control mode
    for motor in motors.values():
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)

    # Enable the position sensors
    timestep = int(robot.getBasicTimeStep())
    for sensor in position_sensors.values():
        sensor.enable(timestep)

    # Initialize previous position variables
    prev_positions = {name: sensor.getValue() for name, sensor in position_sensors.items()}

    return motors, position_sensors, prev_positions

def stop_all_motors(motors):
    """Stop all motors by setting their velocity to zero."""
    for motor in motors.values():
        motor.setVelocity(0)

async def automatic_braking(motors, distance_sensors):
    while True:
        distances = read_distance_sensors(distance_sensors)
        if any(distance < BRAKING_DISTANCE for distance in distances.values()):
            for motor in motors.values():
                motor.setVelocity(0.0)
            print("Automatic braking triggered due to obstacle detection")
        await asyncio.sleep(0.1)