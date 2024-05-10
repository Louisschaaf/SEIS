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


def calculate_velocity(current_position, previous_position, delta_t):
    return (current_position - previous_position) / delta_t