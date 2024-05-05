def setup_motors(robot):
    motors = {
        'front_left': robot.getDevice("front left wheel motor"),
        'front_right': robot.getDevice("front right wheel motor"),
        'rear_left': robot.getDevice("rear left wheel motor"),
        'rear_right': robot.getDevice("rear right wheel motor")
    }
    for motor in motors.values():
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)
    return motors
