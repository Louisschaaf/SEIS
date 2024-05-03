from controller import Robot, Motor

def run_robot(robot):
    timestep = int(robot.getBasicTimeStep())
    
    # Motor setup
    front_left_motor = robot.getDevice("front left wheel motor")
    front_right_motor = robot.getDevice("front right wheel motor")
    rear_left_motor = robot.getDevice("rear left wheel motor")
    rear_right_motor = robot.getDevice("rear right wheel motor")
    
    front_left_motor.setPosition(float('inf'))  # Set to infinity for velocity control
    front_right_motor.setPosition(float('inf'))
    rear_left_motor.setPosition(float('inf'))
    rear_right_motor.setPosition(float('inf'))
    
    # Set initial motor velocities to 0 to ensure controlled start
    front_left_motor.setVelocity(0.0)
    front_right_motor.setVelocity(0.0)
    rear_left_motor.setVelocity(0.0)
    rear_right_motor.setVelocity(0.0)
    
    # Allow some time for initialization
    for i in range(10):
        robot.step(timestep)
    
    # Rotate in place: one wheel forward, the other backward
    front_left_motor.setVelocity(5)    # Positive speed for left wheel
    front_right_motor.setVelocity(5)  # Negative speed for right wheel to rotate in place
    rear_left_motor.setVelocity(5)
    rear_right_motor.setVelocity(5)
    
    # Keep rotating for a few seconds
    for i in range(100):
        if robot.step(timestep) == -1:
            break

if __name__ == "__main__":
    my_robot = Robot()
    run_robot(my_robot)