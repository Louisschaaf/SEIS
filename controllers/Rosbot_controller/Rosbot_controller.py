from controller import Robot, Motor
import websockets
import asyncio
import functools

async def control_robot(websocket, path, robot, motors):
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

async def run_websocket_server(robot, motors):
    async with websockets.serve(functools.partial(control_robot, robot=robot, motors=motors), "localhost", 8765):
        await asyncio.Future()  # This will run forever unless cancelled

def main(robot):
    timestep = int(robot.getBasicTimeStep())
    motors = setup_motors(robot)

    # Manually manage the asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_task = loop.create_task(run_websocket_server(robot, motors))

    # Main simulation loop
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
