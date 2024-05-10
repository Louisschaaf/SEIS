var ws = new WebSocket("ws://localhost:8765/"); // Make sure this matches the address and port of your WebSocket server in the Webots controller

ws.onopen = function() {
    console.log('WebSocket Connection Established');
};

ws.onerror = function(error) {
    console.log('WebSocket Error: ' + error);
};

ws.onmessage = function(e) {
    var data = e.data;
    if (data.startsWith("/9j/")) { // Check if the string looks like base64 JPEG data
        var img = document.getElementById('robotCameraFeed');
        img.src = 'data:image/jpeg;base64,' + data;
    }
    else if (data.startsWith("accelerometer: ")) {
        // Extract and clean up the accelerometer data
        var accelDataString = data.replace("accelerometer: [", "").replace("]", "");
        var accelData = accelDataString.split(", ").map(parseFloat);
    
        // Extract individual components
        var x = accelData[0];
        var y = accelData[1];
        var z = accelData[2];
    
        // Assume the gravity vector is in the Z direction
        var gravityVector = [0, 0, 9.81];
    
        // Subtract gravity from the accelerometer data
        var nonGravitationalAccel = [
            x - gravityVector[0],
            y - gravityVector[1],
            z - gravityVector[2]
        ];
    
        // Compute the magnitude of the non-gravitational acceleration
        var accelNonGravity = Math.sqrt(
            nonGravitationalAccel[0] * nonGravitationalAccel[0] +
            nonGravitationalAccel[1] * nonGravitationalAccel[1] +
            nonGravitationalAccel[2] * nonGravitationalAccel[2]
        );
    
        // Display the non-gravitational magnitude in the HTML element
        document.getElementById('accelerometer').innerHTML = "Accelerometer (Non-Gravitational): " + accelNonGravity.toFixed(2) + " m/s²";
    }
    else if (data.startsWith("velocity: ")) {
        console.log(data);
    }
    else if (data.startsWith("Wheel Velocities: ")) {

        // Extract and clean up the wheel velocity data
        var wheelVelocities = extractWheelVelocities(data);
        //compute average speed
        var averageSpeed = (wheelVelocities.front_left + wheelVelocities.front_right + wheelVelocities.rear_left + wheelVelocities.rear_right) / 4;
        console.log("Average Speed: " + averageSpeed);
        // Display the average speed in the HTML element
        document.getElementById('averageSpeed').innerHTML = "Average Speed: " + averageSpeed.toFixed(2) + " m/s";
    }
    else {
        console.log('Server:', data);
    }
};

function extractWheelVelocities(velocityStr) {
    // Extract the part after "Wheel Velocities: "
    const jsonPart = velocityStr.replace("Wheel Velocities: ", "").trim();
    // Replace single quotes with double quotes for valid JSON
    const jsonString = jsonPart.replace(/'/g, '"');
    // Parse the JSON string into an object
    return JSON.parse(jsonString);
}

function sendCommand(command) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(command);
        console.log("Command sent:", command);
    } else {
        console.log("WebSocket is not open.");
    }
}
