var ws = new WebSocket("ws://localhost:8765/");

ws.onopen = function() {
    console.log('WebSocket Connection Established');
};

ws.onerror = function(error) {
    console.log('WebSocket Error: ' + error);
};

ws.onmessage = function(e) {
    var data = e.data;
    if (data.startsWith("/9j/")) {
        var img = document.getElementById('robotCameraFeed');
        img.src = 'data:image/jpeg;base64,' + data;
    }
    else if (data.startsWith("accelerometer: ")) {
        var accelDataString = data.replace("accelerometer: [", "").replace("]", "");
        var accelData = accelDataString.split(", ").map(parseFloat);

        var x = accelData[0];
        var y = accelData[1];
        var z = accelData[2];

        var gravityVector = [0, 0, 9.81];

        var nonGravitationalAccel = [
            x - gravityVector[0],
            y - gravityVector[1],
            z - gravityVector[2]
        ];

        var accelNonGravity = Math.sqrt(
            nonGravitationalAccel[0] * nonGravitationalAccel[0] +
            nonGravitationalAccel[1] * nonGravitationalAccel[1] +
            nonGravitationalAccel[2] * nonGravitationalAccel[2]
        );

        document.getElementById('accelerometer').innerHTML = "Accelerometer (Non-Gravitational): " + accelNonGravity.toFixed(2) + " m/s²";
    }
    else if (data.startsWith("Wheel Velocities: ")) {
        var wheelVelocities = extractWheelVelocities(data);
        var averageSpeed = (wheelVelocities.front_left + wheelVelocities.front_right + wheelVelocities.rear_left + wheelVelocities.rear_right) / 4;
        document.getElementById('averageSpeed').innerHTML = "Average Speed: " + averageSpeed.toFixed(2) + " m/s";
    }
    else if (typeof e.data === "string" && e.data.startsWith("Lidar Point Cloud:")) {
        const vertices = parseLidarData(e.data);
        if (vertices) {
            updatePointCloud(vertices);
        }
    } else if (e.data instanceof ArrayBuffer) {
        updatePointCloud(parseLidarArrayBuffer(e.data));
    } else {
        console.log('Server:', data);
    }
};

function extractWheelVelocities(velocityStr) {
    const jsonPart = velocityStr.replace("Wheel Velocities: ", "").trim();
    const jsonString = jsonPart.replace(/'/g, '"');
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

var lidarStatus = false;

function changeLidarStatus() {
    lidarStatus = !lidarStatus;
    console.log("Lidar Status:", lidarStatus);

    var lidarButton = document.getElementById('toggleLidarButton');
    lidarButton.className = 'lidar-btn ' + (lidarStatus ? 'on' : 'off');
    lidarButton.innerHTML = lidarStatus ? 'Lidar ON' : 'Lidar OFF';

    if (lidarStatus) {
        sendCommand("enable lidar");
    } else {
        sendCommand("disable lidar");
    }
}

var cameraStatus = false;

function changeCameraStatus() {
    cameraStatus = !cameraStatus;
    console.log("Camera Status:", cameraStatus);

    var cameraButton = document.getElementById('toggleCameraButton');
    cameraButton.className = 'camera-btn ' + (cameraStatus ? 'on' : 'off');
    cameraButton.innerHTML = cameraStatus ? 'Camera ON' : 'Camera OFF';

    if (cameraStatus) {
        sendCommand("enable camera");
    } else {
        sendCommand("disable camera");
    }
}


const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, 400);
document.getElementById('lidar-point-cloud').appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 400, 0.1, 1000);
const scene = new THREE.Scene();

const pointGeometry = new THREE.BufferGeometry();
const pointsMaterial = new THREE.PointsMaterial({ color: 0x888888, size: 0.1 });
const pointCloud = new THREE.Points(pointGeometry, pointsMaterial);
scene.add(pointCloud);

camera.position.set(0, 0, 10);
camera.lookAt(scene.position);

function updatePointCloud(vertices) {
    pointGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    pointGeometry.computeBoundingSphere();
    renderer.render(scene, camera);
}

function parseLidarData(data) {
    const vertices = [];
    const pointCloudData = data.replace("Lidar Point Cloud:", "").trim();
    const pointCloudArray = pointCloudData.split("}, {").map(point => point.replace("{", "").replace("}", ""));
    pointCloudArray[0] = pointCloudArray[0].replace("[", "");
    pointCloudArray[pointCloudArray.length - 1] = pointCloudArray[pointCloudArray.length - 1].replace("]", "");
    for (let i = 0; i < pointCloudArray.length; i++) {
        const point = pointCloudArray[i].replace("'x': ", "").replace("'y': ", "").replace("'z': ","").split(", ").map(parseFloat);
        vertices.push(point[0], point[1], point[2]);
    }
    return vertices;
}

function parseLidarArrayBuffer(buffer) {
    const floatArray = new Float32Array(buffer);
    const vertices = [];

    for (let i = 0; i < floatArray.length; i += 3) {
        vertices.push(floatArray[i], floatArray[i + 1], floatArray[i + 2]);
    }

    return vertices;
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

animate();
