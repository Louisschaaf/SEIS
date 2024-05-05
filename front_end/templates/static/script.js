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
    } else {
        console.log('Server:', data);
    }
};

function sendCommand(command) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(command);
        console.log("Command sent:", command);
    } else {
        console.log("WebSocket is not open.");
    }
}
