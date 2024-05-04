var ws = new WebSocket("ws://0.0.0.0:8765"); // Zorg ervoor dat dit overeenkomt met het adres en de poort van je WebSocket-server in de Webots-controller

ws.onopen = function() {
    console.log('WebSocket Connection Established');
};

ws.onerror = function(error) {
    console.log('WebSocket Error: ' + error);
};

ws.onmessage = function(e) {
    console.log('Server: ' + e.data);
};

function sendCommand(command) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(command);
        console.log("Command sent:", command);
    } else {
        console.log("WebSocket is not open.");
    }
}
