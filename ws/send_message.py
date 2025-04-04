import json
from websockets.sync.client import connect

def hello():
    uri = "ws://localhost:8765"
    with connect(uri) as websocket:
        websocket.send(json.dumps({"id": "client2"}))
        websocket.send(json.dumps({
            "action": "send",
            "to": "client1",
            "message": "Привет, клиент 1!"
        }))
        

if __name__ == "__main__":
    hello()