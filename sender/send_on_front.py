import json
from websockets.sync.client import connect


def send_data(data):
    uri = "ws://localhost:8765"
    with connect(uri) as websocket:
        websocket.send(json.dumps({"id": "back"}))
        websocket.send(json.dumps({
            "action": "send",
            "to": "front",
            "message": data
        }))


if __name__ == "__main__":
    test_data = {"cubes": [[1,1,1],[1,2,1],[1,3,1]],
                 "text": ["a", "б", "в"],}
    send_data(test_data)