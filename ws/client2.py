import asyncio
import websockets
import json

async def client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Отправляем ID клиента
        await websocket.send(json.dumps({"id": "client2"}))

        # Отправляем сообщение другому клиенту
        await websocket.send(json.dumps({
            "action": "send",
            "to": "client1",
            "message": "Привет, клиент 1!"
        }))

        # Читаем входящие сообщения
        async for message in websocket:
            data = json.loads(message)
            print(f"Получено: {data}")

asyncio.run(client())