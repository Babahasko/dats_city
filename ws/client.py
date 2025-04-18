import asyncio
import websockets
import json

async def client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Отправляем ID клиента
        await websocket.send(json.dumps({"id": "client1"}))

        # Отправляем сообщение другому клиенту
        await websocket.send(json.dumps({
            "action": "send",
            "to": "client2",
            "message": "Привет, клиент 2!"
        }))

        # Читаем входящие сообщения
        async for message in websocket:
            data = json.loads(message)
            print(f"Получено: {data}")

asyncio.run(client())