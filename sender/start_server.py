import asyncio
import websockets
import json

# Словарь для хранения подключенных клиентов (ключ - ID клиента, значение - WebSocket соединение)
connected_clients = {}

async def handle_connection(websocket):
    """
    Обработчик подключения нового клиента.
    """
    try:
        # Ожидаем, пока клиент отправит свой ID
        async for message in websocket:
            try:
                data = json.loads(message)
                client_id = data.get("id")

                if not client_id:
                    await websocket.send(json.dumps({"error": "ID клиента не указан"}))
                    continue

                # Регистрируем клиента
                if client_id not in connected_clients:
                    connected_clients[client_id] = websocket
                    print(f"Клиент {client_id} подключен.")
                    await websocket.send(json.dumps({"status": "connected", "id": client_id}))
                else:
                    await websocket.send(json.dumps({"error": "Клиент с таким ID уже существует"}))

                # Обработка команд от клиента
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    action = data.get("action")
                    target_id = data.get("to")
                    text = data.get("message")

                    if action == "send":
                        if target_id == "all":
                            # Отправляем сообщение всем клиентам
                            await broadcast_message(client_id, text)
                        elif target_id in connected_clients:
                            # Отправляем сообщение конкретному клиенту
                            await send_message_to_client(target_id, client_id, text)
                        else:
                            await websocket.send(json.dumps({"error": "Целевой клиент не найден"}))

            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Неверный формат JSON"}))

    except websockets.exceptions.ConnectionClosed:
        # Удаляем клиента из списка при отключении
        for client_id, ws in list(connected_clients.items()):
            if ws == websocket:
                del connected_clients[client_id]
                print(f"Клиент {client_id} отключен.")
                break


async def send_message_to_client(target_id, sender_id, message):
    """
    Отправляет сообщение конкретному клиенту.
    """
    if target_id in connected_clients:
        target_ws = connected_clients[target_id]
        await target_ws.send(json.dumps({"from": sender_id, "message": message}))
        print(f"Сообщение от {sender_id} отправлено клиенту {target_id}: {message}")


async def broadcast_message(sender_id, message):
    """
    Отправляет сообщение всем подключенным клиентам.
    """
    for client_id, ws in connected_clients.items():
        await ws.send(json.dumps({"from": sender_id, "message": message}))
    print(f"Сообщение от {sender_id} отправлено всем клиентам: {message}")

async def main():
    # Запуск сервера
    async with websockets.serve(handle_connection, "localhost", 8765) as server:
        print("Сервер запущен на ws://localhost:8765")
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())