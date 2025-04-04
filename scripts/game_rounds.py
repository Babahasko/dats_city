import asyncio

from sender import GameAPI
game_api = GameAPI()

async def main():
    status, game_data = await game_api.words()
    if status == 200:
        # game_state = GameState(game_data)

        print(game_data)
        # логика...

        # 2. Отправляем запрос с обработк
    elif status != 200:
        print(game_data)
if __name__ == '__main__':
    asyncio.run(main())