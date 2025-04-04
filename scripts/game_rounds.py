import asyncio

from sender import GameAPI
game_api = GameAPI()

async def main():
    game_data = await game_api.rounds()
    if game_data:
        print(game_data)
    if game_data is None:
        print()

if __name__ == '__main__':
    asyncio.run(main())