import asyncio

from sender import GameAPI
from logic.logic import TowerBuilder
game_api = GameAPI()

async def main():
    game_data = await game_api.words()
    if game_data:
        print(game_data.words)
        tower_builder = TowerBuilder(words=game_data.words)
        build_request = tower_builder.build_tower()

    if game_data is None:
        print()

if __name__ == '__main__':
    asyncio.run(main())