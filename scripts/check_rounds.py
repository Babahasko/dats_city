import asyncio
import json

from sender import GameAPI
from logic.logic import TowerBuilder
from sender.game_parser import BuildReq

game_api = GameAPI()

async def main():
    game_data = await game_api.rounds()
    if game_data:
        print(game_data)
    if game_data is None:
        print()

if __name__ == '__main__':
    asyncio.run(main())