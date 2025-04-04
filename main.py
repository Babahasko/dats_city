import asyncio
import time
from utils import logger
from sender import GameAPI, GameState

game_api = GameAPI()

async def main():
    in_game = True
    while in_game:
        try:
            #Основной цикл
            # 1.Получаем данные
            game_data = await game_api.towers()
            if game_data is not None:
                # game_state = GameState(game_data)

                logger.info(game_data)
                print(game_data)
                # логика...

                # 2. Отправляем запрос с обработкой

                # 3. Ждём оставшееся время !!!
                # if game_state.tick_remain_ms > 0:
                #     print(f"Ожидание до конца хода: {game_state.tick_remain_ms} милисекунд")
                #     time.sleep(game_state.tick_remain_ms / 1000)
                time.sleep(10000)
            else:
                in_game = False
        except Exception as e:
            in_game = False
            print(e)
            # print(e)



if __name__ == "__main__":
    asyncio.run(main())