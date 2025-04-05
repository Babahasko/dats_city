import asyncio
import time
from utils import logger
from sender import GameAPI
from logic.algos_tower import WordList
from sender.send_on_front import send_data
from sender.game_parser import BuildReq, WordPosition

game_api = GameAPI()

async def main():
    in_game = True
    while in_game:
        try:
            #Основной цикл
            # 1.Получаем данные
            game_data = await game_api.words()
            if game_data is not None:
                # game_state = GameState(game_data)

                logger.info(game_data)
                # print(game_data.mapSize)
                # print(game_data.words)
                # логика...
                word_list = WordList(game_data.words, game_data.mapSize)
                # Получаем данные для отправки на фронт
                result = word_list.winner_pipeline()
                send_data(result)
                print(word_list.placed_words)
                await game_api.build(BuildReq(done=True, words=word_list.placed_words))
                # 2. Отправляем запрос с обработкой

                # 3. Ждём оставшееся время !!!
                # if game_state.tick_remain_ms > 0:
                #     print(f"Ожидание до конца хода: {game_state.tick_remain_ms} милисекунд")
                #     time.sleep(game_state.tick_remain_ms / 1000)
                time.sleep(10)
            else:
                in_game = False
        except Exception as e:
            in_game = False
            print(e)
            # print(e)



if __name__ == "__main__":
    asyncio.run(main())