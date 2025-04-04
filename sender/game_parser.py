from typing import Literal

from pydantic import BaseModel


class GameState(BaseModel):
    mapSize: list[int]  # [x, y, z] размеры пространства башни
    nextTurnSec: int  # время до следующего хода в секундах
    roundEndsAt: str  # время окончания раунда 2023-10-01T12:00:00Z
    shuffleLeft: int  # сколько раз можно запросить новый набор слов
    turn: int  # текущий ход
    usedIndexes: list[int]  # использованные индексы слов
    words: list[str]  # доступные слова в текущем наборе

class WordPosition(BaseModel):
    dir: Literal[0, 1]  # направление: 0 - горизонтальное, 1 - вертикальное
    id: int  # идентификатор слова
    pos: list[int]  # позиция слова в формате [x, y, z]

# Build tower or add words to existing tower. If tower is done, it will be saved and you can start new tower
class BuildReq(BaseModel):
    done: bool  # флаг завершения строительства башни
    words: list[WordPosition]  # список слов в башне

class BuildResponse(BaseModel):
    shuffleLeft: int  # количество оставшихся перемешиваний
    words: list[str]  # набор оставшихся слов
