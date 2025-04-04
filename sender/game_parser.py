from typing import Literal

from pydantic import BaseModel

# /api/words
# Words list
class GameState(BaseModel):
    mapSize: list[int]  # [x, y, z] размеры пространства башни
    nextTurnSec: int  # время до следующего хода в секундах
    roundEndsAt: str  # время окончания раунда 2023-10-01T12:00:00Z
    shuffleLeft: int  # сколько раз можно запросить новый набор слов
    turn: int  # текущий ход
    usedIndexes: list[int]  # использованные индексы слов
    words: list[str]  # доступные слова в текущем наборе

class WordPosition(BaseModel):
    dir: list[int]  # [0, 0, 1] x y z !!!
    id: int  # идентификатор слова
    pos: list[int]  # позиция слова в формате [x, y, z]

#########################################################
# Build tower or add words to existing tower. If tower is done, it will be saved and you can start new tower
class BuildReq(BaseModel):
    done: bool  # флаг завершения строительства башни
    words: list[WordPosition]  # список слов в башне

class BuildResponse(BaseModel):
    shuffleLeft: int  # количество оставшихся перемешиваний
    words: list[str]  # набор оставшихся слов
##########################################################

##########################################################
# /api/towers GET
# All towers
class TowerWord(BaseModel):
    dir: list[int]  # 0 - горизонтальное, 1 - вертикальное
    pos: list[int]      # [x, y, z] координаты
    text: str           # текст слова

class TowerInfo(BaseModel):
    score: float        # текущий счёт башни
    words: list[TowerWord]  # слова в башне

class DoneTower(BaseModel):
    id: int             # ID завершённой башни
    score: float        # счёт башни

class GameResponse(BaseModel):
    doneTowers: list[DoneTower]  # список завершённых башен
    score: float                 # общий счёт
    tower: TowerInfo             # текущая башня

#############################################################