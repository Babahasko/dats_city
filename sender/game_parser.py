from typing import Literal

from pydantic import BaseModel

# /api/words
# Words list
class WordListResponse(BaseModel):
    mapSize: list[int]  # [x, y, z] размеры пространства башни
    nextTurnSec: int  # время до следующего хода в секундах
    roundEndsAt: str  # время окончания раунда 2023-10-01T12:00:00Z
    shuffleLeft: int  # сколько раз можно запросить новый набор слов
    turn: int  # текущий ход
    usedIndexes: list[int]  # использованные индексы слов
    words: list[str]  # доступные слова в текущем наборе


#########################################################
# /api/build POST
# Build tower or add words to existing tower. If tower is done, it will be saved and you can start new tower
class WordPosition(BaseModel):
    dir: list[int]  # [0, 0, 1] x y z !!!
    id: int  # идентификатор слова
    pos: list[int]  # позиция слова в формате [x, y, z]
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
#/api/rounds

class RoundInfo(BaseModel):
    duration: int                      # Длительность раунда в секундах
    endAt: str                    # Время окончания раунда
    name: str                          # Название раунда
    repeat: int                        # Номер повторения раунда
    startAt: str                  # Время начала раунда
    status: Literal["active", "pending", "finished"]  # Статус раунда

class RoundsGame(BaseModel):
    eventId: str                       # Идентификатор события
    now: str                      # Текущее время сервера
    rounds: list[RoundInfo]            # Список раундов

##################################################################
#/api/shuffle
class ShuffleResponse(BaseModel):
    """
    Модель ответа на запрос перемешивания слов.
    Содержит информацию о доступных словах и оставшихся перемешиваниях.
    """
    shuffleLeft: int #"Количество оставшихся перемешиваний")
    words: list[str] # новый список слов

##################################################################