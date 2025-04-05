from typing import Literal

from pydantic import BaseModel

# /api/words
# Words list
class WordListResponse(BaseModel):
    mapSize: list[int]  # [x, y, z] размеры пространства башни
    nextTurnSec: int = 0  # время до следующего хода в секундах
    roundEndsAt: str = 0 # время окончания раунда 2023-10-01T12:00:00Z
    shuffleLeft: int = 0 # сколько раз можно запросить новый набор слов
    turn: int = 0# текущий ход
    usedIndexes: list[int] = 0# использованные индексы слов
    words: list[str] # доступные слова в текущем наборе

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)


#########################################################
# /api/build POST
# Build tower or add words to existing tower. If tower is done, it will be saved and you can start new tower
class WordPosition(BaseModel):
    dir: int
    id: int  # идентификатор слова
    pos: list[int]  # позиция слова в формате [x, y, z]

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)

class BuildReq(BaseModel):
    # done: bool  # флаг завершения строительства башни
    # words: list[WordPosition]  # список слов в башне

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)

class BuildResponse(BaseModel):
    shuffleLeft: int  # количество оставшихся перемешиваний
    words: list[str]  # набор оставшихся слов

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)
##########################################################

##########################################################
# /api/towers GET
# All towers
class TowerWord(BaseModel):
    dir: int # 0 - горизонтальное, 1 - вертикальное
    pos: list[int]      # [x, y, z] координаты
    text: str           # текст слова

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)

class TowerInfo(BaseModel):
    score: float        # текущий счёт башни
    words: list[TowerWord]  # слова в башне

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)

class DoneTower(BaseModel):
    id: int             # ID завершённой башни
    score: float        # счёт башни

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)

class GameResponse(BaseModel):
    doneTowers: list[DoneTower]  # список завершённых башен
    score: float                 # общий счёт
    tower: TowerInfo             # текущая башня

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)

#############################################################
#/api/rounds

class RoundInfo(BaseModel):
    duration: int                      # Длительность раунда в секундах
    endAt: str                    # Время окончания раунда
    name: str                          # Название раунда
    repeat: int                        # Номер повторения раунда
    startAt: str                  # Время начала раунда
    status: Literal["active", "pending", "finished"]  # Статус раунда

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)

class RoundsGame(BaseModel):
    eventId: str                       # Идентификатор события
    now: str                      # Текущее время сервера
    rounds: list[RoundInfo]            # Список раундов

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)

##################################################################
#/api/shuffle
class ShuffleResponse(BaseModel):
    """
    Модель ответа на запрос перемешивания слов.
    Содержит информацию о доступных словах и оставшихся перемешиваниях.
    """
    shuffleLeft: int #"Количество оставшихся перемешиваний")
    words: list[str] # новый список слов

    def __init__(self, **data):
        object.__setattr__(self, '__dict__', data)

##################################################################