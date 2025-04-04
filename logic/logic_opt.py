import random
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

from pydantic import BaseModel

from sender.game_parser import WordPosition


def direction_xyz_to_number(pos: Tuple[int,int,int]):
    if pos == (0,0,-1):
        return 1
    elif pos == (1,0,0):
        return 2
    elif pos == (0,1,0):
        return 3
    else:
        print(pos)
        return exit(-1)


def get_random_word(length: int):
    import string
    import random
    result = ""
    for _ in range(length):
        result += random.choice(string.ascii_lowercase)
    return result


class MapSize(BaseModel) :
    x : int
    y : int
    z : int

class TowerBuilder:



    def __init__(self, words: list[str]):
        self.z_level = 0
        self.build_requests:List[WordPosition] = []


        self.words = words
        self.used_word_ids = set()
        self.word_objects:List[TowerBuilder.Word] = []
        self.letter_positions = {}
        self.score = 0
        self.bounding_box = [0, 0, 0]  # Текущие границы башни
        self.valid_directions = [
            (1, 0, 0),   # По оси X
            (0, 1, 0),    # По оси Y
            (0, 0, -1)  # По оси Z вниз
        ]

    class Word:
        def __init__(self, text: str, start_pos: Tuple[int, int, int], direction: Tuple[int,int,int]):
            self.text = text
            self.start_pos = start_pos
            self.direction = direction
            self.length = len(text)
            self.end_pos = self._calculate_end_pos()

        def _calculate_end_pos(self) -> Tuple[int, int, int]:
            return (
                self.start_pos[0] + (self.length - 1) * self.direction[0],
                self.start_pos[1] + (self.length - 1) * self.direction[1],
                self.start_pos[2] + (self.length - 1) * self.direction[2]
            )

        def get_letter_positions(self) -> List[Tuple[int, int, int]]:
            return [
                (
                    self.start_pos[0] + i * self.direction[0],
                    self.start_pos[1] + i * self.direction[1],
                    self.start_pos[2] + i * self.direction[2]
                )
                for i in range(self.length)
            ]

    def build_tower(self) -> Dict:
        """Новый алгоритм строительства с более гибкими правилами"""
        # 1. Строим основание из 3 самых длинных горизонтальных слов
        self._build_foundation()

        # 2. Строим этажи снизу вверх
        for floor in range(1, 500):  # Максимум 50 этажей
            self.z_level = -floor
            if not self._build_floor_optimized(self.z_level):
                break

        return {
            'score': self._calculate_score(),
            'words': self._get_tower_structure(),
            'height': self.z_level,
            'bounding_box': self.bounding_box
        }

    def _build_floor_optimized(self, z_level: int) -> bool:
        """Улучшенный алгоритм построения этажа"""
        words_added = 0
        available_words = self._get_available_words_sorted()

        # 1. Сначала пробуем добавить вертикальные слова
        for word_id, word in available_words:
            if words_added >= 2:
                break
            if self._try_place_vertical_optimized(word, z_level, word_id):
                words_added += 1

        # 2. Затем добавляем горизонтальные слова
        for word_id, word in available_words:
            if words_added >= 4:  # Максимум 4 слова на этаж
                break
            if self._try_place_horizontal_optimized(word, z_level, word_id):
                words_added += 1

        return words_added > 0  # Хотя бы одно слово добавили
    def shuffle_words(self):
        for i in range(len(self.word_objects)):
            self.words[i] = get_random_word(len(words[i]))

    def _try_place_vertical_optimized(self, word: str, z_level: int, word_id: int) -> bool:
        """Гибкий алгоритм размещения вертикальных слов"""
        direction = (0, 0, -1)

        # Ищем ВСЕ возможные позиции для этого слова
        possible_positions = []
        for pos in self.letter_positions:
            if pos[2] != z_level + 1:  # Только предыдущий этаж
                continue

            letter = self._get_letter_at_pos(pos)
            if not letter:
                continue

            # Ищем все совпадения букв (кроме первой)
            for i in range(1, len(word)):
                if word[i] == letter:
                    start_pos = (pos[0], pos[1], pos[2] - i)
                    possible_positions.append(start_pos)

        # Пробуем позиции в случайном порядке
        random.shuffle(possible_positions)
        for start_pos in possible_positions:
            if self._can_place_word_optimized(word, start_pos, direction):
                self._place_word(word, start_pos, direction)
                self.build_requests.append(WordPosition(
                    dir=direction_xyz_to_number(direction),
                    id=word_id,
                    pos=start_pos
                ))
                return True
        return False

    def _try_place_horizontal_optimized(self, word: str, z_level: int, word_id: int) -> bool:
        """Гибкий алгоритм размещения горизонтальных слов"""
        direction = (1, 0, 0) if z_level % 2 == 0 else (0, 1, 0)

        # Генерируем все возможные стартовые позиции
        for x in range(0, 30, 3):  # Шаг 3 для оптимизации
            for y in range(0, 30, 3):
                start_pos = (x if direction == (1, 0, 0) else y,
                             y if direction == (0, 1, 0) else x,
                             z_level)

                if self._can_place_word_optimized(word, start_pos, direction):
                    self._place_word(word, start_pos, direction)
                    self.build_requests.append(WordPosition(
                        dir=direction_xyz_to_number(direction),
                        id=word_id,
                        pos=start_pos
                    ))
                    return True
        return False

    def _can_place_word_optimized(self, word: str, start_pos: Tuple[int, int, int],
                                  direction: Tuple[int, int, int]) -> bool:
        """Облегченная проверка размещения"""
        word_obj = self.Word(word, start_pos, direction)

        # Только базовые проверки
        if any(c < 0 for c in word_obj.start_pos[:2]) or word_obj.end_pos[2] < -50:
            return False

        # Проверяем только явные коллизии
        for pos in word_obj.get_letter_positions():
            if pos in self.letter_positions:
                for word_id in self.letter_positions[pos]:
                    if self.word_objects[word_id].direction != direction:
                        return False

        # Для вертикальных слов - хотя бы одно пересечение
        if direction == (0, 0, -1):
            if not any(pos in self.letter_positions for pos in word_obj.get_letter_positions()[1:]):
                return False

        return True



    def continue_build(self):


        # 1 Последовательно добавляем этажи
        current_z = self.z_level
        while current_z >= -300:  # Ограничение на высоту башни
            if not self._build_floor(current_z):
                break
            current_z -= 1

        return {
            'score': self._calculate_score(),
            'words': self._get_tower_structure(),
            'height': abs(current_z) + 1,
            'bounding_box': self.bounding_box
        }

    def _build_foundation(self) -> None:
        """Строим основание башни (только горизонтальные слова)"""
        # Выбираем 3 самых длинных слова для основания
        foundation_words:List[Tuple[int,str]] = sorted(
            [(i,w) for i, w in enumerate(self.words) if i not in self.used_word_ids and len(w) >= 5],
            key=len, reverse=True
        )[:3]

        # Размещаем слова вдоль осей X и Y
        for i, word in enumerate(foundation_words):
            direction = (1, 0, 0) if i % 2 == 0 else (0, 1, 0)  # Чередуем направления
            start_pos = (i * 8, (1 - i % 2) * 5, 0)
            self._place_word(word[1], start_pos, direction)
            self.build_requests.append(WordPosition(dir=direction_xyz_to_number(direction),id=word[0],pos=start_pos))

    def _build_floor(self, z_level: int) -> bool:
        """Строит один этаж башни"""
        words_added = 0
        available_words = self._get_available_words_sorted()

        # Сначала пробуем добавить вертикальные слова (они дают больше пересечений)
        for word_id, word in available_words:
            if words_added >= 2:  # Максимум 2 вертикальных слова на этаж
                break
            if self._try_place_vertical(word, z_level, word_id):
                words_added += 1

        # Затем добавляем горизонтальные слова
        for word_id, word in available_words:
            if words_added >= 4:  # Максимум 4 слова на этаж
                break
            if self._try_place_horizontal(word, z_level, word_id):
                words_added += 1

        return words_added >= 1  # Минимум 1 слово на этаж

    def _try_place_vertical(self, word: str, z_level: int, word_id: int) -> bool:
        """Размещает вертикальное слово с более гибкими условиями"""
        direction = (0, 0, -1)

        # Ищем все возможные пересечения с предыдущими словами
        for pos, word_ids in self.letter_positions.items():
            if pos[2] != z_level + 1:  # Только с предыдущего этажа
                continue

            letter = self._get_letter_at_pos(pos)
            if not letter:
                continue

            # Ищем совпадения букв (кроме первой)
            for i in range(1, len(word)):
                if word[i] == letter:
                    start_pos = (pos[0], pos[1], pos[2] - i)
                    if self._can_place_word(word, start_pos, direction):
                        self._place_word(word, start_pos, direction)
                        self.build_requests.append(WordPosition(
                            dir=direction_xyz_to_number(direction),
                            id=word_id,
                            pos=start_pos
                        ))
                        return True
        return False

    def _try_place_horizontal(self, word: str, z_level: int, word_id: int) -> bool:
        """Размещает горизонтальное слово с более гибкими условиями"""
        direction = (1, 0, 0) if z_level % 2 == 0 else (0, 1, 0)

        # Ищем пересечения с вертикальными словами
        for vertical_word in [w for w in self.word_objects if w.direction == (0, 0, -1)]:
            # Проверяем, что вертикальное слово проходит через этот этаж
            if not (vertical_word.end_pos[2] <= z_level <= vertical_word.start_pos[2]):
                continue

            # Ищем совпадения букв (кроме первой)
            for i in range(1, len(word)):
                for j in range(len(vertical_word.text)):
                    if word[i] == vertical_word.text[j]:
                        if direction == (1, 0, 0):
                            start_pos = (
                                vertical_word.start_pos[0] - i,
                                vertical_word.start_pos[1],
                                z_level
                            )
                        else:
                            start_pos = (
                                vertical_word.start_pos[0],
                                vertical_word.start_pos[1] - i,
                                z_level
                            )

                        if self._can_place_word(word, start_pos, direction):
                            self._place_word(word, start_pos, direction)
                            self.build_requests.append(WordPosition(
                                dir=direction_xyz_to_number(direction),
                                id=word_id,
                                pos=start_pos
                            ))
                            return True
        return False

    def _get_available_words_sorted(self) -> list[Tuple[int,str]]:
        """Возвращает доступные слова, отсортированные по длине"""
        return sorted(
            [(i,w) for i, w in enumerate(self.words) if i not in self.used_word_ids],
            key=len, reverse=True
        )

    def _can_place_word(self, word: str, start_pos: Tuple[int, int, int], direction: Tuple[int, int, int]) -> bool:
        """Облегченная проверка возможности размещения"""
        word_obj = self.Word(word, start_pos, direction)

        # Проверяем границы
        if any(c < 0 for c in word_obj.start_pos[:2]) or word_obj.end_pos[2] < -50:
            return False

        # Проверяем коллизии
        for pos in word_obj.get_letter_positions():
            if pos in self.letter_positions:
                for word_id in self.letter_positions[pos]:
                    if self.word_objects[word_id].direction != direction:
                        return False

        # Для вертикальных слов требуем хотя бы одно пересечение
        if direction == (0, 0, -1):
            has_intersection = any(
                pos in self.letter_positions
                for pos in word_obj.get_letter_positions()[1:]
            )
            if not has_intersection:
                return False

        return True

    def _is_position_occupied(self, pos: Tuple[int, int, int], direction: Tuple[int,int,int]) -> bool:
        """Проверяет, занята ли позиция другими словами"""
        if pos not in self.letter_positions:
            return False

        # Проверяем только слова с другим направлением
        for word_id in self.letter_positions[pos]:
            word_dir = self.word_objects[word_id].direction
            if word_dir != direction:
                return True
        return False

    def _place_word(self, word: str, start_pos: Tuple[int, int, int], direction: Tuple[int,int,int]) -> None:
        """Размещает слово в башне"""
        word_obj = self.Word(word, start_pos, direction)
        word_id = len(self.word_objects)
        self.word_objects.append(word_obj)
        self.used_word_ids.add(self.words.index(word))

        # Обновляем позиции букв - теперь кортеж как ключ
        for pos in word_obj.get_letter_positions():
            if pos not in self.letter_positions.keys() :
                self.letter_positions[pos]= []
            self.letter_positions[pos].append(word_id)  # Теперь pos - это кортеж (x,y,z)

        self._update_bounding_box(word_obj)

    def _update_bounding_box(self, word_obj: 'Word') -> None:
        """Обновляет границы башни"""
        for i in range(3):
            self.bounding_box[i] = max(
                self.bounding_box[i],
                abs(word_obj.start_pos[i]),
                abs(word_obj.end_pos[i])
            )

    def _calculate_score(self) -> float:
        """Новая система подсчета баллов с учетом этажей, пропорций и плотности"""
        floor_scores = {}
        floor_stats = defaultdict(lambda: {'width': 0, 'depth': 0, 'words_x': 0, 'words_y': 0})

        # Собираем статистику по каждому этажу
        for word_obj in self.word_objects:
            z_level = word_obj.start_pos[2]
            if word_obj.direction == (1, 0, 0):  # Слово по X
                floor_stats[z_level]['words_x'] += 1
                floor_stats[z_level]['width'] = max(
                    floor_stats[z_level]['width'],
                    word_obj.end_pos[0] - word_obj.start_pos[0] + 1
                )
                floor_stats[z_level]['depth'] = max(
                    floor_stats[z_level]['depth'],
                    word_obj.end_pos[1] - word_obj.start_pos[1] + 1
                )
            elif word_obj.direction == (0, 1, 0):  # Слово по Y
                floor_stats[z_level]['words_y'] += 1
                floor_stats[z_level]['depth'] = max(
                    floor_stats[z_level]['depth'],
                    word_obj.end_pos[1] - word_obj.start_pos[1] + 1
                )
                floor_stats[z_level]['width'] = max(
                    floor_stats[z_level]['width'],
                    word_obj.end_pos[0] - word_obj.start_pos[0] + 1
                )

        # Рассчитываем баллы для каждого этажа
        total_score = 0
        for z_level, stats in floor_stats.items():
            # Базовые баллы (1 буква = 1 балл)
            base_score = sum(len(w.text) for w in self.word_objects if w.start_pos[2] == z_level)

            # Коэффициент пропорции
            width, depth = stats['width'], stats['depth']
            if width == 0 or depth == 0:
                proportion_coef = 0.5  # Минимальный коэффициент если нет слов по одной оси
            else:
                proportion_coef = min(width, depth) / max(width, depth)

            # Коэффициент плотности
            density_coef = 1 + (stats['words_x'] + stats['words_y']) / 4

            # Итоговые баллы за этаж
            floor_score = base_score * proportion_coef * density_coef
            floor_scores[z_level] = floor_score
            total_score += floor_score

        return round(total_score, 2)

    def _get_tower_structure(self) -> list[Dict]:
        """Возвращает структуру башни"""
        return [
            {
                'text': w.text,
                'pos': list(w.start_pos),
                'dir': w.direction
            }
            for w in self.word_objects
        ]

    def _get_letter_at_pos(self, pos: Tuple[int, int, int]) -> Optional[str]:
        """Возвращает букву в указанной позиции или None, если позиция пуста"""
        if pos not in self.letter_positions:
            return None

        word_id = self.letter_positions[pos][0]  # Берем первое слово в этой позиции
        word_obj = self.word_objects[word_id]

        # Вычисляем индекс буквы в слове
        if word_obj.direction == (1, 0, 0):  # По X
            letter_index = pos[0] - word_obj.start_pos[0]
        elif word_obj.direction == (0, 1, 0):  # По Y
            letter_index = pos[1] - word_obj.start_pos[1]
        else:  # По Z
            letter_index = pos[2] - word_obj.start_pos[2]

        return word_obj.text[letter_index]


    def construct_matrix(self):
        result = [[[""]*100]*100]*100
        for i in self.word_objects:
            index = 0
            start = i.start_pos
            result[start[0]][start[1]][start[2]] = i.text[index]
            while index != len(i.text)-1:
                index+=1
                start += i.direction
                result[start[0]][start[1]][start[2]] = i.text[index]
        return result
    def construct_matrix_2(self):
        result = {
            "cubes":[],
            "text":[],
        }
        for i in self.word_objects:
            index = 0
            start = i.start_pos
            result["cubes"].append([start[0],start[1],start[2]])
            result["text"].append(i.text[index])
            while index != len(i.text)-1:
                index+=1
                start += i.direction
                result["cubes"].append([start[0], start[1], start[2]])
                result["text"].append(i.text[index])
        return result

if __name__ == "__main__":


    # words = ["foundation", "support", "column", "beam", "floor",
    #      "wall", "ceiling", "structure", "building", "tower"]
    words = []
    for _ in range(1000):
        words.append(get_random_word(random.randint(6,10)))


    builder = TowerBuilder(words)
    tower = builder.build_tower()
    builder.shuffle_words()
    tower = builder.continue_build()

    print(f"Башня построена! Счет: {tower['score']}")
    print(f"Высота: {tower['height']} этажей")
    print(f"Габариты: {tower['bounding_box']}")
    print("\nСтруктура башни:")
    for i, word in enumerate(tower['words']):
        direction = "X" if word['dir'] == (1, 0, 0) else "Y" if word['dir'] == (0, 1, 0) else "Z"
        print(f"{i+1}. {word['text']} ({direction}) at {word['pos']}")
    print(builder.build_requests)

    print(builder.construct_matrix_2())