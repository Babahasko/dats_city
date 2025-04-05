import random
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
from functools import lru_cache
from pydantic import BaseModel
import numpy as np
from sender.game_parser import WordPosition


def direction_xyz_to_number(pos: Tuple[int, int, int]):
    if pos == (0, 0, -1):
        return 1
    elif pos == (1, 0, 0):
        return 2
    elif pos == (0, 1, 0):
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


class MapSize(BaseModel):
    x: int
    y: int
    z: int


class TowerBuilder:

    def __init__(self, words: List[str]):
        self.bounding_box = {}
        self.build_requests = []
        self.z_level = 0
        self.words = words
        self.grid = np.zeros((30, 30, 50), dtype=np.int8)

        # Для хранения букв в каждой позиции
        self.letter_grid = np.full((30, 30, 50), '', dtype='U1')
        self.used_word_ids = set()
        self.word_objects: List[TowerBuilder.Word] = []
        self.floor_stats = defaultdict(lambda: {
            'width': 0,
            'depth': 0,
            'words_x': 0,
            'words_y': 0,
            'letters': 0
        })

    class Word:
        def __init__(self, text: str, start_pos: Tuple[int, int, int], direction: Tuple[int, int, int]):
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

    def build_optimized_tower(self) -> Dict:
        """Оптимизированный алгоритм строительства башни"""
        # 1. Строим основание с оптимальными пропорциями
        self._build_optimized_foundation()

        # 2. Строим этажи с учетом коэффициентов
        for floor in range(1, 5000):
            z_level = -floor
            if not self._build_optimized_floor(z_level):
                break

        return {
            'score': self._calculate_score(),
            'words': self._get_tower_structure(),
            'height': abs(self.z_level) + 1,
            # 'bounding_box': self.bounding_box
        }

    def shuffle_words(self):
        for i in range(len(self.word_objects)):
            self.words[i] = get_random_word(len(self.words[i]))

    def _build_optimized_foundation(self):
        """Строим основание с оптимальными пропорциями 1:1"""
        # Выбираем 3 самых длинных слова
        foundation_words = sorted(
            [(i, w) for i, w in enumerate(self.words) if i not in self.used_word_ids],
            key=lambda x: len(x[1]),
            reverse=True
        )[:3]

        # Размещаем слова для достижения пропорций ~1:1
        base_positions = [(0, 0, 0), (0, 10, 0), (10, 5, 0)]
        for idx, ((word_id, word), pos) in enumerate(zip(foundation_words, base_positions)):
            direction = (1, 0, 0) if idx < 2 else (0, 1, 0)
            self._place_word(word, pos, direction, word_id)

    def _build_optimized_floor(self, z_level: int) -> bool:
        """Строит один этаж с оптимизацией баллов"""
        best_score = -1
        best_placement = None

        # Пробуем несколько вариантов размещения
        for _ in range(2):  # Количество попыток найти оптимальное размещение
            placement = self._generate_floor_placement(z_level)

            if placement:
                score = self._evaluate_placement_score(placement, z_level)
                if score > best_score:
                    best_score = score
                    best_placement = placement

        if best_placement:
            for word, pos, direction, word_id in best_placement:
                self._place_word(word, pos, direction, word_id)
            return True
        return False

    def _evaluate_placement_score(self, placement: List[Tuple], z_level: int) -> float:
        """
        Оценивает потенциальное размещение слов на этаже по:
        1. Количеству букв (базовые баллы)
        2. Пропорциям этажа (width/depth ratio)
        3. Плотности слов (количество слов по X и Y)
        4. Количеству пересечений между словами
        """
        # Временные переменные для статистики этажа
        temp_stats = {
            'letters': 0,
            'width': self.floor_stats[z_level]['width'],
            'depth': self.floor_stats[z_level]['depth'],
            'words_x': self.floor_stats[z_level]['words_x'],
            'words_y': self.floor_stats[z_level]['words_y'],
            'intersections': 0
        }

        # Временное хранилище позиций для проверки пересечений
        temp_letter_positions = defaultdict(list)

        # Анализируем каждое слово в предлагаемом размещении
        for word, pos, direction, _ in placement:
            word_obj = self.Word(word, pos, direction)
            temp_stats['letters'] += len(word)

            # Обновляем статистику направления
            if direction == (1, 0, 0):  # Слово по X
                temp_stats['words_x'] += 1
                temp_stats['width'] = max(temp_stats['width'], word_obj.end_pos[0] - word_obj.start_pos[0] + 1)
                temp_stats['depth'] = max(temp_stats['depth'], word_obj.end_pos[1] - word_obj.start_pos[1] + 1)
            elif direction == (0, 1, 0):  # Слово по Y
                temp_stats['words_y'] += 1
                temp_stats['depth'] = max(temp_stats['depth'], word_obj.end_pos[1] - word_obj.start_pos[1] + 1)
                temp_stats['width'] = max(temp_stats['width'], word_obj.end_pos[0] - word_obj.start_pos[0] + 1)

            # Проверяем пересечения с уже существующими словами
            for letter_pos in word_obj.get_letter_positions():
                # if letter_pos in self.letter_positions:
                #     temp_stats['intersections'] += len(self.letter_positions[letter_pos]) fix

                # Добавляем во временное хранилище для проверки пересечений внутри размещения
                temp_letter_positions[letter_pos].append(word_obj)

        # Дополнительно проверяем пересечения между словами в этом размещении
        for pos, words in temp_letter_positions.items():
            if len(words) > 1:
                temp_stats['intersections'] += len(words) - 1

        # Рассчитываем коэффициенты
        width, depth = temp_stats['width'], temp_stats['depth']

        # Коэффициент пропорции (оптимально 1:1)
        if width == 0 or depth == 0:
            proportion_coef = 0.5  # Минимальное значение при отсутствии слов по одной оси
        else:
            proportion_coef = min(width, depth) / max(width, depth)

        # Коэффициент плотности (1 + (words_x + words_y)/4)
        density_coef = 1 + (temp_stats['words_x'] + temp_stats['words_y']) / 4

        # Коэффициент пересечений (поощряем больше пересечений)
        intersection_coef = 1 + temp_stats['intersections'] * 0.1

        # Итоговая оценка
        base_score = temp_stats['letters']  # 1 буква = 1 балл
        score = base_score * proportion_coef * density_coef * intersection_coef

        return score

    def _evaluate_horizontal_position(self, word: str, pos: Tuple[int, int, int],
                                      direction: Tuple[int, int, int], z_level: int) -> float:
        """Оценивает качество позиции по нескольким критериям"""
        # 1. Количество пересечений с вертикальными словами
        intersections = 0
        word_obj = self.Word(word, pos, direction)



        for letter_pos in word_obj.get_letter_positions()[1:]:  # Исключаем первую букву
            if self.grid[letter_pos] > 0:
                intersections += 1

        # 2. Влияние на пропорции этажа
        current_stats = self.floor_stats[z_level]
        new_width = current_stats['width']
        new_depth = current_stats['depth']

        if direction == (1, 0, 0):
            new_width = max(new_width, word_obj.end_pos[0] - word_obj.start_pos[0] + 1)
        else:
            new_depth = max(new_depth, word_obj.end_pos[1] - word_obj.start_pos[1] + 1)

        if new_width == 0 or new_depth == 0:
            proportion_score = 0.5
        else:
            proportion_score = min(new_width, new_depth) / max(new_width, new_depth)

        # 3. Влияние на плотность
        new_density = 1 + (current_stats['words_x'] + (1 if direction == (1, 0, 0) else 0) + (
                current_stats['words_y'] + (1 if direction == (0, 1, 0) else 0))) / 4

        # Комбинированная оценка (веса можно настраивать)
        score = 0.5 * intersections + 0.3 * proportion_score + 0.2 * new_density
        return score

    def _generate_floor_placement(self, z_level: int) -> List[Tuple]:
        """Генерирует вариант размещения слов на этаже"""
        placement = []
        used_words = set()
        available_words = [(i, w) for i, w in enumerate(self.words)
                           if i not in self.used_word_ids and i not in used_words]

        # Пробуем добавить 1-2 вертикальных слова
        for idx in range(len(available_words)):
            if not available_words:
                break
            if idx > len(available_words) - 1:
                break
            if len(placement) > 3:
                break
            word_id, word = available_words[idx]
            position = self._find_best_vertical_position(word, z_level)
            if position:
                placement.append((word, position, (0, 0, -1), word_id))
                used_words.add(word_id)
                available_words.remove((word_id, word))
        print(placement)
        h_words = 0
        # Добавляем горизонтальные слова
        for _ in range(len(available_words)):
            if not available_words:
                break
            if h_words > 3:
                break
            word_id, word = random.choice(available_words)
            direction = (1, 0, 0) if z_level % 2 == 0 else (0, 1, 0)
            position = self._find_horizontal_position(word, z_level, direction)
            if position:
                placement.append((word, position, direction, word_id))
                used_words.add(word_id)
                available_words.remove((word_id, word))
                h_words += 1

        return placement if len(placement) >= 1 else None

    def _find_horizontal_position(self, word: str, z_level: int, direction: Tuple[int, int, int]) -> Optional[
        Tuple[int, int, int]]:
        """Находит оптимальную позицию для горизонтального слова с учетом:
        1. Максимального количества пересечений
        2. Улучшения пропорций этажа
        3. Плотности размещения
        """
        best_position = None
        best_score = -1
        word_length = len(word)

        # Определяем диапазон для поиска позиций
        x_range = range(0, 30 - word_length + 1) if direction == (1, 0, 0) else range(0, 30)
        y_range = range(0, 30 - word_length + 1) if direction == (0, 1, 0) else range(0, 30)

        # Пробуем разные позиции с шагом 3 для оптимизации
        for x in x_range[::]:
            for y in y_range[::]:
                if direction == (1, 0, 0):
                    pos = (x, y, z_level)
                else:
                    pos = (y, x, z_level)

                # Проверяем возможность размещения
                if not self._can_place_word(word, pos, direction):
                    continue

                # Оцениваем качество позиции
                score = self._evaluate_horizontal_position(word, pos, direction, z_level)

                if score > best_score:
                    best_score = score
                    best_position = pos

        return best_position

    def _find_best_vertical_position_(self, word: str, z_level: int) -> Optional[Tuple[int, int, int]]:
        """
        Находит оптимальную позицию для вертикального слова с учетом:
        1. Максимального количества пересечений
        2. Улучшения пропорций этажа
        3. Соответствия направлению
        4. Плотности размещения

        Возвращает лучшую позицию (x, y, z) или None, если размещение невозможно
        """
        best_position = None
        best_score = -1
        word_length = len(word)

        # Собираем все возможные точки пересечения с предыдущим этажом
        potential_connections = []
        # print("here")

        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                for z in range(self.grid.shape[2]):
                    if z == z_level + 1:  # Только с предыдущего этажа
                        letter = self._get_letter_at_pos((x, y, z))
                        if letter in word[0:]:  # Ищем все пересечения кроме первой буквы
                            potential_connections.append((x, y, z, letter))





        # Анализируем каждое потенциальное пересечение
        for x, y, z, letter in potential_connections:
            # Находим все возможные позиции для этого пересечения
            for i in range(0, word_length):  # Первая буква не считается todo(1 -> 0) check
                if word[i] == letter:
                    candidate_pos = (x, y, z - i)

                    # Проверяем возможность размещения
                    if not self._can_place_word(word, candidate_pos, (0, 0, -1)):
                        continue

                    # Оцениваем качество позиции
                    current_score = self._evaluate_vertical_position(
                        word, candidate_pos, z_level)

                    # Обновляем лучшую позицию
                    if current_score > best_score:
                        best_score = current_score
                        best_position = candidate_pos

        return best_position

    def _find_best_vertical_position(self, word: str, z_level: int) -> Optional[Tuple[int, int, int]]:
        """
        #NEW
        Находит оптимальную позицию для вертикального слова с учетом:
        1. Максимального количества пересечений
        2. Улучшения пропорций этажа
        3. Соответствия направлению
        4. Плотности размещения

        Возвращает лучшую позицию (x, y, z) или None, если размещение невозможно
        """
        best_position = None
        best_score = -1
        word_length = len(word)

        # Собираем все возможные точки пересечения с предыдущим этажом
        potential_connections = []
        # print("here")

        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                letter = self._get_letter_at_pos((x, y, z_level - 1))  # todo cehck
                if letter in word[0:]:  # Ищем все пересечения кроме первой буквы
                    potential_connections.append((x, y, z_level, letter))

        # Анализируем каждое потенциальное пересечение
        for x, y, z, letter in potential_connections:
            # Находим все возможные позиции для этого пересечения
            for i in range(0, word_length):  # Первая буква не считается todo(1 -> 0) check
                if word[i] == letter:
                    candidate_pos = (x, y, z - i)

                    # Проверяем возможность размещения
                    if not self._can_place_word(word, candidate_pos, (0, 0, -1)):
                        continue

                    # Оцениваем качество позиции
                    current_score = self._evaluate_vertical_position(
                        word, candidate_pos, z_level)

                    # Обновляем лучшую позицию
                    if current_score > best_score:
                        best_score = current_score
                        best_position = candidate_pos

        return best_position

    def _evaluate_vertical_position(self, word: str, pos: Tuple[int, int, int],
                                    z_level: int) -> float:
        """
        Оценивает качество позиции для вертикального слова по:
        1. Количеству пересечений
        2. Влиянию на пропорции этажа
        3. Плотности слов
        """
        # Создаем временную статистику
        temp_stats = self._simulate_placement(word, pos, (0, 0, -1), z_level)
        if temp_stats is None:
            return -1
        # Рассчитываем коэффициенты
        width, depth = temp_stats['width'], temp_stats['depth']
        prop_coef = min(width, depth) / max(width, depth) if width and depth else 0.5
        density_coef = 1 + (temp_stats['words_x'] + temp_stats['words_y']) / 4
        intersect_coef = 1 + temp_stats['intersections'] * 0.15  # Бонус 15% за пересечение

        # Базовые баллы (учитываем длину слова + 2 балла за вертикальность)
        base_score = len(word) + 2

        # Итоговая оценка
        return base_score * prop_coef * density_coef * intersect_coef

    def _can_place_word(self, word: str, pos: Tuple[int, int, int],
                        direction: Tuple[int, int, int]) -> bool:
        """Проверяет возможность размещения слова с использованием 3D grid"""
        x, y, z = pos
        dx, dy, dz = direction

        # Проверка выхода за границы
        end_x = x + (len(word) - 1) * dx
        end_y = y + (len(word) - 1) * dy
        end_z = z + (len(word) - 1) * dz

        if (x < 0 or y < 0 or end_x >= 30 or end_y >= 30 or z < -49 or end_z > 0):
            return False

        # Проверка последовательного соединения (запрет home+elephant)
        if (self._check_position(x - dx, y - dy, z - dz) or
                self._check_position(end_x + dx, end_y + dy, end_z + dz)):
            return False

        # Проверка каждой буквы слова
        intersections = 0
        for i in range(len(word)):
            px, py, pz = x + i * dx, y + i * dy, z + i * dz

            # Проверка занятости позиции
            if self.grid[px, py, pz] == 1:
                existing_letter = self.letter_grid[px, py, pz]
                if existing_letter != word[i]:
                    return False
                intersections += 1

                # Проверка что пересекаемся только с одним словом
                if self.grid[px, py, pz] > 1:
                    return False

        # Для этажей выше 0 должно быть ровно одно пересечение
        if z < 0 and intersections != 1:
            return False

        # Проверка минимальных расстояний
        for i in range(len(word)):
            px, py, pz = x + i * dx, y + i * dy, z + i * dz
            for dx2, dy2, dz2 in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                if self._check_position(px + dx2, py + dy2, pz + dz2, direction):
                    return False

        return True

    def _check_position(self, x: int, y: int, z: int,
                        exclude_dir: Tuple[int, int, int] = None) -> bool:
        """Проверяет занятость позиции с учетом направления"""
        if not (0 <= x < 30 and 0 <= y < 30 and -50 <= z <= 0):
            return False

        if self.grid[x, y, z] == 0:
            return False

        if exclude_dir:
            for word_id in self._get_words_at_pos(x, y, z):
                if self.word_objects[word_id].direction == exclude_dir:
                    return True
        return True

    def _simulate_placement(self, word: str, pos: Tuple[int, int, int],
                            direction: Tuple[int, int, int], z_level: int) -> Optional[Dict[str, float]]:
        """
        Моделирует добавление слова с проверкой всех правил:
        - На этажах выше 0 должно быть ровно одно пересечение
        - Пересечение только одной буквой
        - Одинаковые буквы в пересечении
        - Нет наложения разных букв
        """
        # Проверка для этажей выше 0
        if z_level < 0:
            intersection_count = 0
            intersection_pos = None

            for i in range(len(word)):
                letter_pos = (
                    pos[0] + i * direction[0],
                    pos[1] + i * direction[1],
                    pos[2] + i * direction[2]
                )

                if self.grid[letter_pos] != 0:
                    # Проверка совпадения букв
                    if self._get_letter_at_pos(letter_pos) != word[i]:
                        return None

                    intersection_count += 1
                    if intersection_count > 1:
                        return None
                    intersection_pos = letter_pos

            # Должно быть ровно одно пересечение
            if intersection_count != 1:
                return None

            # Проверка что пересекаемся только с одним словом
            if len(self.grid[intersection_pos[0], intersection_pos[1], intersection_pos[2]]) > 1:
                return None

        # Остальная логика симуляции...
        temp_stats = {
            'letters': self.floor_stats.get(z_level, {}).get('letters', 0) + len(word),
            'width': self.floor_stats.get(z_level, {}).get('width', 0),
            'depth': self.floor_stats.get(z_level, {}).get('depth', 0),
            'words_x': self.floor_stats.get(z_level, {}).get('words_x', 0),
            'words_y': self.floor_stats.get(z_level, {}).get('words_y', 0),
            'intersections': 1 if z_level < 0 else 0
        }

        # Обновление размеров этажа...
        end_pos = (
            pos[0] + (len(word) - 1) * direction[0],
            pos[1] + (len(word) - 1) * direction[1],
            pos[2] + (len(word) - 1) * direction[2]
        )

        if direction == (1, 0, 0):  # Горизонтальное по X
            temp_stats['words_x'] += 1
            temp_stats['width'] = max(temp_stats['width'], end_pos[0] - pos[0] + 1)
            temp_stats['depth'] = max(temp_stats['depth'], end_pos[1] - pos[1] + 1)
        elif direction == (0, 1, 0):  # Горизонтальное по Y
            temp_stats['words_y'] += 1
            temp_stats['depth'] = max(temp_stats['depth'], end_pos[1] - pos[1] + 1)
            temp_stats['width'] = max(temp_stats['width'], end_pos[0] - pos[0] + 1)

        return temp_stats

    def _calculate_score_details(self) -> Dict:
        """Подробный расчет баллов с разбивкой по этажам"""
        floor_details = {}
        total_score = 0

        for z_level, stats in self.floor_stats.items():
            # Коэффициент пропорции
            width, depth = stats['width'], stats['depth']
            prop_coef = min(width, depth) / max(width, depth) if width and depth else 0.5

            # Коэффициент плотности
            density_coef = 1 + (stats['words_x'] + stats['words_y']) / 4

            # Баллы за этаж
            floor_score = stats['letters'] * prop_coef * density_coef
            floor_details[z_level] = {
                'base_score': stats['letters'],
                'prop_coef': round(prop_coef, 2),
                'density_coef': round(density_coef, 2),
                'floor_score': round(floor_score, 2)
            }
            total_score += floor_score

        return {
            'total_score': round(total_score, 2),
            'height': len(self.floor_stats),
            'floor_details': floor_details,
            'words': self._get_tower_structure()
        }

    def _place_word(self, word: str, pos: Tuple[int, int, int],
                    direction: Tuple[int, int, int], word_id: int):
        """Размещает слово с обновлением всех структур данных"""
        x, y, z = pos
        dx, dy, dz = direction

        # Обновляем 3D grid
        for i in range(len(word)):
            px, py, pz = x + i * dx, y + i * dy, z + i * dz
            self.grid[px, py, pz] += 1
            self.letter_grid[px, py, pz] = word[i]

        # Создаем и сохраняем объект слова
        word_obj = self.Word(word, pos, direction)
        self.word_objects.append(word_obj)
        self.used_word_ids.add(word_id)

        # Обновляем статистику этажа
        self._update_floor_stats(word_obj, z)

    def _update_floor_stats(self, word_obj: 'Word', z_level: int):
        """Обновляет статистику этажа после добавления слова"""
        stats = self.floor_stats[z_level]
        stats['letters'] += len(word_obj.text)

        end_x = word_obj.end_pos[0]
        end_y = word_obj.end_pos[1]

        if word_obj.direction == (1, 0, 0):  # Горизонтальное по X
            stats['words_x'] += 1
            stats['width'] = max(stats['width'], end_x - word_obj.start_pos[0] + 1)
            stats['depth'] = max(stats['depth'], end_y - word_obj.start_pos[1] + 1)
        elif word_obj.direction == (0, 1, 0):  # Горизонтальное по Y
            stats['words_y'] += 1
            stats['depth'] = max(stats['depth'], end_y - word_obj.start_pos[1] + 1)
            stats['width'] = max(stats['width'], end_x - word_obj.start_pos[0] + 1)

        # Подсчет пересечений
        for i in range(len(word_obj.text)):
            pos = (
                word_obj.start_pos[0] + i * word_obj.direction[0],
                word_obj.start_pos[1] + i * word_obj.direction[1],
                word_obj.start_pos[2] + i * word_obj.direction[2]
            )
            if self.grid[pos] > 1:
                stats['intersections'] += 1

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

        return self.letter_grid[pos[0]][pos[1]][pos[2]]

    def construct_matrix_2(self):
        result = {
            "cubes": [],
            "text": [],
        }
        for i in self.word_objects:
            index = 0
            start = i.start_pos

            result["cubes"].append([start[0], abs(start[2]), start[1], ])
            result["text"].append(i.text[index])
            while index != len(i.text) - 1:
                index += 1

                start = (start[0] + i.direction[0], start[1] + i.direction[1], start[2] + i.direction[2])
                result["cubes"].append([start[0], abs(start[2]), start[1], ])
                result["text"].append(i.text[index])
        return result


if __name__ == "__main__":

    # words = ["foundation", "support", "column", "beam", "floor",
    #      "wall", "ceiling", "structure", "building", "tower"]
    words = []
    for _ in range(1000):
        words.append(get_random_word(random.randint(6, 10)))

    builder = TowerBuilder(words)
    tower = builder.build_optimized_tower()
    # builder.shuffle_words()
    # tower = builder.continue_build()

    print(f"Башня построена! Счет: {tower['score']}")
    print(f"Высота: {tower['height']} этажей")
    # print(f"Габариты: {tower['bounding_box']}")
    print("\nСтруктура башни:")
    for i, word in enumerate(tower['words']):
        direction = ""
        if word['dir'] == (1, 0, 0):
            direction = "X"
        elif word['dir'] == (0, 1, 0):
            direction = "Y"
        elif word['dir'] == (0, 0, -1):
            direction = "Z"
        else:
            print("errro", word['dir'])
            # exit(-1)

        print(f"{i + 1}. {word['text']} ({direction}) at {word['pos']}")
    print(builder.build_requests)

    print(builder.construct_matrix_2())
