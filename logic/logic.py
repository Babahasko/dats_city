from typing import Dict

from pydantic import BaseModel

from sender import WordListResponse



class MapSize(BaseModel) :
    x : int
    y : int
    z : int

class TowerBuilder:
    def __init__(self, words: WordListResponse):
        self.map_size = MapSize(x = words.mapSize[0], y = words.mapSize[1], z = words.mapSize[2])
        self.words = words.words

        self.used_word_ids = set()
        self.tower = {
            'horizontal': {},  # z -> {y: [words]}
            'vertical': {}  # z -> {x: [words]}
        }
        self.letter_positions = {}  # (x,y,z) -> [word_ids]
        self.current_z = 0
        self.score = 0

    def build_tower(self) -> Dict:
        """Основной алгоритм строительства башни"""
        # Начинаем с горизонтальных слов на нулевом этаже
        self._build_ground_floor()

        # Строим последующие этажи
        while self._build_next_floor():
            pass

        return {
            'score': self.score,
            'words': self._get_tower_words(),
            'height': abs(self.current_z) + 1
        }

    def _build_ground_floor(self) -> None:
        """Размещаем слова на нулевом этаже"""
        long_words = sorted([w for w in self.words if len(w) >= 5],
                            key=len, reverse=True)[:3]

        for i, word in enumerate(long_words):
            self._place_word(word, 'horizontal', (i * 10, 0, 0))
            self.used_word_ids.add(self.words.index(word))

    def _build_next_floor(self) -> bool:
        """Строит следующий этаж башни"""
        self.current_z -= 1
        placed_words = 0

        # Ищем слова, которые можно разместить на этом этаже
        for word in [w for w in self.words if self.words.index(w) not in self.used_word_ids]:
            if self._try_place_word(word):
                placed_words += 1
                if placed_words >= 2:  # Минимум 2 слова на этаж
                    return True

        return False

    def _try_place_word(self, word: str) -> bool:
        """Пытается разместить слово на текущем этаже"""
        # Пробуем разместить горизонтально
        if self._find_place_for_word(word, 'horizontal'):
            return True

        # Пробуем разместить вертикально
        if self._find_place_for_word(word, 'vertical'):
            return True

        return False

    def _find_place_for_word(self, word: str, direction: str) -> bool:
        """Ищет подходящее место для слова"""
        for letter_pos in self.letter_positions:
            x, y, z = letter_pos
            if z != self.current_z + 1:  # Ищем пересечения с предыдущим этажом
                continue

            for i, letter in enumerate(word):
                if i == 0:  # Первая буква не считается
                    continue

                if any(w[-1] == letter for w in self.words):
                    pos = self._calculate_position(direction, (x, y, self.current_z), i, len(word))
                    if self._can_place_word(word, direction, pos):
                        self._place_word(word, direction, pos)
                        return True
        return False

    def _can_place_word(self, word: str, direction: str, start_pos: Tuple[int, int, int]) -> bool:
        """Проверяет, можно ли разместить слово в указанной позиции"""
        x, y, z = start_pos
        length = len(word)

        # Проверяем границы карты
        if (x < 0 or y < 0 or z > 0 or
                (direction == 'horizontal' and x + length > 30) or
                (direction == 'vertical' and y + length > 30)):
            return False

        # Проверяем пересечения с другими словами
        for i in range(length):
            check_pos = (x + (i if direction == 'horizontal' else 0),
                         y + (i if direction == 'vertical' else 0),
                         z)

            # Проверяем минимальное расстояние между словами
            if self._has_nearby_words(check_pos, direction):
                return False

        return True

    def _place_word(self, word: str, direction: str, start_pos: Tuple[int, int, int]) -> None:
        """Размещает слово в башне"""
        x, y, z = start_pos
        word_id = self.words.index(word)

        # Добавляем слово в структуру башни
        if direction == 'horizontal':
            self.tower['horizontal'][z][y] = (x, word)
            for i, letter in enumerate(word):
                pos = (x + i, y, z)
                self.letter_positions[pos].append(word_id)
        else:
            self.tower['vertical'][z][x] = (y, word)
            for i, letter in enumerate(word):
                pos = (x, y + i, z)
                self.letter_positions[pos].append(word_id)

        self.used_word_ids.add(word_id)
        self._update_score(word, direction)

    def _update_score(self, word: str, direction: str) -> None:
        """Обновляет счет на основе размещенного слова"""
        length = len(word)
        intersections = self._count_intersections(word, direction)
        self.score += length * (1 + intersections * 0.5)

    def _count_intersections(self, word: str, direction: str) -> int:
        """Считает количество пересечений слова с другими словами"""
        # Реализация опущена для краткости
        return min(2, len(word) - 1)  # Максимум 2 пересечения учитываются

    def _get_tower_words(self) -> List[Dict]:
        """Возвращает список слов башни в требуемом формате"""
        result = []
        for direction in ['horizontal', 'vertical']:
            for z, words in self.tower[direction].items():
                for coord, (start, word) in words.items():
                    pos = [start, coord, z] if direction == 'horizontal' else [coord, start, z]
                    result.append({
                        'text': word,
                        'dir': 0 if direction == 'horizontal' else 1,
                        'pos': pos
                    })
        return result