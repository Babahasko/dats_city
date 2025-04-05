import random
from collections import defaultdict
import itertools
import math


class OptimizedTowerBuilder:
    def __init__(self, word_list):
        self.words = [(i, word.lower()) for i, word in enumerate(word_list) if len(word) >= 2]
        self.used_indices = set()
        self.tower = defaultdict(lambda: {
            'x_words': [],
            'y_words': [],
            'vertical_words': []
        })
        self.occupied_positions = set()
        self.buffer_zones = set()

        # Коэффициенты для этажей
        self.floor_coefficients = {
            0: 1.0,
            -1: 1.5,
            -2: 2.0,
            -3: 2.5,
            -4: 3.0
        }

    def calculate_score(self):
        total_score = 0
        for z_level, floor in self.tower.items():
            # Баллы за буквы
            letter_score = sum(len(word['word']) for word in
                               floor['x_words'] + floor['y_words'] + floor['vertical_words'])

            # Коэффициент пропорции
            width, depth = self.calculate_floor_dimensions(floor)
            proportion = min(width, depth) / max(width, depth) if max(width, depth) > 0 else 0

            # Коэффициент плотности
            x_count = len(floor['x_words'])
            y_count = len(floor['y_words'])
            density = 1 + (x_count + y_count) / 4

            # Итоговый балл для этажа
            floor_coeff = self.floor_coefficients.get(z_level, 1.0)
            floor_score = letter_score * proportion * density * floor_coeff
            total_score += floor_score

        return total_score

    def calculate_floor_dimensions(self, floor):
        all_positions = set()
        for word in floor['x_words'] + floor['y_words'] + floor['vertical_words']:
            all_positions.update(word['positions'])

        if not all_positions:
            return 0, 0

        xs = {pos[0] for pos in all_positions}
        ys = {pos[1] for pos in all_positions}

        width = max(xs) - min(xs) + 1 if xs else 0
        depth = max(ys) - min(ys) + 1 if ys else 0

        return width, depth

    def get_word_positions(self, word, start_pos, direction):
        positions = set()
        buffer = set()
        x, y, z = start_pos

        for i in range(len(word)):
            if direction == 'x':
                pos = (x + i, y, z)
            elif direction == 'y':
                pos = (x, y + i, z)
            elif direction == 'z':
                pos = (x, y, z - i)  # Вертикально вниз

            positions.add(pos)

            for dx, dy, dz in itertools.product([-1, 0, 1], repeat=3):
                if dx == dy == dz == 0:
                    continue
                buf_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                if buf_pos not in positions:
                    buffer.add(buf_pos)

        return positions, buffer

    def can_place_word(self, positions, buffer):
        return not (positions & self.occupied_positions or buffer & self.occupied_positions)

    def find_vertical_positions(self, word):
        """Находит все возможные позиции для вертикального слова с пересечениями"""
        candidates = []

        # Ищем пересечения с горизонтальными словами
        for z_level, floor in self.tower.items():
            for h_word in floor['x_words'] + floor['y_words']:
                for h_pos in h_word['positions']:
                    # Пробуем начать вертикальное слово в этой позиции
                    start_pos = (h_pos[0], h_pos[1], h_pos[2])
                    positions, buffer = self.get_word_positions(word, start_pos, 'z')

                    if not self.can_place_word(positions, buffer):
                        continue

                    # Проверяем пересечение хотя бы с одним другим горизонтальным словом
                    other_intersection = False
                    for other_z, other_floor in self.tower.items():
                        if other_z == z_level:
                            continue
                        for other_h_word in other_floor['x_words'] + other_floor['y_words']:
                            if len(positions & other_h_word['positions']) >= 2:
                                other_intersection = True
                                break
                        if other_intersection:
                            break

                    if other_intersection or z_level == 0:
                        # Оцениваем потенциальный вклад в счет
                        self.tower[z_level]['vertical_words'].append({
                            'word': word,
                            'positions': positions,
                            'temp': True
                        })
                        score = self.calculate_score()
                        self.tower[z_level]['vertical_words'].pop()

                        candidates.append({
                            'z_level': z_level,
                            'start_pos': start_pos,
                            'positions': positions,
                            'buffer': buffer,
                            'score': score
                        })

        # Сортируем по убыванию потенциального счета
        candidates.sort(key=lambda x: -x['score'])
        return candidates

    def add_vertical_word(self, word_idx, word):
        candidates = self.find_vertical_positions(word)

        if candidates:
            best = candidates[0]
            self.tower[best['z_level']]['vertical_words'].append({
                'word': word,
                'index': word_idx,
                'position': best['start_pos'],
                'positions': best['positions']
            })
            self.used_indices.add(word_idx)
            self.occupied_positions.update(best['positions'])
            self.buffer_zones.update(best['buffer'])
            return True
        return False

    def build_tower(self):
        # Сортируем слова по убыванию длины
        sorted_words = sorted(self.words, key=lambda x: -len(x[1]))

        # 1. Строим первый этаж
        if len(sorted_words) > 0:
            idx, word = sorted_words[0]
            positions, buffer = self.get_word_positions(word, (0, 0, 0), 'x')
            if self.can_place_word(positions, buffer):
                self.tower[0]['x_words'].append({
                    'word': word,
                    'index': idx,
                    'position': (0, 0, 0),
                    'positions': positions
                })
                self.used_indices.add(idx)
                self.occupied_positions.update(positions)
                self.buffer_zones.update(buffer)

        if len(sorted_words) > 1:
            idx, word = sorted_words[1]
            positions, buffer = self.get_word_positions(word, (0, 0, 0), 'y')
            if self.can_place_word(positions, buffer):
                self.tower[0]['y_words'].append({
                    'word': word,
                    'index': idx,
                    'position': (0, 0, 0),
                    'positions': positions
                })
                self.used_indices.add(idx)
                self.occupied_positions.update(positions)
                self.buffer_zones.update(buffer)

        # 2. Добавляем вертикальные слова
        for idx, word in sorted_words[2:6]:  # Пробуем добавить 4 вертикальных слова
            self.add_vertical_word(idx, word)

        # 3. Строим дополнительные этажи
        for z_level in [-1, -2, -3, -4]:
            for idx, word in sorted_words:
                if idx in self.used_indices:
                    continue

                # Пробуем добавить горизонтальное слово
                for direction in ['x', 'y']:
                    positions, buffer = self.get_word_positions(word, (0, 0, z_level), direction)
                    if self.can_place_word(positions, buffer):
                        self.tower[z_level][f'{direction}_words'].append({
                            'word': word,
                            'index': idx,
                            'position': (0, 0, z_level),
                            'positions': positions
                        })
                        self.used_indices.add(idx)
                        self.occupied_positions.update(positions)
                        self.buffer_zones.update(buffer)
                        break

        return self.tower, self.calculate_score()

    def print_tower(self):
        print("=== ОПТИМИЗИРОВАННАЯ БАШНЯ ===")
        print(f"Общий счет: {self.calculate_score():.2f}")

        for z_level in sorted(self.tower.keys()):
            floor = self.tower[z_level]
            width, depth = self.calculate_floor_dimensions(floor)
            print(f"\nЭтаж {z_level} (ширина: {width}, глубина: {depth}):")

            print("  Слова по X:")
            for word in floor['x_words']:
                print(f"    [{word['index']}] {word['word']} at {word['position']}")

            print("  Слова по Y:")
            for word in floor['y_words']:
                print(f"    [{word['index']}] {word['word']} at {word['position']}")

            print("  Вертикальные слова:")
            for word in floor['vertical_words']:
                print(f"    [{word['index']}] {word['word']} at {word['position']}")



def get_random_word(length: int):
    import string
    import random
    result = ""
    for _ in range(length):
        result += random.choice(string.ascii_lowercase)
    return result
# Пример использования
if __name__ == "__main__":
    words = [
        # "программирование", "алгоритм", "компилятор", "функция",
        # "переменная", "объект", "класс", "наследование",
        # "инкапсуляция", "полиморфизм", "интерфейс", "абстракция"
    ]
    for word in range(1000):
        words.append(get_random_word(random.randint(5, 9)))
    words = [
        "программирование", "алгоритм", "компилятор", "функция",
        "переменная", "объект", "класс", "наследование",
        "инкапсуляция", "полиморфизм", "интерфейс", "абстракция"
    ]
    builder = OptimizedTowerBuilder(words)
    tower, score = builder.build_tower()
    builder.print_tower()
