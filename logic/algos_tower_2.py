from typing import List, Tuple, Dict
import numpy as np
from utils import logger
from sender.send_on_front import send_data
# Отключаем сокращение вывода массива
np.set_printoptions(threshold=np.inf)

class WordList:
    def __init__(self, words: List[str], map_size: List[int], used_words = Dict):
        """
        Инициализация класса.
        :param words: Список слов.
        :param map_size: Размеры трехмерного массива (x, y, z).
        """
        self.words = words
        self.map_size = map_size
        self.array_3d = np.full((map_size[0], map_size[1], map_size[2]), "", dtype=object)
        self.words_dict = self.list_to_dict()
        self.used_indexies = []
        self.used_words = used_words
        self.placed_words = []

    def list_to_dict(self):
        """
        Преобразует список слов в словарь, где ключи - индекс, а значения - слова.
        :return: Словарь, где ключи - слова, а значения - индексы.
        """
        word_dict = {index: word for index, word in enumerate(self.words)}
        return word_dict

    def longest_word(self):
        """
        Находит самое длинное слово в списке.
        :return: Самое длинное слово.
        """
        if not self.words:
            print("Список слов пуст.")
            return None

        # Фильтруем слова, исключая те, которые уже использованы
        available_words = [word_id for word_id in self.words_dict if word_id not in self.used_words]
        available_words_dict = {word_id: self.words_dict[word_id] for word_id in available_words}
        print("Check avaliable words:", available_words)
        print("words_dict:", self.words_dict)
        print("used_words:", self.used_words)


        if not available_words:
            print("Нет доступных слов для использования.")
            return None
        longest_word_index, longest_word = max(available_words_dict.items(), key=lambda item: len(item[1]))
        return longest_word_index, longest_word

    def place_word(self, word_id: int, start_pos: Tuple[int, int, int], direction: int):
        """
        Размещает слово в трехмерном массиве.
        :param word_id: ID слова (индекс в списке слов).
        :param start_pos: Начальная позиция (x, y, z).
        :param direction: Направление размещения (1, 2 или 3).
        """
        # Проверяем, существует ли слово с таким ID
        if word_id < 0 or word_id >= len(self.words):
            print(f"Ошибка: Слово с ID {word_id} не найдено.")
            return

        word = self.words[word_id]
        x, y, z = start_pos

        # Определяем вектор направления
        if direction == 1:
            dir_vector = [0, 0, -1]  # Вдоль оси Z назад
        elif direction == 2:
            dir_vector = [1, 0, 0]  # Вдоль оси X вперед
        elif direction == 3:
            dir_vector = [0, 1, 0]  # Вдоль оси Y вперед
        else:
            print("Ошибка: Неверное направление.")
            return

        # Проверяем, помещается ли слово в массив
        for i, char in enumerate(word):
            new_x = x + i * dir_vector[0]
            new_y = y + i * dir_vector[1]
            new_z = z + i * dir_vector[2]

            if not (0 <= new_x < self.map_size[0] and 0 <= new_y < self.map_size[1] and 0 <= new_z < self.map_size[2]):
                print(f"Ошибка: Слово '{word}' не помещается в массив с начальной позицией {start_pos}.")
                return

            if self.array_3d[new_x, new_y, new_z] != "":
                if self.array_3d[new_x, new_y, new_z] != char:
                    print(f"Ошибка: Ячейка ({new_x}, {new_y}, {new_z}) уже занята.")
                    return

        # Размещаем слово в массиве
        for i, char in enumerate(word):
            new_x = x + i * dir_vector[0]
            new_y = y + i * dir_vector[1]
            new_z = z + i * dir_vector[2]
            self.array_3d[new_x, new_y, new_z] = char

        # Сохраняем информацию о размещенном слове
        self.placed_words.append({
            # "text": word,
            "dir": direction,
            "id": word_id,
            "pos": list(start_pos),
        })
        print(f"Слово '{word}' успешно размещено с ID {word_id}.")

    def fit_the_word(self, word, word_id, start_pos, direction):
        fits = True
        for i in range(len(word)):
            new_x = start_pos[0] + i
            new_y = start_pos[1]
            new_z = start_pos[2]

            if not (0 <= new_x < self.map_size[0] and 0 <= new_y < self.map_size[1] and 0 <= new_z < self.map_size[2]):
                fits = False
                break

            if self.array_3d[new_x, new_y, new_z] != "":
                fits = False
                break
        # Если слово помещается, размещаем его
        if fits:
            self.place_word(word_id, start_pos, direction)
            self.used_words[word_id] = word
            return True
        else:
            # print(f"Ошибка: Не удалось разместить слово '{word}' по оси Z.")
            return False

    def place_longest_word(self):
        """
        Размещает самое длинное слово в массиве по оси X,
        начиная с координаты (0, 0, 0).
        """
        # Находим самое длинное слово
        longest_word_id, longest_word = self.longest_word()
        if longest_word is None:
            print("Ошибка: Список слов пуст.")
            return

        word = longest_word
        word_id = longest_word_id

        # Выбираем направление [1, 0, 0] (вдоль оси X вперед)
        direction = 2

        # Начальная позиция (0, 0, 0)
        start_pos = (0, 0, 0)

        # Проверяем, помещается ли слово в массив
        self.fit_the_word(word, word_id, start_pos, direction)

    def place_vertical_word(self, z_level: int = 0):
        """
        Размещает слово по оси Z
        """
        # Находим слово, которое оканчивается на одну из букв горизонатльного
        coords_and_letters = self.get_letters_on_floor(z_level=z_level)
        coords = coords_and_letters['coords']
        letters = coords_and_letters['letters']

        fit_word = self.find_word_ending_with(letters)

        if fit_word == []:
            print("Ошибка: Нет таких слов")
            return
        for words in fit_word:
            word = words[0]
            word_id = words[1]
            letter = words[2]

            # Устанавливаем направление вдоль оси z вниз
            direction = 1

            # Получаем x,y по букве пересечения, по оси z размещаем по длине слова
            letter_coords = self.letter_coords(coords, letters, letter)

            if len(letter_coords) == 0:
                print("No matching letter found.")
                continue

            print("letter_coords: ", letter_coords)
            z = len(word)
            start_pos = (letter_coords[0],letter_coords[1],z-1)
            print(word, word_id, start_pos)

            # Дублированный код на размещение
            # Проверяем, помещается ли слово в массив
            success = self.fit_the_word(word, word_id, start_pos, direction)
            if success:
                return
            else:
                continue

    def place_words_by_z(self, z_letter_map: dict):
        """
        Размещает слова на уровнях z, используя заданные координаты букв.
        Слово должно пересекать все указанные буквы в соответствующих координатах.
        :param z_letter_map: Словарь, где ключи - уровни z, а значения - списки словарей с координатами букв.
        """
        # Проверка корректности параметра z_letter_map
        if not z_letter_map:
            print("Ошибка: Словарь z_letter_map пуст.")
            return

        # Перебор уровней z
        for z, letters_info in sorted(z_letter_map.items(), reverse=True):
            # print(f"Попытка размещения слова на уровне z = {z} с буквами: {letters_info}")

            # Создаем словарь для быстрого доступа к координатам букв
            letter_coords = {}
            for letter_info in letters_info:
                for letter, coords in letter_info.items():
                    letter_coords[letter] = coords

            # Перебор всех слов
            for word_id, word in enumerate(self.words):
                # Проверяем, размещено ли слово уже
                if any(placed_word["id"] == word_id for placed_word in self.placed_words):
                    continue

                # Проверяем, содержит ли слово все необходимые буквы
                if all(letter in word for letter in letter_coords.keys()):
                    # Пытаемся разместить слово на уровне z с перебором по x
                    for x_offset in range(self.map_size[0]):  # Перебираем возможные значения x
                        start_pos = (x_offset, 0, z)  # Начальная позиция (x_offset, y=0, z)
                        direction = 2  # Вдоль оси X вперед

                        # Проверяем, помещается ли слово в массив
                        fits = True
                        for i, char in enumerate(word):
                            new_x = x_offset + i
                            new_y = 0
                            new_z = z

                            # Проверяем границы массива
                            if not (0 <= new_x < self.map_size[0] and 0 <= new_y < self.map_size[1] and 0 <= new_z <
                                    self.map_size[2]):
                                fits = False
                                break

                            # Проверяем пересечение букв
                            if char in letter_coords:
                                expected_coords = letter_coords[char]
                                if [new_x, new_y, new_z] != expected_coords:
                                    fits = False
                                    break

                            # Проверяем, занята ли ячейка
                            if self.array_3d[new_x, new_y, new_z] != "" and self.array_3d[new_x, new_y, new_z] != char:
                                fits = False
                                break

                        # Если слово помещается, размещаем его
                        if fits:
                            self.place_word(word_id, start_pos, direction)
                            print(f"Слово '{word}' успешно размещено на уровне z = {z} с x_offset = {x_offset}.")
                            self.used_words[word_id] = word
                            return  # Прекращаем попытки размещения для этого уровня

                    else:
                        # Если слово не удалось разместить ни с одним x_offset
                        continue
            else:
                continue
                # print(f"Не удалось разместить слово на уровне z = {z}.")

    def display_plane_xy(self, z_index: int):
        """
        Выводит плоскость XY для фиксированной координаты Z.
        :param z_index: Индекс по оси Z.
        """
        try:
            plane = self.array_3d[:, :, z_index]
            print(f"Плоскость XY для Z = {z_index}:")
            for row in plane:
                print(" ".join(cell if cell != "" else "_" for cell in row))
        except IndexError:
            print(f"Ошибка: Координата Z = {z_index} выходит за границы массива.")

    def display_plane_yz(self, x_index: int):
        """
        Выводит плоскость YZ для фиксированной координаты X.
        :param x_index: Индекс по оси X.
        """
        try:
            plane = self.array_3d[x_index, :, :]
            print(f"Плоскость YZ для X = {x_index}:")
            for row in plane:
                print(" ".join(cell if cell != "" else "_" for cell in row))
        except IndexError:
            print(f"Ошибка: Координата X = {x_index} выходит за границы массива.")

    def get_letters_on_floor(self, z_level: int):
        """
        Возвращает буквы, размещенные по оси X на заданном уровне Z,
        в плоскости Y = 0.
        :param z_level: Уровень (этаж) по оси Z.
        :return: Словарь с двумя списками:
                 - "cubes": список координат [(x, y, z), ...],
                 - "text": список букв [letter, ...].
        """
        # Проверка корректности параметра z_level
        if z_level < 0 or z_level >= self.map_size[2]:
            print(f"Ошибка: Значение Z = {z_level} выходит за границы массива.")
            return {"coords": [], "letters": []}

        cubes = []
        text = []

        # Собираем буквы по оси X (для фиксированного y = 0)
        for x in range(self.map_size[0]):
            cell_value = self.array_3d[x, 0, z_level]
            if cell_value != "":
                cubes.append([x, 0, z_level])
                text.append(cell_value)

        # if not cubes:
        #     print(f"На уровне Z = {z_level} нет размещенных букв.")
        # else:
        #     print(f"Буквы на уровне Z = {z_level} в плоскости Y = 0:")
        #     for i in range(len(cubes)):
        #         print(f"Координаты: ({cubes[i][0]}, {cubes[i][1]}, {cubes[i][2]}), Буква: {text[i]}")
        return {"coords": cubes, "letters": text}

    def find_word_ending_with(self, letters: list):
        """
        Находит слово, которое заканчивается на одну из букв из заданного списка.
        :param letters: Список букв для проверки.
        :return: Найденное слово и его ID, если оно существует.
        """

        word_list = []
        # Проверка корректности параметра letters
        if not letters:
            print("Ошибка: Список букв пуст.")
            return None

        # Перебираем все слова в списке
        for word_id, word in enumerate(self.words):
            if word[-1] in letters:
                # print(f"Найдено слово '{word}' с ID {word_id}, оканчивающееся на букву '{word[-1]}'.")
                # Проверяем, не находится ли слово уже в placed_words
                if any(placed_word["id"] == word_id for placed_word in self.placed_words):
                    # print(f"Слово '{word}' с ID {word_id} уже размещено.")
                    continue
                word_list.append([word, word_id, word[-1]])
        print("Ошибка: Не найдено слов, оканчивающихся на буквы из списка.")
        return word_list

    def letter_coords(self, coords: list, letters: list, target_letter: str) -> list:
        """
        Возвращает координаты для заданной буквы.
        :param coords: Список координат [(x, y, z), ...].
        :param letters: Список букв [letter, ...].
        :param target_letter: Буква, для которой нужно найти координаты.
        :return: Список координат, соответствующих заданной букве.
        """
        if not coords or not letters:
            print("Ошибка: Массивы координат или букв пусты.")
            return []

        if len(coords) != len(letters):
            print("Ошибка: Длины массивов координат и букв не совпадают.")
            return []

            # Находим все координаты, соответствующие заданной букве
        matching_coords = [coord for coord, letter in zip(coords, letters) if letter == target_letter]

        if not matching_coords:
            print(f"Буква '{target_letter}' не найдена.")
            return []
        return matching_coords[0]

    def find_z_with_two_letters(self):
        """
        Находит значения z, на которых находятся ровно две буквы.
        :return: Список значений z, где находятся ровно две буквы.
        """
        letters_by_z = {}

        # Итерация по всему трехмерному массиву
        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                for z in range(self.map_size[2]):
                    # Получаем значение ячейки
                    letter = self.array_3d[x, y, z]
                    if letter != "":
                        # Добавляем букву в соответствующий список для z
                        if z not in letters_by_z:
                            letters_by_z[z] = []
                        letters_by_z[z].append({letter:[x,y,z]})

        # Фильтруем уровни z с ровно двумя буквами
        z_with_two_letters = [z for z, letters in letters_by_z.items() if len(letters) == 2]

        # Создаем отфильтрованный словарь
        filtered_letters_by_z = {z: letters_by_z[z] for z in z_with_two_letters}

        # # Выводим результат
        # if filtered_letters_by_z:
        #     # print("Отфильтрованный словарь letters_by_z:")
        #     for z, letters in filtered_letters_by_z.items():
        #         print(f"z = {z}: {letters}")
        # else:
        #     print("Нет уровней z с ровно двумя буквами.")
        return filtered_letters_by_z


    def get_letter_coordinates(self) -> Dict[str, list]:
        """
        Возвращает координаты каждой буквы и соответствующие символы.
        :return: Словарь с ключами "cubes" и "text".
        """
        cubes = []
        text = []

        # Проходим по всему трехмерному массиву
        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                for z in range(self.map_size[2]):
                    if self.array_3d[x, y, z] != "":
                        cubes.append([x, z, y])
                        text.append(self.array_3d[x, y, z])

        return {"cubes": cubes, "text": text}

    def winner_pipeline(self):
        print("Used_words at start: ", self.used_words)
        # Обнуляем placed_words
        self.placed_words = []
        self.place_longest_word()
        self.place_vertical_word(z_level=0)
        self.place_vertical_word(z_level=0)
        filtered_letters = self.find_z_with_two_letters()
        self.place_words_by_z(filtered_letters)

        result = self.get_letter_coordinates()
        print("Used_words: ", self.used_words)
        return result, self.used_words


# Пример использования
if __name__ == "__main__":
    words = ['сиамец', 'старшекурсница', 'пробковение', 'фитопалеонтолог', 'отлучение', 'маневр', 'капелирование', 'автострада', 'недосягаемость', 'бриг', 'опадение', 'омачивание', 'конверсия', 'отлетание', 'асбестит', 'корсажница', 'пульверизатор', 'химиотерапия', 'законоучитель', 'самец', 'режиссура', 'соединение', 'крачка', 'бортпроводница', 'непогрешимость', 'линкос', 'скрытничество', 'отбельщица', 'будёновка', 'оподление', 'совмещение', 'катамаран', 'вопросник', 'нардек', 'безденежье', 'электрокарщик', 'лжеприсяга', 'строитель', 'необыкновенность', 'скаутизм', 'мобилизованность', 'недостоверность', 'судейство', 'гидромеханика', 'чепан', 'нагребальщица', 'ариетта', 'область', 'документальность', 'присущность', 'учетверение', 'сгущаемость', 'стильщица', 'лишение', 'филипповщина', 'спичрайтер', 'подверженность', 'силур', 'выщупывание', 'подставление', 'аннексия', 'переполненность', 'толщина', 'пластеин', 'прищипывание', 'мужененавистница', 'надклейка', 'палингенезис', 'идиоматизм', 'перевозка', 'индексация', 'содействие', 'угасание', 'чановщик', 'фиглярничание', 'кинограмма', 'бурлачество', 'губитель', 'арка', 'дезинформация', 'чужое', 'багет', 'полива', 'антабус', 'неощутительность', 'пунктир', 'протерозой', 'самозакаливание', 'редова', 'германистка', 'бахча', 'чуня', 'парабеллум', 'электроника', 'мужание', 'наливка', 'полнокровность', 'мирок', 'педократ', 'обезображение', 'камнерез', 'авиатор', 'выспренность', 'кровоподтёк', 'сосед', 'закупщик', 'архиепископ', 'недруг', 'непрозрачность', 'приговаривание', 'декабрь', 'хитпарад', 'пропитывание', 'камнеломка', 'свитка', 'метательница', 'поминание', 'антабус', 'вызубрина', 'рамочник', 'домкрат', 'преследуемый', 'птичница', 'конспектирование', 'шут', 'предпринимательница', 'наливание', 'палаш', 'картография', 'дезактивация', 'слезница', 'цедра', 'эскорт', 'безнадёжность', 'старьёвщик', 'обновляемость', 'щупальце', 'членение', 'морозность', 'газоанализатор', 'выкристаллизовывание', 'термодинамика', 'сурдина', 'несгораемость', 'митрофан', 'сурепа', 'молочность', 'югослав', 'строгание', 'канталупа', 'сталкивание', 'мускатник', 'трепальщик', 'злоба', 'онанистка', 'разливщик', 'кружевница', 'пересвистывание', 'неосновательность', 'безлесье', 'сюрреализм', 'автоматчик', 'буер', 'банкрот', 'водоприёмник', 'рулёна', 'адинамия', 'гистология', 'дифферент', 'газолин', 'крепостник', 'паша', 'анальгин', 'пахота', 'магнитола', 'траволечение', 'ангстрем', 'распев', 'вытискивание', 'яйцо', 'изыскание', 'житель', 'пасение', 'рицинус', 'разводчица', 'сепаратист', 'верстак', 'взыскивание', 'оперативность', 'грузинка', 'ценитель', 'четвертование', 'модистка', 'обязательность', 'обстоятельность', 'ваттметр', 'мюзикл', 'дружность', 'тамарикс', 'континентальность', 'скусывание', 'номинант', 'сахароза', 'динамизм', 'излучатель', 'трок', 'головизна', 'катет', 'мамлюк', 'расценщик', 'пестролистность', 'прошивание', 'конъюнктивит', 'агрегатирование', 'проповедник', 'сурьма', 'королевич', 'тугун', 'сучение', 'левкас', 'перловник', 'слезливость', 'майор', 'странствование', 'радиоактивность', 'подгорье', 'мазание', 'пересказ', 'восьмилетие', 'порочность', 'перелавливание', 'водоотлив', 'кистовяз', 'вывевание', 'актуальность', 'арфист', 'гром', 'забутка', 'командующий', 'тутовод', 'замедленность', 'кантователь', 'энергосистема', 'судоверфь', 'смалец', 'задор', 'вирусолог', 'единовременность', 'метафоричность', 'облачение', 'мездра', 'колоб', 'взволакивание', 'антрекот', 'выклинивание', 'непринятие', 'оцинковывание', 'половица', 'мантилья', 'инцидент', 'фита', 'нападающий', 'полиспаст', 'рейтинг', 'юрисдикция', 'проковывание', 'неоднородность', 'адресант', 'просвирняк', 'вино', 'самоуспокоение', 'лемма', 'донжуанство', 'миропорядок', 'пропивание', 'варение', 'похитительница', 'кремль', 'однородность', 'кордебалет', 'паяльщица', 'психопатизм', 'лгунья', 'плен', 'переорганизация', 'оглушительность', 'легкомыслие', 'прометий', 'состязание', 'бесклассовость', 'флебит', 'припухание', 'нецелесообразность', 'зарядчик', 'требовательность', 'обвальщик', 'жупа', 'шулерство', 'медальщик', 'безнравственность', 'вымогание', 'вагонетчица', 'анекдот', 'холст', 'ивишень', 'синигрин', 'набойщица', 'автомобилизация', 'мавританка', 'доппелькюммель', 'опрессовка', 'останец', 'ординар', 'моцион', 'скоротечность', 'духоборец', 'биогеоценология', 'чрезвычайность', 'клиницист', 'сменяемость', 'подстораживание', 'демократизация', 'микрофон', 'тензиметр', 'картезианка', 'клепало', 'дачница', 'покладистость', 'волеизъявление', 'подсортировывание', 'кобальт', 'вышкомонтажник', 'электролампа', 'полновластие', 'мажара', 'минералог', 'футеровщик', 'взаимообмен', 'вклинение', 'птичница', 'поражение', 'безучастие', 'антисанитария', 'пятиборец', 'прорезинивание', 'комдив', 'расстановщица', 'бренность', 'эпилепсия', 'документ', 'психогенезис', 'нейтрализация', 'прикол', 'выпрягание', 'шерстевед', 'скарпель', 'закрепление', 'конголезка', 'немыслимость', 'комераж', 'обработка', 'благодеяние', 'мостовик', 'девятиклассник', 'никелировщик', 'самум', 'семянка', 'арболит', 'никелировщица', 'фонотека', 'разброд', 'пифос', 'спичечница', 'неодинаковость', 'инвалюта', 'космонавтика', 'футерование', 'подставление', 'правилка', 'новость', 'перебарывание', 'подкосина', 'невропатология', 'кордебалет', 'наутофон', 'домохозяйство', 'конопляник', 'воздухозаборник', 'кацавейка', 'ворошение', 'кляп', 'негоциант', 'остеомиелит', 'подзол', 'приспешник', 'эллин', 'пласт', 'рольня', 'парангон', 'мандрил', 'промокание', 'пакетбот', 'выгонка', 'пахименингит', 'обвяливание', 'уанстеп', 'сгибатель', 'непостоянство', 'перемерка', 'обшлифовывание', 'гуща', 'помол', 'мочалка', 'нутрец', 'полуявь', 'раб', 'прокрашивание', 'браслет', 'продолжительность', 'германизм', 'изверг', 'недожог', 'погонщик', 'солончак', 'промискуитет', 'вундеркинд', 'перепел', 'плавун', 'запухание', 'перегнивание', 'соляризация', 'датчанин', 'глотание', 'патетичность', 'альпинарий', 'сяжок', 'самопрялка', 'ктитор', 'отроек', 'слезание', 'предпочтительность', 'гонобобель', 'клеевар', 'хакас', 'упревание', 'возвещение', 'сандружинница', 'рахатлукум', 'рихтовщица', 'принос', 'живодёр', 'рясофорство', 'обкидывание', 'нагромождённость', 'ботулизм', 'люкарна', 'перо', 'интерпретация', 'увлечённость', 'багряница', 'пенис', 'рапира', 'корнеплод', 'тын', 'загримировывание', 'вычаливание', 'классовость', 'мамлюк', 'семиклассник', 'откомандирование', 'неодарвинизм', 'отвалообразователь', 'неотразимость', 'просёлок', 'бальнеология', 'фотоматериал', 'трог', 'оболонь', 'товарность', 'пузыристость', 'сохранение', 'латекс', 'трансформистка', 'подползание', 'структуралист', 'подсвекольник', 'буран', 'святочник', 'никчемность', 'брабансон', 'убойность', 'фильтровщица', 'суперинтендент', 'еда', 'льносырьё', 'обривание', 'втрамбование', 'лепщик', 'лепщица', 'привинчивание', 'разборщица', 'бутирометр', 'проходчица', 'мишура', 'пневмония', 'селенит', 'панегирик', 'запирательство', 'кишмиш', 'коррупция', 'обмерок', 'троллейкара', 'перегласовка', 'подпирание', 'обтёска', 'глазурование', 'наделённость', 'оаз', 'остроугольник', 'клёв', 'аспект', 'помощница', 'приглушение', 'малоснежье', 'ампутация', 'волнограф', 'зыбление', 'палёное', 'барельеф', 'перекрещенка', 'размагниченность', 'низание', 'мотоциклистка', 'задаток', 'киловаттчас', 'технократ', 'лазер', 'провождение', 'смута', 'пролонгация', 'задурманивание', 'переадресовка', 'извержение', 'стильщица', 'хлебородие', 'форсунщик', 'геомагнетизм', 'контрреволюционность', 'опробковение', 'хлопотание', 'сук', 'телефонистка', 'обогащаемость', 'шлюзовик', 'уличение', 'присыпка', 'стабилизация', 'изолировщик', 'кастрюля', 'погребец', 'оленевод', 'вырастание', 'младенец', 'правовед', 'цитрус', 'ватерлиния', 'затаптывание', 'нерпуха', 'подпёк', 'шутовство', 'платность', 'окрашивание', 'лиственница', 'эрудиция', 'бетатерапия', 'марихуана', 'садоводство', 'обоз', 'темнолицая', 'апокалипсис', 'выкармливание', 'бубон', 'нововведение', 'шахиня', 'перекрутка', 'размягчение', 'основоположение', 'импотенция', 'глинтвейн', 'сыроварение', 'перекрикивание', 'паразитология', 'тривиальность', 'обет', 'акарицид', 'гепард', 'ландштурмист', 'нюхание', 'непотопляемость', 'соланин', 'хиазма', 'крен', 'вошедшая', 'укрывание', 'заявка', 'доверие', 'обсмаливание', 'расхищение', 'новобранец', 'перемеривание', 'праздничность', 'котлета', 'мушкетёр', 'путание', 'светостойкость', 'увинчивание', 'обвислость', 'матрёшка', 'обсыхание', 'сладкозвучность', 'часовня', 'пюпитр', 'новостройка', 'бирманец', 'бесхитростность', 'брошюрование', 'пантомима', 'псовка', 'брусчатка', 'дидактика', 'синовия', 'личность', 'шишак', 'подбойщик', 'героиня', 'низина', 'соответствие', 'канцелярия', 'автоматика', 'конькобежец', 'созвездие', 'пилорама', 'палеоантропология', 'закальщик', 'несварение', 'подзыв', 'торгпред', 'обливальщик', 'смолосемянник', 'предзнаменование', 'полицай', 'истощённость', 'бездеятельность', 'фигурист', 'рафинировка', 'приплющение', 'тесина', 'предрасположенность', 'противостояние', 'бомба', 'заселённость', 'тигрица', 'брюква', 'менеджмент', 'исчисление', 'зелень', 'нахал', 'дворец', 'приселение', 'отсечка', 'плясунья', 'неизъяснимое', 'название', 'ненаучность', 'русофобия', 'деформирование', 'хиромант', 'изгнание', 'горбоносость', 'орфоэпия', 'каин', 'кувыркание', 'муравьед', 'конференцзал', 'квазар', 'национальность', 'смотрительница', 'комбинаторика', 'пилигримка', 'вердикт', 'дадан', 'мягчение', 'картинг', 'коралёк', 'успокоение', 'первое', 'циклевание', 'льносолома', 'канализация', 'надоедание', 'закоулок', 'злодейство', 'туер', 'бессубъектность', 'промеривание', 'сколиоз', 'конфуцианец', 'батон', 'отбояривание', 'конкурент', 'индус', 'секвенция', 'истребительница', 'грибоводство', 'отповедь', 'тенётчик', 'шушера', 'отводка', 'институтка', 'неустанность', 'отливщик', 'застава', 'пырник', 'снайпинг', 'необъятное', 'свирельщица', 'истолкователь', 'тлен', 'прус', 'плечо', 'крап', 'новозеландка', 'презервация', 'озлобление', 'капитал', 'желанная', 'негостеприимность', 'модернистка', 'перерисовка', 'глухарятина', 'парафинирование', 'полукафтанье', 'торевтика', 'гимнастёрка', 'приехавший', 'оппозиционер', 'парашютист', 'практик', 'примирённость', 'радиолюбительство', 'прежнее', 'меланхолик', 'подрезка', 'няня', 'лицеистка', 'фальц', 'пресноводность', 'просьба', 'эклектика', 'припасовка', 'правофланговая', 'падчерица', 'муза', 'завораживание', 'котомка', 'сквозняк', 'замешательство', 'расхолаживание', 'арлекинада', 'пиала', 'недоделанность', 'вышивальщик', 'заступление', 'правосознание', 'монолит', 'койка', 'бедуинка', 'непритязательность', 'саркоплазма', 'перевязочная', 'полусвет', 'неотделимость', 'пуловер', 'диск', 'мумиё', 'бундестаг', 'увод', 'фальцовщица', 'янус', 'окропление', 'оторочка', 'многодомность', 'санврач', 'осведомлённость', 'регистратор', 'тяжелораненый', 'семибоярщина', 'подбор', 'декомпенсация', 'дождевание', 'простодушность', 'митрофанушка', 'абсцисса', 'блюминг', 'окучник', 'патагонец', 'колпик', 'токката', 'обструкционистка', 'оттепель', 'обновленец', 'нависание', 'вскрыша', 'подыскивание', 'облысение', 'перепиливание', 'растачивание', 'купальщица', 'плесень', 'цемянка', 'водостойкость', 'допаривание', 'асессор', 'книжка', 'гелиогравюра', 'капсюль', 'сваебой', 'непроницаемость', 'досылание', 'фазан', 'указ', 'боцман', 'аркада', 'орнитоптер', 'скапливание', 'придерживание', 'голубизна', 'жизнерадостность', 'жестер', 'маловер', 'скандировка', 'зурнист', 'телефотография', 'звуковоспроизведение', 'пробоина', 'калёвочник', 'инфекция', 'надбровье', 'двигатель', 'тасовка', 'целостность', 'сзывание', 'хвост', 'ризалит', 'котлета', 'жизнестроительство', 'прополис', 'кессонщик', 'скупость', 'верстак', 'аттик', 'публика', 'универсализация', 'закрутка', 'лесоукладчик', 'лучистость', 'меблировка', 'серистость', 'русизм', 'мукомолье', 'тархан', 'малярия', 'приспособляемость', 'остепенение', 'галоша', 'бемоль', 'грабен', 'гардероб', 'клоктун', 'проушина', 'автофургон', 'дезавуирование', 'опорос', 'обжим', 'фритюр', 'обстрагивание', 'моджахед', 'тщетность', 'супинация', 'выпивание', 'майорство', 'бригада', 'слаломистка', 'орнитофауна', 'паникадило', 'дек', 'транссексуал', 'единение', 'выгнивание', 'кайла', 'подсолка', 'мезоглея', 'расстраивание', 'глодание', 'пришаркивание', 'нерешительность', 'компактность', 'перестилка', 'пряжа', 'суетливость', 'дож', 'благоприятствование', 'перепрядение', 'загрязнённость', 'трепел', 'подклинка', 'продолжательница', 'электролиз', 'оспина', 'новокрещённая', 'парильщица', 'аноним', 'намораживание', 'расхищение', 'мистер', 'бенгалка', 'индуистка', 'отговаривание', 'укоризна', 'китаист', 'ягельник', 'поддалбливание', 'маловыгодность', 'юродство', 'отстригание', 'многоплодие', 'миндаль', 'идолопоклонница', 'шифровальщик', 'экспатриантка', 'сенсуальность', 'взаимообмен', 'заливка', 'заливание', 'сонет', 'целестин', 'разметчик', 'перуанка', 'шестиборец', 'непрофессиональность', 'необъяснимость', 'взяткополучатель', 'своеземец', 'центровщица', 'фанариот', 'молибден', 'жаркое', 'любование', 'аджарка', 'перегной', 'академичность', 'грызло', 'избитость', 'перемалывание', 'водосток', 'эклектичность', 'узорчатость', 'выколотка', 'филогенез', 'пустоколосица', 'лацкан', 'текст', 'пропагандирование', 'мелодрама', 'керамзитобетон', 'шейх', 'прорезывание', 'ятовь', 'прощание', 'анализатор', 'порнограф', 'оттягивание', 'зычность', 'труборез', 'рысак', 'надписывание', 'перевьючивание', 'повольник', 'румб', 'скоропашка', 'авиатранспорт', 'цензурность', 'переосмысление', 'зигзаг', 'хлопуша', 'шпиговка', 'сексолог', 'компрометирование', 'расценка', 'берсальер']

    map_size = [30, 30, 100]

    # Инициализация класса
    word_list = WordList(words, map_size)

    # Размещаем слово самое длинное слово в основание
    result = word_list.winner_pipeline()
    print(result)
    send_data(result)
    # Находим слова, которые могут пересечь эти буквы

    # Необходимо, чтобы расстояние между горзонтальными словами было более еденицы
    # Проверим по списку размещенных слов с одинаковым dir

