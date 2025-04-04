import random
from typing import Tuple
from logic.logic import TowerBuilder
from utils.custom import convert_matrix_to_test_data
from sender.send_on_front import send_data

def get_random_word(length: int):
    import string
    import random
    result = ""
    for _ in range(length):
        result += random.choice(string.ascii_lowercase[:6])
    return result

# words = ["foundation", "support", "column", "beam", "floor",
#      "wall", "ceiling", "structure", "building", "tower"]
words = []
for _ in range(1000):
    words.append(get_random_word(random.randint(6,8)))


builder = TowerBuilder(words)
tower = builder.build_tower()
# builder.shuffle_words()
tower = builder.continue_build()

print(f"Башня построена! Счет: {tower['score']}")
print(f"Высота: {tower['height']} этажей")
print(f"Габариты: {tower['bounding_box']}")
print("\nСтруктура башни:")
for i, word in enumerate(tower['words']):
    direction = "X" if word['dir'] == [1, 0, 0] else "Y" if word['dir'] == [0, 1, 0] else "Z"
    print(f"{i+1}. {word['text']} ({direction}) at {word['pos']}")
print(builder.build_requests)

matrix = builder.construct_matrix_2()
send_data(matrix)


