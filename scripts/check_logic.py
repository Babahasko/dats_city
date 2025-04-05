from logic.logic_opt4 import *
from sender.send_on_front import send_data

if __name__ == "__main__":


    # words = ["foundation", "support", "column", "beam", "floor",
    #      "wall", "ceiling", "structure", "building", "tower"]
    words = []
    for _ in range(1000):
        words.append(get_random_word(random.randint(6,10)))


    builder = TowerBuilder(words)
    tower = builder.build_optimized_tower()
    builder.shuffle_words()
    tower = builder.continue_build()

    print(f"Башня построена! Счет: {tower['score']}")
    print(f"Высота: {tower['height']} этажей")
    # print(f"Габариты: {tower['bounding_box']}")
    print("\nСтруктура башни:")
    for i, word in enumerate(tower['words']):
        direction = "X" if word['dir'] == (1, 0, 0) else "Y" if word['dir'] == (0, 1, 0) else "Z"
        print(f"{i+1}. {word['text']} ({direction}) at {word['pos']}")

    matrix = builder.construct_matrix_2()
    send_data(matrix)