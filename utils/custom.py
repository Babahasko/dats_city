def convert_matrix_to_test_data(matrix):
    cubes = []
    text = []

    # Проходим по всему трехмерному массиву
    for x in range(len(matrix)):
        for y in range(len(matrix[x])):
            for z in range(len(matrix[x][y])):
                if matrix[x][y][z]:  # Если ячейка не пустая
                    cubes.append([x, y, z])  # Добавляем координаты
                    text.append(matrix[x][y][z])  # Добавляем текст

    # Формируем результат в нужном формате
    test_data = {
        "cubes": cubes,
        "text": text,
    }
    return test_data