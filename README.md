# Работа с веб-сокетами

Запуск сервера
```cmd
python sender/start_server.py
```

Отправка в веб-сокет
```cmd
python sender/send_on_front.py
```

Импоритруем функцию send_data из send_on_front
Пихаем в неё данные и радуемся

Пример из файла scripts/check_logic
```python
 print(f"Башня построена! Счет: {tower['score']}")
    print(f"Высота: {tower['height']} этажей")
    # print(f"Габариты: {tower['bounding_box']}")
    print("\nСтруктура башни:")
    for i, word in enumerate(tower['words']):
        direction = "X" if word['dir'] == (1, 0, 0) else "Y" if word['dir'] == (0, 1, 0) else "Z"
        print(f"{i+1}. {word['text']} ({direction}) at {word['pos']}")

    matrix = builder.construct_matrix_2()
    send_data(matrix)
```