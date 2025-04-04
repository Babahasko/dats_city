class GameState:
    def __init__(self, data):
        """
        Инициализация парсера с URL API.
        :param api_url: URL для запроса данных.
        """
        self.map_size =data.get("mapSize")
        self.name =data.get("name")
        self.points =data.get("points")
        self.fences =data.get("fences")
        self.snakes =data.get("snakes")
        self.snake_1 = SnakeCollection(self.snakes)[0]
        self.snake_2 = SnakeCollection(self.snakes)[1]
        self.snake_3 = SnakeCollection(self.snakes)[1]
        self.enemies =data.get("enemies")
        self.food =data.get("food")
        self.special_food =data.get("specialFood")
        self.turn =data.get("turn")
        self.tick_remain_ms =data.get("tickRemainMs")
        self.revive_timeout_sec =data.get("reviveTimeoutSec")
        self.errors =data.get("errors")

class Snake:
    def __init__(self, data):
        self.id = data['id']
        self.direction = data['direction']
        self.oldDirection = data['oldDirection']
        self.geometry = data['geometry']
        self.deathCount = data['deathCount']
        self.status = data['status']
        self.reviveRemainMs = data['reviveRemainMs']

    def __repr__(self):
        return f"Snake(id={self.id}, status={self.status}, geometry={self.geometry})"

class SnakeCollection:
    def __init__(self, snakes_data):
        self.snakes = [Snake(snake_data) for snake_data in snakes_data]

    def __getitem__(self, index):
        return self.snakes[index]

    def __iter__(self):
        return iter(self.snakes)