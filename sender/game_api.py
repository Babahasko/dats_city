import json
from config.config import settings
import aiohttp
from dataclasses import dataclass

from sender.game_parser import BuildReq, TowerInfo


@dataclass
class GameURI:
    build = "/api/build"
    shuffle = "/api/shuffle"
    towers = "/api/towers"
    words = "/api/words"
    rounds = "/api/rounds"


class GameAPI:
    def __init__(self, api_key=settings.api.api_key, base_url=settings.api.server_url):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-Auth-Token": self.api_key,
            "Content-Type": "application/json"
        }

    async def _make_request(self, method: str, endpoint: str, payload: BuildReq = None):
        """
        Универсальная функция для выполнения HTTP-запросов.
        :param method: Метод запроса ('GET', 'POST').
        :param endpoint: Конечная точка API.
        :param payload: Данные для POST-запроса (опционально).
        :return: Кортеж (статус, данные или сообщение об ошибке).
        """
        url = self.base_url + endpoint
        async with aiohttp.ClientSession() as session:
            if method.upper() == "GET":
                async with session.get(url, headers=self.headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == "POST":
                json_payload = json.dumps(payload) if payload else None
                async with session.post(url, headers=self.headers, data=json_payload) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

    async def _handle_response(self, response) -> any :
        """
        Обработка HTTP-ответа.
        :param response: Ответ от сервера.
        :return: Кортеж (статус, данные или сообщение об ошибке).
        """
        if response.status == 200:
            print(f"{response.url.path}: {response.status}")
            data = await response.json()
            return data
        else:
            print(f"{response.url.path}: {response.status}")
            # error_message = await response.text()
            return None

    # Методы для работы с API
    async def rounds(self):
        """Получить информацию о раундах."""

        return await self._make_request("GET", GameURI.rounds)

    async def towers(self) -> TowerInfo:
        """Получить информацию о башнях."""
        return await self._make_request("GET", GameURI.towers)

    async def words(self):
        """Получить информацию о словах."""
        return await self._make_request("GET", GameURI.words)

    async def shuffle(self):
        """Выполнить shuffle."""
        return await self._make_request("POST", GameURI.shuffle)

    async def build(self, payload: BuildReq):
        """Выполнить build."""
        return await self._make_request("POST", GameURI.build, payload)