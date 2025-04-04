import json

from config.config import settings
import aiohttp
from dataclasses import dataclass


@dataclass
class GameURI:
    build_tower = "/api/build"
    new_words = "/api/shuffle"
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

    async def game_rounds(self):
        """Выбрать направление змейки"""
        url = self.base_url + GameURI.rounds
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return response.status, data # Возвращаем JSON, если всё ок
                elif response.status != 200:
                    error_message = await response.text()
                    return response.status, error_message

    async def get_game_rounds(self):
        """Получить расписание раундов"""
        url = self.base_url+GameURI.game_rounds
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                return await response.json()