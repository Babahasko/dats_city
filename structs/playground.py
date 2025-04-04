class Playground:
    map3D: list[list[list[str]]]



    def __init__(self, x: int, y: int, z: int):
        self.map3D = [[[""] * x] * y]*z