import matplotlib.pyplot as plt

class Map():
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots()  # Создание объекта "карты"
        self.ax.set_xlabel('X Coordinate')  # Подпись оси X
        self.ax.set_ylabel('Y Coordinate')  # Подпись оси Y
        self.ax.set_title('Camera Position')  # Заголовок графика

    def update_plot(self, x_coords, y_coords):
        self.ax.clear()  # Очистка предыдущего графика
        self.ax.plot(x_coords, y_coords)  # Построение нового графика
        plt.pause(0.1)  # Пауза для обновления графика

    def add_params(self, x, y, param_name):
        self.ax.scatter(x, y, label=param_name)  # Добавление параметров на карту
        self.ax.legend()  # Добавление легенды

