import matplotlib.pyplot as plt

class Map():
    def __init__(self) -> None:
        pass
    def update_plot(x_coords, y_coords):
        plt.clf()  # Очистка предыдущего графика
        plt.plot(x_coords, y_coords)  # Построение нового графика
        plt.xlabel('X Coordinate')  # Подпись оси X
        plt.ylabel('Y Coordinate')  # Подпись оси Y
        plt.title('Camera Position')  # Заголовок графика
        plt.pause(0.1)  # Пауза для обновления графика


