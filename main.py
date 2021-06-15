import cv2
import os
import numpy as np
import os.path
import pandas as pd
import mahotas as mt
from time import time

# Классы, которые не будут учитываться при обучение,
# потому что классификатор их часто путает.
# Должны соответствовать названиям папок.
BAN_LIST = {'stone2', 'oatmeal1', 'rug1', 'stone3'}


def extract_features(image, n_neighbours):
    """
    Функция вычисляет признаки Харалика для заданного изображения
    :param image: Изображение на вход
    :param n_neighbours: Сколько соседей учитывать при подсчете
                        признаков Харалика
    :return: Возвращает вектор из n_neighbours * 13 признаков
    """
    # Вычисляем признаки Харалика для первого соседа
    ht_features = mt.features.haralick(image)
    ht_mean = ht_features.mean(axis=0)

    for i in range(2, n_neighbours + 1):
        # Вычисляем признаки харалика для i-го соседа
        ht_temp_features = mt.features.haralick(image, distance=i)
        ht_temp_mean = ht_temp_features.mean(axis=0)

        # Складываем признаки в массив
        ht_mean = np.append(ht_mean, ht_temp_mean)
    return ht_mean


def form_training_df(training_dir, n_neighbours, n_images):
    """
    Функция формирует обучающий DataFrame и вектор меток,
    соответствующий выборке
    :param n_images: Число изображений с каждого класса
    :param training_dir: Директория, в которой находятся папки с классами.
                        Должна заканчиваться слэшем '\'!
    :param n_neighbours: Сколько соседей учитывать при подсчете
                        признаков Харалика
    :return: Возвращает 2 DataFrame с признаками и метками классов
    """
    x_train = []
    y_train = []
    filenames = []
    for class_dir_name in os.listdir(training_dir):
        if class_dir_name in BAN_LIST:
            continue
        class_dir = training_dir + class_dir_name
        num_of_segments = len(os.listdir(class_dir))
        objects_count = 0
        for i in range(0, num_of_segments, num_of_segments // n_images - 1):
            image_file = os.listdir(class_dir)[i]
            image = cv2.imread(class_dir + '\\' + image_file)

            # Перевод в серый и извлечение признаков Харалика
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features = extract_features(gray, n_neighbours)

            # Выводит имя файла, попавшего в выборку (можно закомментировать)
            print(image_file)

            # Добавляем в массив признаки, метку класса и название файла
            x_train.append(features)
            y_train.append(class_dir_name)
            filenames.append(image_file)

            objects_count += 1
            if objects_count == n_images:
                break

    # Переводим массивы в структуру DataFrame из pandas
    x_train_df = pd.DataFrame(x_train, columns=[
        str(i) for i in range(1, n_neighbours * 13 + 1)
    ])
    y_train_df = pd.DataFrame(y_train, columns=['Class'])
    filenames_df = pd.DataFrame(filenames, columns=['Filename'])

    return x_train_df, y_train_df, filenames


if __name__ == '__main__':
    # Пример использования программы
    tic = time()  # Засекаем время
    X_train, Y_train = form_training_df('D:\\projects\\cutted_texture_dataset1_128\\', 3, 200)
    toc = time()
    print(toc-tic)  # Выводим время обработки
    print(X_train)  # Выводим обучающую выбокру
