import cv2
import os
import re

"""
Скрипт формирует сегменты из изображений.
[I_WIDTH, J_WIDTH] - размер сегмента
I_STEP, J_STEP - шаги, с которыми берутся сегменты
"""
I_WIDTH = 128
J_WIDTH = 128
I_STEP = 130
J_STEP = 130

# Директория в которой находятся изображения по классам
source = os.path.join('D:\\projects\\VKR\\texture_dataset1\\')

# Директория, куда будут кладываться сегменты
cutted = os.path.join('D:\\projects\\VKR\\cutted_texture_dataset1_128\\')


def rename_classes(dir_name):
    """
    Функция переименовывает папки
    'stone1-without-rotations' -> 'stone1'
    :param dir_name: Директория, в которой лежат папки классов
    :return: None
    """
    for class_dir_name in os.listdir(dir_name):
        class_dir = dir_name + class_dir_name
        clear_class = re.sub(r'-without-rotations', r'', class_dir_name)
        clear_class_dir = dir_name + clear_class
        os.rename(class_dir, clear_class_dir)


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def cut_images(source_images_dir, cutted_images_dir):
    """
    Функция делит изображения на сегменты
    :param source_images_dir: Исходная папка с классами
    :param cutted_images_dir: Результирующая папка (куда будут складываться сегменты)
    :return: None
    """
    create_dir(cutted_images_dir)
    for class_dir_name in os.listdir(source_images_dir):
        class_dir = source_images_dir + class_dir_name
        cutted_class_dir = cutted_images_dir + class_dir_name
        create_dir(cutted_class_dir)
        for image_file in os.listdir(class_dir):
            try:
                img = cv2.imread(class_dir + '\\' + image_file)
            except Exception:
                continue
            for i in range(0, img.shape[0] - I_WIDTH, I_STEP):
                for j in range(0, img.shape[1] - J_WIDTH, J_STEP):
                    temp_image = img[
                        i:i + I_WIDTH,
                        j:j + J_WIDTH,
                        :
                    ]
                    temp_name = os.path.splitext(image_file)[0] + \
                                "_[" + \
                                str(i) + \
                                "_" + \
                                str(j) + \
                                "].bmp"
                    try:
                        cv2.imwrite(
                            cutted_class_dir + '\\' + temp_name,
                            temp_image
                        )
                    except Exception:
                        continue


if __name__ == '__main__':
    # Пример использования
    rename_classes('D:\\projects\\texture_dataset1\\')
    cut_images(
        'D:\\projects\\texture_dataset2\\',
        'D:\\projects\\cutted_texture_dataset2_128\\'
    )
