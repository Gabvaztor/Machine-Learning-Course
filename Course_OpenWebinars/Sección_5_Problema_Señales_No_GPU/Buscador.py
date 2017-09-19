import os
import numpy as np

class Buscador():

    def __init__(self, path, numero_clases):
        self.path = path
        self.numero_clases = numero_clases
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def encuentra_conjuntos_entrenamiento_test_desde_path(self):
        for path in self.path:
            for root, dirs, files in os.walk(path):
                for file_name in files:
                    if (file_name.endswith(".png")):
                        full_path = os.path.join(root, file_name)
                        self._obtiene_conjuntos_desde_path_de_señales(full_path)

    def _obtiene_conjuntos_desde_path_de_señales(self, path):
        """
        If path contains 'train', y_label is two dir up. Else if path contains 'test', y_label is one dir up.
        :param path: the full path
        """
        labels = np.zeros(self.numero_clases, dtype=np.float32)
        if 'train' in path: # If 'train' in path
            y_label_dir = os.path.dirname(os.path.dirname(path))  # Directory of directory of file
            y_label = os.path.basename(y_label_dir)
            labels[int(y_label)] = 1
            self.y_train.append(list(labels))
            self.x_train.append(path)
        elif 'test' in path: # If 'test' in path
            y_label_dir = os.path.dirname(path)  # Directory of file
            y_label = os.path.basename(y_label_dir)
            labels[int(y_label)] = 1
            self.y_test.append(list(labels))
            self.x_test.append(path)
