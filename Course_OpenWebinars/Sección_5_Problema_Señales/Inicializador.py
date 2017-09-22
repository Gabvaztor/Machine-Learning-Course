"""
Author: @gabvaztor
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""
import Course_OpenWebinars.Sección_5_Problema_Señales.Modelo_Convolucional as modelo
import Course_OpenWebinars.Sección_5_Problema_Señales.Buscador as buscador
from Course_OpenWebinars.UsefulTools.TensorFlowUtils import pt
import tensorflow as tf

"""
Ubicaciones
"""
conjunto_entrenamiento = "D:\\Machine_Learning\\Competitions\\German_Signal\\DITS-classification\\classification_train\\"
conjunto_test = "D:\\Machine_Learning\\Competitions\\German_Signal\\DITS-classification\\classification_test\\"

numero_clases = 59 # Start in 0

searcher = buscador.Buscador(path=[conjunto_entrenamiento, conjunto_test], numero_clases=numero_clases)
searcher.encuentra_conjuntos_entrenamiento_test_desde_path()

percentages_sets = None  # Example

"""
Getting train, validation (if necessary) and test set.
"""
test_set = [searcher.x_test, searcher.y_test]  # Test Set
train_set = [searcher.x_train, searcher.y_train]  # Train Set

pt(test_set[0])
pt(test_set[1])
del searcher
option_problem = "Problema de señales"

models = modelo.Modelo(input=train_set[0],test=test_set[0],
                         input_labels=train_set[1],test_labels=test_set[1],
                         number_of_classes=numero_clases,
                         option_problem=option_problem)
with tf.device('/gpu:0'):
    models.convolucion_imagenes()



