# Importamos las librerias necesarias
from __future__ import print_function
from argparse import ArgumentParser
from functools import partial
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import tensorflow as tf
import random
import math
# Manipulacion de imagenes.
import PIL.Image
from scipy.ndimage.filters import gaussian_filter

#!wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip

import inception5h
# inception.data_dir = 'models'
inception5h.maybe_download()

model = inception5h.Inception5h()

session = tf.InteractiveSession(graph=model.graph)
'''Funciones de ayuda para la manipulación de imágenes'''


# Esta funcion carga una imagen y la retorna como un arreglo numpy de floats.
def cargar_imagen(filename):
    image = PIL.Image.open(filename)

    return np.float32(image)


# guarda una imagen en formato jpeg.
def guardar_imagen(image, filename):
    # Nos aseguramos de que los valores de los pixeles esten en el rango 0-255.
    image = np.clip(image, 0.0, 255.0)

    # Convertimos a bytes.
    image = image.astype(np.uint8)

    # Escribimos el archivo de imagen en formato jpeg.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


# Esta función muestra una imagen. El uso de matplotlib proporciona imágenes de baja resolución.
# El uso de PIL da imágenes bonitas.
def mostrar_imagen(image):
    # Suponemos que los valores de píxel se escalan entre 0 y 255.

    if False:
        # Convierte los valores de píxel al rango entre 0,0 y 1,0.
        image = np.clip(image / 255.0, 0.0, 1.0)

        # muestra utilizando matplotlib..
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:
        # verifica que los valores de píxel estén entre 0 y 255.
        image = np.clip(image, 0.0, 255.0)

        # Convierte pixeles a bytes.
        image = image.astype(np.uint8)

        # Convierte a una imagen tipo PIL y la muestra en pantalla.

        try:
            imagen = PIL.Image.fromarray(image)
            imagen.show()
        except:
            print("No ha sido posible cargar la imagen")


# Normalizar una imagen para que sus valores estén entre 0,0 y 1,0. Esto es útil para trazar el degradado.
def normalizar_imagen(x):
    # Obtenga los valores mínimo y máximo para todos los píxeles de la entrada.
    x_min = x.min()
    x_max = x.max()

    # Normalizar para que todos los valores estén entre 0,0 y 1,0.
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


# Esta función traza el gradient después de normalizarlo.
def mostrar_gradiente(gradient):
    # Normalice el gradient de modo que esté entre 0,0 y 1,0.
    gradient_normalized = normalizar_imagen(gradient)

    # muestre el gradient normalizado.
    plt.imshow(gradient_normalized, interpolation='bilinear')
    plt.show()


# Esta funcion reescala una imagen.
def reescalar_imagen(image, size=None, factor=None):
    # Si se proporciona un factor de reescalado, utilírelo.
    if factor is not None:
        # Escale la forma de la matriz NumPy para la altura y el ancho.
        size = np.array(image.shape[0:2]) * factor

        # El tamaño es de punto flotante porque se ha escalado.
        # PIL requiere que el tamaño sea números enteros.
        size = size.astype(int)
    else:
        # Asegúrese de que el tamaño tiene una longitud de 2.
        size = size[0:2]

    # La altura y la anchura se invierten en NumPy frente a PIL.
    size = tuple(reversed(size))

    # EAsegúrese los valores de píxel están entre 0 y 255.
    img = np.clip(image, 0.0, 255.0)

    # Convierta los píxeles en bytes de 8 bits.
    img = img.astype(np.uint8)

    # Cree PIL-Object a partir de una matriz NumPy.
    img = PIL.Image.fromarray(img)

    # Cambie el tamaño de la imagen.
    img_resized = img.resize(size, PIL.Image.LANCZOS)

    # Convierta los valores de píxel de 8 bits de nuevo a punto flotante.
    img_resized = np.float32(img_resized)

    return img_resized

'''DeepDream Algorithm'''


# Se trata de una función auxiliar para determinar un tamaño de mosaico adecuado.
# El tamaño de mosaico deseado es, por ejemplo, 400x400 píxeles, pero el tamaño real
# de la tesela dependerá de las dimensiones de imagen.
def obtener_tamano_mozaico(num_pixels, tile_size=400):
    """
    num_pixels es el número de píxeles en una dimensión de la imagen.
    tile_size es el tamaño de mosaico deseado.
    """

    # ¿Cuántas veces podemos repetir un mosaico del tamaño deseado?.
    num_tiles = int(round(num_pixels / tile_size))

    # Asegúrese de que haya al menos 1 mosaico.
    num_tiles = max(1, num_tiles)

    # El tamaño real del mosaico.
    actual_tile_size = math.ceil(num_pixels / num_tiles)

    return actual_tile_size


# Esta función auxiliar calcula el degradado para una imagen de entrada.
def gradiente_mozaico(gradient, image, tile_size=400):
    # Asigne una matriz para el degradado de toda la imagen.
    grad = np.zeros_like(image)

    # Número de píxeles para los ejes x e y.
    x_max, y_max, _ = image.shape

    # Tamaño de mosaico para el eje x.
    x_tile_size = obtener_tamano_mozaico(num_pixels=x_max, tile_size=tile_size)
    # 1/4 del tamaño del mozaico.
    x_tile_size4 = x_tile_size // 4

    # Tamaño de mosaico para el eje y.
    y_tile_size = obtener_tamano_mozaico(num_pixels=y_max, tile_size=tile_size)
    # 1/4 del tamaño del mosaico
    y_tile_size4 = y_tile_size // 4

    # La posición de inicio aleatoria de los mosaicos del eje x.
    # El valor aleatorio está entre-3/4 y-1/4 del tamaño de mosaico.
    # Esto es así que los azulejos de borde son al menos 1/4 de la baldosa-tamaño,
    # de lo contrario, las baldosas pueden ser demasiado pequeñas que crea degradados ruidosos.
    x_start = random.randint(-3 * x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        # La posición final del mosaico actual.
        x_end = x_start + x_tile_size

        # Asegúrese de que las posiciones de inicio y fin de la ventana son válidas.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        # Posición de inicio aleatoria para las teselas en el eje y.
        # El valor aleatorio está entre-3/4 y-1/4 del tamaño de tesela.
        y_start = random.randint(-3 * y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            # La posición final del mosaico actual.
            y_end = y_start + y_tile_size

            # Asegúrese de que las posiciones de inicio y fin de la ventana son válidas.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Obtenga el mosaico de imágenes.
            img_tile = image[x_start_lim:x_end_lim,
                       y_start_lim:y_end_lim, :]

            # Crear un feed-dict con el mosaico de imágenes.
            feed_dict = model.create_feed_dict(image=img_tile)

            # Utilice TensorFlow para calcular el valor de degradado.
            g = session.run(gradient, feed_dict=feed_dict)

            # Normalice el degradado para el mosaico. Esto es
            # necesario porque las baldosas pueden tener muy diferentes
            # valores. La normalización da un gradiente más coherente.
            g /= (np.std(g) + 1e-8)

            # Almacene el degradado del mosaico en la ubicación adecuada.
            grad[x_start_lim:x_end_lim,
            y_start_lim:y_end_lim, :] = g

            # Avance de la posición inicial para el eje y.
            y_start = y_end

        # Avance de la posición inicial del eje x.
        x_start = x_end

    return grad



'''Optimizar Imagen'''

# Esta función es el principal bucle de optimización para el algoritmo DeepDream.
# Calcula el degradado de la capa dada del modelo de inicio con respecto a la imagen de entrada.
# A continuación, el degradado se añade a la imagen de entrada para aumentar el valor medio del tensor de capa.
# Este proceso se repite varias veces y amplifica los patrones que el modelo de inicio ve en la imagen de entrada.
def optimizar_imagen(layer_tensor, image,
                     num_iterations=10, step_size=3.0, tile_size=400,
                     show_gradient=False):
    """
    Utilice la subida de degradado para optimizar una imagen para maximizar la
    valor medio de la determinada layer_tensor.

    Parametros:
    layer_tensor: Referencia a un tensor que sera maximizado.
    image: La imagen de entrada utilizada como punto de partida.
    num_iterations: Número de iteraciones de optimización para realizar.
    step_size: Escala para cada paso del ascenso del degradado.
    tile_size: Tamaño de los mosaicos al calcular el degradado.
    show_gradient: Trace el degradado en cada iteración.
    """

    # Copie la imagen para que no sobrescriba la imagen original.
    img = image.copy()

    #print("Image before:")
    #mostrar_imagen(img)

    #print("Processing image: ", end="")

    # Utilice TensorFlow para obtener la función matemática de la
    # gradiente del tensor de capa dado con respecto a la
    # imagen de entrada. Esto puede provocar que TensorFlow agregue el mismo
    # expresiones matemáticas al gráfico cada vez que se llama a esta función.
    # Se puede utilizar una gran cantidad de RAM y podría ser movido fuera de la función.
    gradient = model.get_gradient(layer_tensor)

    for i in range(num_iterations):
        # Calcule el valor del degradado.
        # Esto nos dice cómo cambiar la imagen para
        # maximizar la media del tensor de capa dado.
        grad = gradiente_mozaico(gradient=gradient, image=img,
                                 tile_size=tile_size)

        # Difumina el degradado con diferentes cantidades y añade
        # ellos juntos. La cantidad de desenfoque también se incrementa
        # durante la optimización. Esto fue encontrado para dar
        # imágenes agradables y suaves. Puede intentar cambiar las fórmulas.
        # El desenfoque-cantidad se llama Sigma (0 = sin desenfoque, 1 = desenfoque bajo, etc.)
        # Podríamos llamar a gaussian_filter (Grad, Sigma = (Sigma, Sigma, 0,0))
        # que no difuminaría el canal de color. Esto tiende a
        # dar colores psicodélicos/pastel en las imágenes resultantes.
        # Cuando el canal de color también se desdibuja los colores de la
        # imagen de entrada se conservan principalmente en la imagen de salida.
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma * 2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma * 0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        # Escale el tamaño de paso según los valores de degradado.
        # Esto puede no ser necesario porque el degradado de mosaico
        # ya está normalizado.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Actualice la imagen siguiendo el degradado.
        img += grad * step_size_scaled

        if show_gradient:
            # Imprima estadísticas para el degradado.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))

            # Trace el degradado.
            mostrar_gradiente(grad)
        else:
            # De lo contrario, muestra un pequeño indicador de progreso.
            print(". ", end="")

    print()
    #print("Image after:")
    #mostrar_imagen(img)

    return img


'''Optimizacion Recursiva de Imagenes'''


# Esta función auxiliar reduce la escala de la imagen de entrada varias veces y ejecuta cada
# versión a escala descendente a través de la función ' optimize_image () ' anterior. Esto da
# como resultado patrones más grandes en la imagen final. También acelera el cálculo.

def optimizacion_recursiva(layer_tensor, image,
                           num_repeats=4, rescale_factor=0.7, blend=0.2,
                           num_iterations=10, step_size=3.0,
                           tile_size=400):
    """
    Desenfoque recursivamente y reducir la escala de la imagen de entrada.
    Cada imagen a escala reducida se ejecuta a través de la optimize_image ()
    función para amplificar los patrones que el modelo de inicio ve.

    Parámetros:
    image: imagen de entrada utilizada como punto de partida.
    rescale_factor: factor de reducción de escala de la imagen.
    num_repeats: número de veces que se desea reducir la escala de la imagen.
    Blend: factor para mezclar las imágenes originales y procesadas.

    Parámetros pasados a optimize_image ():
    layer_tensor: referencia a un tensor que será maximizado.
    num_iterations: número de iteraciones de optimización que se realizarán.
    step_size: escala para cada paso del ascenso del degradado.
    tile_size: tamaño de las teselas al calcular el degradado.
    """

    # ¿hacer un paso de recursividad?
    if num_repeats > 0:
        # Desenfoca la imagen de entrada para evitar artefactos al reducir el escalado.
        # La cantidad de desenfoque es controlada por Sigma. Tenga en cuenta que el
        # el canal de color no está difuminado ya que haría que la imagen fuera gris.
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        # Reducir la escala de la imagen.
        img_downscaled = reescalar_imagen(image=img_blur,
                                          factor=rescale_factor)

        # Llamada recursiva a esta función.
        # Reste uno de num_repeats y utilice la imagen a escala descendente.
        img_result = optimizacion_recursiva(layer_tensor=layer_tensor,
                                            image=img_downscaled,
                                            num_repeats=num_repeats - 1,
                                            rescale_factor=rescale_factor,
                                            blend=blend,
                                            num_iterations=num_iterations,
                                            step_size=step_size,
                                            tile_size=tile_size)

        # De nuevo la imagen resultante a su tamaño original.
        img_upscaled = reescalar_imagen(image=img_result, size=image.shape)

        # Mezcle las imágenes originales y procesadas.
        image = blend * image + (1.0 - blend) * img_upscaled

    #print("Recursive level:", num_repeats)

    # Procese la imagen utilizando el algoritmo DeepDream.
    img_result = optimizar_imagen(layer_tensor=layer_tensor,
                                  image=image,
                                  num_iterations=num_iterations,
                                  step_size=step_size,
                                  tile_size=tile_size)

    return img_result

