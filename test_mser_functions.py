import logging

logging.basicConfig(level=logging.WARNING)

import mser_functions


# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019_1.png'
# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019.png'
IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-024.jpg'
# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-038.jpg'

# IMG_INPUT_PATH = "/home/elias/Downloads/pie_charts/test_1.png"
# IMG_INPUT_PATH = "/home/elias/Downloads/pie_charts/test_2.jpg"
# IMG_INPUT_PATH = "/home/elias/Downloads/pie_charts/test_3.jpg"
# IMG_INPUT_PATH = "/home/elias/Downloads/pie_charts/test_4.jpg"


mser_functions.main(IMG_INPUT_PATH)

