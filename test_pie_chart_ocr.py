import logging

logging.basicConfig(level=logging.WARNING)

import pie_chart_ocr


# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019_1.png'
# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019.png'
# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-024.jpg'
# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-038.jpg'

# IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_1.png"
# IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_2.jpg"
# IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_3.jpg"
# IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_4.jpg"
# IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_5.png"  # not well detected, a little better with scaling at step 1 and scaling factor 4
# IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_6.png"
# IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_7.png"  # percentages detected, but false positive for descriptions
# IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_8.png"
# IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_9.png"  # good, but a few false positives
IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_10.png"


pie_chart_ocr.main(IMG_INPUT_PATH)


