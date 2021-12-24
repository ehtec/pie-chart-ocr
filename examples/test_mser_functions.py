import logging

logging.basicConfig(level=logging.WARNING)

from piechartocr import mser_functions


def main():

    # IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019_1.png'
    # IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019.png'
    # IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-024.jpg'
    # IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-038.jpg'

    # IMG_INPUT_PATH = "/home/elias/Downloads/pie_charts/test_1.png"
    # IMG_INPUT_PATH = "/home/elias/Downloads/pie_charts/test_2.jpg"
    IMG_INPUT_PATH = "/home/elias/git/pie-chart-ocr/pie_charts/test_3.jpg"
    # IMG_INPUT_PATH = "/home/elias/Downloads/pie_charts/test_4.jpg"

    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/temp/1c7933ac-2b99-4745-ac12-38e158b5f0f8.jpg"
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/temp1/af34f3b9-a9df-45db-8d32-c0c1eaf35f62.jpg"

    mser_functions.main(IMG_INPUT_PATH)


if __name__ == "__main__":
    main()
