from piechartocr import mser_functions
from piechartocr import superreshelper


def main():

    # n = int(input("Image id: "))
    #
    # superreshelper.upscale_test_image_file(n)

    superreshelper.upscale_all_images()

    # mser_functions.main('temp2/upscaled{0}.png'.format(n))


if __name__ == "__main__":
    main()
