import mser_functions
import superreshelper


n = int(input("Image id: "))

superreshelper.upscale_test_image_file(n)

mser_functions.main('temp2/upscaled{0}.png'.format(n))
