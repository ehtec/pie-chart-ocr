import logging
logging.basicConfig(level=logging.DEBUG)
from examples import test_helperfunctions, test_ellipse_detection, test_mser_functions, test_polygon_calc_wrapper, \
    test_superreshelper, test_pie_chart_ocr


test_helperfunctions.main()
test_ellipse_detection.main()
test_mser_functions.main()
test_polygon_calc_wrapper.main()
test_superreshelper.main()
test_pie_chart_ocr.main()
