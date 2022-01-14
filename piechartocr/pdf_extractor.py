from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from io import BytesIO
import logging
import copy
import re
from .helperfunctions import isfloat
import gc
gc.enable()


# maximum number of pages before pdf is discarded
MAX_PDF_PAGES = 250

# maximum words per headline
MAX_HEADLINE_WORDS = 5


# get the font name and size of a LTTextBox object
def get_textbox_font(obj):

    # we need to average in case a text box has multiple different font sizes and names
    all_font_names = []
    all_font_sizes = []

    if not isinstance(obj, LTTextBox):
        raise TypeError("Supplied object is not a LTTextBox, but {0}".format(type(obj)))

    for o in obj._objs:
        if isinstance(o, LTTextLine):
            text = o.get_text()
            if text.strip():
                for c in o._objs:
                    if isinstance(c, LTChar):
                        all_font_names.append(c.fontname)
                        all_font_sizes.append(int(round(c.size)))

    if (not bool(all_font_sizes)) or (not bool(all_font_sizes)):
        logging.warning("Unable to determine font size of object because it contains no text")
        return None, None

    font_name = max(all_font_names, key=all_font_names.count)
    font_size = max(all_font_sizes, key=all_font_sizes.count)

    return font_name, font_size


# get the font name and size of a LTTextLine object
# get the font name and size of a LTTextBox object
def get_textline_font(obj):

    # we need to average in case a text box has multiple different font sizes and names
    all_font_names = []
    all_font_sizes = []

    if not isinstance(obj, LTTextLine):
        raise TypeError("Supplied object is not a LTTextLine, but {0}".format(type(obj)))

    text = obj.get_text()
    if text.strip():
        for c in obj._objs:
            if isinstance(c, LTChar):
                all_font_names.append(c.fontname)
                all_font_sizes.append(int(round(c.size)))

    if (not bool(all_font_sizes)) or (not bool(all_font_sizes)):
        logging.warning("Unable to determine font size of object because it contains no text")
        return None, None

    font_name = max(all_font_names, key=all_font_names.count)
    font_size = max(all_font_sizes, key=all_font_sizes.count)

    return font_name, font_size


# extract tuples with formatting info from pdf
#   order is preserved.
# if sort_direction is specified, unsorted_output will be sorted. sort_direction can be top-to-bottom or left-to-right
# x_coordinate, y_coordinate: coordinates used for sorting, x_coordinate = left | right, y_coordinate = top | bottom
def extract_tuples_from_pdf(path, max_pdf_pages=MAX_PDF_PAGES, return_unsorted_output=False, sort_direction=None,
                            x_coordinate='right', y_coordinate='top'):

    with open(path, 'rb') as orig_fp:
        content = orig_fp.read()

    with BytesIO(content) as fp:
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        setattr(laparams, 'all_texts', True)
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        pages = PDFPage.get_pages(fp)

        # count pdf pages
        pages_list = list(pages)
        pages_nr = len(pages_list)
        logging.warning("pages_nr: {0}".format(pages_nr))

        if not pages_nr:
            raise Exception("pages number could not be determined")

        if pages_nr > max_pdf_pages:
            raise Exception("Too many pdf pages: {0}".format(pages_nr))

        output = []
        orig_output = []

        for i, page in enumerate(pages_list):
            logging.info('Processing page {0}...'.format(i))
            interpreter.process_page(page)
            layout = device.get_result()
            for lobj in layout:
                if isinstance(lobj, LTTextBox):
                    for o in lobj._objs:

                        if not isinstance(o, LTTextLine):
                            continue

                        # x0, y0, x1, y1
                        bbox = o.bbox
                        the_text = o.get_text()
                        # x, y = bbox[0], bbox[3]

                        orig_text = copy.deepcopy(the_text)

                        # strip the_text
                        the_text = the_text.strip()

                        # discard if text is empty
                        if not bool(the_text):
                            continue

                        # discard text if it is only a float (might be a page number)
                        if isfloat(the_text.replace(' ', '').replace('\n', '').replace('\r', '')):
                            continue

                        # discard text if it has no alphanumeric character
                        if not bool(re.findall(r'[A-z0-9]+', the_text)):
                            continue

                        font_name, font_size = get_textline_font(o)

                        # we don't have anything implemented to fetch font style
                        font_style = None

                        output.append((the_text, font_size, font_name, font_style, i, bbox))
                        orig_output.append((orig_text, font_size, font_name, font_style, i, bbox))

    words_tuples = []
    font_sizes = []

    for elem in output:

        the_words = re.findall(r'\w+', " {0} ".format(elem[0]))

        for item in the_words:
            words_tuples.append((item, elem[1]))
            font_sizes.append(elem[1])

    # for row in words_tuples:
    #     print(row)

    if not bool(font_sizes):
        logging.warning("No texts found")
        return None

    p_font_size = max(font_sizes, key=font_sizes.count)

    logging.info("p font size: {0}".format(p_font_size))

    unsorted_output = copy.deepcopy(output)

    # sort unsorted_output if a sort direction is specified
    if sort_direction is not None:

        old_unsorted_output = copy.deepcopy(unsorted_output)

        distinct_pages = list(range(pages_nr))

        pagified_output = [[el for el in old_unsorted_output if el[4] == i] for i in distinct_pages]

        sorted_pagified_output = []

        if x_coordinate == "left":
            x_coord = 0

        elif x_coordinate == "right":
            x_coord = 2

        else:
            raise NotImplementedError("Invalid x_coordinate: {0}".format(x_coordinate))

        if y_coordinate == "top":
            y_coord = 3

        elif y_coordinate == "bottom":
            y_coord = 1

        else:
            raise NotImplementedError("Invalid y_coordinate: {0}".format(y_coordinate))

        if sort_direction == "top-to-bottom":

            for elem in pagified_output:
                elem_copy = copy.deepcopy(elem)
                elem_copy.sort(key=lambda x: x[5][x_coord])
                elem_copy.sort(key=lambda x: x[5][y_coord], reverse=True)
                sorted_pagified_output.append(elem_copy)

        elif sort_direction == "left-to-right":

            for elem in pagified_output:
                elem_copy = copy.deepcopy(elem)
                elem_copy.sort(key=lambda x: x[5][y_coord], reverse=True)
                elem_copy.sort(key=lambda x: x[5][x_coord])
                sorted_pagified_output.append(elem_copy)

        else:
            raise NotImplementedError("Unknown sort direction: {0}".format(sort_direction))

        unsorted_output = [el for elem in sorted_pagified_output for el in elem]

    output.sort(key=lambda x: x[1], reverse=True)

    res_dict = {}

    for row in output:
        is_bold = False

        if row[2] is not None:
            if 'bold' in row[2].lower():
                is_bold = True

        if row[1] > p_font_size or (is_bold and row[1] >= p_font_size):
            logging.info(row)

            if len(re.findall(r'\w+', row[0])) > MAX_HEADLINE_WORDS:
                continue

            if row[1] in res_dict.keys():
                res_dict[row[1]].append(row)
            else:
                res_dict.update({row[1]: [row]})

    logging.info("res_dict: {0}".format(res_dict))

    gc.collect()

    if return_unsorted_output:
        for el in unsorted_output:
            logging.info("unsorted_output: {0}".format(el))

        logging.warning("len(unsorted_output): {0}".format(len(unsorted_output)))

        return res_dict, orig_output, pages_nr, unsorted_output, p_font_size

    return res_dict, orig_output, pages_nr
