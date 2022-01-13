from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from io import BytesIO
import logging


# maximum number of pages before pdf is discarded
MAX_PDF_PAGES = 250


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
def extract_tuples_from_pdf(path, max_pdf_pages=MAX_PDF_PAGES, return_unsorted_output=False):

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
        pages_nr = len(pages)
        logging.warning("pages_nr: {0}".format(pages_nr))

    if not pages_nr:
        raise Exception("pages number could not be determined")

    if pages_nr > max_pdf_pages:
        raise Exception("Too many pdf pages: {0}".format(pages_nr))
