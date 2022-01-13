# path = "/home/elias/collected-whitepapers/Finminity-Brochure(1).pdf"
path = "/home/elias/collected-whitepapers/METASEER_Whitepaper_v7.7.pdf"

from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from io import BytesIO
import logging


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

# fp = open(path, 'rb')


with open(path, 'rb') as orig_fp:
    content = orig_fp.read()

with BytesIO(content) as fp:

    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pages = PDFPage.get_pages(fp)

    for i, page in enumerate(pages):
        print('Processing next page...')
        interpreter.process_page(page)
        layout = device.get_result()
        for lobj in layout:
            if isinstance(lobj, LTTextBox):
                # x0, y0, x1, y1
                bbox = lobj.bbox
                text = lobj.get_text()
                # x, y = bbox[0], bbox[3]

                if not bool(text.strip()):
                    continue

                font_name, font_size = get_textbox_font(lobj)
                print('Page %s: Font %s: At %r is text: %s' % (i, (font_name, font_size), bbox, text))
