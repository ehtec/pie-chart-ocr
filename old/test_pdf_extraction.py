# path = "/home/elias/collected-whitepapers/Finminity-Brochure(1).pdf"
path = "/home/elias/collected-whitepapers/METASEER_Whitepaper_v7.7.pdf"

from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from io import BytesIO

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
                print('Page %s: At %r is text: %s' % (i, bbox, text))
