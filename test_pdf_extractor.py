import logging
logging.basicConfig(level=logging.INFO)
from piechartocr.pdf_extractor import extract_tuples_from_pdf


# path = "/home/elias/collected-whitepapers/Finminity-Brochure(1).pdf"
path = "/home/elias/collected-whitepapers/METASEER_Whitepaper_v7.7.pdf"
# path = "/home/elias/collected-whitepapers/7plusWhitePaper_compressed(1).pdf"

extract_tuples_from_pdf(path, return_unsorted_output=True)
