import logging
from piechartocr.pdf_extractor import extract_tuples_from_pdf


logging.basicConfig(level=logging.WARNING)
path = "/home/elias/collected-whitepapers/METASEER_Whitepaper_v7.7.pdf"
extract_tuples_from_pdf(path)
