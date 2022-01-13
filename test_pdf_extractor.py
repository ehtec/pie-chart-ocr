import logging
logging.basicConfig(level=logging.INFO)
from piechartocr.pdf_extractor import extract_tuples_from_pdf


path = "/home/elias/collected-whitepapers/METASEER_Whitepaper_v7.7.pdf"
extract_tuples_from_pdf(path)
