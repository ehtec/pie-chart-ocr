class PDFParsingError(Exception):
    pass


class PagesNumberUnknownError(PDFParsingError):

    def __init__(self):
        self.message = "Pages number could not be determined"


class TooManyPagesError(PDFParsingError):

    def __init__(self, n):
        self.message = "Too many pages: {0}".format(n)
