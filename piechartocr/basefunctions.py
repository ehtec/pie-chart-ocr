import logging
import os
import platform


# get root path
def get_root_path():

    return os.path.dirname(os.path.dirname(__file__))


# convert complex to real, throw error if imaginary part is not 0. Return orignial number if it is not a complex.
def complex_to_real(c):

    if not isinstance(c, complex):
        return c

    if c.imag != 0:
        raise ValueError("Imaginary part not zero")

    return c.real


# find lib by keyword in search path
def find_lib(search_path, keyword):

    if not os.path.isdir(search_path):
        logging.warning("Path {0} is not a directory".format(search_path))  # pragma: no cover
        return None

    # files = os.listdir(search_path)

    if platform.system().upper() == "WINDOWS":  # pragma: no cover
        fileext = ".dll"

    else:
        fileext = ".so"

    # files = [el for el in files if el.lower().endswith(fileext)]

    # files = [el for el in files if keyword in el]

    files = []

    for root, dirs, the_files in os.walk(search_path):
        for a_file in the_files:

            if not a_file.lower().endswith(fileext):
                continue

            if keyword not in a_file:
                continue

            files.append(os.path.join(root, a_file))

    if not bool(files):
        return None

    if len(files) > 1:
        logging.warning("Multiple matches found: {0}".format(files))  # pragma: no cover

    logging.info("Matching library found: {0}".format(files[0]))

    return files[0]
