import logging


# convert complex to real, throw error if imaginary part is not 0. Return orignial number if it is not a complex.
def complex_to_real(c):

    if not isinstance(c, complex):
        return c

    assert c.imag == 0

    return c.real
