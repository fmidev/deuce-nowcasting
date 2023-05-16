"""
Miscellaneous functions, that do not belong elsewhere.
"""
from itertools import product

def enumerated_product(*args):
        yield from zip(product(*(range(len(x)) for x in args)), product(*args))