from random import sample
from string import ascii_letters


def get_random_hash(length=11):
    return ''.join(sample(ascii_letters +
                          ''.join(str(i) for i in range(10)), length))
