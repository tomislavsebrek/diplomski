from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def gcd(p, q):
    if q == 0:
        return p
    return gcd(q, p % q)
