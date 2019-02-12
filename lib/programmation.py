# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:01:45 2018

@author: Mahery
"""


def set_range(lst):
    """Convertit un range de liste en un ensemble pour que les tests
    d'appartenance soit en O(1)."""
    return set(range(lst))


def prnt_79(fgr='_'):
    print(79*f'{fgr}')


def zip_to_dict(lst_keys, lst_val):
    """Create a dictionnary from keys and values lists.

    Keywords arguments:
        lst_keys -- liste des clés
        lst_val -- liste des valeurs
    """
    zp = zip(lst_keys, lst_val)  # regroupement sous forme de tuples
    dct = dict(zp)
    return dct


def reshape_one_feat(arr):
    """Reshape single featured data either using array.reshape(-1, 1)."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def reshape_one_sample(arr):
    """Reshape single sampled data either using array.reshape(1, -1)."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def keys_from_dct_val(dct, vl):
    """Returns Keys from a dict linked to its value."""
    return list(dct.keys())[list(dct.values()).index(vl)]
