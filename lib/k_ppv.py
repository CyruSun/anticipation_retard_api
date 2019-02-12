# -*- coding: utf-8 -*-
'''
Created on 16 août 2018

@author: Mahery
'''
# # K-nearest neighbors module

# %%

import numpy as np
import math as mt
import itertools as it
import operator as op
import copy as cp
import scipy.spatial.distance as sci_dist
# from numpy import * # Pour PyDev
# from sklearn.metrics.pairwise import pairwise_distances as dist_pairwise

# %%

# NaN (not a number) que l'on convertit en float
nan = float('nan')

# %%


def set_range(liste):
    """Convertit un range de liste en un ensemble pour que les tests
    d'appartenance soit en O(1)."""
    return set(range(liste))

# %%


def table_dimension_2d(list_tab):
    """Return the height and the width of a 2d list of lists."""

    # !!! A améliorer pour pouvoir recevoir une liste en 1D !!!
    list_height = len(list_tab)
    list_width = len(list_tab[0])
    return list_height, list_width

# %%


def remove_lin_col(list_tab, ind_lin=[], ind_col=[]):
    """Elimine les variables des lignes ou colonnes spécifiées d'une liste
    et retourne une table tronquée des lignes ou des colonnes voulues.

    Keywords arguments:
        list_table -- liste python
        ind_lin -- liste des indices des lignes des variables à
        tronquer (default [])
        ind_col -- liste des indices des colonnes des variables à
        tronquer (default [])

    Returns:
        list_tab_tronq -- table tronquée des lignes ou des colonnes
        voulues
    """

    # table sans les variables qualitatives
    list_tab_tronq = [list_tab[i][j] for i in set_range(
            table_dimension_2d(list_tab)[0]) for j in set_range(
                    table_dimension_2d(list_tab)[1])
        if i not in set(ind_lin) and j not in set(ind_col)]

    return list_tab_tronq

# %%


def conversion_array(list_tab, new_shape_lin=-1, new_shape_col=-1):
    """Convertit une liste en un tableau de numpy et le reforme.

    Keywords arguments:
        new_shape_lin -- nouvelle hauteur voulue du tableau (default -1)
        new_shape_col -- nouvelle largeur voulue du tableau (default -1)

    Returns:
        array_list_tab -- tableau numpy reformé
    """

    # Conversion de la table en array
    array_list_tab = np.array(list_tab)
    array_list_tab = array_list_tab.reshape(new_shape_lin, new_shape_col)

    return array_list_tab

# %%


def detecter_nan(array_input):
    """Détecte les nan dans un array et retourne une liste des index des
    lignes correspondants.

    Returns:
        coord_nan -- coordonnées des valeurs numériques
        ind_lin_nan -- indice des lignes des valeurs numèriques
        ind_col_nan -- indice des colonnes des valeurs numèriques
        coord_not_nan -- coordonnées des valeurs non numèriques
    """

    ind_lin_nan = []  # indices des lignes qui ont au moins un nan
    ind_col_nan = []  # indices des colonnes qui ont au moins un nan
    coord_nan = []  # coordonnées des valeurs nan
    coord_not_nan = []  # coordonnées des valeurs non nan

    # Extraction des indices de lignes qui contiennent un nan
    for ind, e in np.ndenumerate(array_input):
        if mt.isnan(e):
            # print (f'isnan: ind = {ind}, val = {e}')
            ind_lin_nan.append(ind[0])
            ind_col_nan.append(ind[1])
            coord_nan.append(ind)
        else:
            # print (f'is not nan: ind = {ind}, val = {e}')
            coord_not_nan.append(ind)

    # Conversion en set pour ne garder que les éléments uniques
    ind_lin_nan = list(set(ind_lin_nan))
    ind_col_nan = list(set(ind_col_nan))

    return coord_nan, ind_lin_nan, ind_col_nan, coord_not_nan

# %%


def group_indcol_tuple(tupl_list):
    """ Regroupe les coordonnées des valeurs d'une liste de tuples
    sous la forme: [(première valeur du tuple, [liste des autres valeurs
    de tuple ayant la première valeur en commun])].

        Returns:
            tupl_list_group -- coordonnées des valeurs d'une liste de tuples
    sous la forme: [(première valeur du tuple, [liste des autres valeurs
    de tuple ayant la première valeur en commun])]
            len_tupl_list_group -- taille de la liste de sortie
    """

    tupl_list_group = [(n, list(list(zip(*g))[1])) for n, g in it.groupby(
            tupl_list, op.itemgetter(0))]
    # print(f'tupl_list_group =\n{tupl_list_group}\n')
    len_tupl_list_group = len(tupl_list_group)

    return tupl_list_group, len_tupl_list_group

# %%


def distance(array_ref, array_pointtest, ind_lin_pttest, ind_col_pttest,
             sr_height_array_ref, metrique='euclidean'):
    """Retourne la distance de chaque point test aux pts de références.

    Keywords arguments:
        array_ref -- tableau des points de référence
        array_pointtest -- tableau des points de test
        ind_lin_pttest -- indice des lignes des valeurs des points test
        ind_col_pttest -- liste des indices des colonnes des valeurs des
        points test
        sr_height_array_ref -- ensemble du range de la hauteur du tableau de
        référence
        metrique
    """

#        print(f'array_ref[0]=\n{array_ref[0]}\nshape = {array_ref[0].shape} ')
#        test_pttest = array_pointtest[ind_lin_pttest,ind_col_pttest]
#        print\
#        (f'array_pointtest[ind_lin_pttest,ind_col_pttest]=\n{test_pttest}\n\
#         shape = {test_pttest.shape} ')

    if metrique == 'euclidean':
        distances = [np.linalg.norm(array_ref[i]-array_pointtest[
                ind_lin_pttest, ind_col_pttest]) for i in sr_height_array_ref]
    elif metrique == 'braycurtis':
        distances = [sci_dist.braycurtis(array_ref[i], array_pointtest[
                ind_lin_pttest, ind_col_pttest]) for i in sr_height_array_ref]
    elif metrique == 'cityblock':
        distances = [sci_dist.cityblock(array_ref[i], array_pointtest[
                ind_lin_pttest, ind_col_pttest]) for i in sr_height_array_ref]
    elif metrique == 'canberra':
        distances = [sci_dist.canberra(array_ref[i], array_pointtest[
                ind_lin_pttest, ind_col_pttest]) for i in sr_height_array_ref]
    elif metrique == 'chebyshev':
        distances = [sci_dist.chebyshev(array_ref[i], array_pointtest[
                ind_lin_pttest, ind_col_pttest]) for i in sr_height_array_ref]
    elif metrique == 'cosine':
        distances = [sci_dist.cosine(array_ref[i], array_pointtest[
                ind_lin_pttest, ind_col_pttest]) for i in sr_height_array_ref]
    elif metrique == 'correlation':
        distances = [sci_dist.correlation(array_ref[i], array_pointtest[
                ind_lin_pttest, ind_col_pttest]) for i in sr_height_array_ref]
    elif metrique == 'minkowski':
        distances = [sci_dist.minkowski(array_ref[i], array_pointtest[
                ind_lin_pttest, ind_col_pttest]) for i in sr_height_array_ref]
    elif metrique == 'jaccard':
        distances = [sci_dist.jaccard(array_ref[i], array_pointtest[
                ind_lin_pttest, ind_col_pttest]) for i in sr_height_array_ref]

    return distances

# %%


def k_nn(pointtest, list_references, K, ind_col_quali=[], metrique='euclidean',
         imputation=False):
    """Retourne un tableau imputé, calcule les K plus proches voisins
    d'un point test, et retourne les indices des lignes et les
    distances à ceux-ci.

    Keywords arguments:
        pointtest -- liste 2d ou 2darray des points test;
        pour l'instant les points test n'ont pas de variables qualitatives,
        mais doivent avoir le MÊME nombre de modalités quantitatives que
        les valeurs de référence.
        Il faudrait modifier le programme de façon à ce qu'il puisse
        éliminer d'éventuelles modalités qualitatives des points test
        list_references -- liste 2d des points de référence
        ind_col_quali -- liste des indices des colonnes des variables
        qualitatives
        K -- rang du K plus proche voisin voulu pour le calcul de distance
        metrique -- {'braycurtis', 'cityblock', 'canberra', 'chebyshev',
                     'cosine', 'correlation', 'minkowski', 'jaccard'},
        optional (default='euclidean')
        imputation -- True si on veut déclencher l'imputation des valeurs Nan

    Return:
        imputed_points -- nd_array des points considérés comme imputés
        ind_lin_k_nearest -- indices des lignes des k plus proches voisins
        distances_k_nearest -- k distances les plus proches

    """
    # print(f'list_references =\n{list_references}\n')

    # Dimensions de la table des points tests
    height_pointtest, width_pointtest = table_dimension_2d(pointtest)

    # print(f'nb_ligne_pttest = {height_pointtest}')
    # print(f'nb_col_pttest = {width_pointtest}\n')

    # ## Détection des lignes avec nan

    # tab ref sans colonne quali
    list_ref_quanti = remove_lin_col(list_references, ind_col=ind_col_quali)

    # conversion de tab_ref en array sans valeurs qualitatives
    array_ref_quanti = conversion_array(
            list_ref_quanti, new_shape_lin=table_dimension_2d(
                    list_references)[0])
    # print(f'array_ref_quanti =\n{array_ref_quanti}\n')

    # conversion de la liste pointtest en array
    array_pointtest = conversion_array(pointtest,
                                       new_shape_lin=height_pointtest)
    # print(f'\narray_pointtest=\n{array_pointtest}\narray_pointtest.shape =\n\
# {array_pointtest.shape}')

    # coordonnées des valeurs nan dans points_test
    # ind colonnes avec nan des points_test
    # coordonnées des valeurs non nan dans points_test
    coord_pttest_nan, _, ind_col_pttest_nan, coord_pttest_no_nan = \
        detecter_nan(array_pointtest)

    # print(f'coord_pttest_nan = \n{coord_pttest_nan}\n')
    # print(f'ind_col_pttest_nan = \n{ind_col_pttest_nan}\n')
    # print(f'coord_pttest_no_nan = \n{coord_pttest_no_nan}\n')

    # ind ligne avec nan de la table de reference
    ind_lin_tab_ref_nan = detecter_nan(array_ref_quanti)[1]
    # print(f'ind_lin_tab_ref_nan = \n{ind_lin_tab_ref_nan}\n')

    # Suppression des lignes avec nan dans la table de référence:
    array_ref_no_nan = remove_lin_col(array_ref_quanti,
                                      ind_lin=ind_lin_tab_ref_nan)

    # Reshape en array
    array_ref_no_nan = conversion_array(
            array_ref_no_nan, new_shape_col=table_dimension_2d(
                    array_ref_quanti)[1])
    # print(f'array_ref_no_nan =\n{array_ref_no_nan}\n')

    # ### Distance des points de référence aux points de test

    array_ref_tronq = list()
    # Rassemblement des K plus faibles distances
    distances_k_nearest_all = list()
    # Rassemblement de tous les indices des lignes des K plus faibles distances
    ind_lin_k_nearest_all = list()

    # Données des valeurs nan des points test à traiter
    coord_pttest_nan_grouped, len_coord_pttest_nan_grouped = \
        group_indcol_tuple(coord_pttest_nan)

    # Indices des lignes des points test
    ind_lin_pttest_nan = [coord_pttest_nan_grouped[i][0] for i in set_range(
            len_coord_pttest_nan_grouped)]
    # print(f'Indice des lignes des points test "ind_lin_pttest_nan" =\n\
# {ind_lin_pttest_nan}\n')

    # Indices des colonnes des points test
    ndlist_ind_col_pttest_nan = [coord_pttest_nan_grouped[i][1] for i in
                                 set_range(len_coord_pttest_nan_grouped)]
    # print(f'Indice des colonnes des points test
    # "ndlist_ind_col_pttest_nan"=\n\{ndlist_ind_col_pttest_nan}\n')

    # Données des valeurs numériques des points test à traiter
    coord_pttest_no_nan_grouped, len_coord_pttest_no_nan_grouped =\
        group_indcol_tuple(coord_pttest_no_nan)

    # Points à imputer après calculs
    imputed_points = cp.deepcopy(array_pointtest)  # deep copy

    if imputation:
        # Moyennes des valeurs du tableau de référence par colonnes
        moyenne_colonne_ref = array_ref_no_nan.mean(axis=0)

    for j in set_range(len_coord_pttest_no_nan_grouped):
        # Indice de la ligne du point test en cours de traitement
        ind_lin_pttest_no_nan = coord_pttest_no_nan_grouped[j][0]
        # print('Indice de la ligne du point test en cours:')
        # print(f'ind_lin_pttest_no_nan = {ind_lin_pttest_no_nan}\n')

        # Indice de la colonne du point test en cours de traitement
        ind_col_pttest_no_nan = coord_pttest_no_nan_grouped[j][1]
        # print('Indice de la colonne du point test en cours:')
        # print(f'ind_col_pttest_no_nan = {ind_col_pttest_no_nan}\n')

        # Donne un tableau des points de référence tronqué des colonnes de
        # mêmes indices que les colonnes du pointtest qui contiennent
        # au moins un nan.
        array_ref_tronq = array_ref_no_nan[:, ind_col_pttest_no_nan]
        height_array_ref_tronq = len(array_ref_tronq)
        sr_height_array_ref_tronq = set_range(height_array_ref_tronq)

        print('\n', 79*'-', '\n')
        print(f'tour = {j+1}\n')
        # print(f'array_ref_tronq=\n{array_ref_tronq}\n')
        # print(f'shape(array_ref_tronq) = {np.shape(array_ref_tronq)}\n')

        # Distances de chaque point test aux pts de références:
        list_distances = distance(
                array_ref_tronq, array_pointtest, ind_lin_pttest_no_nan,
                ind_col_pttest_no_nan,
                sr_height_array_ref=sr_height_array_ref_tronq,
                metrique=metrique)
        # print(f'distances {metrique} =\n{list_distances}\n')

        # ## Classement des distances:
        # TROUVER les INDICES des lignes de la table de référence, pour les K
        # plus faibles distances au point test!

        # Distances classées des points de références aux points test
        distances_sorted = sorted(list_distances)

        # Indices classés des lignes du tableau de référence qui correspondent
        # aux distances des points de référence aux points test:
        indices_distances = [indx for dstncs_srtd in set(distances_sorted)
                             for indx, vl in enumerate(list_distances)
                             if vl == dstncs_srtd]
        # print(f'indices_distances =\n{indices_distances}\n')

        # On ne garde que les K plus faibles distances:
        distances_k_nearest = distances_sorted[:K]
        # print(f'distances_k_nearest =\n{distances_k_nearest}\n')

        # Placement de toutes les distances dans une liste
        distances_k_nearest_all.append(distances_k_nearest)

        # Indices des lignes du tableau de référence qui correspondent aux K
        # plus faibles distances des points de référence aux points test:
        ind_lin_k_nearest = sorted(indices_distances[:K])
        # print(f'ind_lin_k_nearest =\n{ind_lin_k_nearest}\n')

        # Placement de tous les indices dans une liste
        ind_lin_k_nearest_all.append(ind_lin_k_nearest)

        if imputation:
            # Ne fonctionne pas !
            # imputed_points = imputation_knn(
            #        imputed_points, moyenne_colonne_ref, array_ref_no_nan,
            #        len_coord_pttest_nan_grouped, ndlist_ind_col_pttest_nan,
            #        array_ref_quanti, ind_lin_pttest_nan,
            #        ind_lin_pttest_no_nan, ind_lin_k_nearest)

            # # Imputation des valeurs manquantes
            # Calcul des valeurs d'imputation
            for m in set_range(len_coord_pttest_nan_grouped):
                # Si toutes les colonnes d'une ligne sont nan, on calcule les
                # moyennes entre toutes les valeurs des colonnes des points de
                # référence afin d'imputer le point
                if len(ndlist_ind_col_pttest_nan[m]) ==\
                        np.shape(array_ref_quanti)[1]:
                            imputed_points[ind_lin_pttest_nan[m]][:] =\
                                moyenne_colonne_ref[:]

                # Conditionner l'indice de la colonne à traiter de façon à ce
                # qu'on ne traite que les points de mêmes indices de ligne à
                # chaque tour.
                if ind_lin_pttest_nan[m] == ind_lin_pttest_no_nan:
                    nb_nan_in_col = len(ndlist_ind_col_pttest_nan[m])
                    # print(
                    # f'nombre de nan dans la colonne = {nb_nan_in_col}\n')

                    for p in set_range(nb_nan_in_col):
                        # Calcul de la moyenne
                        valeur_imputation = np.mean(
                                [array_ref_no_nan[n][
                                        ndlist_ind_col_pttest_nan[m][p]] for n
                                    in set(ind_lin_k_nearest)])
                        # print(f'Coordonnées pt test = \
        # <{ind_lin_pttest_nan[m]},{ndlist_ind_col_pttest_nan[m][p]}>')
                        # print(f'valeur d imputation = {valeur_imputation}\n')

                        # On remplace chaque nan par les valeurs d'imputation
                        imputed_points[ind_lin_pttest_nan[m]][
                                ndlist_ind_col_pttest_nan[m][p]] =\
                            valeur_imputation

    # print(f'ind_lin_k_nearest_all =\n{ind_lin_k_nearest_all}\n')
    # print(f'distances_k_nearest_all =\n{distances_k_nearest_all}\n')
    # print(f'imputed_points =\n{imputed_points}\n')

    return imputed_points, ind_lin_k_nearest_all, distances_k_nearest_all
