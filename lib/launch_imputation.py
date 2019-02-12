#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 16 août 2018

@author: Mahery
'''

# import time
import pandas as pd
import numpy as np
import os
import k_ppv as knn
# import knn_mod_beta as knn

# from pandas import * # pour PyDev
# from root.analyse_donnees import knn_mod # pour PyDev
from random import sample

nan = float(np.nan)

# %%

# Ecriture d'une DataFrame pour le test

# knn n'accepte que les 2d_array
#val_1d = np.array([[2.6, -12, 0, 'z', nan, 28.57, 28.57, 0.0,
#                    0.018000000000000002, 64.29, 14.29, 3.6, 3.57, 0.0, 0.0,
#                    0.0, 0.0214, 0.0, 0.0012900000000000001, 14.0, 14.0]])
#
#val_nd = np.array([[
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan],
#       [0.0, 0.0, 0.0, 'd', 2243.0, 28.57, 28.57, 0.0, 0.018000000000000002,
#        64.29, 14.29, 3.6, 3.57, 0.0, 0.0, 0.0, 0.0214, 0.0,
#        0.0012900000000000001, 14.0, 14.0],
#       [0.0, 0.0, 0.0, 'b', 1941.0, 17.86, 0.0, 0.0, 0.0, 60.71, 17.86,
#        7.1, 17.86, 0.635, 0.25, 0.0, 0.0, 0.071, 0.0012900000000000001,
#        0.0, 0.0],
#       [0.0, 0.0, 0.0, 'b', 1941.0, 17.86, 0.0, 0.0, 0.0, 60.71, 17.86,
#        7.1, 17.86, 0.635, 0.25, 0.0, 0.0, 0.071, 0.0012900000000000001,
#        0.0, 0.0],
#       [0.0, 0.0, 0.0, 'd', 2540.0, 57.14, 5.36, nan, nan, 17.86, 3.57,
#        7.1, 17.86, 1.22428, 0.48200000000000004, nan, nan,
#        0.14300000000000002, 0.00514, 12.0, 12.0],
#       [0.0, 0.0, 0.0, nan, 1552.0, 1.43, nan, nan, nan, 77.14, nan, 5.7,
#        8.57, nan, nan, nan, nan, nan, nan, nan, nan],
#       [0.0, 0.0, 0.0, nan, 1933.0, 18.27, 1.92, nan, nan, 63.46, 11.54,
#        7.7, 13.46, nan, nan, nan, nan, 0.038, 0.00346, nan, nan],
#       [0.0, 0.0, 0.0, nan, 1490.0, nan, nan, nan, nan, 80.0, nan, nan,
#        8.89, nan, nan, nan, 0.0027, 0.044000000000000004, nan, nan, nan],
#       [2.0, 0.0, 0.0, 'c', 1833.0, 18.75, 4.69, nan, nan, 57.81, 15.62,
#        9.4, 14.06, 0.1397, 0.055, nan, nan, 0.062, 0.004220000000000001,
#        7.0, 7.0],
#       [0.0, 0.0, 0.0, nan, 2406.0, 37.5, 22.5, nan, nan, 55.0, 42.5,
#        7.5, 5.0, nan, nan, nan, nan, 0.05, 0.01125, nan, nan],
#       [0.0, 0.0, 0.0, nan, 3586.0, 100.0, 7.14, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan]])
#
#val = val_1d
#val = val_nd
#df_pointtest = pd.DataFrame(
#        val, columns=[
#                'additives_n', 'ingredients_from_palm_oil_n',
#                'ingredients_that_may_be_from_palm_oil_n',
#                'nutrition_grade_fr', 'energy_100g', 'fat_100g',
#                'saturated-fat_100g', 'trans-fat_100g', 'cholesterol_100g',
#                'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
#                'proteins_100g', 'salt_100g', 'sodium_100g', 'vitamin-a_100g',
#                'vitamin-c_100g', 'calcium_100g', 'iron_100g',
#                'nutrition-score-fr_100g', 'nutrition-score-uk_100g'])
#df_pointtest
# %%

#data_file = 'data_nutri_sup_20_pct_pca_ref_no_outlier.csv'
#ind_col_quali = [3]
#ind_col_choisies = None
#sep = '\t'
#nb_lignes_reference = 10

# %%


def launch_knn(
        data, df_pointtest, K, folder_name="data", fromcsv=True,
        ind_col_quali=[], ind_col_choisies=None, sep='\t', imputation=False,
        metrique='euclidean', nb_lignes_reference=10):

    """Lecture de parties choisies du fichier de données, ainsi que
    tronquage afin d'épargner les ressources du calculateur. Tronquage
    aussi des points test afin de ne retenir que les variables qui
    correspondent à celles de la data de référence.
    Lancement de l'imputation.

    Keywords arguments:
        data -- nom du fichier csv ou de la dataframe des valeurs de
        référence dans l'emplacement du dossier des données
        df_pointtest -- dataframe des valeurs à imputer
        K -- rang du K plus proche voisin voulu pour le calcul de distance
        folder_name -- folder containing the data file
        fromcsv -- option si la data est un fichier csv
        ind_col_quali -- liste des indices des colonnes des variables
        qualitatives
        ind_col_choisies -- liste des indices des colonnes sélectionnées
        pour l'analyse
        sep -- séparateur de champs dans le fichier csv
        imputation -- déclenchement de l'imputation
        metrique -- type de distance
        nb_lignes_reference -- nombre de ligne de la data de référence

    Returns:
        imputed_points -- array imputé
        ind_lin_k_nearest_all
        distances_k_nearest
    """

    # start_time = time.time()  # Pour calculer le temps d'exécution

    # data_file = input("Entrer le nom du fichier placé dans le répertoire\
    # data:\n")

    # we get the right path, directory name of pathname path:
    directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # Assignation de la dataframe de réfé&rence
    dtfrm = data
    if fromcsv:
        # Chargement du fichier de données
        # with this path, we go inside the folder in `folder_name` and get
        # the file in 'data':
        path_to_file = os.path.join(directory, folder_name, data)
        # print(f'path_to_file=\n{path_to_file}\n')

        # Lecture que pour certaines colonnes sélectionnées
        # On précise la première colonne en tant que labels des lignes
        dtfrm = pd.read_csv(
                path_to_file, index_col=0, sep=sep, usecols=ind_col_choisies,
                engine='python')
    # print(f'dtfrm.head() = {dtfrm.head()}\n')

    # Conversion de la dataframe pointtest en liste
    list_pointtest = df_pointtest.values.tolist()
    # print(f'list_pointtest =\n{list_pointtest}\n')

    # Conversion de la dataframe pandas en liste
    list_references = dtfrm.values.tolist()

    height_dtfrm = dtfrm.shape[0]  # hauteur de la dataframe
    # Lignes prises au hasard
    lst_indx_rndm = sample(range(height_dtfrm), nb_lignes_reference)
    # On utilise une liste de référence tronquée afin d'éviter une pression
    # trop importante sur les capacités de la machine
    list_references_tronq = [list_references[i][:] for i in lst_indx_rndm]
    # print(f'list_references_tronq =\n{list_references_tronq}\n')

    # Tronquage des points test l'indice des colonnes à supprimer doit être une
    # LISTE
    list_test_tronq = knn.remove_lin_col(list_pointtest, ind_col=ind_col_quali)
    array_test_tronq = knn.conversion_array(list_test_tronq, new_shape_col=len(
            list_references_tronq[0]) - len(ind_col_quali))

    # On convertit le tableau en réel
    float_array_test_tronq = array_test_tronq.astype(float)

    # Points imputés
    imputed_points, ind_lin_k_nearest_all, distances_k_nearest = knn.k_nn(
            float_array_test_tronq, list_references_tronq, K, ind_col_quali,
            metrique=metrique, imputation=True)

    # K plus proches voisins
    # df_ppv = nutritional_data.iloc[ind_lin_k_nearest_all,:]

    # print(f'Fin\n')
    # print("--- %s seconds ---" % (time.time() - start_time))
    return imputed_points, ind_lin_k_nearest_all, distances_k_nearest
# , df_ppv

# %%


#path = '../../data/'
#
#launch_knn(
#        df_pointtest=df_pointtest, ind_col_quali=ind_col_quali, K=5,
#        data=data_file, nb_lignes_reference=10,
#        ind_col_choisies=None, sep='\t', imputation=False,
#        metrique='braycurtis')
