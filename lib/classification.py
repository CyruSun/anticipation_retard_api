#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:35:54 2018

@author: Mahery
"""
import os.path as os_path
import sys
# import programmation as prgrm
import numpy as np

from sklearn import preprocessing as prprcss, cluster as clst

# Recherche et enregistrement de CHEMIN de module à importer
pckg_pth_explrtn = os_path.abspath(
                    os_path.join('../../Exploration_nettoyage_donnees/lib'))
pckg_pth_prgrmmtn = os_path.abspath(
                    os_path.join('../../programmation/lib'))

pckg_pth = [pckg_pth_explrtn, pckg_pth_prgrmmtn]
# print(f'module_path:')

# Ajout des chemins dans le path du système s'ils n'y sont pas encore
for pckg in pckg_pth:
    # print(f"{pckg}")
    if pckg not in sys.path:
        sys.path.append(pckg)
# print()
# print(f"sys.path:\n{sys.path}\n")

import programmation as prgrm


def fit_label_encode(df, ind_quali, n_clusters):
    """Compute k-means clustering à partir des labels encodés d'une df.

    Keywords arguments:
        df -- dataframe dont on encodera les labels de chaque attributs en
        entier
        ind_quali -- indices des variables qualitatives
        n_clusters -- nombre de cluster
    Returns:
        fit_kmean -- Compute k-means clustering.
    """

    # Création d'un objet LabelEncoder object et application à chaque feature
    # (caractéristique) de la df

    # 1. instanciation
    # Encodage de chaque type de variables(label) des attributs en des valeurs
    # entières entre 0 et n_classes-1:
    # CHAQUE label a sa PROPRE classe
    le = prprcss.LabelEncoder()

    # 2.APPLIQUER A UN TABLEAU ET TRANSFORMER
    # On utilise df.apply() pour appliquer le.fit_transform sur toutes les
    # valeurs des colonnes de la df
    df_transformed_int = df.apply(le.fit_transform)

    enc = prprcss.OneHotEncoder(categorical_features=ind_quali)
    mat_creuse = enc.fit_transform(df_transformed_int)

    # kmean qui sépare en n_clusters classes.
    kmean = clst.KMeans(n_clusters=n_clusters)

    # kmeans clustering pour la matrice
    fit_kmean = kmean.fit(mat_creuse)

    return fit_kmean


def dist_centroid_label_encode(df, ind_quali, n_clusters):
    """Retourne les distances aux centroides déduites par KMeans des labels
    encodés d'une df.

    Keywords arguments:
        df -- dataframe dont on encodera les labels de chaque attributs en
        entier
        ind_quali -- indices des variables qualitatives
        n_clusters -- nombre de cluster
    Returns:
        dist_centroid -- distances au centroide
    """

    # Création d'un objet LabelEncoder object et application à chaque feature
    # (caractéristique) de la df

    # 1. instanciation
    # Encodage de chaque type de variables(label) des attributs en des valeurs
    # entières entre 0 et n_classes-1:
    # CHAQUE label a sa PROPRE classe
    le = prprcss.LabelEncoder()

    # 2.APPLIQUER A UN TABLEAU ET TRANSFORMER
    # On utilise df.apply() pour appliquer le.fit_transform sur toutes les
    # valeurs des colonnes de la df
    df_transformed_int = df.apply(le.fit_transform)

    enc = prprcss.OneHotEncoder(categorical_features=ind_quali)
    mat_creuse = enc.fit_transform(df_transformed_int)

    # kmean qui sépare en n_clusters classes.
    kmean = clst.KMeans(n_clusters=n_clusters)

    # Chaque dimension est la distance au centre du cluster
    dist_centroid = kmean.fit_transform(mat_creuse)

    return dist_centroid


def predict_cluster_encode(df, ind_quali, kmean):
    """Retourne la classe la plus proche à laquelle chaque échantillon
    encodé d'une df appartient.


    Keywords arguments:
        df -- dataframe dont on encodera les labels de chaque attributs
        en entier
        ind_quali -- indices des variables qualitatives
        kmean -- K-Means clustering

    Returns:
        cluster_prediction -- clusters les plus proches pour chaque
        échantillon
    """

    # Création d'un objet LabelEncoder object et application à chaque feature
    # (caractéristique) de la df

    # 1. instanciation
    # Encodage de chaque type de variables(label) des attributs en des valeurs
    # entières entre 0 et n_classes-1:
    # CHAQUE label a sa PROPRE classe
    le = prprcss.LabelEncoder()

    # 2.APPLIQUER A UN TABLEAU ET TRANSFORMER
    # On utilise df.apply() pour appliquer le.fit_transform sur toutes les
    # valeurs des colonnes de la df
    df_transformed_int = df.apply(le.fit_transform)

    enc = prprcss.OneHotEncoder(categorical_features=ind_quali)
    mat_creuse = enc.fit_transform(df_transformed_int)

    # Classes les plus proches
    cluster_prediction = kmean.fit_predict(mat_creuse)

    return cluster_prediction


def standardisaton_nd(lst_xs, x_cmpt=np.zeros([])):
    """Standardisation de plusieurs variables.
    Keyword
    lst_xs -- list of data used to scale along the features axis
    x_cmpt -- The data used to compute the mean and standard deviation
    used for later scaling along the features axis
    """
    x_cmpt = prgrm.reshape_one_feat(x_cmpt)
    lst_scaled = []
    for x in lst_xs:
        x = prgrm.reshape_one_feat(x)
        # Uses x as data to compute the mean and std as default
        if x_cmpt == np.zeros([]):
            lst_scaled.append(prprcss.StandardScaler().fit_transform(x))
        else:
            lst_scaled.append(
                prprcss.StandardScaler().fit(x_cmpt).transform(x))
    return lst_scaled


def prnt_perf_rgrssn_mdl(rgrssn, scr):
    """Affiche les performances d'un modèle de régression vis à vis d'un
    hyperparamètre et retourne le meilleur coefficient de régularisation.

    Keyword arguments:
        rgrssn -- mdl_selct.GridSearchCV
        scr -- type de métrique de la régression
    """

    # coefficient le plus performant
    best_rglrstn = rgrssn.best_params_
    print(f"coefficient le plus performant:\n{rgrssn.best_params_}\n")
    # Affichage des performances des modèles
    print("Performances des modèles :")
    for mean, std, params in zip(
        rgrssn.cv_results_['mean_test_score'],  # score moyen
        rgrssn.cv_results_['std_test_score'],  # écart-type du score
        rgrssn.cv_results_['params']  # valeur de l'hyperparamètre
    ):
        print("\t%s = %0.10f (+/-%0.10f) for %r" % (
            scr,  # critère utilisé
            mean,  # score moyen
            std * 2,  # barre d'erreur
            params  # hyperparamètre
        ))
    return best_rglrstn
