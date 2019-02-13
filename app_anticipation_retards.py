# -*- coding: utf-8 -*-
'''
Created on 21 janvier 2019

@author: Mahery Andriana
'''

# %%
import numpy as np
import pandas as pd
import time
import os.path as os_path
import sys
import pickle
import re

from flask import Flask, render_template, request
from sklearn import preprocessing as prprcssng, model_selection as mdl_slctn
from sklearn.linear_model import Ridge as rdg

# %%
app = Flask(__name__)

# %% Recherche et enregistrement de CHEMIN de module à importer
# Chemin des packages
pckg_pth = os_path.abspath(os_path.join('./lib'))
if pckg_pth not in sys.path:
    sys.path.append(pckg_pth)
pth_dt = './data/'  # Chemin des données
# print(f"module_path:\n")
# print(f"{pckg_pth}\n")


# %%
import k_ppv as knn
import programmation as prgrm
import launch_imputation as impute
import classification as clssfctn

# %%
# nan de type nombre et non de type str
nan = float(np.nan)


# %%
@app.route('/')  # décorateur permets d'entrer des métadonnées
def home():
    return render_template('pages/home.html')


# %%
def predict_delay(orgn, dstntn, mnth, d_mnth, dp_hr, crrr):
    """Calcule une prédiction de retards pour une date, une heure de
    départ, un transporteur et un trajet donné.

    Keywords arguments:
        mnth -- numero du mois
        d_mnth -- jour du mois
        dp_hr -- heure de départ
        crrr -- identifiant de la compagine aérienne
        orgn -- aéroport de départ
        dstntn -- aéroport d'arrivée
    """

    start_time_origin = time.time()  # Pour calculer le temps d'exécution

    # chargement des données
#    start_time = time.time()  # Pour calculer le temps d'exécution
    # Df de départ
    dtfrm = pd.read_pickle(pth_dt + 'dtfrm_smpl_flght_1_rdct.pkl')
#    print("chargement des données:\n--- %s seconds ---\n" % (
#            time.time() - start_time))

    # index d'entraînement (sans les retards au départ et à l'arrivée)
    lst_X_clmns = [
            'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CARRIER', 'ORIGIN',
            'TAXI_OUT', 'TAXI_IN', 'DISTANCE', 'DEST', 'CARRIER_DELAY',
            'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY',
            'LATE_AIRCRAFT_DELAY', 'DEP_HOUR']
    ln_X_clmns = len(lst_X_clmns)

    lst_quali = ['CARRIER', 'ORIGIN', 'DEST']  # Features qualitatifs

    # la standardisation ne concerne que les var quanti
    lst_x_clmns_quanti = [
            'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'TAXI_OUT', 'TAXI_IN',
            'DISTANCE', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY',
            'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'DEP_HOUR']

    # df échantillon
    smpl_sz = 5000  # taille de l'echantillon
    dtfrm_smpl = dtfrm.sample(smpl_sz)
    dtfrm_smpl.sort_index(inplace=True)
    dtfrm_rgrssn = dtfrm_smpl.loc[:, lst_X_clmns]  # dataframe de la régression
    dtfrm_qlttf = dtfrm_rgrssn.loc[:, lst_quali]  # df des valeurs qualitatifs
    # %% df qualitatif utilisateur
    dct_usr = prgrm.zip_to_dict(lst_quali, [[crrr], [orgn], [dstntn]])
    dtfrm_usr_qlttf = pd.DataFrame(dct_usr)

    # %% Encodage des variables
    x_qlttf = dtfrm_qlttf.values  # Conversion pour encodage
    # Encodeur
    enc = prprcssng.OneHotEncoder(handle_unknown='ignore')
    ohe = enc.fit(x_qlttf)
    # encodage binarisation
    bnr_x_qlttf = ohe.transform(x_qlttf).toarray()
    # encodage des données utilisateurs
    arr_usr = prgrm.reshape_one_sample(dtfrm_usr_qlttf.values)
    bnr_usr = ohe.transform(arr_usr).toarray()

    # %%Recherche des K-NN du vol de l'utilisateur
    # lignes des vols les plus proches
    K = 5
    lst_ind_k_nrst = knn.k_nn(bnr_usr, bnr_x_qlttf, K)[1]
    # %% Standardisation des données
    # Conversion en array numpy des données quantitatives
    x_quanti = dtfrm_rgrssn.loc[:, lst_x_clmns_quanti].values
    # Standardiseur
    x_stdrd = prprcssng.StandardScaler().fit_transform(x_quanti)
    # dtfrm des valeurs standardisées
    dct_stdrd = prgrm.zip_to_dict(lst_x_clmns_quanti, x_stdrd.transpose())
    dtfrm_rgrssn_stdrd = pd.DataFrame(data=dct_stdrd)
    # %% Imputation du vol utilisateur
    # lst des valeurs entrées par l'utilisateur
    lst_vlrs = [nan for i in knn.set_range(ln_X_clmns)]
    lst_vlrs[0], lst_vlrs[1], lst_vlrs[3], lst_vlrs[4], lst_vlrs[8],\
        lst_vlrs[14] = [mnth, d_mnth, crrr, orgn, dstntn, dp_hr]

    # dtfrm de l'utilisateur
    dct = prgrm.zip_to_dict(lst_X_clmns, lst_vlrs)
    dtfrm_sim = pd.DataFrame(data=dct, index=[0])
    dtfrm_usr = dtfrm_sim.loc[:, lst_x_clmns_quanti]

    # Référence pour imputation
    dtfrm_rgrssn_rfrnc = dtfrm_rgrssn_stdrd.iloc[lst_ind_k_nrst[0], :]

    # Imputation
    imputed_X = impute.launch_knn(
            dtfrm_rgrssn_rfrnc, dtfrm_usr, K, fromcsv=False, imputation=True,
            nb_lignes_reference=K, folder_name=pth_dt)[0]

    # %% Régression Ridge
    # la cible est le retard à l'arrivée: ARR_DElAY
    y = np.asarray(dtfrm_smpl['ARR_DELAY'])
    x_rgrssn_stdrd = dtfrm_rgrssn_stdrd.values

    # Séparation en jeux de test et d'entraînement
#    start_time = time.time()  # Pour calculer le temps d'exécution
    x_train, x_test, y_train, y_test = mdl_slctn.train_test_split(
        x_rgrssn_stdrd, y, test_size=0.3)
#    print("--- %s seconds ---" % (time.time() - start_time))

    # Régression Ridge pour lambda
    lmbd = 5
    rgrssn = rdg().set_params(alpha=lmbd).fit(x_train, y=y_train)
# %%
    with open(pth_dt + 'dct_jrs.pkl', 'rb') as p:
        dct_jrs = pickle.load(p)

    # numero du jour de la semaine
    d_wk = dtfrm[
            (dtfrm['MONTH'] == mnth) & (dtfrm['DAY_OF_MONTH'] == d_mnth)][
                    'DAY_OF_WEEK'].iloc[0]

    jr = prgrm.keys_from_dct_val(dct_jrs, d_wk)  # Nom du jour

    # nom du mois
    with open(pth_dt + 'dct_nmr_mnth.pkl', 'rb') as p:
        dct_mnth = pickle.load(p)
    mnth_nm = prgrm.keys_from_dct_val(dct_mnth, mnth)

    # nom de la compagnie aérienne
    with open(pth_dt + 'dct_crrrs.pkl', 'rb') as p:
        dct_crrrs = pickle.load(p)
        crrr_nm = dct_crrrs[crrr]

# %% Prediction de retard en min
    rtrd_prdct = rgrssn.predict(imputed_X)
#    print(f"Retard prédit au départ de {dp_hr} heure de {orgn} à destination \
#          de {dstntn} pour la compagnie {crrr_nm} le {jr} {d_mnth} \
#          {mnth_nm} 2016:\n{rtrd_prdct[0]} min")
    print("Temps d'éxecution total--- %s seconds ---" % (
            time.time() - start_time_origin))

    return rtrd_prdct, jr, mnth_nm, crrr_nm


# %%
@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        orgn = request.form['orgn']
        dstntn = request.form['dstntn']
        date_dp = request.form['date']
#        mnth = request.form['mnth']
#        d_mnth = request.form['d_mnth']
        dp_hr = request.form['dp_hr']
        crrr = request.form['crrr']

        regexp = r"[2][0][1][6][-]+(?P<mois>\d+)+[-]+(?P<jour>\d+)"
        mnth = re.match(regexp, date_dp).group('mois')
        d_mnth = re.match(regexp, date_dp).group('jour')

        # conversion de type !!
        crrr = str(crrr)
        mnth = int(mnth)
        d_mnth = int(d_mnth)
        dp_hr = int(dp_hr)

        rtrd, nm_jr, nm_mnth, crrr_nm = predict_delay(orgn, dstntn, mnth,
                                                      d_mnth, dp_hr, crrr)
        rtrd = rtrd[0]
        return render_template(
                'pages/predict_delay.html', nom_mois=nm_mnth, nom_jour=nm_jr,
                heure=dp_hr, orgn=orgn, dstntn=dstntn, compagnie=crrr_nm,
                d_mnth=d_mnth, rtrd_prdt=round(rtrd,2))
    return render_template('pages/index.html')


# %%
# Fonctionnement pour exécution en ligne de commande
if __name__ == '__main__':
    app.run(debug=True, port=5000)
