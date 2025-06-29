import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import pulp

# ---------- Authentication ----------
PASSWORD = "1928"
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if not st.session_state.auth_ok:
    st.markdown("## 🔒 Accès Sécurisé")
    with st.form("login_form"):
        pw = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Accéder"):
            if pw == PASSWORD:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
    st.stop()

st.title("🎯 EuroMillions – Prédictions (Algo Perio V2-LP)")

# ---------- File Upload ----------
uploaded_file = st.file_uploader("Téléversez l'historique Euromillions (Excel)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    lambda_decay = 0.20
    draws = df[[f"boule_{i}" for i in range(1,6)]].values.tolist()
    n = len(draws)
    weights = np.exp(-lambda_decay * np.arange(n)[::-1])
    weights /= weights.sum()
    boule_cnt = Counter()
    for w, draw in zip(weights, draws):
        for b in draw:
            boule_cnt[b] += w
    total_boule = sum(boule_cnt.values()) or 1
    p = {i: boule_cnt[i]/total_boule for i in range(1,51)}
    prob = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (range(6), range(1,51)), cat="Binary")
    prob += pulp.lpSum(p[i] * x[g][i] for g in range(6) for i in range(1,51))
    for g in range(6):
        prob += pulp.lpSum(x[g][i] for i in range(1,51)) == 5
    for i in range(1,51):
        prob += pulp.lpSum(x[g][i] for g in range(6)) <= 2
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    grids = []
    for g in range(6):
        nums = sorted([i for i in range(1,51) if x[g][i].value() == 1])
        grids.append(nums)
    stars = df[["etoile_1","etoile_2"]].values.tolist()
    star_cnt = Counter()
    for w, sdraw in zip(weights, stars):
        for e in sdraw:
            star_cnt[e] += w
    total_star = sum(star_cnt.values()) or 1
    star_p = {i: star_cnt[i]/total_star for i in range(1,13)}
    top2 = sorted(star_p, key=star_p.get, reverse=True)[:2]
    candidates = sorted(star_p, key=star_p.get, reverse=True)[2:6]
    import itertools
    star_pairs = [sorted(top2)]
    for pair in itertools.combinations(candidates, 2):
        if len(star_pairs) < 6:
            star_pairs.append(sorted(pair))
    st.subheader("🎯 Système 6 grilles optimisé")
    for i, nums in enumerate(grids):
        stars_sel = star_pairs[i]
        st.write(f"Grille {i+1} : {', '.join(map(str,nums))}    ⭐ {stars_sel[0]} & {stars_sel[1]}")
    if st.checkbox("Afficher les probabilités boules/étoiles"):
        df_b = pd.DataFrame(sorted(p.items(), key=lambda x:-x[1])[:15], columns=["Numéro","Probabilité"])
        df_s = pd.DataFrame(sorted(star_p.items(), key=lambda x:-x[1])[:6], columns=["Étoile","Probabilité"])
        col1, col2 = st.columns(2)
        col1.table(df_b)
        col2.table(df_s)
else:
    st.info("Veuillez téléverser votre fichier Excel pour générer les grilles.")
