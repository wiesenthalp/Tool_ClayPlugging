import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Modell zur Pfropfenbildung in bindigen Böden")
st.warning('Wiesenthal, P., Henke, S. Concept on plug development in jacked open-ended piles in clay considering total stresses. Acta Geotech. 20, 1019–1033 (2025). https://doi.org/10.1007/s11440-024-02455-0')
# -------------------------
# Eingabefunktionen
# -------------------------
def eingabe_bodenparameter():
    st.header("Bodenparameter")
    c_u = st.number_input("Undränierte Scherfestigkeit c_u (kPa)", value=10.0, min_value=0.0)
    gamma = st.number_input("Wichte γ' (kN/m³)", value=10.0, min_value=0.0)
    gamma_w = st.number_input("Wichte Wasser γ_w (kN/m³)", value=0.0, min_value=0.0, max_value=0.0)
    K_0 = st.number_input("K_0 (-)", value=1.0, min_value=0.0)
    N_c_max = st.number_input("Maximaler Widerstandsfaktor N_c", value=9.0, min_value=0.0)
    N_c_option = st.radio("Verlauf von N_c:", ["Hyperbolisch", "Konstant"])
    return c_u, gamma, gamma_w, N_c_max, N_c_option, K_0

def eingabe_pfahlparameter(c_u, gamma):
    st.header("Pfahlparameter")
    D = st.number_input("Durchmesser D (m)", value=0.5, min_value=0.0)
    L = st.number_input("Einbindelänge L (m)", value=10.0, min_value=0.0)
    t_mm = st.number_input("Wandstärke t (mm)", value=40.0, min_value=0.0)
    t = t_mm / 1000
    D_plug = D - 2*t
    A = np.pi * D**2 / 4
    A_ann = ((D/2)**2 - (D/2 - t)**2) * np.pi
    A_plug = (D/2 - t)**2 * np.pi
    U = np.pi * D
    U_plug = np.pi * (D - 2 * t)

    h_1_default = 2 * c_u / gamma
    return D, D_plug, L, t, A, A_ann, A_plug, U, U_plug, h_1_default

def eingabe_interaktion():
    st.header("Interaktion Pfahl-Boden")
    mu_raw = st.number_input("Reibungskoeffizient μ", value=0.1, min_value=0.00001, max_value=1.0)
    stressratio = st.number_input("Stress ratio zeta (Verhältnis von effektiver zu totaler Normalspannung)", value=1.0, min_value=0.00001)
    mu = mu_raw * stressratio
    return mu
    

def eingabe_pfropfenmodell(A_ann, A_plug, h_1_default):
    st.header("Modell Pfropfenbildung")
    st.info('Bilinearer Verlauf von IFR. ' \
    'Ermittlung der Pfropfenhöhe erfolgt durch Integration von IFR. ' \
    'Grenzwert wird iterativ über Gleichgewichtsbetrachtung bestimmt.')
    IFR_0_default = 1 + A_ann / A_plug
    IFR_0 = st.number_input("IFR_0 (-)", value=IFR_0_default)
    h_1 = st.number_input("h₁ (m)", value=h_1_default)
    return IFR_0, h_1

def eingabe_berechnungseinstellungen(case):
    st.header("Berechnungseinstellungen")
    n_steps_default = 501
    z_initial_default = 6.0
    threshold_default = 0.0001
    max_iterations_default = 100
    if case == 'long':
        n_steps = st.number_input("Anzahl Tiefeninkremente", value=n_steps_default, min_value=10, max_value=1000, step=10)
        z_initial = st.number_input("Startwert Tiefe volle Pfropfenbildung für Iteration", value=z_initial_default, min_value=0.0, max_value=1000.0, step=1.0)
        threshold= st.number_input("Threshold", value=threshold_default, min_value=0.0000001, max_value=1.0, step=.001)
        max_iterations= st.number_input("Begrenzung Anzahl Iterationen", value=max_iterations_default, min_value=1, max_value=1000, step=10)
        return n_steps, z_initial, threshold, max_iterations
    elif case == 'short':
        n_steps = st.number_input("Anzahl Tiefeninkremente", value=n_steps_default, min_value=10, max_value=1000, step=10)
        return n_steps, None, None, None

# -------------------------
# Sonstige Funktionen
# -------------------------

# Berechne Spitzendruck offener Pfahl
def cal_q_b_open(q_plug, q_ann, A, A_plug, A_ann):
    return (q_plug*A_plug+q_ann*A_ann)/A

def cal_IFR_from_ratio_qbopen_qbclosed(qb_open, qb_closed):
    x=qb_open/qb_closed
    IFR = -5.1298 * x**2 + 4.9121 * x + 0.3277
    return IFR

# -------------------------
# Hauptcode
# -------------------------
# Eingaben sammeln
c_u, gamma, gamma_w, N_c_max, N_c_option, K_0 = eingabe_bodenparameter()
D, D_plug, L, t, A, A_ann, A_plug, U, U_plug, h_1_default = eingabe_pfahlparameter(c_u, gamma)
mu = eingabe_interaktion()
#IFR_0, h_1 = eingabe_pfropfenmodell(A_ann, A_plug, h_1_default)

n_steps, z_initial, threshold, max_iterations = eingabe_berechnungseinstellungen(case='short')

# -------------------------
# Berechnung geschlossener Pfahl
# -------------------------
z = np.linspace(0, L, n_steps)

if N_c_option == "Konstant":
    N_c_z = np.full_like(z, N_c_max)
else:
    N_c_raw = 6.2 + (z/D) / (0.2454 * (z/D) + 0.4296)
    N_c_z_based_9 = np.minimum(N_c_raw, 9.0)
    N_c_z = N_c_z_based_9 / 9.0 *  N_c_max

q_b = z * gamma + N_c_z * c_u



# ------------------------------------------------------------------------
# Auswahl IFR Ermittlung
# ------------------------------------------------------------------------
st.header("Eingabe des IFR-Verlaufs")
ifr_mode = st.selectbox("Art der IFR-Eingabe wählen:", ["Widerstand"])

# Darstellung
if ifr_mode == "Widerstand":
    ratio = np.arange(0,1.01, 0.01)
    IFR_demo = np.zeros(len(ratio))
    for i in range(len(ratio)):
        IFR_demo[i] =  cal_IFR_from_ratio_qbopen_qbclosed(ratio[i], 1.0)
fig, axes = plt.subplots( figsize=(16, 8))
ax=axes
ax.plot(ratio, IFR_demo, label="closed-ended", color = 'black')
ax.set_xlabel("Ratio q_b [kN/m²]")
ax.set_ylabel("IFR z [m]")
ax.set_title("Pfahlspitzendruck")
ax.set_xlim(left=0, right=1)
ax.set_ylim(bottom=0)
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Berechne q_plug über h
h = np.zeros(n_steps)
q_plug = np.zeros(n_steps)
IFR = np.zeros(n_steps)
dh = np.zeros(n_steps)
for i in range(len(h)):
    if i == 0:
        h[i] = 0
        q_plug[i] = 0  # Startwert oben
        IFR[i] = None
        dh[i] = None
    else:
        # ------------------------------------------------------------------------
        # calculate h and IFR
        # ------------------------------------------------------------------------
        dz_local = z[i] - z[i-1]
        if ifr_mode == "Widerstand":
            qbopen = cal_q_b_open(q_plug[i-1], q_b[i-1], A, A_plug, A_ann)
            qbclosed = q_b[i-1]
            IFR[i] =  cal_IFR_from_ratio_qbopen_qbclosed(qbopen, qbclosed)
        dh[i] = IFR[i] * dz_local
        h[i] = h[i-1]+ dh[i]
        # ------------------------------------------------------------------------
        # plug resistance
        # ------------------------------------------------------------------------
        p = q_plug[i-1]
        # Exponentieller Anstieg
        delta_stress_exp = np.exp(4 * mu * K_0 * dz_local / D) * (4 * mu * K_0 / D * p + gamma)
        # Grenzwert für Schubspannung
        delta_stress_limit = gamma + c_u * U_plug / A
        if delta_stress_exp <= delta_stress_limit:
            q_plug[i] = p + (np.exp(4 * mu * K_0 * dz_local / D) - 1) * (p + gamma * D / (4 * mu * K_0))
        else:
            q_plug[i] = p + (gamma + c_u * U_plug / A) * dz_local
        # Grenzwert geschlossener Pfahl
        max_q = q_b[i]
        if q_plug[i]> max_q:
            q_plug[i] = max_q

# Umwandlung in NumPy-Array für weitere Verwendung
#q_plug = np.array([q_plug[i] for i in range(len(z))])

q_ann = q_b
q_b_open = cal_q_b_open(q_plug, q_ann, A, A_plug, A_ann)

# -------------------------
# Plot
# -------------------------
df_q_plug = pd.DataFrame({
    "Tiefe z [m]": z,
    "IFR [-]": IFR,
    "dh [m]": dh ,
    "h [m]": h ,
    "q_plug [kN/m²]": q_plug,
    "q_b_open [kN/m²]": q_b_open, 
    "q_b_closed [kN/m²]": q_b,
})


st.subheader("Ergebnisse")

fig, axes = plt.subplots(1,4, figsize=(16, 8))
ax=axes[0]
ax.plot(q_b, z, label="closed-ended", color = 'black')
ax.plot(df_q_plug["q_b_open [kN/m²]"], df_q_plug["Tiefe z [m]"], 
        label="open-ended", color = 'blue')
ax.set_xlabel("Spitzendruck q_b [kN/m²]")
ax.set_ylabel("Tiefe z [m]")
ax.set_title("Pfahlspitzendruck")
ax.set_xlim(left=0)
ax.set_ylim([z[0], z[-1]])
ax.grid(True)
ax.invert_yaxis()
ax.legend()
ax=axes[1]
ratio = q_plug / q_b
ax.plot(ratio, z, label="Pfropen/geschlossen", color = 'red')
ratio = q_b_open / q_b
ax.plot(ratio, z, label="offen/geschlossen", color = 'blue')
ax.set_ylim([z[0], z[-1]])
ax.set_xlabel("Anteil Spitzendruck [-]")
ax.set_title("Anteil Spitzendruck")
ax.grid(True)
ax.invert_yaxis()
ax.legend(loc = 'best')

ax=axes[2]
ax.plot(h, z, color = 'blue')
ax.set_ylim([z[0], z[-1]])
ax.set_xlabel("h [m]")
ax.set_title("Pfropfenhöhe")
ax.grid(True)
ax.invert_yaxis()
ax=axes[3]
ax.plot(IFR, z, label="q_b_open/q_b", color = 'blue')
ax.set_ylim([z[0], z[-1]])
ax.set_xlabel("IFR [-]")
ax.set_title("Incremental filling ratio")
ax.grid(True)
ax.invert_yaxis()
st.pyplot(fig)

# -------------------------
# Parameterübersicht
# -------------------------
if st.checkbox("Parameterdetails anzeigen"):
    param_data = {
        "Parameter": [
            "A", "A_ann", "A_plug",
            "U", "U_plug"
        ],
        "Wert": [
            A, A_ann, A_plug, U, U_plug
        ]
    }
    df_param = pd.DataFrame(param_data)
    st.table(df_param)


st.dataframe(df_q_plug)