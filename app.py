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

    A = np.pi * D**2 / 4
    A_ann = ((D/2)**2 - (D/2 - t)**2) * np.pi
    A_plug = (D/2 - t)**2 * np.pi
    U = np.pi * D
    U_plug = np.pi * (D - 2 * t)

    h_1_default = 2 * c_u / gamma
    return D, L, t, A, A_ann, A_plug, U, U_plug, h_1_default

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



st.header('Hintergrund')
st.info('Der Pfropfenwiderstand wird nach folgenden Gleichungen berechnet. ')
st.latex(r'''
    \sigma_{v,plug,1}=(\gamma^\prime+\gamma_w\ )h    \\
    \sigma_{v,plug,2}=\gamma^\prime h_{c,1}+\gamma_wh+\gamma\prime\frac{A}{UK_0\mu}\left(e^\frac{\left(h-h_{c,1}\right)UK_0\mu}{A}-1\right)\\
    \sigma_{v,plug,3}=\gamma\prime h_{c,1}+\gamma_wh+\frac{c_u}{\mu K_0}+\left(h-h_{c,1}-h_{c,2}\right)\left(\gamma\prime+c_u\frac{U}{A}\right) 
        ''')
st.info('Zudem wird der Pfahlspitzendruck eines offenen Profils angesetzt, um die maximale Belastung auf den Pfropfen zu begrenzen.')
st.latex(r'''
    \sigma_{v,closed}=(\gamma^\prime+\gamma_w\ )z + N_c c_u    
        ''')
# -------------------------
# Hauptcode
# -------------------------
# Eingaben sammeln
c_u, gamma, gamma_w, N_c_max, N_c_option, K_0 = eingabe_bodenparameter()
D, L, t, A, A_ann, A_plug, U, U_plug, h_1_default = eingabe_pfahlparameter(c_u, gamma)
mu = eingabe_interaktion()
IFR_0, h_1 = eingabe_pfropfenmodell(A_ann, A_plug, h_1_default)

n_steps, z_initial, threshold, max_iterations = eingabe_berechnungseinstellungen(case='long')

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
# Iterative Berechnung der Gleichgewichtstiefe z_iterated
# ------------------------------------------------------------------------

# Anfangswert für Iteration
z_iterated = z_initial
threshold = threshold  # Abbruchbedingung [m]
max_iterations = max_iterations
iteration = 0

while iteration < max_iterations:
    iteration += 1
    
    # Hilfsgrößen
    z_grenz1 = h_1 / IFR_0
    h_2 = np.log(1 + c_u / gamma * U_plug / A_plug) * A_plug / U_plug / mu / K_0 + h_1

    # Berechnung der Bedingungen, mit gamma_w funktioniert das hier noch nicht, ist noch zu implementieren!
    h_cond_1 = h_2 + (N_c_max * c_u - gamma * h_1 - c_u / mu / K_0 + gamma * z_iterated) / (gamma + c_u * U_plug / A_plug)
    h_cond_2 =  np.log( 1 + 
                       (N_c_max * c_u - gamma * h_1 + gamma * z_iterated) 
                       / (gamma * A_plug / U_plug / mu / K_0)
                        ) * (A_plug / U_plug / mu / K_0)  + h_1

    h_cond = h_cond_2 if h_cond_2 < h_2 else h_cond_1

    z_plug = 2 * h_cond / IFR_0 - z_grenz1
    diff = z_plug - z_iterated

    # Update Iterationswert
    z_iterated = z_iterated + 0.5 * diff  # sanfte Anpassung zur Stabilität

    if abs(diff) < threshold:
        break
# Ausgabe
st.subheader("Ermittelte Gleichgewichtstiefe")
st.write(f"Gleichgewicht erreicht bei z = {z_iterated:.3f} m nach {iteration} Iterationen.")

dIFR = IFR_0 / (z_iterated - z_grenz1)

# -------------------------
# Berechnung offener Pfahl
# -------------------------
dz = z[1] - z[0]
h={}
IFR = {}
q_plug = {}
# Berechne IFR und h über die Tiefe
for i in range(len(z)):
    if i == 0 or z[i] <= z_grenz1:
        IFR[i] = IFR_0
        h[i] = z[i] * IFR_0
    else:
        IFR[i] = max(IFR[i-1] - dz * dIFR, 0)
        if z[i] <= z_iterated:
            h[i] = z_grenz1*IFR_0 + (IFR[i]+IFR_0)/2 * (z[i]-z_grenz1)
        else:
            h[i] = z_grenz1*IFR_0 + (IFR_0)/2 * (z_iterated-z_grenz1)

    #h[i] = IFR[i] * dz + h[i-1] if i > 0 else z[i] * IFR[i]
IFR = np.array([IFR[i] for i in range(len(IFR))])
h = np.array([h[i] for i in range(len(h))])

# Funktionen für q_plug
def cal_q_plug_1(gamma, gamma_w, h_val):
    return (gamma+gamma_w) * h_val

def cal_q_plug_2(gamma, gamma_w, h_val, A, U, mu, K0, h_c1):
    return gamma * h_c1 + gamma * A/U/K0/mu * (np.exp((h_val - h_c1)/(A/(U*K0*mu))) - 1) + gamma_w * h_val

def cal_q_plug_3(gamma, gamma_w, h_val, A, U, c_u, mu, K_0, h_c1, h_c2):
    return gamma * h_c1 + c_u / mu / K_0 + (h_val - h_c2) * (gamma + c_u * U / A) + gamma_w * h_val
def cal_q_plugged(gamma, gamma_w, dz, q_plug_old):
    return q_plug_old + (gamma + gamma_w) * dz

# Berechne q_plug über h
for i in range(len(h)):
    h_i = h[i]
    if z[i]> z_iterated and i>0: # fully plugged, just add soil weight
        q_plug[i] =cal_q_plugged(gamma, gamma_w, dz, q_b[i-1]) 
    elif h_i <= h_1:
        q_plug[i] = cal_q_plug_1(gamma, gamma_w, h_i)
    elif h_i <= h_2:
        q_plug[i] = cal_q_plug_2(gamma, gamma_w, h_i, A_plug, U_plug, mu, K_0, h_1)
    else:
        q_plug[i] = cal_q_plug_3(gamma, gamma_w, h_i, A_plug, U_plug, c_u, mu, K_0, h_1, h_2)
q_plug = np.array([q_plug[i] for i in range(len(z))])

# Berechne Spitzendruck offener Pfahl
def cal_q_b_open(q_plug, q_ann, A, A_plug, A_ann):
    return (q_plug*A_plug+q_ann*A_ann)/A

q_ann = q_b
q_b_open = cal_q_b_open(q_plug, q_ann, A, A_plug, A_ann)

# -------------------------
# Plot
# -------------------------
df_q_plug = pd.DataFrame({
    "Tiefe z [m]": z,
    "IFR [-]": IFR,
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

# fig, axes = plt.subplots(1,2, figsize=(8, 6))
# ax=axes[3]
# ratio = q_b_open / q_b
# ax.plot(ratio, IFR, label='Calculation', color = 'blue')
# #reference curve min
# x = np.arange(0.5, 1.21, 0.01)
# y = -4.4826*x**2 + 4.1909*x+0.2353
# ax.plot(x, y, label = 'ref min')
# #reference curve max
# x = np.arange(0.5, 1.21, 0.01)
# y = -5.1298*x**2 + 4.9121*x+0.3277
# ax.plot(x, y, label = 'ref max')
# ax.set_xlim([0,1.2])
# ax.set_ylim([0,1.5])
# ax.set_ylabel("IFR")
# ax.set_xlabel("q_b_open / q_b")
# ax.set_title("IFR")
# ax.grid(True)
# ax.legend()

# st.pyplot(fig)

# -------------------------
# Parameterübersicht
# -------------------------
if st.checkbox("Parameterdetails anzeigen"):
    param_data = {
        "Parameter": [
            "A", "A_ann", "A_plug",
            "U", "U_plug", "h₁", "h₂"
        ],
        "Wert": [
            A, A_ann, A_plug, U, U_plug, h_1, h_2
        ]
    }
    df_param = pd.DataFrame(param_data)
    st.table(df_param)


st.dataframe(df_q_plug)