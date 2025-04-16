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

def eingabe_berechnungseinstellungen():
    st.header("Berechnungseinstellungen")
    n_steps = st.number_input("Anzahl Tiefeninkremente", value=501, min_value=10, max_value=1000, step=10)
    z_initial = st.number_input("Startwert Tiefe volle Pfropfenbildung für Iteration", value=6.0, min_value=0.0, max_value=1000.0, step=1.0)
    threshold= st.number_input("Threshold", value=0.0001, min_value=0.0000001, max_value=1.0, step=.001)
    max_iterations= st.number_input("Begrenzung Anzahl Iterationen", value=100, min_value=1, max_value=1000, step=10)
    return n_steps, z_initial, threshold, max_iterations



st.header('Hintergrund')
st.info('Der Pfropfenwiderstand wird nach folgenden Gleichungen berechnet. ')
st.latex(r'''
    \sigma_{v,plug,1}=p+(\gamma^\prime+\gamma_w\ )h    \\
    \sigma_{v,plug,2}=p+\gamma^\prime h_{c,1}+\gamma_wh+\left(p+\gamma\prime\frac{A}{UK_0\mu}\right)\left(e^\frac{\left(h-h_{c,1}\right)UK_0\mu}{A}-1\right)\\
    \sigma_{v,plug,3}=p+\gamma\prime h_{c,1}+\gamma_wh+\frac{c_u}{\mu K_0}+\left(h-h_{c,1}-h_{c,2}\right)\left(\gamma\prime+c_u\frac{U}{A}\right) 
        ''')
st.latex(r'''
    \frac{d\sigma_{v}}{dz}=\gamma^\prime+\gamma_w\    \\
    \frac{d\sigma_{v}}{dz}=\left(p + \left(e^\left(4\mu K_0 \Delta z / D\right)-1\right)\right)\left(p+D*\gamma/\left(4 \mu k_0\right)\right)\\
    \sigma_{v,plug,3}=p+\gamma\prime h_{c,1}+\gamma_wh+\frac{c_u}{\mu K_0}+\left(h-h_{c,1}-h_{c,2}\right)\left(\gamma\prime+c_u\frac{U}{A}\right) 
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

n_steps, z_initial, threshold, max_iterations = eingabe_berechnungseinstellungen()

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
ifr_mode = st.selectbox("Art der IFR-Eingabe wählen:", ["Konstant", "Multilinear", "Punktuell", "Polynom"])

if ifr_mode == "Konstant":
    IFR_0 = st.number_input("IFR_0 (konstanter Wert)", value=0.9, min_value=0.0)
    z_ifr = z
    IFR_values = np.full_like(z, IFR_0)
elif ifr_mode == "Multilinear":
    IFR_0 = st.number_input("Startwert IFR_0", value=1.5)
    n_segments = st.number_input("Anzahl Segmente", min_value=1, max_value=10, value=2, step=1)
    
    deltas = []
    slopes = []
    for i in range(n_segments):
        dz = st.number_input(f"Δz Segment {i+1} [m]", value=2.0, key=f"dz_{i}")
        m = st.number_input(f"Steigung Segment {i+1}", value=-0.1, key=f"m_{i}")
        deltas.append(dz)
        slopes.append(m)

    z_ifr = [0]
    IFR_values = [IFR_0]
    for dz, m in zip(deltas, slopes):
        z_end = z_ifr[-1] + dz
        z_new = np.linspace(z_ifr[-1], z_end, int(dz / (L / n_steps)))
        for zi in z_new[1:]:
            z_ifr.append(zi)
            IFR_values.append(IFR_values[-1] + m * (zi - z_ifr[-2]))
elif ifr_mode == "Punktuell":
    n_points = st.number_input("Anzahl diskreter Punkte", min_value=2, max_value=20, value=3)
    z_points = []
    ifr_points = []
    for i in range(n_points):
        z_pt = st.number_input(f"Tiefe z{i+1} [m]", value=float(i * L / (n_points - 1)), key=f"z{i}")
        ifr_pt = st.number_input(f"IFR an z{i+1}", value=1.5, key=f"ifr{i}")
        z_points.append(z_pt)
        ifr_points.append(ifr_pt)

    z_ifr = z
    IFR_values = np.interp(z_ifr, z_points, ifr_points)
elif ifr_mode == "Polynom":
    poly_degree = st.number_input("Polynomgrad", min_value=1, max_value=6, value=2)
    coeffs = []
    for i in range(poly_degree + 1):
        coeff = st.number_input(f"Koeffizient für z^{i}", value=0.0, key=f"coeff_{i}")
        coeffs.append(coeff)

    z_ifr = z
    IFR_values = np.polyval(coeffs[::-1], z_ifr)

IFR = IFR_values

dz = np.diff(z)
IFR_avg = 0.5 * (IFR_values[1:] + IFR_values[:-1])
h = np.concatenate([[0], np.cumsum(dz * IFR_avg)])



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