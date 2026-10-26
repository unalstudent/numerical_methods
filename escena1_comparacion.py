# -*- coding: utf-8 -*-
"""
escena1_comparacion.py
========================
ESCENA 1 -- Comparacion clasica vs relativista.

Dos paneles lado a lado muestran al MISMO muon, naciendo a h0 = 15 km
y descendiendo a la misma rapidez v, bajo dos hipotesis fisicas:

  Panel izquierdo  (FISICA CLASICA, incorrecta):
      El reloj propio del muon avanza al mismo ritmo que el reloj de
      la Tierra (tiempo absoluto newtoniano). El muon se desintegra
      mucho antes de llegar al suelo.

  Panel derecho  (RELATIVIDAD ESPECIAL, correcta):
      El reloj propio del muon, visto desde la Tierra, avanza mas
      lento por el factor de Lorentz gamma (dilatacion temporal).
      El muon alcanza el detector en la superficie.

Cada panel incluye un "reloj" analogico cuya manecilla gira al ritmo
del TIEMPO PROPIO del muon, para visualizar directamente la
dilatacion temporal (la manecilla derecha gira gamma veces mas lento
que la izquierda, en terminos del tiempo de la Tierra).

Ejecutar directamente:
    python3 escena1_comparacion.py
guarda la animacion en ../salidas/escena1_comparacion.mp4 (o .gif) y,
si hay una pantalla disponible, tambien la muestra en una ventana.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

import fisica_relatividad as fr

# ------------------------------------------------------------------
# 1) Parametros fisicos (ver fisica_relatividad.py para las ecuaciones)
# ------------------------------------------------------------------
V = fr.V
C = fr.C
H0 = fr.H0
TAU0 = fr.TAU0
GAMMA = fr.factor_lorentz(V)
T_VIAJE = fr.tiempo_viaje_laboratorio(H0, V)          # tiempo (sistema Tierra) para llegar al suelo

# Tiempo PROPIO de decaimiento del muon representativo de esta escena.
# Se elige deliberadamente un valor T_DECAY_PROPIO tal que:
#
#     T_VIAJE / GAMMA   <   T_DECAY_PROPIO   <   T_VIAJE
#
# Esto garantiza que el MISMO muon (mismo tiempo propio de vida)
# decaiga clasicamente mucho antes de llegar al suelo, pero
# sobreviva en la descripcion relativista -- la esencia misma de la
# paradoja. El valor 3.5 us corresponde a ~1.6 vidas medias, un
# evento con probabilidad individual razonable
# (P = exp(-3.5/2.2) ~ 20%), no un caso extremo o irreal.
T_DECAY_PROPIO = 3.5e-6  # s

assert T_VIAJE / GAMMA < T_DECAY_PROPIO < T_VIAJE, (
    "T_DECAY_PROPIO debe cumplir T_VIAJE/GAMMA < T_DECAY_PROPIO < T_VIAJE "
    "para ilustrar la paradoja correctamente."
)

T_DECAY_LAB_REL = fr.dilatacion_temporal(T_DECAY_PROPIO, GAMMA)  # tiempo (lab) en que decaeria, vision relativista

# ------------------------------------------------------------------
# 2) Linea temporal de la animacion (tiempo de laboratorio = tiempo Tierra)
# ------------------------------------------------------------------
T_TOTAL = T_VIAJE * 1.18
N_FRAMES = 260
t_frames = np.linspace(0, T_TOTAL, N_FRAMES)

# Periodo visual del reloj analogico: una vuelta completa de la
# manecilla cada 1 us de TIEMPO PROPIO transcurrido (puramente
# estetico, no afecta a la fisica).
PERIODO_RELOJ = 1.0e-6


def estado_clasico(t):
    """
    Devuelve (altura, tiempo_propio_transcurrido, vida_restante, status)
    para el panel clasico en el instante de laboratorio t.
    Clasicamente tiempo_propio = tiempo_lab (no hay dilatacion).
    """
    if t < T_DECAY_PROPIO:
        t_propio = t
        altura = H0 - V * t
        status = "EN VUELO"
    else:
        t_propio = T_DECAY_PROPIO
        altura = H0 - V * T_DECAY_PROPIO
        status = "DESINTEGRADO"
    vida_restante = max(T_DECAY_PROPIO - t_propio, 0.0)
    return altura, t_propio, vida_restante, status


def estado_relativista(t):
    """
    Devuelve (altura, tiempo_propio_transcurrido, vida_restante, status)
    para el panel relativista en el instante de laboratorio t.
    El tiempo propio avanza GAMMA veces mas lento que el tiempo lab:
        t_propio = t / GAMMA   (consecuencia directa de Delta t = gamma * Delta t0)
    """
    t_lab_efectivo = min(t, T_VIAJE)
    altura = max(H0 - V * t_lab_efectivo, 0.0)
    t_propio = t_lab_efectivo / GAMMA

    if altura <= 0.0:
        status = "DETECTADO"
    elif t_propio >= T_DECAY_PROPIO:
        status = "DESINTEGRADO"
    else:
        status = "EN VUELO"
    vida_restante = max(T_DECAY_PROPIO - t_propio, 0.0)
    return altura, t_propio, vida_restante, status


# ------------------------------------------------------------------
# 3) Construccion de la figura
# ------------------------------------------------------------------
def crear_animacion():
    fig = plt.figure(figsize=(13, 7.5), facecolor="#0b1020")
    gs = fig.add_gridspec(
        nrows=2, ncols=2, height_ratios=[5, 1.15], width_ratios=[1, 1],
        hspace=0.06, wspace=0.18, left=0.07, right=0.95, top=0.88, bottom=0.07,
    )

    ax_izq = fig.add_subplot(gs[0, 0])
    ax_der = fig.add_subplot(gs[0, 1])
    ax_reloj_izq = fig.add_subplot(gs[1, 0])
    ax_reloj_der = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "La paradoja del muon atmosferico: ¿por que sobreviven el viaje?",
        color="white", fontsize=16, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.925,
        f"v = 0.998c   |   h$_0$ = {H0/1000:.0f} km   |   "
        f"vida media propia $\\tau_0$ = {TAU0*1e6:.1f} $\\mu$s   |   "
        f"factor de Lorentz $\\gamma$ = {GAMMA:.2f}",
        color="#9aa7c7", fontsize=11, ha="center",
    )

    paneles = []
    for ax, titulo, color_track in (
        (ax_izq, "FISICA CLASICA  (tiempo absoluto)", "#3a4366"),
        (ax_der, "RELATIVIDAD ESPECIAL  (dilatacion temporal)", "#3a4366"),
    ):
        ax.set_facecolor("#0b1020")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1200, H0 + 1500)
        ax.set_xticks([])
        ax.set_yticks([0, 5000, 10000, 15000])
        ax.set_yticklabels(["0", "5 km", "10 km", "15 km"])
        ax.tick_params(colors="#9aa7c7", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#2a3258")
        ax.set_title(titulo, color="white", fontsize=12.5, fontweight="bold", pad=10)

        # Pista vertical (atmosfera)
        ax.add_patch(Rectangle((-0.06, 0), 0.12, H0, facecolor=color_track,
                                edgecolor="none", zorder=1))
        # Suelo / detector
        ax.add_patch(Rectangle((-0.9, -350), 1.8, 350, facecolor="#1c2444",
                                edgecolor="#4c5694", linewidth=1.2, zorder=2))
        ax.text(0, -700, "DETECTOR (superficie)", color="#7d8ac4",
                fontsize=9, ha="center", va="center")
        # Punto de creacion
        ax.scatter([0], [H0], marker="^", s=70, color="#ffd166", zorder=3)
        ax.text(0.18, H0, "creacion\n(rayo cosmico)", color="#ffd166",
                 fontsize=8.5, va="center")

        paneles.append(ax)

    # Marcador del muon (un punto + etiqueta) en cada panel
    muon_izq, = ax_izq.plot([0], [H0], marker="o", markersize=16,
                             color="#5ee0c0", zorder=5, markeredgecolor="white",
                             markeredgewidth=1.2)
    muon_der, = ax_der.plot([0], [H0], marker="o", markersize=16,
                             color="#5ee0c0", zorder=5, markeredgecolor="white",
                             markeredgewidth=1.2)

    # Marca de desintegracion (estrella roja, oculta al inicio)
    burst_izq, = ax_izq.plot([], [], marker="*", markersize=26, color="#ef476f",
                              zorder=6, markeredgecolor="white", markeredgewidth=0.6)
    burst_der, = ax_der.plot([], [], marker="*", markersize=26, color="#ef476f",
                              zorder=6, markeredgecolor="white", markeredgewidth=0.6)

    # Halo verde al ser detectado
    halo_der, = ax_der.plot([], [], marker="o", markersize=34, color="#06d6a0",
                             alpha=0.35, zorder=4)

    # Cajas de texto con las 4 magnitudes pedidas
    caja_izq = ax_izq.text(
        -0.95, H0 + 1250, "", color="white", fontsize=9.6, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#141a33", edgecolor="#3a4366"),
    )
    caja_der = ax_der.text(
        -0.95, H0 + 1250, "", color="white", fontsize=9.6, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#141a33", edgecolor="#3a4366"),
    )

    # ---- relojes analogicos (visualizan el ritmo del tiempo propio) ----
    manecillas = []
    for ax_rel, etiqueta, color_mano in (
        (ax_reloj_izq, "reloj propio del muon\n(ritmo NORMAL)", "#5ee0c0"),
        (ax_reloj_der, "reloj propio del muon\n(ritmo LENTO, dilatado)", "#5ee0c0"),
    ):
        ax_rel.set_facecolor("#0b1020")
        ax_rel.set_xlim(-1.3, 1.3)
        ax_rel.set_ylim(-1.25, 1.55)
        ax_rel.set_aspect("equal")
        ax_rel.axis("off")
        ax_rel.add_patch(Circle((0, 0), 1.0, facecolor="#141a33",
                                 edgecolor="#5a6694", linewidth=1.6))
        for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x0, y0 = 0.86 * np.cos(ang), 0.86 * np.sin(ang)
            x1, y1 = 0.98 * np.cos(ang), 0.98 * np.sin(ang)
            ax_rel.plot([x0, x1], [y0, y1], color="#5a6694", linewidth=1.2)
        mano, = ax_rel.plot([0, 0], [0, 0.78], color=color_mano, linewidth=3,
                             solid_capstyle="round")
        ax_rel.add_patch(Circle((0, 0), 0.045, facecolor=color_mano, edgecolor="none", zorder=5))
        ax_rel.text(0, -1.22, etiqueta, color="#9aa7c7", fontsize=9, ha="center", va="top")
        manecillas.append(mano)

    mano_izq, mano_der = manecillas

    def texto_info(t, altura, t_propio, vida_restante, status, gamma_mostrado):
        return (
            f"t (Tierra)     = {t*1e6:6.2f} us\n"
            f"distancia      = {(H0-altura):6.0f} m\n"
            f"vida restante  = {vida_restante*1e6:6.2f} us\n"
            f"gamma          = {gamma_mostrado:6.2f}\n"
            f"estado: {status}"
        )

    def init():
        muon_izq.set_data([0], [H0])
        muon_der.set_data([0], [H0])
        burst_izq.set_data([], [])
        burst_der.set_data([], [])
        halo_der.set_data([], [])
        caja_izq.set_text("")
        caja_der.set_text("")
        return (muon_izq, muon_der, burst_izq, burst_der, halo_der,
                caja_izq, caja_der, mano_izq, mano_der)

    def actualizar(i):
        t = t_frames[i]

        # ---- panel clasico ----
        altura_i, tp_i, vida_i, st_i = estado_clasico(t)
        muon_izq.set_data([0], [altura_i])
        if st_i == "DESINTEGRADO":
            burst_izq.set_data([0], [altura_i])
        else:
            burst_izq.set_data([], [])
        caja_izq.set_text(texto_info(t, altura_i, tp_i, vida_i, st_i, 1.00))
        ang_i = -(tp_i / PERIODO_RELOJ) * 2 * np.pi
        mano_izq.set_data([0, 0.78 * np.sin(-ang_i)], [0, 0.78 * np.cos(-ang_i)])

        # ---- panel relativista ----
        altura_d, tp_d, vida_d, st_d = estado_relativista(t)
        muon_der.set_data([0], [altura_d])
        if st_d == "DESINTEGRADO":
            burst_der.set_data([0], [altura_d])
        else:
            burst_der.set_data([], [])
        if st_d == "DETECTADO":
            halo_der.set_data([0], [altura_d])
        else:
            halo_der.set_data([], [])
        caja_der.set_text(texto_info(t, altura_d, tp_d, vida_d, st_d, GAMMA))
        ang_d = -(tp_d / PERIODO_RELOJ) * 2 * np.pi
        mano_der.set_data([0, 0.78 * np.sin(-ang_d)], [0, 0.78 * np.cos(-ang_d)])

        return (muon_izq, muon_der, burst_izq, burst_der, halo_der,
                caja_izq, caja_der, mano_izq, mano_der)

    anim = animation.FuncAnimation(
        fig, actualizar, frames=N_FRAMES, init_func=init,
        interval=35, blit=True,
    )
    return fig, anim


if __name__ == "__main__":
    fig, anim = crear_animacion()

    os.makedirs("../salidas", exist_ok=True)
    ruta = fr.guardar_animacion(anim, "../salidas/escena1_comparacion", fps=24)
    print(f"Animacion guardada en: {ruta}")

    try:
        plt.show()
    except Exception:
        pass
