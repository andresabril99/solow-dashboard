# app.py
# Dashboard del modelo neoclásico de Solow en Streamlit con dinámica completa

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 

# 1) Parámetros por defecto (baseline)
alpha0, n0, s0, delta0, g0 = 0.34, 0.013, 0.27, 0.04, 0.0017
T_pre, T_post = 10, 90
times_full = np.arange(-T_pre, T_post + 1)

# Callback para resetear a defaults
def reset_defaults():
    for key, val in dict(alpha=alpha0, n=n0, s=s0, delta=delta0, g=g0).items():
        st.session_state[key] = val

# Función principal
def main():
    st.sidebar.header("Controles del modelo")
    # Inicializar estado
    for key, val in dict(alpha=alpha0, n=n0, s=s0, delta=delta0, g=g0).items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Sliders de parámetros
    st.sidebar.slider('Elasticidad del capital (α)', 0.1, 1.0, key='alpha', step=0.01)
    st.sidebar.slider('Tasa de crecimiento poblacional (n)', 0.0, 0.05, key='n', step=0.005)
    st.sidebar.slider('Tasa de ahorro (s)', 0.0, 1.0, key='s', step=0.05)
    st.sidebar.slider('Tasa de depreciación (δ)', 0.0, 0.1, key='delta', step=0.005)
    st.sidebar.slider('Tasa de crecimiento tecnológico (g)', 0.0, 0.05, key='g', step=0.005)
    st.sidebar.button('Reset defaults', on_click=reset_defaults)

    # Título y ecuación
    st.title('Modelo Neoclásico de Solow Intensivo')
    st.text("Tarea de Macroeconomía Avanzada. Elaborado por: Andrés Abril")
    st.latex(r"""
    \dot{k} = s \; k^{\alpha} - (n + g + \delta) \; k
    """)

    # Parámetros actuales
    alpha = st.session_state.alpha
    n     = st.session_state.n
    s     = st.session_state.s
    delta = st.session_state.delta
    g     = st.session_state.g

    # 2) Simulación dinámica
    # Estado estacionario inicial (base)
    k_base = np.zeros(T_post + 1)
    k_base[0] = (s0 / (n0 + g0 + delta0))**(1 / (1 - alpha0))
    for i in range(T_post):
        k_base[i + 1] = k_base[i] + s0 * k_base[i]**alpha0 - (n0 + g0 + delta0) * k_base[i]
    y_base = k_base**alpha0

    # Dinámica tras shock
    k_new = np.zeros(T_post + 1)
    k_new[0] = k_base[0]
    for i in range(T_post):
        k_new[i + 1] = k_new[i] + s * k_new[i]**alpha - (n + g + delta) * k_new[i]
    y_new = k_new**alpha

    # Series completas pre y post
    k_full_base = np.concatenate((np.full(T_pre, k_base[0]), k_base))
    y_full_base = np.concatenate((np.full(T_pre, y_base[0]), y_base))
    k_full_new  = np.concatenate((np.full(T_pre, k_new[0]),  k_new))
    y_full_new  = np.concatenate((np.full(T_pre, y_new[0]),  y_new))

    # Estado estacionario de ambos
    k_star0, y_star0 = k_base[0], y_base[0]
    k_star1, y_star1 = k_new[-1],   y_new[-1]

    # Detectar cambio
    changed = any([alpha != alpha0, n != n0, s != s0, delta != delta0, g != g0])

    # 3) Graficar 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: inversión vs depreciación
    k_max = max(k_full_base.max(), k_full_new.max()) * 1.2
    k_seq = np.linspace(0, k_max, 200)
    ax1.plot(k_seq, s0 * k_seq**alpha0, color='gray', label='s₀ k^α₀')
    ax1.plot(k_seq, (n0 + g0 + delta0) * k_seq, color='gray', label='(n₀+g₀+δ₀)k')
    if changed:
        ax1.plot(k_seq, s * k_seq**alpha, '--', color='black', label='s k^α')
        ax1.plot(k_seq, (n + g + delta) * k_seq, '--', color='red',   label='(n+g+δ)k')
    ax1.set(title='Inversión Real vs. Inversión en Reposición', xlabel='k', ylabel='Δk')
    ax1.legend()

    # Panel 2: evolución de k
    ax2.plot(times_full, k_full_base, color='gray')
    if changed:
        ax2.plot(times_full, k_full_new, '--', color='black')
    ax2.axvline(0, linestyle='--')
    ax2.set(title='Capital por Unidad de Trabajo Intensivo', xlabel='t', ylabel='k')

    # Panel 3: función de producción + marcadores
    ax3.plot(k_seq, k_seq**alpha0, color='gray', label='y₀=k^α₀')
    if changed:
        ax3.plot(k_seq, k_seq**alpha, '--', color='black', label='y=k^α')
    ax3.plot(k_star0, y_star0, 'o', color='gray', markersize=8)
    if changed:
        ax3.plot(k_star1, y_star1, 'x', color='black', markersize=8)
    ax3.set(title='Función de Producción', xlabel='k', ylabel='y')
    if changed:
        ax3.legend()

    # Panel 4: evolución de y
    ax4.plot(times_full, y_full_base, color='gray')
    if changed:
        ax4.plot(times_full, y_full_new, '--', color='black')
    ax4.axvline(0, linestyle='--')
    ax4.set(title='Producción por Unidad de Trabajo Intensivo', xlabel='t', ylabel='y')

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == '__main__':
    main()
