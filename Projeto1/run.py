# %%
# Arquivo que roda a simulação 
# Importa os métodos de main.py 


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from main import CenarioSimulacao


AMBIENTE = "indoor"
PHI0_GRAUS = 30
MARGEM_ESPACO = 1.3
N_COMPONENTES = 100

VELOCIDADE_KMH = 30
PHI_V_GRAUS = 200
THETA_V_GRAUS = 0

DELTAS_T = [1e-7, 1e-6, 1e-5, 1e-4]        # larguras de pulso 
VELOCIDADES_TEMPO_COERENCIA_KMH = [18, 180]  # ~5 m/s e ~50 m/s

PASTA_SAIDA = "figuras"
SEMENTE = 42

np.random.seed(SEMENTE)
os.makedirs(PASTA_SAIDA, exist_ok=True)


def salvar(fig, nome):
    caminho = os.path.join(PASTA_SAIDA, nome)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> salvo: {caminho}")


def rodar_pipeline(cenario):
    cenario.mobilibdade(VELOCIDADE_KMH, PHI_V_GRAUS, THETA_V_GRAUS)
    cenario.variavel_aleatoria_LoS()
    cenario.gerar_espalhamento_de_atraso()
    cenario.gerar_fator_rice()
    cenario.gerar_atrasos_multipercursos(N=N_COMPONENTES)
    cenario.gerar_espalhamento_angular()
    cenario.gerar_potencia_multipercurso()
    cenario.gerar_angulos_chegada()
    cenario.gerar_vetor_chegada()
    cenario.gerar_desvio_doppler()
    return cenario


# Espaço 3D
def figura_espaco_3d(cenario):
    cenario.plotar_angulos_LoS()
    cenario.plotar_vetor_velocidade()
    cenario.mostrar()
    salvar(cenario.espaco.fig, "01_espaco_3d.png")


# Probabilidade de LOS vs. distância
def figura_prob_los(cenario):
    distancias = np.linspace(0.1, 300, 400)
    probs = [cenario.probabilidade_LoS(d_2D=d) for d in distancias]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(distancias, probs, color="blue", linewidth=2)
    ax.set_xlabel("Distância 2D BS-UT (m)")
    ax.set_ylabel(r"$P_{LOS}$")
    ax.set_title(f"Probabilidade de LOS vs. distância ({cenario.tipo_do_ambiente})")
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    salvar(fig, "02_probabilidade_los.png")


# Perfil de Potência de Atraso
def figura_pdp(cenario):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stem(cenario.atrasos * 1e6, cenario.potencias, basefmt=" ")
    ax.set_yscale("log")
    ax.set_xlabel(r"Domínio de Atraso - $\tau$ ($\mu$s)")
    ax.set_ylabel("PDP")
    ax.set_title(fr"$\sigma_\tau$ = {cenario.std_espalhamento*1e9:.2f} ns "
                 f"({'LOS' if cenario.estado_LoS else 'NLOS'})")
    ax.grid(alpha=0.3, which="both")
    salvar(fig, "03_pdp.png")

# Espectro Angular de Potência (Azimute e Elevação) 
def figura_espectro_angular(cenario):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    ax.stem(np.radians(cenario.angulo_azimutal_chegada), cenario.potencias, bottom=1e-8)
    ax.set_rscale("log")
    ax.set_rlim(bottom=1e-8, top=1.5)
    ax.set_title(fr"Espectro Angular de Potência (Azimute) - "
                 fr"$\sigma_{{\phi,AoA}}$={cenario.sigma_phi_AoA:.2f}°")
    salvar(fig, "04_espectro_angular_azimute.png")

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    ax.stem(np.radians(cenario.angulo_elevacional_chegada), cenario.potencias, bottom=1e-8)
    ax.set_rscale("log")
    ax.set_rlim(bottom=1e-8, top=1.5)
    ax.set_title(fr"Espectro Angular de Potência (Elevação) - "
                 fr"$\sigma_{{\theta,AoA}}$={cenario.sigma_theta_AoA:.2f}°")
    salvar(fig, "05_espectro_angular_elevacao.png")


# Vetores Direção de Chegada em 3D - Fig. 4
def figura_vetores_chegada(cenario):
    r_n = cenario.vetores_direcao_chegada

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(cenario.N):
        cor = "blue" if (i == 0 and cenario.estado_LoS) else "red"
        ax.quiver(0, 0, 0, r_n[i, 0], r_n[i, 1], r_n[i, 2],
                  color=cor, arrow_length_ratio=0.15, linewidth=1)

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("Eixo X"); ax.set_ylabel("Eixo Y"); ax.set_zlabel("Eixo Z")
    ax.set_title(f"Vetores Direção de Chegada (N={cenario.N}, "
                 f"{'LOS' if cenario.estado_LoS else 'NLOS'})")
    ax.set_box_aspect([1, 1, 1])
    salvar(fig, "06_vetores_direcao_chegada.png")


#  Espectro Doppler 
def figura_espectro_doppler(cenario):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stem(cenario.desvio_doppler, cenario.potencias, basefmt=" ")
    ax.set_yscale("log")
    ax.set_xlabel(r"Domínio de Desvio Doppler - $\nu$ (Hz)")
    ax.set_ylabel("Espectro Doppler")
    ax.set_title(f"Espectro Doppler (v={cenario.velocidade:.0f} km/h)")
    ax.grid(alpha=0.3, which="both")
    salvar(fig, "07_espectro_doppler.png")


# Sinal Recebido para diferentes larguras de pulso - Fig. 8-12
def figura_sinal_recebido(cenario):
    for i, delta_t in enumerate(DELTAS_T, start=1):
        t, s_tx, r_t = cenario.gerar_sinal_recebido(delta_t=delta_t, N_t=int(1e5))
        B_w = 1 / delta_t
 
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, s_tx, color="blue", label="Sinal Transmitido")
        ax.plot(t, np.abs(r_t), color="red", label="Sinal Recebido")
        ax.set_xlabel("Tempo absoluto - t (s)")
        ax.set_ylabel(r"$|\tilde{r}(t)|$")
        ax.set_title(fr"$\delta t$ = {delta_t:.0e} s, $B_w \approx$ {B_w:.0e} Hz, "
                     fr"$\sigma_\tau \approx$ {cenario.std_espalhamento*1e9:.1f} ns")
        ax.legend()
        ax.grid(alpha=0.3)
        salvar(fig, f"08_sinal_recebido_{i}_deltat_{delta_t:.0e}.png")
 


# Banda de Coerência do Canal - Fig. 13-15
def figura_banda_coerencia(cenario):
    banda = cenario.calcular_banda_coerencia(limiares=(0.95, 0.8))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(cenario._kappas_teste, cenario._rho_kappa)
    for lim, k in banda.items():
        ax.axvline(k, color="gray", linestyle="--")
    ax.axhline(0.95, color="gray", linestyle="-.", linewidth=0.8)
    ax.axhline(0.8, color="gray", linestyle="-.", linewidth=0.8)
    ax.set_xlabel(r"Desvio de Frequência - $\kappa$ (Hz)")
    ax.set_ylabel(r"$|\rho_{TT}(\kappa,0)|$")
    ax.set_title(fr"$B_C(0.95)$={banda[0.95]/1e6:.2f} MHz, "
                 fr"$B_C(0.8)$={banda[0.8]/1e6:.2f} MHz, "
                 fr"$\sigma_\tau$={cenario.std_espalhamento*1e9:.1f} ns")
    ax.grid(alpha=0.3, which="both")
    salvar(fig, "09_banda_coerencia.png")


# Tempo de Coerência do Canal 
def figura_tempo_coerencia(cenario_base):
    for i, v_kmh in enumerate(VELOCIDADES_TEMPO_COERENCIA_KMH, start=1):
        cenario_base.mobilibdade(v_kmh, PHI_V_GRAUS, THETA_V_GRAUS)
        cenario_base.gerar_desvio_doppler()

        tempo = cenario_base.calcular_tempo_coerencia(limiares=(0.95, 0.8))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogx(cenario_base._sigmas_teste, cenario_base._rho_sigma)
        for lim, s in tempo.items():
            ax.axvline(s, color="gray", linestyle="--")
        ax.axhline(0.95, color="gray", linestyle="-.", linewidth=0.8)
        ax.axhline(0.8, color="gray", linestyle="-.", linewidth=0.8)
        ax.set_xlabel(r"Desvio de Tempo - $\sigma$ (s)")
        ax.set_ylabel(r"$|\rho_{TT}(0,\sigma)|$")
        ax.set_title(fr"$T_C(0.95)$={tempo[0.95]*1e3:.2f} ms, "
                     fr"$T_C(0.8)$={tempo[0.8]*1e3:.2f} ms, v={v_kmh} km/h")
        ax.grid(alpha=0.3, which="both")
        salvar(fig, f"10_tempo_coerencia_{i}_v_{v_kmh}kmh.png")




parametros_ambiente = CenarioSimulacao.PARAMETROS_DO_AMBIENTE[AMBIENTE]
limite_espaco = max(parametros_ambiente["h_bs"], parametros_ambiente["d_2D"]) * MARGEM_ESPACO

cenario = CenarioSimulacao(AMBIENTE, phi0_graus=PHI0_GRAUS, limite_espaco=limite_espaco)
rodar_pipeline(cenario)


figura_espaco_3d(cenario)
figura_prob_los(cenario)
figura_pdp(cenario)
figura_espectro_angular(cenario)
figura_vetores_chegada(cenario)
figura_espectro_doppler(cenario)    
figura_sinal_recebido(cenario)
figura_banda_coerencia(cenario)
figura_tempo_coerencia(cenario)


# %%