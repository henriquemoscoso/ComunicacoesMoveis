# %% 
from main import CenarioSimulacao
import numpy as np
import matplotlib.pyplot as plt



cenario = CenarioSimulacao("indoor", phi0_graus=30, limite_espaco=35)
cenario.plotar_angulos_LoS()
cenario.mostrar()

# Verificando os valores de cada componente de theta e phi
angulos = cenario.angulos_LoS()
for nome, valor in angulos.items():
    print(f"{nome} = {valor:.2f}°")
# %%
# Testando a velocidade e o plot dela na UT 
cenario.mobilibdade(5,90)
cenario.plotar_vetor_velocidade(10,"orange")
cenario.mostrar()
# %%
# Testando probabilidade de LoS e se o estado_LoS é aleatório 
cenario.probabilidade_LoS()
cenario.variavel_aleatoria_LoS()
# %%
# Testando probabilidade com UT se movendo 

distancias = np.linspace(0.1, 200, 300)
# Probabilidade para cada distância 
probs_dist = [cenario.probabilidade_LoS(d_2D=d) for d in distancias]

plt.figure(figsize=(7, 4))
plt.plot(distancias, probs_dist, color='blue', linewidth=2)
plt.xlabel("Distância 2D BS-UT (m)")
plt.ylabel("P_LOS")
plt.title(f"P_LOS vs. distância ({cenario.tipo_do_ambiente})")
plt.ylim([0, 1.05])
plt.grid(alpha=0.3)
plt.show()

v_ms = cenario.velocidade / 3.6  

duracao_s = 60
passo_s = 1
tempos = np.arange(0, duracao_s, passo_s)

pos_inicial = np.array(cenario.posicao_ut)
posicoes = pos_inicial + np.outer(tempos, cenario.v_hat * v_ms)

d_2D_t = np.sqrt(posicoes[:, 0]**2 + posicoes[:, 1]**2)
probs_tempo = [cenario.probabilidade_LoS(d_2D=d) for d in d_2D_t]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

ax1.plot(tempos, d_2D_t, color='green')
ax1.set_ylabel("Distância 2D (m)")
ax1.set_title(f"Mobilidade do UT ({cenario.tipo_do_ambiente})")
ax1.grid(alpha=0.3)

ax2.plot(tempos, probs_tempo, color='blue')
ax2.set_xlabel("Tempo (s)")
ax2.set_ylabel("P_LOS")
ax2.set_ylim([0, 1.05])
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# %%
# Testando o cálculo de geração de atraso (já em escala linear)
cenario.gerar_espalhamento_de_atraso()
# %%
# Testando fator de Rice 
lista_teste = []
for i in range(1,4):
    cenario.variavel_aleatoria_LoS()
    lista_teste.append(cenario.gerar_fator_rice())
lista_teste
# %%
# Testando média e std para AoD elevacional 
cenario = CenarioSimulacao("indoor", 30, 35)
cenario._media_std_AoD_saida()
# %%
# Testando potência multipercurso 

cenario.variavel_aleatoria_LoS()
cenario.gerar_espalhamento_de_atraso()
cenario.gerar_fator_rice()
cenario.gerar_atrasos_multipercursos(N=100)

potencias = cenario.gerar_potencia_multipercurso()

plt.figure(figsize=(8, 4))
plt.stem(cenario.atrasos * 1e6, potencias, basefmt=" ") 
plt.yscale('log')
plt.xlabel("Domínio de Atraso - τ (μs)")
plt.ylabel("PDP")
plt.title(f"σ_τ = {cenario.std_espalhamento*1e9:.2f} ns")
plt.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# %%
# Testando ângulo azimutal e elevacional de chegada 
cenario.gerar_espalhamento_angular()
phi, theta = cenario.gerar_angulos_chegada()

plt.figure(figsize=(6,6))
ax = plt.subplot(111, projection='polar')
ax.stem(np.radians(cenario.angulo_azimutal_chegada), cenario.potencias, bottom=1e-8)  # <- bottom aqui
ax.set_rscale('log')
ax.set_rlim(bottom=1e-8, top=1.5)
ax.set_title(f"Espectro Angular de Potência (Azimute) - σ_phi,AoA={cenario.sigma_phi_AoA:.2f}°")
plt.show()

plt.figure(figsize=(6,6))
ax = plt.subplot(111, projection='polar')
ax.stem(np.radians(cenario.angulo_elevacional_chegada), cenario.potencias, bottom=1e-8)  # <- bottom aqui
ax.set_rscale('log')
ax.set_rlim(bottom=1e-8, top=1.5)
ax.set_title(f"Espectro Angular de Potência (Elevação) - σ_theta,AoA={cenario.sigma_theta_AoA:.2f}°")
plt.show()

# %%
r_n = cenario.gerar_vetor_chegada()
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

for i in range(cenario.N):
    cor = 'blue' if (i == 0 and cenario.estado_LoS) else 'red'
    ax.quiver(0, 0, 0, r_n[i, 0], r_n[i, 1], r_n[i, 2],
              color=cor, arrow_length_ratio=0.15, linewidth=1)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
ax.set_zlabel("Eixo Z")
ax.set_title(f"Vetores Direção de Chegada (N={cenario.N}, {'LOS' if cenario.estado_LoS else 'NLOS'})")
ax.set_box_aspect([1, 1, 1])

plt.show()
# %%

cenario.mobilibdade(v_kmh=3, phi0v_graus=200, theta0v_graus=0)
nu_n = cenario.gerar_desvio_doppler()
plt.figure(figsize=(8,4))
plt.stem(nu_n, cenario.potencias, basefmt=" ")
plt.yscale('log')
plt.xlabel("Domínio de Desvio Doppler - ν (Hz)")
plt.ylabel("Espectro Doppler")
plt.title(f"Espectro Doppler (v={cenario.velocidade} km/h)")
plt.grid(alpha=0.3, which='both')
plt.show()
# %%
t, s_tx, r_t = cenario.gerar_sinal_recebido(delta_t=1e-7, N_t=100000)
plt.figure(figsize=(9, 5))
plt.plot(t, s_tx, color='blue', label='Sinal Transmitido')
plt.plot(t, np.abs(r_t), color='red', label='Sinal Recebido')
plt.xlabel("Tempo absoluto - t (s)")
plt.ylabel("|r̃(t)|")
plt.title(f"δt = 1e-07 s, σ_τ = {cenario.std_espalhamento*1e9:.2f} ns")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# %%
banda = cenario.calcular_banda_coerencia(limiares=(0.95, 0.8))
tempo = cenario.calcular_tempo_coerencia(limiares=(0.95, 0.8))

plt.figure(figsize=(8,4))
plt.semilogx(cenario._kappas_teste, cenario._rho_kappa)
for lim, k in cenario.banda_coerencia.items():
    plt.axvline(k, color='gray', linestyle='--')
plt.axhline(0.95, color='gray', linestyle='-.', linewidth=0.8)
plt.axhline(0.8, color='gray', linestyle='-.', linewidth=0.8)
plt.xlabel("Desvio de Frequência - κ (Hz)")
plt.ylabel("|ρ_TT(κ,0)|")
plt.title(f"B_C(0.95)={banda[0.95]/1e6:.2f} MHz, B_C(0.8)={banda[0.8]/1e6:.2f} MHz")
plt.grid(alpha=0.3)
plt.show()
# %%
plt.figure(figsize=(8,4))
plt.semilogx(cenario._sigmas_teste, cenario._rho_sigma)
for lim, s in cenario.tempo_coerencia.items():
    plt.axvline(s, color='gray', linestyle='--')
plt.axhline(0.95, color='gray', linestyle='-.', linewidth=0.8)
plt.axhline(0.8, color='gray', linestyle='-.', linewidth=0.8)
plt.xlabel("Desvio de Tempo - σ (s)")
plt.ylabel("|ρ_TT(0,σ)|")
plt.title(f"T_C(0.95)={tempo[0.95]*1e3:.2f} ms, T_C(0.8)={tempo[0.8]*1e3:.2f} ms")
plt.grid(alpha=0.3)
plt.show()