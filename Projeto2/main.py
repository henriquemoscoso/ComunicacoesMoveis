# %% 
import os
import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import comb, expn, factorial, erfc

# Semente da simulação 
rng = np.random.default_rng(42)

# Parâmetros gerais
N = int(1e8)                    # número de realizações do canal
tamanho_do_bloco = int(5e6)     # realizações processadas por vez (N inteiro não cabe na memória)
M = 16                          # ordem do M-ASK
lista_de_L = [1, 2, 3, 4]       # números de ramos de diversidade
combinadores = ["SC", "EGC", "MRC"]
cores = ["C0", "C1", "C2", "C3"]

os.makedirs("figuras", exist_ok=True)


def db2lin(valor_em_db):
    """Converte de dB para linear."""
    return 10**(np.asarray(valor_em_db)/10)


def blocos(N, tamanho=tamanho_do_bloco):
    """Divide as N realizações em blocos de no máximo 'tamanho'."""
    tamanho = min(tamanho, N)
    return [tamanho]*(N//tamanho) + ([N % tamanho] if N % tamanho else [])


def gerar_canal(N,L,K=0,rng=rng):
    """
    Canal com L ramos independentes, gerado N vezes. K é o fator de Rice. Como a maior 
    parte da simulação é de Rayleigh, então vamos considerá-lo K = 0 por definição. 
    Só no último caso iremos considerá-lo.

    Retorna uma matriz h de N x L normalizado para energia ser unitária.
    """ 

    # Componente difusa
    difusa = (rng.standard_normal((N,L)) + 1j*rng.standard_normal((N,L)))/np.sqrt(2)

    if K == 0: 
        return difusa 

    # Componente de visada direta com fase aleatória 
    fase = rng.uniform(0, 2*np.pi,(N,L))
    los = np.exp(1j*fase)

    return np.sqrt(K / (K + 1)) * los + np.sqrt(1 / (K + 1)) * difusa


def snr_combinador(h,snr_media,tipo_do_combinador): 
    """
    SNR instantânea da saída do combinador. Depende da SNR média e do canal h, além de ter um modelo diferente para cada 
    tipo de combinador.
    """

    # SNR instantânea
    modulo_de_h = np.abs(h)
    snr = snr_media*modulo_de_h**2

    L = h.shape[1]

    if tipo_do_combinador.lower()=="sc": 
        return snr.max(axis=1)
    elif tipo_do_combinador.lower()=="mrc": 
        return snr.sum(axis=1)
    elif tipo_do_combinador.lower()=="egc": 
        return np.sqrt(snr).sum(axis=1)**2/L
    else:
        raise ValueError(f"Combinador não existente: {tipo_do_combinador}")


def modelo_teorico(tipo_do_combinador, L, snr_media): 
    """
    Calcula a SNR média teórica na saída de cada combinador (Rayleigh). Para o EGC 
    não existem expressões das métricas, mas a média tem forma fechada.
    """

    if tipo_do_combinador.lower() == "sc": 
        return snr_media*np.sum(1/np.arange(1,L+1))
    elif tipo_do_combinador.lower() == "mrc": 
        return L * snr_media
    elif tipo_do_combinador.lower() == "egc": 
        return snr_media*(1 + (L - 1)*np.pi/4)
    else: 
        raise ValueError(f"Combinador não existente: {tipo_do_combinador}")
    
# %% 
# Testando primeiras funções (comparando as médias)
snr_media = db2lin(10)     # 10 dB 
erros = {}
for tipo in combinadores: 
    # Testando L entre 1 e 4 ramos
    for L in lista_de_L:
        # Média acumulada bloco a bloco
        soma = 0
        for n in blocos(N): 
            soma += snr_combinador(gerar_canal(N=n,L=L),snr_media,tipo).sum()
        simulacao = soma/N
        teorica = modelo_teorico(tipo,L,snr_media)

        # Erro relativo 
        erros[(tipo,L)] = 100*abs(simulacao - teorica)/teorica
erros


# %%
# Constantes do M-ASK: Pe(snr) = aM * Q(sqrt(bM * snr))
aM = 2*(M - 1)/M
bM = 6/(M**2 - 1)


def outage_teorica(tipo_do_combinador, limiar, snr_media, L):
    """Probabilidade de outage teórica."""

    if tipo_do_combinador.lower() == "sc": 
        return (1 - np.exp(-limiar/snr_media))**L
    elif tipo_do_combinador.lower() == "mrc": 
        x = np.asarray(limiar)/snr_media
        soma = sum(x**k/factorial(k) for k in range(L))
        return 1 - np.exp(-x)*soma
    else: 
        raise ValueError(f"Sem expressão teórica para: {tipo_do_combinador}")


def sep_teorica(tipo_do_combinador, snr_media, L):
    """Probabilidade de erro de símbolo teórica."""

    snr_media = np.asarray(snr_media, dtype=float)

    if tipo_do_combinador.lower() == "sc": 
        soma = np.zeros_like(snr_media)
        for l in range(L):
            c = bM*snr_media/(2*(l + 1))
            soma += (-1)**l*comb(L - 1, l)*L/(2*(l + 1))*(1 - np.sqrt(c/(1 + c)))
        return aM*soma
    elif tipo_do_combinador.lower() == "mrc": 
        mi = np.sqrt((bM*snr_media/2)/(1 + bM*snr_media/2))
        soma = sum(comb(L - 1 + k, k)*((1 + mi)/2)**k for k in range(L))
        return aM*((1 - mi)/2)**L*soma
    else: 
        raise ValueError(f"Sem expressão teórica para: {tipo_do_combinador}")


def capacidade_teorica(tipo_do_combinador, snr_media, L):
    """Capacidade ergódica teórica em bits/s/Hz."""

    snr_media = np.asarray(snr_media, dtype=float)

    if tipo_do_combinador.lower() == "sc": 
        soma = np.zeros_like(snr_media)
        for k in range(L):
            a = (k + 1)/snr_media
            soma += comb(L - 1, k)*(-1)**k/(k + 1)*np.exp(a)*expn(1, a)
        return L/np.log(2)*soma
    elif tipo_do_combinador.lower() == "mrc": 
        soma = sum(expn(L - k + 1, 1/snr_media) for k in range(1, L + 1))
        return np.exp(1/snr_media)/np.log(2)*soma
    else: 
        raise ValueError(f"Sem expressão teórica para: {tipo_do_combinador}")


# %%
def outage_simulada(tipo_do_combinador, limiares, snr_media, L, K=0):
    """Fração das realizações em que a SNR de saída fica abaixo do limiar."""

    contagem = np.zeros(len(limiares))
    for n in blocos(N): 
        snr = snr_combinador(gerar_canal(n,L,K), snr_media, tipo_do_combinador)
        contagem += np.array([np.sum(snr < limiar) for limiar in limiares])
    return contagem/N


def capacidade_simulada(tipo_do_combinador, lista_snr_media, L, K=0):
    """
    Média de log2(1 + snr) na saída do combinador. Cada bloco do canal é gerado uma vez só e 
    reaproveitado em todos os pontos de SNR.
    """

    soma = np.zeros(len(lista_snr_media))
    for n in blocos(N): 
        h = gerar_canal(n,L,K)
        soma += np.array([np.sum(np.log2(1 + snr_combinador(h, snr_media, tipo_do_combinador)))
                          for snr_media in lista_snr_media])
    return soma/N


def sep_simulada_um_ponto(tipo_do_combinador, snr_media, L, K=0, bloco=int(1e6)):
    """
    SEP do M-ASK simulada em nível de símbolo. Cada ramo recebe r = hs + ruído.
    Simula em blocos para não estourar a memória.
    """

    # Constelação M-ASK normalizada para energia unitária
    niveis = np.arange(-(M - 1), M, 2)
    constelacao = niveis/np.sqrt(np.mean(niveis**2))
    N0 = 1/snr_media

    erros = 0
    for _ in range(N//bloco):
        indices = rng.integers(0, M, bloco)
        s = constelacao[indices]
        h = gerar_canal(bloco,L,K)
        ruido = np.sqrt(N0/2)*(rng.standard_normal((bloco,L)) + 1j*rng.standard_normal((bloco,L)))
        r = h*s[:,None] + ruido

        if tipo_do_combinador.lower() == "sc": 
            # Escolhe o ramo com maior |h|
            melhor_ramo = np.argmax(np.abs(h), axis=1)
            linhas = np.arange(bloco)
            y = r[linhas,melhor_ramo]/h[linhas,melhor_ramo]
        elif tipo_do_combinador.lower() == "mrc": 
            # Pesos iguais a h*
            y = np.sum(np.conj(h)*r, axis=1)/np.sum(np.abs(h)**2, axis=1)
        elif tipo_do_combinador.lower() == "egc": 
            # Só corrige a fase, pesos de mesma amplitude
            y = np.sum(np.exp(-1j*np.angle(h))*r, axis=1)/np.sum(np.abs(h), axis=1)
        else:
            raise ValueError(f"Combinador não existente: {tipo_do_combinador}")

        # Detecção por mínima distância
        detectado = np.argmin(np.abs(y.real[:,None] - constelacao[None,:]), axis=1)
        erros += np.sum(detectado != indices)

    return erros/N


def sep_simulada(tipo_do_combinador, lista_snr_media, L, K=0):
    return np.array([sep_simulada_um_ponto(tipo_do_combinador, snr_media, L, K) for snr_media in lista_snr_media])


# %%
# Eixos (em dB)
limiares_db = np.linspace(-10, 20, 31)      # limiar da outage
snr_outage = db2lin(10)                     # SNR média por ramo fixa na outage
snr_sep_db = np.arange(0, 51, 2.5)          # 16-ASK precisa de bastante SNR
snr_cap_db = np.linspace(-5, 25, 31)

# N no nome do arquivo: evita carregar resultados de outra configuração
arquivo = f"resultados_rayleigh_{M}ASK_N{N:.0e}.npz"
resultados = dict(np.load(arquivo)) if os.path.exists(arquivo) else {}

# Salva após cada caso: se a execução parar, recomeça de onde parou
for tipo in combinadores: 
    for L in lista_de_L: 
        if f"cap_{tipo}_{L}" in resultados: 
            continue
        print(f"Simulando {tipo}, L = {L}...")
        resultados[f"op_{tipo}_{L}"] = outage_simulada(tipo, db2lin(limiares_db), snr_outage, L)
        resultados[f"sep_{tipo}_{L}"] = sep_simulada(tipo, db2lin(snr_sep_db), L)
        resultados[f"cap_{tipo}_{L}"] = capacidade_simulada(tipo, db2lin(snr_cap_db), L)
        np.savez(arquivo, **resultados)
print("Resultados em", arquivo)


# %%
def sem_zeros(y):
    """Pontos sem nenhum evento viram NaN para não quebrar o eixo log."""
    return np.where(y > 0, y, np.nan)


# Informações de cada métrica: eixo x, função teórica e rótulos
info_metricas = {
    "op": {
        "x_db": limiares_db,
        "teorica": lambda tipo, L: outage_teorica(tipo, db2lin(limiares_db), snr_outage, L),
        "xlabel": r"Limiar $\gamma_{th}$ (dB)",
        "ylabel": "Probabilidade de outage",
        "titulo": r"Probabilidade de outage ($\bar{\gamma}$ = 10 dB por ramo)",
        "log": True,
    },
    "sep": {
        "x_db": snr_sep_db,
        "teorica": lambda tipo, L: sep_teorica(tipo, db2lin(snr_sep_db), L),
        "xlabel": "SNR média por ramo (dB)",
        "ylabel": "SEP",
        "titulo": f"SEP {M}-ASK",
        "log": True,
    },
    "cap": {
        "x_db": snr_cap_db,
        "teorica": lambda tipo, L: capacidade_teorica(tipo, db2lin(snr_cap_db), L),
        "xlabel": "SNR média por ramo (dB)",
        "ylabel": "Capacidade ergódica (bits/s/Hz)",
        "titulo": "Capacidade ergódica",
        "log": False,
    },
}


def formatar_eixos(info):
    if info["log"]: 
        plt.yscale("log")
        plt.ylim(1e-7, 1)
    plt.xlabel(info["xlabel"])
    plt.ylabel(info["ylabel"])
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()


# %%

for metrica, info in info_metricas.items(): 
    for tipo in combinadores: 
        plt.figure(figsize=(6.4, 4.5))
        for L, cor in zip(lista_de_L, cores): 
            simulado = resultados[f"{metrica}_{tipo}_{L}"]
            if info["log"]: 
                simulado = sem_zeros(simulado)

            if tipo == "EGC": 
                plt.plot(info["x_db"], simulado, "-", color=cor, label=f"L = {L}")
            else: 
                plt.plot(info["x_db"], info["teorica"](tipo, L), "-", color=cor, label=f"L = {L}")
                plt.plot(info["x_db"], simulado, "o", color=cor, mfc="none")

        if tipo != "EGC": 
            # Entradas da legenda para explicar linha x marcador
            plt.plot([], [], "k-", label="Teórico")
            plt.plot([], [], "ko", mfc="none", label="Simulado")

        plt.title(f"{info['titulo']} – {tipo} (Rayleigh)")
        formatar_eixos(info)
        plt.savefig(f"figuras/item1_{metrica}_{tipo}.png", dpi=150)
        plt.show()


# %%
estilo = {"SC": ("C0", "s"), "EGC": ("C2", None), "MRC": ("C3", "o")}

for metrica, info in info_metricas.items(): 
    for L in lista_de_L[1:]:                    # L = 1 é igual para todos
        plt.figure(figsize=(6.4, 4.5))
        for tipo in combinadores: 
            cor, marcador = estilo[tipo]
            simulado = resultados[f"{metrica}_{tipo}_{L}"]
            if info["log"]: 
                simulado = sem_zeros(simulado)

            if tipo == "EGC": 
                plt.plot(info["x_db"], simulado, "-", color=cor, label="EGC (simulado)")
            else: 
                plt.plot(info["x_db"], info["teorica"](tipo, L), "-", color=cor, label=f"{tipo} (teórico)")
                plt.plot(info["x_db"], simulado, marcador, color=cor, mfc="none", label=f"{tipo} (simulado)")

        # Referência sem diversidade
        plt.plot(info["x_db"], info["teorica"]("SC", 1), "k:", label="Sem diversidade (L = 1)")

        plt.title(f"{info['titulo']} – L = {L} (Rayleigh)")
        formatar_eixos(info)
        plt.savefig(f"figuras/item2_{metrica}_L{L}.png", dpi=150)
        plt.show()


# %%
def snr_para_alvo(x_db, y, alvo):
    """SNR (dB) em que a curva cruza o valor alvo. Retorna NaN se estiver fora da faixa."""
    validos = y > 0
    if not np.any(validos): 
        return np.nan
    if alvo < y[validos].min() or alvo > y[validos].max(): 
        return np.nan
    return np.interp(np.log10(alvo), np.log10(y[validos])[::-1], x_db[validos][::-1])


alvo = 1e-3
print(f"SNR média por ramo necessária para SEP = {alvo:g} ({M}-ASK):")
for L in lista_de_L: 
    snr_alvo = {tipo: snr_para_alvo(snr_sep_db, resultados[f"sep_{tipo}_{L}"], alvo) for tipo in combinadores}
    print(f"  L = {L}:  SC = {snr_alvo['SC']:5.2f} dB | EGC = {snr_alvo['EGC']:5.2f} dB | "
          f"MRC = {snr_alvo['MRC']:5.2f} dB | perda EGC = {snr_alvo['EGC'] - snr_alvo['MRC']:.2f} dB | "
          f"perda SC = {snr_alvo['SC'] - snr_alvo['MRC']:.2f} dB")

indice_20db = np.argmin(np.abs(snr_cap_db - 20))
print("\nCapacidade a 20 dB por ramo (bits/s/Hz):")
for L in lista_de_L: 
    print(f"  L = {L}:  " + " | ".join(f"{tipo} = {resultados[f'cap_{tipo}_{L}'][indice_20db]:.3f}" for tipo in combinadores))


# %%

def ganho_de_diversidade(metrica_com_L, metrica_sem_diversidade, tipo_da_metrica):
    """
    Ganho de diversidade em dB, comparando L ramos com L = 1.

    - Capacidade: quanto maior, melhor. G = 10 log10(C_L / C_1)
    - SEP e outage: quanto menor, melhor. G = 10 log10(P_1 / P_L) 

    Pontos com probabilidade zero (sem eventos na simulação) viram NaN.
    """

    metrica_com_L = np.asarray(metrica_com_L, dtype=float)
    metrica_sem_diversidade = np.asarray(metrica_sem_diversidade, dtype=float)

    if tipo_da_metrica.lower() == "cap": 
        razao = metrica_com_L/metrica_sem_diversidade
    elif tipo_da_metrica.lower() in ["sep", "op"]: 
        razao = metrica_sem_diversidade/np.where(metrica_com_L > 0, metrica_com_L, np.nan)
    else: 
        raise ValueError(f"Métrica não existente: {tipo_da_metrica}")

    return 10*np.log10(razao)


def sep_monte_carlo(tipo_do_combinador, lista_snr_media, L, K=0):
    """
    SEP por Monte Carlo do canal: média da Pe condicional exata do M-ASK,
    Pe(snr) = aM*Q(sqrt(bM*snr)), com Q(x) = 0.5*erfc(x/sqrt(2)).
    """

    soma = np.zeros(len(lista_snr_media))
    for n in blocos(N): 
        ganho_do_canal = snr_combinador(gerar_canal(n,L,K), 1, tipo_do_combinador)
        soma += np.array([np.sum(aM*0.5*erfc(np.sqrt(bM*snr_media*ganho_do_canal/2)))
                          for snr_media in lista_snr_media])
    return soma/N


def outage_vs_snr(tipo_do_combinador, lista_snr_media, L, K=0, limiar=1):
    """Outage em função da SNR média por ramo, com limiar fixo (padrão: 0 dB)."""

    contagem = np.zeros(len(lista_snr_media))
    for n in blocos(N): 
        ganho_do_canal = snr_combinador(gerar_canal(n,L,K), 1, tipo_do_combinador)
        contagem += np.array([np.sum(snr_media*ganho_do_canal < limiar) for snr_media in lista_snr_media])
    return contagem/N


lista_de_K = [0, 1, 5, 10]                  # K = 0 é Rayleigh
K_destaque = 5                              # Rice usado nas figuras por combinador
lista_de_K_varredura = [0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
snr_ganho_db = np.arange(0, 91, 1.0)        # 16-ASK com L = 1 precisa de ~86 dB para SEP = 1e-7
snr_op_db = np.arange(-30, 81, 0.5)         # outage com limiar em 0 dB (L = 1 precisa de ~70 dB para 1e-7)
limite_confiavel_mc = 1e-8                  # conferir na figura de validação após rodar
limite_confiavel_op = 1e-7                  # abaixo disso há menos de 10 eventos de outage
estilo_marcador = {"SC": ("C0", "s"), "EGC": ("C2", "^"), "MRC": ("C3", "o")}

# Cache do Item 3: só simula o que ainda não estiver salvo
arquivo_item3 = f"resultados_item3_{M}ASK_N{N:.0e}.npz"
cache_item3 = dict(np.load(arquivo_item3)) if os.path.exists(arquivo_item3) else {}


def obter_resultado(metrica, tipo_do_combinador, L, K):
    """Busca no cache ou simula. Capacidade em snr_cap_db, SEP em snr_ganho_db e outage em snr_op_db."""

    chave = f"{metrica}_{tipo_do_combinador}_{L}_K{K}"
    if chave not in cache_item3: 
        print(f"Simulando {chave}...")
        if metrica == "cap": 
            cache_item3[chave] = capacidade_simulada(tipo_do_combinador, db2lin(snr_cap_db), L, K)
        elif metrica == "sep": 
            cache_item3[chave] = sep_monte_carlo(tipo_do_combinador, db2lin(snr_ganho_db), L, K)
        elif metrica == "op": 
            cache_item3[chave] = outage_vs_snr(tipo_do_combinador, db2lin(snr_op_db), L, K)
        else: 
            raise ValueError(f"Métrica não existente: {metrica}")
        salvar_cache_item3()       # salva a cada simulação nova
    return cache_item3[chave]


def sep_confiavel(sep):
    """Descarta (NaN) os pontos de SEP abaixo do limite de confiança do Monte Carlo."""
    return np.where(sep >= limite_confiavel_mc, sep, np.nan)


def op_confiavel(op):
    """Descarta (NaN) os pontos de outage abaixo do limite de confiança."""
    return np.where(op >= limite_confiavel_op, op, np.nan)


def salvar_cache_item3():
    np.savez(arquivo_item3, **cache_item3)


# %%
for tipo in combinadores: 
    plt.figure(figsize=(6.4, 4.5))
    for L, cor in zip(lista_de_L[1:], cores[1:]): 
        ganho_rayleigh = ganho_de_diversidade(obter_resultado("cap", tipo, L, 0),
                                              obter_resultado("cap", tipo, 1, 0), "cap")
        if tipo == "EGC": 
            plt.plot(snr_cap_db, ganho_rayleigh, "-", color=cor, label=f"L = {L}")
        else: 
            ganho_teorico = ganho_de_diversidade(capacidade_teorica(tipo, db2lin(snr_cap_db), L),
                                                 capacidade_teorica(tipo, db2lin(snr_cap_db), 1), "cap")
            plt.plot(snr_cap_db, ganho_teorico, "-", color=cor, label=f"L = {L}")
            plt.plot(snr_cap_db[::2], ganho_rayleigh[::2], "o", color=cor, mfc="none")

        # Rice (referência L = 1 também em Rice)
        ganho_rice = ganho_de_diversidade(obter_resultado("cap", tipo, L, K_destaque),
                                          obter_resultado("cap", tipo, 1, K_destaque), "cap")
        plt.plot(snr_cap_db, ganho_rice, "--", color=cor)

    plt.plot([], [], "k-", label="Rayleigh")
    plt.plot([], [], "k--", label=f"Rice (K = {K_destaque})")
    if tipo != "EGC": 
        plt.plot([], [], "ko", mfc="none", label="Rayleigh simulado")

    plt.xlabel("SNR média por ramo (dB)")
    plt.ylabel(r"$G_C = 10\log_{10}(\bar{C}_L/\bar{C}_1)$ (dB)")
    plt.title(f"Ganho de diversidade na capacidade – {tipo}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"figuras/item3_ganho_cap_{tipo}.png", dpi=150)
    plt.show()

salvar_cache_item3()


# %%
snr_fixa_db = 10
L_fixo = 4
indice_snr_fixa = np.argmin(np.abs(snr_cap_db - snr_fixa_db))

plt.figure(figsize=(6.4, 4.5))
for tipo in combinadores: 
    ganhos = [ganho_de_diversidade(obter_resultado("cap", tipo, L_fixo, K)[indice_snr_fixa],
                                   obter_resultado("cap", tipo, 1, K)[indice_snr_fixa], "cap")
              for K in lista_de_K_varredura]
    cor, marcador = estilo_marcador[tipo]
    plt.plot(lista_de_K_varredura, ganhos, marcador + "-", color=cor, mfc="none", label=tipo)

arranjo_cap = 10*np.log10(np.log2(1 + L_fixo*db2lin(snr_fixa_db))/np.log2(1 + db2lin(snr_fixa_db)))
plt.axhline(arranjo_cap, color="k", ls=":", label=f"Ganho de arranjo ({arranjo_cap:.2f} dB)")

plt.xscale("symlog", linthresh=1); plt.xlim(left=0)
plt.xlabel("Fator K de Rice (linear)  –  K = 0 é Rayleigh")
plt.ylabel(r"$G_C$ (dB)")
plt.title(f"Ganho na capacidade x K  (L = {L_fixo}, SNR = {snr_fixa_db} dB por ramo)")
plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("figuras/item3_ganho_cap_vs_K.png", dpi=150)
plt.show()

salvar_cache_item3()


# %%

for tipo in combinadores: 
    plt.figure(figsize=(6.4, 4.5))
    for L, cor in zip(lista_de_L, cores): 
        sep_mc = obter_resultado("sep", tipo, L, 0)
        if tipo == "EGC": 
            plt.plot(snr_ganho_db, sep_mc, "-", color=cor, label=f"L = {L}")
        else: 
            plt.plot(snr_ganho_db, sep_teorica(tipo, db2lin(snr_ganho_db), L), "-", color=cor, label=f"L = {L}")
            plt.plot(snr_ganho_db[::3], sep_mc[::3], "o", color=cor, mfc="none")
        plt.plot(snr_sep_db, sem_zeros(resultados[f"sep_{tipo}_{L}"]), "x", color=cor)

    if tipo != "EGC": 
        plt.plot([], [], "k-", label="Teórico")
        plt.plot([], [], "ko", mfc="none", label="Monte Carlo do canal")
    plt.plot([], [], "kx", label="Simulação de símbolos")

    plt.axhline(limite_confiavel_mc, color="gray", ls=":", label="Limite de confiança do MC")
    plt.yscale("log"); plt.ylim(1e-11, 1); plt.xlim(0, 80)
    plt.xlabel("SNR média por ramo (dB)"); plt.ylabel("SEP")
    plt.title(f"Validação da SEP {M}-ASK – {tipo} (Rayleigh)")
    plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"figuras/item3_validacao_sep_{tipo}.png", dpi=150)
    plt.show()

salvar_cache_item3()


# %%
for tipo in combinadores: 
    plt.figure(figsize=(6.4, 4.5))
    for L, cor in zip(lista_de_L[1:], cores[1:]): 
        ganho_rayleigh = ganho_de_diversidade(sep_confiavel(obter_resultado("sep", tipo, L, 0)),
                                              sep_confiavel(obter_resultado("sep", tipo, 1, 0)), "sep")
        ganho_rice = ganho_de_diversidade(sep_confiavel(obter_resultado("sep", tipo, L, K_destaque)),
                                          sep_confiavel(obter_resultado("sep", tipo, 1, K_destaque)), "sep")
        plt.plot(snr_ganho_db, ganho_rayleigh, "-", color=cor, label=f"L = {L}")
        plt.plot(snr_ganho_db, ganho_rice, "--", color=cor)

    plt.plot([], [], "k-", label="Rayleigh")
    plt.plot([], [], "k--", label=f"Rice (K = {K_destaque})")
    plt.xlim(0, 50)
    plt.xlabel("SNR média por ramo (dB)")
    plt.ylabel(r"$G_P = 10\log_{10}(\bar{P}_1/\bar{P}_L)$ (dB)")
    plt.title(f"Ganho de diversidade na SEP ({M}-ASK) – {tipo}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"figuras/item3_ganho_sep_razao_{tipo}.png", dpi=150)
    plt.show()


# %%

lista_de_alvos = np.logspace(-1, -7, 25)     

def economia_de_snr(sep_com_L, sep_sem_diversidade, alvos):
    """Pontos abaixo do limite de confiança são descartados antes de interpolar."""
    sep_com_L = sep_confiavel(sep_com_L)
    sep_sem_diversidade = sep_confiavel(sep_sem_diversidade)
    return np.array([snr_para_alvo(snr_ganho_db, sep_sem_diversidade, alvo)
                     - snr_para_alvo(snr_ganho_db, sep_com_L, alvo) for alvo in alvos])

for tipo in combinadores: 
    plt.figure(figsize=(6.4, 4.5))
    for L, cor in zip(lista_de_L[1:], cores[1:]): 
        economia_rayleigh = economia_de_snr(obter_resultado("sep", tipo, L, 0),
                                            obter_resultado("sep", tipo, 1, 0), lista_de_alvos)
        economia_rice = economia_de_snr(obter_resultado("sep", tipo, L, K_destaque),
                                        obter_resultado("sep", tipo, 1, K_destaque), lista_de_alvos)
        plt.plot(lista_de_alvos, economia_rayleigh, "-", color=cor, label=f"L = {L}")
        plt.plot(lista_de_alvos, economia_rice, "--", color=cor)

    plt.plot([], [], "k-", label="Rayleigh")
    plt.plot([], [], "k--", label=f"Rice (K = {K_destaque})")
    plt.xscale("log"); plt.gca().invert_xaxis()      
    plt.xlabel("SEP alvo")
    plt.ylabel(r"Economia de SNR $\Delta$ (dB)")
    plt.title(f"Ganho em SNR para atingir a SEP alvo ({M}-ASK) – {tipo}")
    plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"figuras/item3_ganho_sep_snr_{tipo}.png", dpi=150)
    plt.show()


# %%
alvo_fixo = 1e-3

plt.figure(figsize=(6.4, 4.5))
for tipo in combinadores: 
    economias = [economia_de_snr(obter_resultado("sep", tipo, L_fixo, K),
                                 obter_resultado("sep", tipo, 1, K), [alvo_fixo])[0]
                 for K in lista_de_K_varredura]
    cor, marcador = estilo_marcador[tipo]
    plt.plot(lista_de_K_varredura, economias, marcador + "-", color=cor, mfc="none", label=tipo)

# K -> infinito: canal AWGN, o MRC e o EGC somam a energia dos L ramos (10 log10 L)
plt.axhline(10*np.log10(L_fixo), color="k", ls=":", label=f"Ganho de arranjo ({10*np.log10(L_fixo):.2f} dB)")

plt.xscale("symlog", linthresh=1); plt.xlim(left=0)
plt.xlabel("Fator K de Rice (linear)  –  K = 0 é Rayleigh")
plt.ylabel(r"Economia de SNR $\Delta$ (dB)")
plt.title(f"Economia de SNR x K  (L = {L_fixo}, SEP alvo = {alvo_fixo:g})")
plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("figuras/item3_ganho_sep_vs_K.png", dpi=150)
plt.show()

salvar_cache_item3()


# %%

lista_de_alvos_op = np.logspace(-1, -6, 21)

def economia_de_snr_op(op_com_L, op_sem_diversidade, alvos):
    """Mesma ideia da economia de SNR da SEP, mas para a outage."""
    op_com_L = op_confiavel(op_com_L)
    op_sem_diversidade = op_confiavel(op_sem_diversidade)
    return np.array([snr_para_alvo(snr_op_db, op_sem_diversidade, alvo)
                     - snr_para_alvo(snr_op_db, op_com_L, alvo) for alvo in alvos])

for tipo in combinadores: 
    plt.figure(figsize=(6.4, 4.5))
    for L, cor in zip(lista_de_L[1:], cores[1:]): 
        economia_rayleigh = economia_de_snr_op(obter_resultado("op", tipo, L, 0),
                                               obter_resultado("op", tipo, 1, 0), lista_de_alvos_op)
        economia_rice = economia_de_snr_op(obter_resultado("op", tipo, L, K_destaque),
                                           obter_resultado("op", tipo, 1, K_destaque), lista_de_alvos_op)
        plt.plot(lista_de_alvos_op, economia_rayleigh, "-", color=cor, label=f"L = {L}")
        plt.plot(lista_de_alvos_op, economia_rice, "--", color=cor)

    plt.plot([], [], "k-", label="Rayleigh")
    plt.plot([], [], "k--", label=f"Rice (K = {K_destaque})")
    plt.xscale("log"); plt.gca().invert_xaxis()
    plt.xlabel("Probabilidade de outage alvo")
    plt.ylabel(r"Economia de SNR $\Delta$ (dB)")
    plt.title(f"Ganho em SNR para atingir a outage alvo – {tipo}")
    plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"figuras/item3_ganho_op_snr_{tipo}.png", dpi=150)
    plt.show()

salvar_cache_item3()


# %%
alvo_op_fixo = 1e-2

plt.figure(figsize=(6.4, 4.5))
for tipo in combinadores: 
    economias = [economia_de_snr_op(obter_resultado("op", tipo, L_fixo, K),
                                    obter_resultado("op", tipo, 1, K), [alvo_op_fixo])[0]
                 for K in lista_de_K_varredura]
    cor, marcador = estilo_marcador[tipo]
    plt.plot(lista_de_K_varredura, economias, marcador + "-", color=cor, mfc="none", label=tipo)

plt.axhline(10*np.log10(L_fixo), color="k", ls=":", label=f"Ganho de arranjo ({10*np.log10(L_fixo):.2f} dB)")

plt.xscale("symlog", linthresh=1); plt.xlim(left=0)
plt.xlabel("Fator K de Rice (linear)  –  K = 0 é Rayleigh")
plt.ylabel(r"Economia de SNR $\Delta$ (dB)")
plt.title(f"Economia de SNR na outage x K  (L = {L_fixo}, OP alvo = {alvo_op_fixo:g})")
plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("figuras/item3_ganho_op_vs_K.png", dpi=150)
plt.show()

salvar_cache_item3()

# %%