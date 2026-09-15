# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
# %%
# Frequência central da simulação (em GHz)
FREQUENCIA_CENTRAL = 3
# %%
# Classe para criar o espaço 3D
class Espaco3D:
    '''
    Cria o espaço 3D com determinado limite nos eixos x, y e z (contidos em [0, limite])
    e identificado com um título específico
    '''
    def __init__(self, limite, titulo):
        self.limite = limite
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(titulo)
        self._configurar_eixos()
        plt.close(self.fig)  

    def _configurar_eixos(self):
        L = self.limite

        self.ax.set_xlim([0, L])
        self.ax.set_ylim([0, L])
        self.ax.set_zlim([0, L])

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_box_aspect([1, 1, 1])

        # Eixos principais
        self.ax.plot([0, L], [0, 0], [0, 0], color='gray', linewidth=0.8)
        self.ax.plot([0, 0], [0, L], [0, 0], color='gray', linewidth=0.8)
        self.ax.plot([0, 0], [0, 0], [0, L], color='gray', linewidth=0.8)

        # Origem
        self.ax.scatter([0], [0], [0], color='black', s=20)

        y = np.linspace(0, L, 2)
        z = np.linspace(0, L, 2)
        Y, Z = np.meshgrid(y, z)
        X = np.zeros_like(Y)
        self.ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.15)

    def adicionar_pontos(self, x, y, z, cor, rotulo, tamanho=40, e_bs=False):
        """
        e_bs: marque True se este ponto representa a estação base.
        Só é permitido ter UMA estação base por espaço.
        """
        if e_bs:
            if self._bs_adicionada:
                raise ValueError(
                    "Já existe uma estação base registrada neste espaço. "
                    "Remova ou recrie o espaço antes de adicionar outra."
                )
            self._bs_adicionada = True

        self.ax.scatter([x], [y], [z], color=cor, s=tamanho)
        self.ax.text(x, y, z, f"    {rotulo}", fontsize=10)

    def mostrar(self):
        display(self.fig)

# %%

# Colocando a BS e a UT no espaço 

def adicionar_BS_e_UT(tipo_do_ambiente,phi0_rad,show_image = False):
    
    # Criando o espaço 3D
    espaco = Espaco3D(35,"Espaço 3D para simulação")
    phi0 = np.radians(phi0_rad)

    if tipo_do_ambiente == "UMa": 
        # Posição da BS
        x_BS = y_BS = 0
        z_BS = 25
        espaco.adicionar_pontos(0,0,z_BS,"red",rotulo="BS")

        # Posição inicial da UT
        d = 35 
        # Coordenadas x e y
        x_UT = d*np.cos(phi0)
        y_UT = d*np.sin(phi0)
        z_UT = 1.5
        espaco.adicionar_pontos(x_UT,y_UT,z_UT,"blue",rotulo="UT")

    elif tipo_do_ambiente == "UMi": 
        x_BS = y_BS = 0
        z_BS = 10
        espaco.adicionar_pontos(0,0,z_BS,"red",rotulo="BS")

        d = 10
        x_UT = d*np.cos(phi0)
        y_UT = d*np.sin(phi0)
        z_UT = 1.5
        espaco.adicionar_pontos(x_UT,y_UT,z_UT,"blue",rotulo="UT")

    elif tipo_do_ambiente == "indoor": 
        x_BS = y_BS = 0
        z_BS = 3
        espaco.adicionar_pontos(0,0,z_BS,"red",rotulo="BS")

        d = 0
        x_UT = d*np.cos(phi0)
        y_UT = d*np.sin(phi0)
        z_UT = 1
        espaco.adicionar_pontos(x_UT,y_UT,z_UT,"blue",rotulo="UT")


    dic_posicoes_iniciais = {
        "Posicoes BS" : [0,0,z_BS], 
        "Posicoes UT" : [x_UT,y_UT,z_UT]
    }
    if show_image:
        return espaco.mostrar(), dic_posicoes_iniciais
    else: 
        return dic_posicoes_iniciais

# %%
# Criando o cenário de simulação: classe que irá guardar todos esses
# parâmetros 

class CenarioSimulacao: 
    '''
    Representa o cenário completo de simulação: posição da BS, UT, ambiente, 
    ângulos de chegada e saída e etc.
    '''

    # Dicionário com todas as informações de posicionamento de BS e UT
    PARAMETROS_DO_AMBIENTE = {
        "UMa":    {"h_bs": 25, "h_ut": 1.5, "d_2D": 35},
        "UMi":    {"h_bs": 10, "h_ut": 1.5, "d_2D": 10},
        "indoor": {"h_bs": 3,  "h_ut": 1,   "d_2D": 5},
    }

    # Parâmetro dos espalhamentos de atraso
    PARAMETROS_AESP = {
        ("UMi", True):     {"mu": lambda fc: -0.24*np.log10(1+fc) - 7.14,  "sigma": 0.38},
        ("UMi", False):    {"mu": lambda fc: -0.24*np.log10(1+fc) - 6.83,  "sigma": lambda fc: 0.16*np.log10(1+fc) + 0.28},
        ("UMa", True):     {"mu": lambda fc: -6.955 - 0.0963*np.log10(fc), "sigma": 0.66},
        ("UMa", False):    {"mu": lambda fc: -6.28 - 0.204*np.log10(fc),   "sigma": 0.39},
        ("indoor", True):  {"mu": lambda fc: -0.01*np.log10(1+fc) - 7.692, "sigma": 0.18},
        ("indoor", False): {"mu": lambda fc: -0.28*np.log10(1+fc) - 7.173, "sigma": lambda fc: 0.10*np.log10(1+fc) + 0.055},
    }

    # Parâmetros do fator de Rice 
    PARAMETROS_RICE = {
        "UMi":    {"mu": 9, "sigma": 5},
        "UMa":    {"mu": 9, "sigma": 3.5},
        "indoor": {"mu": 7, "sigma": 4},
    }

    def __init__(self,tipo_do_ambiente, phi0_graus, limite_espaco):

        parametros = self.PARAMETROS_DO_AMBIENTE[tipo_do_ambiente] 
        self.tipo_do_ambiente = tipo_do_ambiente

        # Posição da BS 
        self.posicao_BS = (0,0,parametros["h_bs"])
        # Ângulo inicial em radianos 
        phi0 = np.radians(phi0_graus)
        # Distância no plano xy entre BS e UT
        d = parametros["d_2D"]
        x = d*np.cos(phi0)
        y = d*np.sin(phi0)

        # Posição de UT
        self.posicao_ut = (x,y,parametros["h_ut"])

        # Criando espaço 3D 

        self.espaco = Espaco3D(limite_espaco, f"Espaço 3D para simulação ({tipo_do_ambiente})")
        self.espaco.adicionar_pontos(*self.posicao_BS, "red", rotulo="BS")
        self.espaco.adicionar_pontos(*self.posicao_ut, "blue", rotulo="UT")

    def angulos_LoS(self): 
        x_bs, y_bs, z_bs = self.posicao_BS
        x_ut, y_ut, z_ut = self.posicao_ut

        dx, dy, dz = x_ut - x_bs, y_ut - y_bs, z_ut - z_bs
        d_2D = np.sqrt(dx**2 + dy**2)

        # Retornando os ângulos de chegada e saída
        return {
            "phi_aod": np.degrees(np.arctan2(dy, dx)),
            "phi_aoa": np.degrees(np.arctan2(-dy, -dx)),
            "theta_eod": np.degrees(np.arctan2(dz, d_2D)),
            "theta_eoa": np.degrees(np.arctan2(-dz, d_2D)),
        }
    
    # Faz o plot no espaço 3D dos ângulos em questão
    def plotar_angulos_LoS(self, raio_azimute=None, raio_elevacao=None):
        angulos = self.angulos_LoS()
        phi_aod = np.radians(angulos["phi_aod"])
        theta_eod = np.radians(angulos["theta_eod"])

        x_bs, y_bs, z_bs = self.posicao_BS
        x_ut, y_ut, z_ut = self.posicao_ut
        ax = self.espaco.ax

        d_2D = np.sqrt(x_ut**2 + y_ut**2)
        if raio_azimute is None:
            raio_azimute = d_2D * 0.35
        if raio_elevacao is None:
            raio_elevacao = d_2D * 0.25

        # linha reta BS para UT (o vetor LoS)
        ax.plot([x_bs, x_ut], [y_bs, y_ut], [z_bs, z_ut], color='green', linewidth=1.5)

        # projeção do UT no plano horizontal da BS (mostra a diferença de altura)
        ax.plot([x_bs, x_ut], [y_bs, y_ut], [z_bs, z_bs], 'k--', linewidth=0.8)
        ax.plot([x_ut, x_ut], [y_ut, y_ut], [z_bs, z_ut], 'k--', linewidth=0.8)

        # linha vertical tracejada ligando a origem até a BS 
        ax.plot([0, x_bs], [0, y_bs], [0, z_bs], color='gray', linestyle=':', linewidth=0.8)

        t = np.linspace(0, phi_aod, 30)
        xs = raio_azimute * np.cos(t)
        ys = raio_azimute * np.sin(t)
        zs = np.zeros_like(t)
        ax.plot(xs, ys, zs, color='orange', linewidth=2)

        ax.plot([x_bs, x_bs], [y_bs, y_bs], [0, z_bs], color='gold', linestyle='--', linewidth=1.3)
        t_meio = phi_aod / 2
        ax.text(raio_azimute*1.15*np.cos(t_meio),
                raio_azimute*1.15*np.sin(t_meio),
                0,
                f"φ={angulos['phi_aod']:.1f}°", color='orange', fontsize=9)

        e_h = np.array([np.cos(phi_aod), np.sin(phi_aod), 0])
        e_v = np.array([0, 0, 1])

        t2 = np.linspace(0, theta_eod, 30)
        origem_bs = np.array([x_bs, y_bs, z_bs])
        pontos_arco = origem_bs + raio_elevacao * (np.cos(t2)[:, None]*e_h + np.sin(t2)[:, None]*e_v)
        ax.plot(pontos_arco[:, 0], pontos_arco[:, 1], pontos_arco[:, 2], color='purple', linewidth=2)

        t2_meio = theta_eod / 2
        ponto_texto = origem_bs + raio_elevacao*1.2*(np.cos(t2_meio)*e_h + np.sin(t2_meio)*e_v)
        ax.text(*ponto_texto, f"θ={angulos['theta_eod']:.1f}°", color='purple', fontsize=9)

    def mobilibdade(self, v_kmh, phi0v_graus, theta0v_graus = 0):
        ''''
        Método que define a velocidade e sua orientação (azimutal por phi0v_graus e elevacional por 
        theta0v_graus)
        '''
        phi_v = np.radians(phi0v_graus)
        theta_v = np.radians(theta0v_graus)

        # Definindo o vetor unitário da velocidade
        v_hat = np.array([
            np.cos(theta_v) * np.cos(phi_v),
            np.cos(theta_v) * np.sin(phi_v),
            np.sin(theta_v)
        ])

        self.velocidade = v_kmh
        self.v_hat = v_hat 
        return v_hat

    
    def plotar_vetor_velocidade(self, escala=None, cor='magenta'):
        ''''
        Plota o vetor no espaço 3D 
        '''
        x_ut, y_ut, z_ut = self.posicao_ut
        ax = self.espaco.ax

        # escala automática: proporcional à distância BS-UT, se não for passada na mão
        if escala is None:
            d_2D = np.sqrt(x_ut**2 + y_ut**2)
            escala = d_2D * 0.3

        dx, dy, dz = self.v_hat * escala

        ax.quiver(x_ut, y_ut, z_ut, dx, dy, dz,
                color=cor, arrow_length_ratio=0.2, linewidth=2)

        ax.text(x_ut + dx*1.15, y_ut + dy*1.15, z_ut + dz*1.15,
                f"v = {self.velocidade:.1f} km/h", color=cor, fontsize=9)

    # Cálculos das probabilidades de LoS (visada direta)
    def probabilidade_LoS(self,d_2D = None): 

        if d_2D is None:
            x_ut, y_ut, _ = self.posicao_ut
            d_2D = np.sqrt(x_ut**2 + y_ut**2)


        _,_,h_ut = self.posicao_ut

        if self.tipo_do_ambiente == "UMi":
            if d_2D <= 18:
                return 1.0
            return 18/d_2D + np.exp(-d_2D/36) * (1 - 18/d_2D)

        elif self.tipo_do_ambiente == "UMa":
            if d_2D <= 18:
                return 1.0
            if h_ut <= 13:
                C = 0.0
            else:
                C = ((h_ut - 13) / 10) ** 1.5

            return  (18/d_2D + np.exp(-d_2D/63) * (1 - 18/d_2D))*(1 + C * (5/4) * (d_2D/100)**3 * np.exp(-d_2D/150))

        elif self.tipo_do_ambiente == "indoor":
            if d_2D <= 1.2:
                return 1.0
            elif d_2D < 6.5:
                return np.exp(-(d_2D - 1.2) / 4.7)
            else:
                return np.exp(-(d_2D - 6.5) / 32.6) * 0.32

    # Cospe a variável aleatória (com distribuição uniforme para definir se é LoS (1) ou NLoS (0))
    def variavel_aleatoria_LoS(self):

        p_los = self.probabilidade_LoS()
        x = np.random.uniform(0,1)

        # Se estado_LoS = True: LoS
        # Se estado_LoS = False: NLoS
        self.estado_LoS = x < p_los

        return self.estado_LoS

    def gerar_espalhamento_de_atraso(self):
        '''
        Gera o espalhamento de atraso baseado no estado gerado pela variável aleatória (LoS ou NLoS)
        '''

        fc = FREQUENCIA_CENTRAL
        # Tupla para pegar o equivalente no dicionário PARAMETROS_AESP
        chave = (self.tipo_do_ambiente,self.estado_LoS)
        parametros = self.PARAMETROS_AESP[chave]

        media = parametros["mu"](fc)

        if callable(parametros["sigma"]):
            desvio_padrao = parametros["sigma"](fc)
        else: 
            desvio_padrao = parametros["sigma"]

        # Espalhamento de atraso em escala logarítmica (distribuição normal de média = media e desvio
        # padrão = desvio_padrao)
        log_EA = np.random.normal(media, desvio_padrao)

        self.std_espalhamento = 10**log_EA

        return self.std_espalhamento


    def gerar_fator_rice(self):
        '''
        Calcula o fator de Rice com base no ambiente
        '''
        # Se não é visada direta, então K = 0 
        if not self.estado_LoS: 
            self.K_R = None
            return self.K_R

        parametros = self.PARAMETROS_RICE[self.tipo_do_ambiente]
        K_dB = np.random.normal(parametros["mu"], parametros["sigma"])

        self.K_R = 10**(K_dB/10)
        return self.K_R
    
        
    def mostrar(self):
        self.espaco.mostrar()


# Teste do plot dos ângulos de saída
cenario = CenarioSimulacao("UMi", phi0_graus=30, limite_espaco=35)
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
