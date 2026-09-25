# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
# %%
# Frequência central da simulação (em GHz)
FREQUENCIA_CENTRAL = 3
# Velocidade da luz: constante global
C = 3e8
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

    # Parâmetros de espalhamento angular AoD (Azimutal de saída)
    PARAMETROS_AOD_AZI = {
        ("UMi", True):     {"mu": lambda fc: -0.05*np.log10(1+fc)+1.21, "sigma": 0.41},
        ("UMi", False):    {"mu": lambda fc: -0.23*np.log10(1+fc)+1.53, "sigma": lambda fc: 0.11*np.log10(1+fc)+0.33},
        ("UMa", True):     {"mu": lambda fc: 1.06+0.1114*np.log10(fc),  "sigma": 0.28},
        ("UMa", False):    {"mu": lambda fc: 1.5-0.1144*np.log10(fc),   "sigma": 0.28},
        ("indoor", True):  {"mu": 1.60, "sigma": 0.18},
        ("indoor", False): {"mu": 1.62, "sigma": 0.25},
    }

    # Parâmetros de espalhamento angular AoA (Azimutal de chegada)
    PARAMETROS_AOA_AZI = {
        ("UMi", True):     {"mu": lambda fc: -0.08*np.log10(1+fc)+1.73, "sigma": lambda fc: 0.014*np.log10(1+fc)+0.28},
        ("UMi", False):    {"mu": lambda fc: -0.08*np.log10(1+fc)+1.81, "sigma": lambda fc: 0.05*np.log10(1+fc)+0.3},
        ("UMa", True):     {"mu": 1.81, "sigma": 0.20},
        ("UMa", False):    {"mu": lambda fc: 2.08-0.27*np.log10(fc),    "sigma": 0.11},
        ("indoor", True):  {"mu": lambda fc: -0.19*np.log10(1+fc)+1.781, "sigma": lambda fc: 0.12*np.log10(1+fc)+0.119},
        ("indoor", False): {"mu": lambda fc: -0.11*np.log10(1+fc)+1.863, "sigma": lambda fc: 0.12*np.log10(1+fc)+0.059},
    }

    # Parâmetros de espalhamento angular AoA (elevacional de chegada)
    PARAMETROS_AOA_ELE = {
        ("UMi", True):     {"mu": lambda fc: -0.1*np.log10(1+fc)+0.73,  "sigma": lambda fc: -0.04*np.log10(1+fc)+0.34},
        ("UMi", False):    {"mu": lambda fc: -0.04*np.log10(1+fc)+0.92, "sigma": lambda fc: -0.07*np.log10(1+fc)+0.41},
        ("UMa", True):     {"mu": 0.95, "sigma": 0.16},
        ("UMa", False):    {"mu": lambda fc: -0.3236*np.log10(fc)+1.512, "sigma": 0.16},
        ("indoor", True):  {"mu": lambda fc: -0.26*np.log10(1+fc)+1.44, "sigma": lambda fc: -0.04*np.log10(1+fc)+0.264},
        ("indoor", False): {"mu": lambda fc: -0.15*np.log10(1+fc)+1.387, "sigma": lambda fc: -0.09*np.log10(1+fc)+0.746},
    }

    # Parâmetro do fator de proporcionalidade
    PARAMETRO_R_TAU = {
        ("UMi", True):     3.0,
        ("UMi", False):    2.1,
        ("UMa", True):     2.5,
        ("UMa", False):    2.3,
        ("indoor", True):  3.6,
        ("indoor", False): 3.0,         
    }

    # Parâmetros para sombra
    PARAMETROS_SOMBREAMENTO = {
        ("UMi", True):     3,
        ("UMi", False):    3,
        ("UMa", True):     3,
        ("UMa", False):    3,
        ("indoor", True):  6,
        ("indoor", False): 3,
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

    def variavel_aleatoria_LoS(self):
        """
        Cospe a variável aleatória (com distribuição uniforme para definir se é LoS (1) ou NLoS (0))
        """
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

    def _avaliar(self,parametro, fc): 
        '''
        Função auxiliar: resolve um parâmetro que pode ser fixo ou em função de fc
        '''
        if callable(parametro): 
            return parametro(fc)
        else: 
            return parametro

    def _media_std_AoD_saida(self):
        '''
        Calcula a média (mu) e o desvio-padrão (std ou sigma) para o ângulo de saída 
        elevacional 
        '''

        fc = FREQUENCIA_CENTRAL

        # Posições da UT 
        x_ut, y_ut, h_ut = self.posicao_ut
        # Posição da BS
        _,_,h_bs = self.posicao_BS

        # Calcula a distância no plano xy em km
        d_2D_km = np.sqrt(x_ut**2 + y_ut**2) / 1000

        amb, LoS = self.tipo_do_ambiente , self.estado_LoS

        if amb == "UMa":
            if LoS:
                media = max(-0.5, -2.1*d_2D_km - 0.01*(h_ut-1.5) + 0.75)
                std = 0.40
            else:
                media = max(-0.5, -2.1*d_2D_km - 0.01*(h_ut-1.5) + 0.9)
                std = 0.49

        elif amb == "UMi":
            if LoS:
                media = max(-0.21, -14.8*d_2D_km + 0.01*abs(h_ut - h_bs) + 0.83)
                std = 0.35
            else:
                media = max(-0.5, -3.1*d_2D_km + 0.01*max(h_ut - h_bs, 0) + 0.2)
                std = 0.35

        elif amb == "indoor":
            if LoS:
                media = -1.43*np.log10(1+fc) + 2.228
                std = 0.13*np.log10(1+fc) + 0.30
            else:
                media = 1.08
                std = 0.36
        
        return media, std   

    def gerar_espalhamento_angular(self): 
        '''
        Gera o espalhamento angular para os quatro casos (AoA e AoD elevacional 
        e azimutal)
        ''' 

        # Depende da geração do estado LoS
        if not hasattr(self, "estado_LoS"): 
            raise RuntimeError("Estado LoS não gerado")

        fc = FREQUENCIA_CENTRAL
        # Identificando o problema com tipo do ambiente e o estaod LoS
        chave = (self.tipo_do_ambiente, self.estado_LoS)

        # Pegando o espalhamento azimutal de chegada
        p_aoa_azi = self.PARAMETROS_AOA_AZI[chave]
        # Pegando o espalhamento azimutal de saída
        p_aod_azi = self.PARAMETROS_AOD_AZI[chave]
        # Pegando o espalhamento elevacional de chegada
        p_aoa_ele = self.PARAMETROS_AOA_ELE[chave]

        # A partir dos parâmetros, retiramos as médias e desvios padrão
        media_aoa_azi , std_aoa_azi = self._avaliar(p_aoa_azi["mu"], fc), self._avaliar(p_aoa_azi["sigma"], fc)
        media_aod_azi , std_aod_azi = self._avaliar(p_aod_azi["mu"], fc), self._avaliar(p_aod_azi["sigma"], fc)
        media_aoa_ele , std_aoa_ele = self._avaliar(p_aoa_ele["mu"], fc), self._avaliar(p_aoa_ele["sigma"], fc)

        media_aod_ele, std_aod_ele = self._media_std_AoD_saida()

        # Distribuições normais dos ângulos de chegada e saída 
        sigma_phi_AoD   = 10 ** np.random.normal(media_aod_azi, std_aod_azi)
        sigma_phi_AoA   = 10 ** np.random.normal(media_aoa_azi, std_aoa_azi)
        sigma_theta_AoD = 10 ** np.random.normal(media_aod_ele, std_aod_ele)
        sigma_theta_AoA = 10 ** np.random.normal(media_aoa_ele, std_aoa_ele)

        # Implementando os limites de 104 graus para azimute e 52 graus para elevação
        self.sigma_phi_AoD   = min(sigma_phi_AoD, 104)
        self.sigma_phi_AoA   = min(sigma_phi_AoA, 104)
        self.sigma_theta_AoD = min(sigma_theta_AoD, 52)
        self.sigma_theta_AoA = min(sigma_theta_AoA, 52)

        return {
            "sigma_phi_AoD": self.sigma_phi_AoD,
            "sigma_phi_AoA": self.sigma_phi_AoA,
            "sigma_theta_AoD": self.sigma_theta_AoD,
            "sigma_theta_AoA": self.sigma_theta_AoA,
        }

    
    def gerar_atrasos_multipercursos(self, N=100):
        '''
        Gera as N componentes de atrasos multipercursos. Requer que o espalhamento de atraso já tenha sido 
        calculado. Retorna os atrasados ordenados. 
        '''

        if not hasattr(self, 'std_espalhamento'): 
            raise RuntimeError('Espalhamento de atraso ainda não gerado.')

        chave = (self.tipo_do_ambiente, self.estado_LoS)
        r_tau = self.PARAMETRO_R_TAU[chave]

        # A média é o produto do fator de proporcionalidade pelo sigma_tau (desvio-padrão do espalhamento de atraso)
        # Essa é a média da distribuição exponencial dos atrasos
        media_tau = r_tau * self.std_espalhamento

        # PDF dos atrasos
        f_tau = np.random.exponential(scale=media_tau, size=N)

        tau_p = f_tau - np.min(f_tau)
        tau_n = np.sort(tau_p)

        self.N = N 
        self.r_tau = r_tau
        self.atrasos = tau_n

        return tau_n

    def gerar_potencia_multipercurso(self): 
        """
        Gera a potência de cada componente multipercurso, dependendo do caso de LoS ou NLoS.
        """

        # Precisa gerar o atraso multipercurso 
        if not hasattr(self,'atrasos'): 
            raise RuntimeError("Atrasos ainda não foram gerados.")

        # Sombreamento 
        std_sombreamento = self.PARAMETROS_SOMBREAMENTO[(self.tipo_do_ambiente, self.estado_LoS)]

        # Gerando os termos de sombreamento (variável aleatória com desvio-padrão de std_sombreamento)
        n_somb = np.random.normal(0,std_sombreamento,self.N)

        # Potência preliminar 
        alpha_2 = np.exp(-self.atrasos * (self.r_tau - 1)/(self.r_tau * self.std_espalhamento)) * 10**(-n_somb/10)

        # Normalizar a potência dependendo do estado de visada direta 
        # Se tiver visada direta
        if self.estado_LoS: 
            omega_c = np.sum(alpha_2[1:])

            alpha_2_norm = np.zeros(self.N)
            alpha_2_norm[1:] = (1 / (self.K_R + 1)) * alpha_2[1:] / omega_c
            alpha_2_norm[0] = self.K_R / (self.K_R + 1)
        else: 
            omega_c = np.sum(alpha_2[1:])
            alpha_2_norm = alpha_2/omega_c

        self.potencias = alpha_2_norm

        return alpha_2_norm

    def gerar_angulos_chegada(self): 
        """
        Gera o ângulo de chegada azimutal e elevacional (phi e theta), precisando dos valores das potências 
        e dos desvios-padrão dos ângulos de chegadas azimutais.
        """

        # Verificando se as potências e o desvio-padrão do ângulo azimutal de 
        # chegada já foram gerados 
        if not hasattr(self,"sigma_theta_AoA"):
            raise RuntimeError("Gere antes o desvio-padrão dos ângulos elevacionais de chegada")
        if not hasattr(self,'potencias'): 
            raise RuntimeError("Gere primeiramente as potências.")
        if not hasattr(self,"sigma_phi_AoA"): 
            raise RuntimeError("Gere o desvio-padrão do ângulo de chegada primeiramente.")

        # Valor máximo da potência 
        max_alpha_2 = max(self.potencias)

        # Valores dos ângulos iniciais 
        phi_inicial_n = 1.42*self.sigma_phi_AoA*np.sqrt(-np.log(self.potencias/max_alpha_2))
        theta_inicial_n = -self.sigma_theta_AoA*np.log(self.potencias/max_alpha_2)

        # Gerando sinais aleatórios
        U_n = np.random.choice([-1, 1], size=self.N)
        # Flutuações aleatórias 
        Y_n_phi = np.random.normal(0, self.sigma_phi_AoA/7, self.N)
        Y_n_theta = np.random.normal(0, self.sigma_theta_AoA/7, self.N)

        phi_LoS = self.angulos_LoS()["phi_aoa"]
        theta_LoS = self.angulos_LoS()["theta_eoa"]

        phi_n = U_n*phi_inicial_n + Y_n_phi + phi_LoS
        theta_n = U_n*theta_inicial_n + Y_n_theta + theta_LoS

        if self.estado_LoS:
            phi_n[0] = phi_LoS 
            theta_n[0] = theta_LoS

        self.angulo_azimutal_chegada = phi_n
        self.angulo_elevacional_chegada = theta_n

        return phi_n, theta_n

    def gerar_vetor_chegada(self): 
        """
        Gera o vetor de chegada com base nos ângulos elevacionais e azimutais 
        de chegada.
        """

        # Precisa dos theta e phi 
        if not hasattr(self,'angulo_azimutal_chegada'): 
            raise RuntimeError("Precisa gerar ângulo azimutal de chegada")
        if not hasattr(self,"angulo_elevacional_chegada"): 
            raise RuntimeError("Precisa gerar ângulo elevacional de chegada")

        # ângulo azimutal em radiano 
        phi_rad = np.radians(self.angulo_azimutal_chegada)
        # ângulo elevacional está representado por enquanto na horizontal, quando precisaria estar na vertical
        # solução é atrasar 90 graus antes de passar para radiano 
        theta_rad = np.radians(90-self.angulo_elevacional_chegada)

        x = np.cos(phi_rad)*np.sin(theta_rad)
        y = np.sin(phi_rad)*np.sin(theta_rad)
        z = np.cos(theta_rad)

        r_n = np.column_stack((x,y,z))

        self.vetores_direcao_chegada = r_n

        return r_n

    def gerar_desvio_doppler(self): 
        """
        Gera o desvio doppler f_n, com velocidade v_rx do receptor. 
        """
        if not hasattr(self,"vetores_direcao_chegada"): 
            raise RuntimeError("Precisa gerar o vetor de direção de chegada")
        if not hasattr(self,"v_hat"): 
            raise RuntimeError("Precisa gerar a mobilidade (vetor velocidade)")

        comp_de_onda = C/(FREQUENCIA_CENTRAL*1e9)

        # velocidade em m/s
        v = self.velocidade/3.6
        v_vetor = v*self.v_hat

        # Produto escalar
        f_n = (1/comp_de_onda)*np.sum(self.vetores_direcao_chegada*v_vetor, axis=1)

        self.desvio_doppler = f_n
        return f_n

    def gerar_sinal_recebido(self, delta_t, N_t = 1e5, t_max_factor = 5): 
        """"
        Gera o sinal recebido, combinando todos os parâmetros calculados anteriormente.
        O pulso gerado (com largura delta_t) e transmitido é s(t) e o recebido é r(t), considerando as componentes 
        multipercurso.


        """

        # Precisamos gerar o desvio doppler 
        if not hasattr(self, "desvio_doppler"): 
            raise RuntimeError("Precisa gerar o desvio doppler")

        # Precisamos gerar os atrasos 
        if not hasattr(self, "atrasos"): 
            raise RuntimeError("Precisa gerar os atrasos")
        # Precisamos gerar as potências 
        if not hasattr(self, "potencias"): 
            raise RuntimeError("Precisa gerar as potências")
        fc_hz = FREQUENCIA_CENTRAL * 1e9   # GHz -> Hz, essencial aqui!

        fases_estaticas = 2 * np.pi * (fc_hz + self.desvio_doppler) * self.atrasos
        # Vetor temporal 
        t = np.linspace(0,t_max_factor*delta_t, N_t)

        # Sinal transmitido. Gera o pulso de amplitude 1 se t está entre 0 e delta_t
        s_t = ((t >= 0) & (t < delta_t)).astype(float)

        # Amplitudes: raiz quadrada da potência
        alpha_n = np.sqrt(self.potencias)

        # Sinal recebido
        r_t = np.zeros(N_t, dtype=complex)

        for n in range(self.N): 
            # Deslocando o pulso no tempo, devido aos atrasos
            s_deslocado = ((t - self.atrasos[n] >= 0) & (t - self.atrasos[n] < delta_t)).astype(float)

            # A fase completa 
            fase_n_t = fases_estaticas[n] - 2 * np.pi * self.desvio_doppler[n] * t

            # Agora, o sinal recebido completo
            r_t += alpha_n[n] * np.exp(-1j * fase_n_t) * s_deslocado

        self.tempo = t
        self.sinal_transmitido = s_t
        self.sinal_recebido = r_t
        return t, s_t, r_t

    def gerar_autocorrelacao(self, kappa = 0, sigma = 0):
        """
        Calcula a função de autocorrelação normalizada do canal.
        """ 

        if not hasattr(self, 'potencias') or not hasattr(self, 'atrasos') or not hasattr(self, 'desvio_doppler'):
            raise RuntimeError("Gere potências, atrasos e desvios Doppler antes.")

        Omega_c = np.sum(self.potencias)
        kappa = np.atleast_1d(kappa)
        sigma = np.atleast_1d(sigma)


        fase_atraso = np.exp(-1j * 2*np.pi * np.outer(kappa, self.atrasos))   
        fase_doppler = np.exp(1j * 2*np.pi * np.outer(sigma, self.desvio_doppler))  

        if len(kappa) >= len(sigma):
            # Produto matricial, por isso o @
            rho = (fase_atraso @ (self.potencias * fase_doppler[0])) / Omega_c
        else:
            rho = (fase_doppler @ (self.potencias * fase_atraso[0])) / Omega_c

        return rho if rho.size > 1 else rho[0]

    def calcular_banda_coerencia(self, limiares=(0.95, 0.8), kappa_max=1e10, n_pontos=3000):
        """
        Calcula a banda de coerência.
        """
        Omega_c = np.sum(self.potencias)
        kappas = np.logspace(0, np.log10(kappa_max), n_pontos)  

        fase = np.exp(-1j * 2*np.pi * np.outer(kappas, self.atrasos))  
        rho = np.abs(fase @ self.potencias) / Omega_c

        resultado = {}
        for rho_B in limiares:
            abaixo = np.where(rho < rho_B)[0]
            resultado[rho_B] = kappas[abaixo[0]] if len(abaixo) > 0 else kappa_max

        self.banda_coerencia = resultado
        self._kappas_teste, self._rho_kappa = kappas, rho  
        return resultado

    def calcular_tempo_coerencia(self, limiares=(0.95, 0.8), sigma_max=1.0, n_pontos=3000):
        """
        Calcula o tempo de coerência.
        """
        Omega_c = np.sum(self.potencias)
        sigmas = np.logspace(-6, np.log10(sigma_max), n_pontos)

        fase = np.exp(1j * 2*np.pi * np.outer(sigmas, self.desvio_doppler))  
        rho = np.abs(fase @ self.potencias) / Omega_c

        resultado = {}
        for rho_T in limiares:
            abaixo = np.where(rho < rho_T)[0]
            resultado[rho_T] = sigmas[abaixo[0]] if len(abaixo) > 0 else sigma_max

        self.tempo_coerencia = resultado
        self._sigmas_teste, self._rho_sigma = sigmas, rho
        return resultado




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
# %%
