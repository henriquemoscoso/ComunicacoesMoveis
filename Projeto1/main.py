# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
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

def adicionar_BS_e_UT(tipo_do_ambiente,phi0_rad):

    # Criando o espaço 3D
    espaco = Espaco3D(35,"Espaço 3D para simulação")
    phi0 = np.radians(phi0_rad)

    if tipo_do_ambiente == "UMa": 
        # Posição da BS
        espaco.adicionar_pontos(0,0,25,"red",rotulo="BS")

        # Posição inicial da UT
        d = 35 
        # Coordenadas x e y
        x = d*np.cos(phi0)
        y = d*np.sin(phi0)
        espaco.adicionar_pontos(x,y,1.5,"blue",rotulo="UT")

    elif tipo_do_ambiente == "UMi": 

        espaco.adicionar_pontos(0,0,10,"red",rotulo="BS")

        d = 10
        x = d*np.cos(phi0)
        y = d*np.sin(phi0)
        espaco.adicionar_pontos(x,y,1.5,"blue",rotulo="UT")

    elif tipo_do_ambiente == "indoor": 

        espaco.adicionar_pontos(0,0,3,"red",rotulo="BS")

        d = 0
        x = d*np.cos(phi0)
        y = d*np.sin(phi0)
        espaco.adicionar_pontos(x,y,1,"blue",rotulo="UT")


    return espaco.mostrar()


adicionar_BS_e_UT("indoor",30)
# %%
