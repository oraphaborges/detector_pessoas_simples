import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

QUANTIDADE_PALAVRAS_VIRTUAIS = 512 

# Pegando os detectores de uma imagem
# Utiliando o ORB
def get_descritores(caminho):
    LARGURA = 350
    ALTURA = 350

    img_test = cv2.imread(caminho,0)

    redimencionada = cv2.resize(img_test,(LARGURA,ALTURA), interpolation=cv2.INTER_CUBIC)
    equalizada = cv2.equalizeHist(redimencionada)
    suavizada = cv2.GaussianBlur(equalizada, (9,9),1)

    ORB = cv2.ORB_create(nfeatures=512)
    pontos_chave = ORB.detect(suavizada, None)

    pontos_chave, descritores = ORB.compute(suavizada, pontos_chave)

    return descritores

# Tecnica dos pacotes de palavras visual
class PacoteDePalavras:
    def gerar_dicionario(self, lista_descritores):
        kmeans = KMeans(n_clusters = QUANTIDADE_PALAVRAS_VIRTUAIS)
        kmeans = kmeans.fit(lista_descritores)
        self.dicionario = kmeans.cluster_centers_

    def histograma_de_frequencia(self, descritor):
        try:
            KNN = NearestNeighbors(n_neighbors=1)
            KNN.fit(self.dicionario)
            mais_prox = KNN.kneighbors(descritor, return_distance=False).flatten()

            histograma_caracteristicas = np.histogram(mais_prox, bins = np.arange(self.dicionario.shape[0]+1) )[0]

            return histograma_caracteristicas
        except AttributeError:
            print("O atributo dicionario não foi definido")
            
    def salvar_dicionario(self, caminho='', nome_dicionario = 'dicionario.csv'):
        try:
            np.savetxt(os.path.join(caminho, nome_dicionario), self.dicionario, delimiter=',', fmt='%f')
            print("Dicionário salvo")
        except AttributeError:
            print("Dicionário vazio")

    def carregar_dicionario(self, caminho="",nome_dicionario="dicionario.csv"):
        self.dicionario = np.loadtxt(os.path.join(caminho,nome_dicionario),delimiter=",")

caminho = 'dadosImagem/Treinamento/positivos/crop_000010.png'
descritor = get_descritores(caminho)
TPV = PacoteDePalavras()
TPV.gerar_dicionario(descritor)

histograma_caracteristicas = TPV.histograma_de_frequencia(descritor)
print(histograma_caracteristicas)