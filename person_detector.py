import cv2
import numpy as np

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

caminho = 'dadosImagem/Treinamento/positivos/crop_000010.png'
descritor = get_descritores(caminho)

print("TIPO:", type(descritor))
print("TAMANHO:",descritor.shape)
print('\nPONTO[0]', descritor[0])

