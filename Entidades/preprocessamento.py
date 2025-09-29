import os
import cv2
import numpy as np

class Preprocessamento:
    def __init__(self, pxcm, corte):
        self.pxcm = pxcm
        self.corte = corte

    def criar_imagem_fundo(self, imagens, primeira_imagem_com_spray, num_injecoes, imagens_por_ciclo, num_imagens_criarfundo):
        altura, largura = imagens.shape[1], imagens.shape[2]
        sem_spray = np.zeros((altura, largura, num_injecoes), dtype=np.float32)

        for j in range(num_injecoes):
            inicio = primeira_imagem_com_spray + imagens_por_ciclo * j - num_imagens_criarfundo
            fim = primeira_imagem_com_spray + imagens_por_ciclo * j - 1
            soma = np.sum(imagens[inicio:fim].astype(np.float32), axis=0)
            sem_spray[:, :, j] = soma / num_imagens_criarfundo

        return sem_spray.astype(np.uint8)

    def subtrair_fundo(self, imagens, sem_spray, inicio, num_injecoes, imagens_por_ciclo, frames_por_injecao, pasta_saida, captura_numero):
        for j in range(num_injecoes):
            frame_inicial = inicio + imagens_por_ciclo * j
            for i in range(frames_por_injecao):
                img_com_spray = imagens[frame_inicial + i]
                img_subtraida = cv2.subtract(sem_spray[:, :, j], img_com_spray)
                img_ajustada = cv2.normalize(img_subtraida, None, 0, 300, cv2.NORM_MINMAX)
                img_filtrada = cv2.medianBlur(img_ajustada, 3)

                nome_arquivo = f"captura{captura_numero}_inj{j+1}_frame{i+1}.pgm"
                caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
                if not os.path.exists(caminho_arquivo):
                    cv2.imwrite(caminho_arquivo, img_filtrada)
