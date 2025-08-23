import os
import numpy as np
import cv2
import tifffile

class PreProcessamento:
    def __init__(self, primeira_imagem_com_spray, num_imagens_subtracao):
        self.primeira_imagem_com_spray = primeira_imagem_com_spray
        self.num_imagens_subtracao = num_imagens_subtracao

    def carregar_imagens_tif(self, caminho_arquivo_tif):
        """Carrega as imagens do tif"""
        if not os.path.exists(caminho_arquivo_tif):
            print(f"\nArquivo {caminho_arquivo_tif} nao encontrado")
            return None
        print(f"\nCarregando {caminho_arquivo_tif}...")
        with tifffile.TiffFile(caminho_arquivo_tif) as tif:
            return tif.asarray()

    def criar_imagem_fundo(self, imagens, num_injecoes, imagens_por_ciclo):
        """Cria a imagem de fundo sem spray"""
        print("Criando imagens de fundo...")
        altura, largura = imagens.shape[1], imagens.shape[2]
        sem_spray = np.zeros((altura, largura, num_injecoes), dtype=np.float32)

        for j in range(num_injecoes):
            inicio = self.primeira_imagem_com_spray + imagens_por_ciclo * j - self.num_imagens_subtracao
            fim = self.primeira_imagem_com_spray + imagens_por_ciclo * j - 1
            
            inicio = max(0, inicio)

            soma = np.sum(imagens[inicio:fim].astype(np.float32), axis=0)
            sem_spray[:, :, j] = soma / (fim - inicio) if (fim - inicio) > 0 else 0

        return sem_spray.astype(np.uint8)

    def subtrair_fundo(self, imagens, sem_spray, inicio_frame, num_injecoes, imagens_por_ciclo, frames_por_injecao, pasta_saida, captura_numero):
        """Subtrai o fundo e salva as imagens tratadas em pgm"""
        print("Subtraindo fundo das imagens...")
        for j in range(num_injecoes):
            frame_inicial_ciclo = inicio_frame + imagens_por_ciclo * j

            for i in range(frames_por_injecao):
                img_com_spray = imagens[frame_inicial_ciclo + i]
                img_subtraida = cv2.subtract(sem_spray[:, :, j], img_com_spray)
                img_ajustada = cv2.normalize(img_subtraida, None, 0, 300, cv2.NORM_MINMAX)
                img_filtrada = cv2.medianBlur(img_ajustada, 3)

                nome_arquivo = f"captura{captura_numero}_inj{j+1}_frame{i+1}.pgm"
                caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)

                if not os.path.exists(caminho_arquivo):
                    cv2.imwrite(caminho_arquivo, img_filtrada)
                    print(f"  Criado: {caminho_arquivo}")
                else:
                    print(f"  Ja existe, pulando {caminho_arquivo}")