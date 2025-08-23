import os
import numpy as np
import matplotlib.pyplot as plt

class Visualizacao:
    def __init__(self, pxcm):
        self.pxcm = pxcm

    def salvar_visualizacao_retas(self, img_original, img_tratada, m_e, m_d, caminho_saida, nome_base, dados_bordas):
        """Salva imagens com as retas da regressao linear das bordas"""
        if not dados_bordas or not dados_bordas.get('dy_e'):
            return

        # Coordenadas do bico em cm
        y_bico_cm = min(dados_bordas['dy_e'])
        idx_e = np.argmin(dados_bordas['dy_e'])
        idx_d = np.argmin(dados_bordas['dy_d'])
        x_bico_cm = np.mean([dados_bordas['dx_e'][idx_e], dados_bordas['dx_d'][idx_d]])
        
        # Converte para pixels
        x_bico_px = x_bico_cm * self.pxcm
        y_bico_px = y_bico_cm * self.pxcm

        altura, _ = img_original.shape
        y_vals = np.array([y_bico_px, altura - 1])
        
        x_e = m_e * ((y_vals / self.pxcm) - y_bico_cm) + x_bico_cm
        x_d = m_d * ((y_vals / self.pxcm) - y_bico_cm) + x_bico_cm
        
        x_e_px = x_e * self.pxcm
        x_d_px = x_d * self.pxcm

        self._plotar_e_salvar(img_original, x_e_px, x_d_px, y_vals, os.path.join(caminho_saida, f"{nome_base}_original_retas.png"))
        self._plotar_e_salvar(img_tratada, x_e_px, x_d_px, y_vals, os.path.join(caminho_saida, f"{nome_base}_tratada_retas.png"))

    def _plotar_e_salvar(self, imagem, x_e, x_d, y_vals, caminho_arquivo):
        """Plotar e salva uma imagem"""
        plt.figure(figsize=(8, 8))
        plt.imshow(imagem, cmap='gray')
        plt.plot(x_e, y_vals, color='lime', linewidth=1)
        plt.plot(x_d, y_vals, color='lime', linewidth=1)
        plt.axis('off')
        plt.savefig(caminho_arquivo, bbox_inches='tight', pad_inches=0)
        plt.close()