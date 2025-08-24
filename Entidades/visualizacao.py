import os
import numpy as np
import matplotlib.pyplot as plt

class Visualizacao:
    @staticmethod
    def salvar_visualizacao_retas(img_original, img_tratada, m_e, m_d, caminho_saida, nome_base, x_bico, y_bico):
        
        os.makedirs(caminho_saida, exist_ok=True)

        altura = img_original.shape[0]
        y_vals = np.array([y_bico, altura - 1], dtype=float)
        x_e = m_e * (y_vals - y_bico) + x_bico
        x_d = m_d * (y_vals - y_bico) + x_bico

        # original + retas
        plt.figure(figsize=(8, 8))
        plt.imshow(img_original, cmap='gray')
        plt.plot(x_e, y_vals, linewidth=2)
        plt.plot(x_d, y_vals, linewidth=2)
        plt.axis('off')
        plt.savefig(os.path.join(caminho_saida, f"{nome_base}_original_retas.png"), bbox_inches='tight')
        plt.close()

        # tratada + retas
        plt.figure(figsize=(8, 8))
        plt.imshow(img_tratada, cmap='gray')
        plt.plot(x_e, y_vals, linewidth=2)
        plt.plot(x_d, y_vals, linewidth=2)
        plt.axis('off')
        plt.savefig(os.path.join(caminho_saida, f"{nome_base}_tratada_retas.png"), bbox_inches='tight')
        plt.close()
