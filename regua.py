import cv2
import numpy as np

def calcular_px_por_cm(caminho_imagem, tamanho_interno_tabuleiro, tamanho_real_quadrado_cm):

    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro: Nao foi possivel carregar a imagem.")
        return None

    ret, corners = cv2.findChessboardCorners(img, tamanho_interno_tabuleiro, None)

    if ret:
        print(f"Tabuleiro de xadrez {tamanho_interno_tabuleiro} encontrado")

        distancias_pixels = []

        # Horizontal
        for i in range(tamanho_interno_tabuleiro[1]): # para cada linha
            for j in range(tamanho_interno_tabuleiro[0] - 1): # para cada coluna - 1
                idx1 = i  * tamanho_interno_tabuleiro[0] + j
                idx2 = i * tamanho_interno_tabuleiro[0] + j + 1
                dist = np.linalg.norm(corners[idx1] - corners[idx2])
                distancias_pixels.append(dist)

        # Vertical
        for i in range(tamanho_interno_tabuleiro[1] - 1): # para cada linha - 1
            for j in range(tamanho_interno_tabuleiro[0]): # para cada coluna
                idx1 = i * tamanho_interno_tabuleiro[0] + j
                idx2 = (i + 1) * tamanho_interno_tabuleiro[0] + j
                dist = np.linalg.norm(corners[idx1] - corners[idx2])
                distancias_pixels.append(dist)

        # Media
        distancia_media_pixels = np.mean(distancias_pixels)

        # Calcular px/cm
        px_cm = distancia_media_pixels / tamanho_real_quadrado_cm

        # Desenhar os cantos para verificacao visual
        img_desenhada = cv2.drawChessboardCorners(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), tamanho_interno_tabuleiro, corners, ret)
        cv2.imwrite("calibracao_verificacao2.png", img_desenhada)
        print("Imagem 'calibracao_verificacao.png' salva para inspecao")

        return px_cm
    else:
        print("Erro: Tabuleiro de xadrez nao encontrado na imagem")
        return None

def main():
    IMAGEM_CALIBRACAO = "injetorreguaatras.tif"
    # um tabuleiro com 9x7 quadrados tem (8, 6) cantos internos
    CANTOS_INTERNOS = (6, 8) 
    TAMANHO_QUADRADO_CM = 1.0 

    pxcm_calculado = calcular_px_por_cm(IMAGEM_CALIBRACAO, CANTOS_INTERNOS, TAMANHO_QUADRADO_CM)

    if pxcm_calculado is not None:
        print(f"Px/Cm calculado: {pxcm_calculado:.4f}")

main()