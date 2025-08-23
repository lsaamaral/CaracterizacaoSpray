import cv2
import numpy as np
import math

def criar_spray_sintetico(largura, altura, desvio_graus, angulo_cone_graus, nome_arquivo):
    # bico do injetor
    bico_x = largura // 2
    bico_y = 50  # comeca no px50

    # imagem preta
    imagem = np.zeros((altura, largura), dtype=np.uint8)

    # grau para radiano
    desvio_rad = math.radians(desvio_graus)
    metade_cone_rad = math.radians(angulo_cone_graus / 2)

    # angulo de cada borda
    angulo_borda_esquerda_rad = desvio_rad - metade_cone_rad
    angulo_borda_direita_rad = desvio_rad + metade_cone_rad

    # inclinacao das bordas
    m_esquerda = math.tan(angulo_borda_esquerda_rad)
    m_direita = math.tan(angulo_borda_direita_rad)

    p1 = (bico_x, bico_y)
    y_base = altura - 1
    delta_y = y_base - bico_y

    x_base_esquerda = bico_x + delta_y * m_esquerda
    x_base_direita = bico_x + delta_y * m_direita

    # extremidades do spray na base da imagem
    p2 = (int(x_base_esquerda), y_base)
    p3 = (int(x_base_direita), y_base)
    
    pontos_poligono = np.array([[p1, p2, p3]], dtype=np.int32)
    
    cv2.fillPoly(imagem, [pontos_poligono], 255)
    imagem_borrada = cv2.GaussianBlur(imagem, (15, 15), 0)

    cv2.imwrite(nome_arquivo, imagem_borrada)
    print(f"Imagem '{nome_arquivo}' criada com sucesso!")

def main():
    criar_spray_sintetico(largura=480, altura=480, desvio_graus=5.0, angulo_cone_graus=20.0, nome_arquivo="spray_sintetico_5deg.png")
    criar_spray_sintetico(largura=480, altura=480, desvio_graus=10.0, angulo_cone_graus=20.0, nome_arquivo="spray_sintetico_10deg.png")
    criar_spray_sintetico(largura=480, altura=480, desvio_graus=15.0, angulo_cone_graus=20.0, nome_arquivo="spray_sintetico_15deg.png")

main()