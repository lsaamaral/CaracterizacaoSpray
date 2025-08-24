import os
import numpy as np
from sklearn.linear_model import RANSACRegressor
import cv2
import pandas as pd
import tifffile
import json
import matplotlib.pyplot as plt

class Spray:
    def __init__(self, pxcm, corte):
        self.pxcm = pxcm  # pixel/centimetro
        self.corte = corte  # valor de corte para deteccao de bordas
        # dicionarios que armazenam todas as penetracoes, angulos de cone e desvios respectivamente
        self.penetracoes = {}
        self.angulos_cone = {}
        self.desvios = {}
        self.borda_adicionada = {}

    def criar_imagem_fundo(self, imagens, primeira_imagem_com_spray, num_injecoes, imagens_por_ciclo, num_imagens_subtracao):
        print("Criando imagens de fundo")
        altura, largura = imagens.shape[1], imagens.shape[2]
        sem_spray = np.zeros((altura, largura, num_injecoes), dtype=np.float32)

        for j in range(num_injecoes):
            inicio = primeira_imagem_com_spray + imagens_por_ciclo * j - num_imagens_subtracao
            fim = primeira_imagem_com_spray + imagens_por_ciclo * j - 1

            soma = np.sum(imagens[inicio:fim].astype(np.float32), axis=0)
            sem_spray[:, :, j] = soma / num_imagens_subtracao

        return sem_spray.astype(np.uint8)

    def subtrair_fundo(self, imagens, sem_spray, inicio, num_injecoes, imagens_por_ciclo, frames_por_injecao, pasta_saida, captura_numero):

        for j in range(num_injecoes):
            frame_inicial = inicio + imagens_por_ciclo * j

            for i in range(frames_por_injecao):
                img_com_spray = imagens[frame_inicial + i]  # Selecionar frame atual
                img_subtraida = cv2.subtract(sem_spray[:, :, j], img_com_spray)  # Subtrair a imagem de fundo
                img_ajustada = cv2.normalize(img_subtraida, None, 0, 300, cv2.NORM_MINMAX)  # Normalizar contraste para 0-255
                img_filtrada = cv2.medianBlur(img_ajustada, 3)  # Suavizar ruido sem perder a borda (mediana 3x3)

                nome_arquivo = f"captura{captura_numero}_inj{j+1}_frame{i+1}.pgm"
                caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)

                if not os.path.exists(caminho_arquivo):
                    cv2.imwrite(caminho_arquivo, img_filtrada)
                    print(f"  Criado: {caminho_arquivo}")
                else:
                    print(f"  Ja existe, pulando: {caminho_arquivo}")

    def detectar_bordas(self, img_path):
        """
        Contorna o spray com bordas em branco utilizando bibliotecas de processamento de imagem. Armazena valores de min e max de cada direcao no dicionario dados_bordas. Retorna esse dicionario e a imagem com borda.
        """
        # Carregar a imagem
        if isinstance(img_path, str):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = img_path.copy()
        
        if img is None:
            raise ValueError("Nao foi possivel carregar a imagem")

        # Tirar nevoa dos primeiros frames
        if isinstance(img_path, str) and img_path.endswith('_frame1.pgm'):
            metade_altura = img.shape[0] // 2
            nova_img = np.zeros_like(img)
            nova_img[:metade_altura, :] = img[:metade_altura, :]
            img = nova_img

        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        _, img_thresh = cv2.threshold(img_blur, self.corte, 255, cv2.THRESH_BINARY) # pixels acima do corte ficam brancos
        
        kernel = np.ones((3,3), np.uint8) # define a area que vai ser considerada um buraco ou um ponto branco
        img_processed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel) # preenche buracos dentro do spray
        img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_OPEN, kernel) # remove pontos brancos fora do spray
    
        # Retorna o maior contorno
        contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            print("Aviso: Nenhum contorno encontrado na imagem")
            return {}, img.copy()
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Extrair pontos das bordas
        dados_bordas = {
            'dx_e': [], 'dy_e': [],  # borda esquerda
            'dx_d': [], 'dy_d': [],  # borda direita
            'dx_i': [], 'dy_i': [],  # borda inferior
            'dx_s': [], 'dy_s': []   # borda superior
        }
        
        # Converter contorno para array de pontos
        contour_points = main_contour[:,0,:]
        x_px = contour_points[:,0]
        y_px = contour_points[:,1]
        
        # Converter para cm
        x_cm = x_px / self.pxcm
        y_cm = y_px / self.pxcm
        
        # Borda esquerda
        unique_y = np.unique(y_px) # garante que a linha seja processada apenas uma vez
        for y in unique_y:
            mask = (y_px == y) # seleciona todos os pontos da linha atual
            if np.any(mask):
                idx = np.argmin(x_px[mask]) # seleciona apenas o ponto que esta mais a esquerda
                dados_bordas['dx_e'].append(x_cm[mask][idx])
                dados_bordas['dy_e'].append(y_cm[mask][idx])
        
        # Borda direita
        for y in unique_y:
            mask = (y_px == y)
            if np.any(mask):
                idx = np.argmax(x_px[mask])
                dados_bordas['dx_d'].append(x_cm[mask][idx])
                dados_bordas['dy_d'].append(y_cm[mask][idx])
        
        # Borda superior
        unique_x = np.unique(x_px)
        for x in unique_x:
            mask = (x_px == x)
            if np.any(mask):
                idx = np.argmin(y_px[mask])
                dados_bordas['dx_s'].append(x_cm[mask][idx])
                dados_bordas['dy_s'].append(y_cm[mask][idx])
        
        # Borda inferior
        for x in unique_x:
            mask = (x_px == x)
            if np.any(mask):
                idx = np.argmax(y_px[mask])
                dados_bordas['dx_i'].append(x_cm[mask][idx])
                dados_bordas['dy_i'].append(y_cm[mask][idx])
        
        # Criar visualizacao
        img_visualizacao = img.copy()

        img_visualizacao = cv2.drawContours(img_visualizacao, [main_contour], -1, 255, 1)
        
        return dados_bordas, img_visualizacao
    
    def calcular_penetracao(self, dados_bordas):
        """ Calcula a penetracao do spray em cm"""
        todos_y = []
        for key in dados_bordas:
            if key.startswith('dy_'):
                todos_y.extend(dados_bordas[key])

        if not todos_y:
            return 0.0
    
        y_bico_injetor = min(todos_y)
        y_final_spray = max(todos_y)

        return y_final_spray - y_bico_injetor
    
    def salvar_visualizacao_retas(self, img_original, img_tratada, m_e, m_d, caminho_saida, nome_base, x_bico, y_bico):
        """
        Salva duas imagens:
        - Uma com a imagem original + retas da regressao linear das bordas
        - Outra com a imagem tratada + retas da regressao linear das bordas
        """
        altura, largura = img_original.shape
        
        y_vals = np.array([y_bico, altura-1])
        x_e = m_e * (y_vals - y_bico) + x_bico
        x_d = m_d * (y_vals - y_bico) + x_bico

        # Imagem original
        plt.figure(figsize=(8, 8))
        plt.imshow(img_original, cmap='gray')
        plt.plot(x_e, y_vals, color='green', linewidth=2)
        plt.plot(x_d, y_vals, color='green', linewidth=2)
        plt.axis('off')
        plt.savefig(os.path.join(caminho_saida, f"{nome_base}_original_retas.png"), bbox_inches='tight')
        plt.close()

        # Imagem tratada
        plt.figure(figsize=(8, 8))
        plt.imshow(img_tratada, cmap='gray')
        plt.plot(x_e, y_vals, color='green', linewidth=2)
        plt.plot(x_d, y_vals, color='green', linewidth=2)
        plt.axis('off')
        plt.savefig(os.path.join(caminho_saida, f"{nome_base}_tratada_retas.png"), bbox_inches='tight')
        plt.close()
    
    def calcular_angulo_desvio(self, dados_bordas, frame_num=None):
        """
        Calcula o angulo de cone e desvio usando uma altura fixa
        """
        if frame_num is not None and frame_num <= 3:
            return None, None, None, None
        
        if not dados_bordas or not dados_bordas.get('dy_e') or not dados_bordas.get('dy_d'):
            return None, None, None, None

        try:
            x_esquerda_todos = np.array(dados_bordas['dx_e'])
            y_esquerda_todos = np.array(dados_bordas['dy_e'])
            x_direita_todos = np.array(dados_bordas['dx_d'])
            y_direita_todos = np.array(dados_bordas['dy_d'])
            
            if len(y_esquerda_todos) < 10 or len(y_direita_todos) < 10:
                return None, None, None, None

            y_min = min(np.min(y_esquerda_todos), np.min(y_direita_todos))

            altura_fixa_roi_cm = 2.0

            limite_y_roi = y_min + altura_fixa_roi_cm

            mascara_esquerda = y_esquerda_todos <= limite_y_roi
            mascara_direita = y_direita_todos <= limite_y_roi

            x_esquerda = x_esquerda_todos[mascara_esquerda]
            y_esquerda = y_esquerda_todos[mascara_esquerda]
            x_direita = x_direita_todos[mascara_direita]
            y_direita = y_direita_todos[mascara_direita]

            if len(y_esquerda) < 5 or len(y_direita) < 5:
                print(f"Altura fixa com poucos pontos no frame {frame_num}. Angulo nao calculado.")
                return None, None, None, None
            
            ransac_e = RANSACRegressor()
            ransac_e.fit(y_esquerda.reshape(-1, 1), x_esquerda)
            m_e = ransac_e.estimator_.coef_[0]

            ransac_d = RANSACRegressor()
            ransac_d.fit(y_direita.reshape(-1, 1), x_direita)
            m_d = ransac_d.estimator_.coef_[0]

            angulo_e_rad = np.arctan(m_e)
            angulo_d_rad = np.arctan(m_d)
            
            angulo_cone_rad = np.abs(angulo_d_rad - angulo_e_rad)
            angulo_cone_graus = np.degrees(angulo_cone_rad)
            
            angulo_central_rad = (angulo_e_rad + angulo_d_rad) / 2
            desvio_graus = np.degrees(angulo_central_rad)
                    
            return angulo_cone_graus, desvio_graus, m_e, m_d
        
        except Exception as e:
            print(f"Erro ao calcular angulo e desvio: {str(e)}")
            return None, None, None, None

    def processar_todas_bordas(self, pasta_captura, captura_numero, imagens, inicio, imagens_por_ciclo):
        print(f"\nProcessando bordas para captura {captura_numero}...")
        
        arquivos_pgm = sorted([f for f in os.listdir(pasta_captura) 
                           if f.startswith(f"captura{captura_numero}_inj") 
                           and f.endswith(".pgm")])

        if captura_numero not in self.penetracoes:
            self.penetracoes[captura_numero] = {}
        if captura_numero not in self.angulos_cone:
            self.angulos_cone[captura_numero] = {}
        if captura_numero not in self.desvios:
            self.desvios[captura_numero] = {}
        
        for arquivo in arquivos_pgm:
            caminho = os.path.join(pasta_captura, arquivo)
            partes = arquivo.split('_')
            injecao = int(partes[1][3:])
            frame = int(partes[2][5:-4])
            
            print(f"Processando bordas em: {arquivo}")
            dados_bordas, img_com_bordas = self.detectar_bordas(caminho)
            
            if injecao not in self.penetracoes[captura_numero]:
                self.penetracoes[captura_numero][injecao] = {}
            if injecao not in self.angulos_cone[captura_numero]:
                self.angulos_cone[captura_numero][injecao] = {}
            if injecao not in self.desvios[captura_numero]:
                self.desvios[captura_numero][injecao] = {}
                
            penetracao = self.calcular_penetracao(dados_bordas)
            self.penetracoes[captura_numero][injecao][frame] = penetracao
            
            if dados_bordas:
                angulo_cone, desvio, m_e, m_d = self.calcular_angulo_desvio(dados_bordas, frame)
                self.angulos_cone[captura_numero][injecao][frame] = angulo_cone
                self.desvios[captura_numero][injecao][frame] = desvio

                if m_e is not None and m_d is not None:

                    if dados_bordas['dy_e'] and dados_bordas['dy_d']:
                        # Coordenadas do bico em cm
                        y_bico_cm = min(min(dados_bordas['dy_e']), min(dados_bordas['dy_d']))
                        idx_e = np.argmin(dados_bordas['dy_e'])
                        idx_d = np.argmin(dados_bordas['dy_d'])
                        x_bico_cm = np.mean([dados_bordas['dx_e'][idx_e], dados_bordas['dx_d'][idx_d]])
                        
                        # Converter coordenadas do bico para pixels para a visualizacao
                        x_bico_px = x_bico_cm * self.pxcm
                        y_bico_px = y_bico_cm * self.pxcm
                    else:
                        altura, largura = imagens.shape[1], imagens.shape[2]
                        x_bico_px = largura // 2
                        y_bico_px = 0
                    
                    original_image_index = inicio + imagens_por_ciclo * (injecao - 1) + (frame - 1)
                    
                    if original_image_index < len(imagens):
                        img_original = imagens[original_image_index]
                    else:
                        print(f"Aviso: Indice da imagem original ({original_image_index}) fora dos limites. Usando imagem preta.")
                        img_original = np.zeros((imagens.shape[1], imagens.shape[2]), dtype=np.uint8)

                    img_tratada = img_com_bordas
                    nome_base = arquivo.replace('.pgm', '')
                    caminho_saida = os.path.join(pasta_captura, "visualizacoes")
                    os.makedirs(caminho_saida, exist_ok=True)

                    self.salvar_visualizacao_retas(
                        img_original,
                        img_tratada,
                        m_e,
                        m_d,
                        caminho_saida,
                        nome_base,
                        x_bico_px,
                        y_bico_px
                    )
            else:
                self.angulos_cone[captura_numero][injecao][frame] = None
                self.desvios[captura_numero][injecao][frame] = None
            
            img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
            num_borda = np.sum(img == 255)

            frame_num = int(arquivo.split('_frame')[1].split('.pgm')[0])

            if frame_num in [1, 2]:
                limiar = 50
            else:
                limiar = 200

            if num_borda > limiar:
                print(f"Borda já adicionada em {arquivo}, pulando.")
            else:
                dados_bordas, img_com_bordas = self.detectar_bordas(caminho)
                cv2.imwrite(caminho, img_com_bordas)
                print(f"Borda adicionada em {arquivo}")

        for arquivo in arquivos_pgm:
            partes = arquivo.split('_')
            injecao = int(partes[1][3:])
            frame = int(partes[2][5:-4])
            
            penetracao = self.penetracoes[captura_numero][injecao][frame]
            angulo = self.angulos_cone[captura_numero][injecao][frame]
            desvio = self.desvios[captura_numero][injecao][frame]
            
            print(f"  {arquivo}: Penetracao: {penetracao:.3f} cm | Angulo: {angulo if angulo is not None else 'N/A'} | Desvio: {desvio if desvio is not None else 'N/A'}")

    def exportar_resultados(self, fps, saida_dir="."):
        """
        Cria os arquivos Excel com os resultados de penetracao, angulo de cone e desvio.
        """
        os.makedirs(saida_dir, exist_ok=True)

        escritor_pen = pd.ExcelWriter(os.path.join(saida_dir, "Penetracao.xlsx"), engine='openpyxl')
        escritor_ang = pd.ExcelWriter(os.path.join(saida_dir, "AnguloCone.xlsx"), engine='openpyxl')
        escritor_desv = pd.ExcelWriter(os.path.join(saida_dir, "Desvio.xlsx"), engine='openpyxl')

        for captura in sorted(self.penetracoes.keys()):
            frames_disponiveis = set()
            for inj in self.penetracoes[captura]:
                frames_disponiveis.update(self.penetracoes[captura][inj].keys())
            
            frames_ordenados = sorted(frames_disponiveis)
            frames_penetracao = frames_ordenados
            frames_ang_desv = [f for f in frames_ordenados if f >= 4]

            dados_pen = []
            for frame in frames_penetracao:
                tempo = frame / fps
                linha_pen = [frame, tempo]
                for inj in range(1, max(self.penetracoes[captura].keys()) + 1):
                    p = self.penetracoes[captura].get(inj, {}).get(frame, None)
                    linha_pen.append(p if p is not None else "")
                dados_pen.append(linha_pen)

            dados_ang = []
            dados_desv = []
            for frame in frames_ang_desv:
                tempo = frame / fps
                linha_ang = [frame, tempo]
                linha_desv = [frame, tempo]
                for inj in range(1, max(self.penetracoes[captura].keys()) + 1):
                    a = self.angulos_cone[captura].get(inj, {}).get(frame, None)
                    d = self.desvios[captura].get(inj, {}).get(frame, None)
                    linha_ang.append(a if a is not None else "")
                    linha_desv.append(d if d is not None else "")
                dados_ang.append(linha_ang)
                dados_desv.append(linha_desv)

            colunas = ["Frame", "Tempo (s)"] + [f"Inj{inj}" for inj in range(1, max(self.penetracoes[captura].keys()) + 1)]
            
            df_pen = pd.DataFrame(dados_pen, columns=colunas)
            df_ang = pd.DataFrame(dados_ang, columns=colunas)
            df_desv = pd.DataFrame(dados_desv, columns=colunas)

            nome_aba = f"captura{captura}"
            df_pen.to_excel(escritor_pen, sheet_name=nome_aba, index=False)
            df_ang.to_excel(escritor_ang, sheet_name=nome_aba, index=False)
            df_desv.to_excel(escritor_desv, sheet_name=nome_aba, index=False)

        escritor_pen.close()
        escritor_ang.close()
        escritor_desv.close()


def carregar_config(caminho_config):
        with open(caminho_config, 'r') as f:
            return json.load(f)

def main():
        
    config = carregar_config("config.json")

    nome_pasta = config["nome_pasta"]
    num_capturas = config["num_capturas"]
    primeira_imagem_com_spray = config["primeira_imagem_com_spray"]
    inicio = config["inicio"]
    num_imagens_subtracao = config["num_imagens_subtracao"]
    num_injecoes = config["num_injecoes"]
    frames_por_injecao = config["frames_por_injecao"]
    fps = config["fps"]

    spray = Spray(pxcm=config["pxcm"], corte=config["corte"])

    for captura_numero in range(1, num_capturas + 1):
        try:
            imagens_por_ciclo = config["imagens_por_ciclo"][str(captura_numero)]
        except KeyError:
            print(f"\nNumero de imagens por ciclo nao configurado para captura {captura_numero}")
            continue

        pasta_captura = os.path.join(nome_pasta, f"captura{captura_numero}")
        arquivo_tif = os.path.join(pasta_captura, f"captura{captura_numero}.tif")

        if not os.path.exists(arquivo_tif):
            print(f"\nArquivo {arquivo_tif} nao encontrado, pulando captura {captura_numero}")
            continue

        print(f"\nProcessando {arquivo_tif}...")

        with tifffile.TiffFile(arquivo_tif) as tif:
            imagens = tif.asarray()  # (imagens.shape) = (n_frames, altura, largura)

        sem_spray = spray.criar_imagem_fundo(
            imagens,
            primeira_imagem_com_spray,
            num_injecoes,
            imagens_por_ciclo,
            num_imagens_subtracao
        )

        spray.subtrair_fundo(
            imagens,
            sem_spray,
            inicio,
            num_injecoes,
            imagens_por_ciclo,
            frames_por_injecao,
            pasta_captura,
            captura_numero
        )

        spray.processar_todas_bordas(pasta_captura, captura_numero, imagens, inicio, imagens_por_ciclo)

        spray.exportar_resultados(fps=config["fps"], saida_dir=os.path.join("resultados_excel", f"dados_{nome_pasta}"))


main()