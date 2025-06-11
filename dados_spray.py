import os
import numpy as np
from sklearn.linear_model import RANSACRegressor
import cv2
import pandas as pd
import tifffile
import json

class Spray:
    def __init__(self, pxcm, corte):
        self.pxcm = pxcm  # pixel/centimetro
        self.corte = corte  # valor de corte para deteccao de bordas
        # dicionarios que armazenam todas as penetracoes, angulos de cone e desvios respectivamente
        self.penetracoes = {}
        self.angulos_cone = {}
        self.desvios = {}

    def criar_imagem_fundo(self, imagens, primeira_imagem_com_spray, num_injecoes, imagens_por_ciclo, num_imagens_subtracao):
        print("Criando imagens de fundo")
        altura, largura = imagens.shape[1], imagens.shape[2]
        sem_spray = np.zeros((altura, largura, num_injecoes), dtype=np.float32)

        for j in range(num_injecoes):
            inicio = primeira_imagem_com_spray + imagens_por_ciclo * j - num_imagens_subtracao
            fim = primeira_imagem_com_spray + imagens_por_ciclo * j - 1

            #print(f"Processando injecao {j+1}: frames {inicio} a {fim}")
            soma = np.sum(imagens[inicio:fim].astype(np.float32), axis=0)
            sem_spray[:, :, j] = soma / num_imagens_subtracao

        return sem_spray.astype(np.uint8)

    def subtrair_fundo(self, imagens, sem_spray, inicio, num_injecoes, imagens_por_ciclo, frames_por_injecao, pasta_saida, captura_numero):
        #print(f"\nSubtraindo fundo para captura {captura_numero}...")

        for j in range(num_injecoes):
            frame_inicial = inicio + imagens_por_ciclo * j

            for i in range(frames_por_injecao):
                img_com_spray = imagens[frame_inicial + i]  # Selecionar frame atual
                img_subtraida = cv2.subtract(sem_spray[:, :, j], img_com_spray)  # Subtrair a imagem de fundo
                img_ajustada = cv2.normalize(img_subtraida, None, 0, 255, cv2.NORM_MINMAX)  # Normalizar contraste para 0-255
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
        
        # Converter para coordenadas em cm
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

        if not self.imagem_ja_processada(img_path):
            img_visualizacao = cv2.drawContours(img_visualizacao, [main_contour], -1, 255, 1)
        
        return dados_bordas, img_visualizacao

    def imagem_ja_processada(self, img_path):
        """
        Verifica se a imagem ja foi processada procurando por qualquer pixel branco (255).
        As imagens nunca tem pixels 255.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        
        return np.any(img == 255)  # True se encontrar pelo menos 1 pixel branco
    
    def calcular_penetracao(self, dados_bordas, frame_num=None, injecao_num=None, captura_num=None):
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
    
    def calcular_angulo_desvio(self, dados_bordas, frame_num=None, injecao_num=None, captura_num=None):
        """Calcula o angulo de cone e desvio do spray usando regressao RANSAC"""
        if frame_num is not None and frame_num <= 3:
            return None, None
        
        if not dados_bordas or 'dx_e' not in dados_bordas or 'dx_d' not in dados_bordas:
            return self._herdar_valores(frame_num, injecao_num, captura_num, (None, None))

        try:
            x_esquerda = np.array(dados_bordas['dx_e'][:-20]) if len(dados_bordas['dx_e']) > 200 else np.array(dados_bordas['dx_e'])
            y_esquerda = np.array(dados_bordas['dy_e'][:-20]) if len(dados_bordas['dy_e']) > 200 else np.array(dados_bordas['dy_e'])
            x_direita = np.array(dados_bordas['dx_d'][:-20]) if len(dados_bordas['dx_d']) > 200 else np.array(dados_bordas['dx_d'])
            y_direita = np.array(dados_bordas['dy_d'][:-20]) if len(dados_bordas['dy_d']) > 200 else np.array(dados_bordas['dy_d'])
            
            if len(x_esquerda) < 8 or len(x_direita) < 8:
                return self._herdar_valores(frame_num, injecao_num, captura_num, (0.0, 0.0))
            
            x_esquerda, y_esquerda = self._remover_outliers(x_esquerda, y_esquerda)
            x_direita, y_direita = self._remover_outliers(x_direita, y_direita)
            
            if len(x_esquerda) < 5 or len(x_direita) < 5:
                return self._herdar_valores(frame_num, injecao_num, captura_num, (0.0, 0.0))
            
            ransac_e = RANSACRegressor()
            ransac_e.fit(x_esquerda.reshape(-1, 1), y_esquerda)
            a_e = ransac_e.estimator_.coef_[0]  # Coeficiente angular da reta esquerda
            
            ransac_d = RANSACRegressor()
            ransac_d.fit(x_direita.reshape(-1, 1), y_direita)
            a_d = ransac_d.estimator_.coef_[0]  # Coeficiente angular da reta direita

            denominador = 1 + a_d * a_e
            if abs(denominador) < 1e-6 or np.isnan(a_d) or np.isnan(a_e):
                return self._herdar_valores(frame_num, injecao_num, captura_num, (0.0, 0.0))
            
            angulo_cone_rad = np.arctan(abs((a_d - a_e) / (1 + a_d * a_e)))
            angulo_cone_graus = np.degrees(angulo_cone_rad)
            
            # Angulo da reta direita em relacao a vertical
            angulo_direita_graus = np.degrees(np.arctan(abs(1 / a_d))) if abs(a_d) > 1e-6 else 90.0
            
            # Angulo da reta esquerda em relacao a vertical
            angulo_esquerda_graus = np.degrees(np.arctan(abs(1 / a_e))) if abs(a_e) > 1e-6 else 90.0
            
            desvio_graus = (angulo_direita_graus - angulo_esquerda_graus) / 2
            
            return angulo_cone_graus, desvio_graus
        
        except Exception as e:
            print(f"Erro ao calcular angulo e desvio: {str(e)}")
            return self._herdar_valores(frame_num, injecao_num, captura_num, (0.0, 0.0))

    def _remover_outliers(self, x, y):
        """Remove outliers usando o metodo IQR"""
        if len(x) < 5:
            return x, y
        
        # Calcular residuos de uma regressao linear inicial
        coeff = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeff, x)
        residuos = y - y_pred
        
        # Calcular quartis e IQR
        q1 = np.percentile(residuos, 25)
        q3 = np.percentile(residuos, 75)
        iqr = q3 - q1
        
        # Definir limites para outliers
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        
        # Filtrar pontos dentro dos limites
        mascara = (residuos >= limite_inferior) & (residuos <= limite_superior)
        return x[mascara], y[mascara]

    def _herdar_valores(self, frame_num, injecao_num, captura_num, default):
        """Versão simplificada que faz frames 1-3 herdarem do frame 4"""
        if frame_num is None or injecao_num is None or captura_num is None:
            return default
        
        # Se for frame 1, 2 ou 3, tenta pegar o valor do frame 4
        if frame_num in [1, 2, 3]:
            try:
                if (captura_num in self.angulos_cone and 
                    injecao_num in self.angulos_cone[captura_num] and 
                    4 in self.angulos_cone[captura_num][injecao_num]):
                    
                    angulo = self.angulos_cone[captura_num][injecao_num][4]
                    desvio = self.desvios[captura_num][injecao_num][4]
                    
                    if angulo is not None and desvio is not None:
                        print(f"Herança simplificada: usando frame 4 para frame {frame_num}")
                        return angulo, desvio
            except KeyError:
                pass
        return default
        

    def processar_todas_bordas(self, pasta_captura, captura_numero):
        print(f"\nProcessando bordas para captura {captura_numero}...")
        
        arquivos_pgm = [f for f in os.listdir(pasta_captura) 
                    if f.startswith(f"captura{captura_numero}_inj") 
                    and f.endswith(".pgm")]
        
        ordem_prioridade = {"frame4": 0, "frame2": 1, "frame3": 2}
        arquivos_pgm.sort(key=lambda x: ordem_prioridade.get(x.split("_")[-1][:-4], 3))


        # Inicializa estruturas de dados
        if captura_numero not in self.penetracoes:
            self.penetracoes[captura_numero] = {}
        if captura_numero not in self.angulos_cone:
            self.angulos_cone[captura_numero] = {}
        if captura_numero not in self.desvios:
            self.desvios[captura_numero] = {}
        
        # Primeiro processa todos os frames para coletar dados
        for arquivo in arquivos_pgm:
            caminho = os.path.join(pasta_captura, arquivo)
            partes = arquivo.split('_')
            injecao = int(partes[1][3:])
            frame = int(partes[2][5:-4])
            
            print(f"Processando bordas em: {arquivo}")
            dados_bordas, img_com_bordas = self.detectar_bordas(caminho)
            
            # Inicializa estruturas para esta injecao
            if injecao not in self.penetracoes[captura_numero]:
                self.penetracoes[captura_numero][injecao] = {}
            if injecao not in self.angulos_cone[captura_numero]:
                self.angulos_cone[captura_numero][injecao] = {}
            if injecao not in self.desvios[captura_numero]:
                self.desvios[captura_numero][injecao] = {}
                
            penetracao = self.calcular_penetracao(dados_bordas, frame, injecao, captura_numero)
            self.penetracoes[captura_numero][injecao][frame] = penetracao
            
            if dados_bordas:
                angulo_cone, desvio = self.calcular_angulo_desvio(dados_bordas, frame, injecao, captura_numero)
                self.angulos_cone[captura_numero][injecao][frame] = angulo_cone
                self.desvios[captura_numero][injecao][frame] = desvio
            else:
                self.angulos_cone[captura_numero][injecao][frame] = None
                self.desvios[captura_numero][injecao][frame] = None
            
            cv2.imwrite(caminho, img_com_bordas)

        for arquivo in arquivos_pgm:
            partes = arquivo.split('_')
            injecao = int(partes[1][3:])
            frame = int(partes[2][5:-4])
            
            if (self.angulos_cone[captura_numero][injecao][frame] is None or 
                self.desvios[captura_numero][injecao][frame] is None):
                
                angulos_validos = [v for k, v in self.angulos_cone[captura_numero][injecao].items() 
                                if v is not None and v > 0]
                
                desvios_validos = [v for k, v in self.desvios[captura_numero][injecao].items() 
                                if v is not None]
                
                if angulos_validos:
                    media_angulo = np.mean(angulos_validos)
                    self.angulos_cone[captura_numero][injecao][frame] = media_angulo
                    print(f"Substituindo angulo None do frame {frame} pela media {media_angulo:.2f}")
                
                if desvios_validos:
                    media_desvio = np.mean(desvios_validos)
                    self.desvios[captura_numero][injecao][frame] = media_desvio
                    print(f"Substituindo desvio None do frame {frame} pela media {media_desvio:.2f}")

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

        spray.processar_todas_bordas(pasta_captura, captura_numero)

        spray.exportar_resultados(fps=config["fps"], saida_dir="resultados_excel")


main()