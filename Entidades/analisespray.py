import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
import os
from Entidades.visualizacao import Visualizacao

class AnaliseSpray:
    def __init__(self, pxcm, corte, salvar_visualizacoes):
        self.pxcm = pxcm
        self.corte = corte
        self.salvar_visualizacoes = salvar_visualizacoes
        self.penetracoes = {}
        self.angulos_cone = {}
        self.desvios = {}

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
            coef_ang_esquerda = ransac_e.estimator_.coef_[0]

            ransac_d = RANSACRegressor()
            ransac_d.fit(y_direita.reshape(-1, 1), x_direita)
            coef_ang_direita = ransac_d.estimator_.coef_[0]

            angulo_e_rad = np.arctan(coef_ang_esquerda)
            angulo_d_rad = np.arctan(coef_ang_direita)
            
            angulo_cone_rad = np.abs(angulo_d_rad - angulo_e_rad)
            angulo_cone_graus = np.degrees(angulo_cone_rad)
            
            angulo_central_rad = (angulo_e_rad + angulo_d_rad) / 2
            desvio_graus = np.degrees(angulo_central_rad)
                    
            return angulo_cone_graus, desvio_graus, coef_ang_esquerda, coef_ang_direita
        
        except Exception as e:
            print(f"Erro ao calcular angulo e desvio: {str(e)}")
            return None, None, None, None

    def processar_todas_bordas(self, pasta_captura, captura_numero, imagens, inicio, imagens_por_ciclo):
        print(f"\nProcessando bordas para captura {captura_numero}...")
        
        arquivos_filtrados = []
        todos_os_arquivos = os.listdir(pasta_captura)

        for nome_do_arquivo in todos_os_arquivos:
            
            se_captura_certa = nome_do_arquivo.startswith(f"captura{captura_numero}_inj")
            se_arquivo_pgm = nome_do_arquivo.endswith(".pgm")
            
            if se_captura_certa and se_arquivo_pgm:
                arquivos_filtrados.append(nome_do_arquivo)

        arquivos_pgm = sorted(arquivos_filtrados)

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
                
            penetracao = self.calcular_penetracao(dados_bordas)
            self.penetracoes[captura_numero][injecao][frame] = penetracao
            
            if dados_bordas:
                angulo_cone, desvio, coef_ang_esquerda, coef_ang_direita = self.calcular_angulo_desvio(dados_bordas, frame)
                self.angulos_cone[captura_numero][injecao][frame] = angulo_cone
                self.desvios[captura_numero][injecao][frame] = desvio

                if coef_ang_esquerda is not None and coef_ang_direita is not None:

                    if dados_bordas['dy_e'] and dados_bordas['dy_d']:
                        # Coordenadas do bico em cm
                        y_bico_cm = min(min(dados_bordas['dy_e']), min(dados_bordas['dy_d']))
                        # indices dos pontos mais baixos em cada borda (que estao mais proximos do bico)
                        menor_ponto_esquerda = np.argmin(dados_bordas['dy_e'])
                        menor_ponto_direita = np.argmin(dados_bordas['dy_d'])
                        x_bico_cm = np.mean([dados_bordas['dx_e'][menor_ponto_esquerda], dados_bordas['dx_d'][menor_ponto_direita]])
                        
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

                    if self.salvar_visualizacoes and coef_ang_esquerda is not None and coef_ang_direita is not None:
                        Visualizacao.salvar_visualizacao_retas(
                            img_original,
                            img_tratada,
                            coef_ang_esquerda,
                            coef_ang_direita,
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

            # tentativa de nao adicionar borda se ja tiver
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