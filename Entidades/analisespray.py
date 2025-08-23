import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

class AnaliseSpray:
    def __init__(self, pxcm, corte):
        self.pxcm = pxcm
        self.corte = corte

    def detectar_bordas(self, img_path):
        """Detecta a borda do spray e retorna os pontos em cm"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Nao foi possivel carregar a imagem {img_path}")

        # Remover nevoa nos primeiros frames
        if img_path.endswith('_frame1.pgm'):
            metade_altura = img.shape[0] // 2
            nova_img = np.zeros_like(img)
            nova_img[:metade_altura, :] = img[:metade_altura, :]
            img = nova_img

        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        _, img_thresh = cv2.threshold(img_blur, self.corte, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3,3), np.uint8)
        img_processed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_OPEN, kernel)
    
        contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return {}, img.copy()
        
        main_contour = max(contours, key=cv2.contourArea)
        img_visualizacao = cv2.drawContours(img.copy(), [main_contour], -1, 255, 1)

        # Extrai os pontos da borda e converte para cm
        contour_points = main_contour[:, 0, :]
        x_px, y_px = contour_points[:, 0], contour_points[:, 1]
        x_cm, y_cm = x_px / self.pxcm, y_px / self.pxcm

        dados_bordas = self._extrair_pontos_bordas(x_px, y_px, x_cm, y_cm)
        
        return dados_bordas, img_visualizacao

    def _extrair_pontos_bordas(self, x_px, y_px, x_cm, y_cm):
        """Organiza os pontos das bordas"""
        dados_bordas = {'dx_e': [], 'dy_e': [], 'dx_d': [], 'dy_d': []}
        
        unique_y = np.unique(y_px)
        for y_val in unique_y:
            mask = (y_px == y_val)
            if np.any(mask):
                # Borda esquerda
                idx_e = np.argmin(x_px[mask])
                dados_bordas['dx_e'].append(x_cm[mask][idx_e])
                dados_bordas['dy_e'].append(y_cm[mask][idx_e])
                # Borda direita
                idx_d = np.argmax(x_px[mask])
                dados_bordas['dx_d'].append(x_cm[mask][idx_d])
                dados_bordas['dy_d'].append(y_cm[mask][idx_d])
        return dados_bordas

    def calcular_penetracao(self, dados_bordas):
        """Calcula a penetracao do spray em cm"""
        todos_y = dados_bordas.get('dy_e', []) + dados_bordas.get('dy_d', [])
        if not todos_y:
            return 0.0
        return max(todos_y) - min(todos_y)

    def calcular_angulo_desvio(self, dados_bordas, frame_num=None):
        """Calcula o angulo de cone e o desvio"""
        if frame_num is not None and frame_num <= 3:
            return None, None, None, None
        
        if not all(k in dados_bordas for k in ['dx_e', 'dy_e', 'dx_d', 'dy_d']):
            return None, None, None, None

        try:
            x_e, y_e = np.array(dados_bordas['dx_e']), np.array(dados_bordas['dy_e'])
            x_d, y_d = np.array(dados_bordas['dx_d']), np.array(dados_bordas['dy_d'])
            
            if len(y_e) < 10 or len(y_d) < 10:
                return None, None, None, None

            y_min = min(np.min(y_e), np.min(y_d))
            limite_y_roi = y_min + 2.0

            mask_e = y_e <= limite_y_roi
            mask_d = y_d <= limite_y_roi

            if np.sum(mask_e) < 5 or np.sum(mask_d) < 5:
                print(f"Altura fixa com poucos pontos no frame {frame_num}. Angulo nao calculado.")
                return None, None, None, None
            
            # Regressao borda esquerda
            ransac_e = RANSACRegressor().fit(y_e[mask_e].reshape(-1, 1), x_e[mask_e])
            m_e = ransac_e.estimator_.coef_[0]

            # Regressao borda direita
            ransac_d = RANSACRegressor().fit(y_d[mask_d].reshape(-1, 1), x_d[mask_d])
            m_d = ransac_d.estimator_.coef_[0]

            angulo_e_rad = np.arctan(m_e)
            angulo_d_rad = np.arctan(m_d)
            
            angulo_cone_graus = np.degrees(np.abs(angulo_d_rad - angulo_e_rad))
            desvio_graus = np.degrees((angulo_e_rad + angulo_d_rad) / 2)
                    
            return angulo_cone_graus, desvio_graus, m_e, m_d
        
        except Exception as e:
            print(f"Erro no calculo do angulo e desvio no frame {frame_num}: {e}")
            return None, None, None, None