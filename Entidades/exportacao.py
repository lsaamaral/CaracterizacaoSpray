import os
import pandas as pd

class Exportacao:
    def __init__(self, fps, diretorio_saida):
        self.fps = fps
        self.diretorio_saida = diretorio_saida
        self.dados_agregados = {
            "penetracoes": {},
            "angulos_cone": {},
            "desvios": {}
        }
        os.makedirs(self.diretorio_saida, exist_ok=True)

    def adicionar_resultados_captura(self, num_captura, penetracoes, angulos_cone, desvios):
        """Adiciona resultados de uma captura inteira"""
        self.dados_agregados["penetracoes"][num_captura] = penetracoes
        self.dados_agregados["angulos_cone"][num_captura] = angulos_cone
        self.dados_agregados["desvios"][num_captura] = desvios

    def exportar_para_excel(self):
        """Cria Excel com todos os resultados"""
        print("\nExportando resultados para Excel")
        self._exportar_arquivo("Penetracao.xlsx", self.dados_agregados["penetracoes"], False)
        self._exportar_arquivo("AnguloCone.xlsx", self.dados_agregados["angulos_cone"], True)
        self._exportar_arquivo("Desvio.xlsx", self.dados_agregados["desvios"], True)
        print("Arquivos Excel criados")

    def _exportar_arquivo(self, nome_arquivo, dados_dict, ignorar_primeiros_frames):
        caminho_completo = os.path.join(self.diretorio_saida, nome_arquivo)
        with pd.ExcelWriter(caminho_completo, engine='openpyxl') as escritor:
            for captura, dados_captura in sorted(dados_dict.items()):
                df = self._criar_dataframe(dados_captura, ignorar_primeiros_frames)
                df.to_excel(escritor, sheet_name=f"captura{captura}", index=False)
    
    def _criar_dataframe(self, dados_captura, ignorar_primeiros_frames):
        """Cria um DataFrame a partir do dicionario de dados da captura"""
        if not dados_captura:
            return pd.DataFrame()
        
        frames_disponiveis = set()
        max_injecao = 0
        for inj, dados_inj in dados_captura.items():
            frames_disponiveis.update(dados_inj.keys())
            if inj > max_injecao:
                max_injecao = inj
        
        frames_ordenados = sorted(list(frames_disponiveis))
        if ignorar_primeiros_frames:
            frames_ordenados = [f for f in frames_ordenados if f >= 4]

        colunas = ["Frame", "Tempo (s)"] + [f"Inj{i}" for i in range(1, max_injecao + 1)]
        dados_tabela = []

        for frame in frames_ordenados:
            tempo = frame / self.fps
            linha = [frame, tempo]
            for inj in range(1, max_injecao + 1):
                valor = dados_captura.get(inj, {}).get(frame, "")
                linha.append(valor if valor is not None else "")
            dados_tabela.append(linha)
            
        return pd.DataFrame(dados_tabela, columns=colunas)