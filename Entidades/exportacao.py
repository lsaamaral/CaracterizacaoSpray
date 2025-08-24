import os
import pandas as pd

class Exportacao:
    @staticmethod
    def exportar_resultados(penetracoes, angulos_cone, desvios, fps, saida_dir="."):
        os.makedirs(saida_dir, exist_ok=True)

        escritor_pen = pd.ExcelWriter(os.path.join(saida_dir, "Penetracao.xlsx"), engine='openpyxl')
        escritor_ang = pd.ExcelWriter(os.path.join(saida_dir, "AnguloCone.xlsx"), engine='openpyxl')
        escritor_desv = pd.ExcelWriter(os.path.join(saida_dir, "Desvio.xlsx"), engine='openpyxl')

        for captura in sorted(penetracoes.keys()):
            frames_disponiveis = set()
            for inj in penetracoes[captura]:
                frames_disponiveis.update(penetracoes[captura][inj].keys())
            
            frames_ordenados = sorted(frames_disponiveis)
            frames_penetracao = frames_ordenados
            frames_ang_desv = [f for f in frames_ordenados if f >= 4]

            dados_pen = []
            for frame in frames_penetracao:
                tempo = frame / fps
                linha_pen = [frame, tempo]
                for inj in range(1, max(penetracoes[captura].keys()) + 1):
                    p = penetracoes[captura].get(inj, {}).get(frame, None)
                    linha_pen.append(p if p is not None else "")
                dados_pen.append(linha_pen)

            dados_ang = []
            dados_desv = []
            for frame in frames_ang_desv:
                tempo = frame / fps
                linha_ang = [frame, tempo]
                linha_desv = [frame, tempo]
                for inj in range(1, max(penetracoes[captura].keys()) + 1):
                    a = angulos_cone[captura].get(inj, {}).get(frame, None)
                    d = desvios[captura].get(inj, {}).get(frame, None)
                    linha_ang.append(a if a is not None else "")
                    linha_desv.append(d if d is not None else "")
                dados_ang.append(linha_ang)
                dados_desv.append(linha_desv)

            colunas = ["Frame", "Tempo (s)"] + [f"Inj{inj}" for inj in range(1, max(penetracoes[captura].keys()) + 1)]
            
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
