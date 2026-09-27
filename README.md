# Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs

Protótipo de triagem semântica (representações vetoriais, busca semântica, detecção de duplicatas e chunking)
para as manifestações da Ouvidoria Municipal.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `análise_comparativa.ipynb` | **Entrega 1**: BoW × TF-IDF × embeddings (3 modelos) nos pares do enunciado e em toda a base |
| `deteccao_duplicatas.ipynb` | **Entrega 2**: `detectar_duplicatas`, heatmap, calibração do limiar, análise de FP/FN |
| `chunking_manifestacoes.ipynb` | **Entrega 3**: truncamento em 128 tokens, configurações de chunking, coesão, visualização 2D |
| `app_ouvidoria.py` | **Entrega 4**: app Streamlit (Busca, Base Completa, Espaço Vetorial, Chunking) |
| `RELATORIO.pdf` | Síntese de decisões, dificuldades e aprendizados |
| `ouvidoria_core.py` | Funções compartilhadas por notebooks e app (mesmos números em todos) |
| `manifestacoes.json` | Base com 40 manifestações (sintética, seguindo a especificação do desafio) |
| `duplicatas_gabarito.json` | Gabarito: 6 pares de duplicatas reais + 6 "falsos amigos" |
| `figuras/` | Figuras geradas pelos notebooks e usadas no relatório |

## Como executar

```bash
pip install -r requirements.txt
streamlit run app_ouvidoria.py
```

Os notebooks rodam de cima para baixo (Jupyter/VS Code) a partir desta pasta. Na primeira execução, os modelos
são baixados do Hugging Face Hub (~0,5 GB cada para MiniLM e E5-small, ~1,1 GB para MPNet).

## Principais resultados

- **Esparsas × densas:** o par duplicado M003×M017 tem cosseno 0,06 no BoW e 0,71–0,76 com MiniLM/MPNet.
- **Duplicatas:** o limiar padrão 0,85 não encontra nenhuma duplicata com o MiniLM. Com **MPNet + limiar 0,75**, as
  6 duplicatas são encontradas sem falsos positivos (ver ressalvas no notebook).
- **Chunking:** MiniLM/MPNet leem só 128 tokens e ignoram 20–34% das manifestações longas. Com chunking, os fatos
  do final do texto voltam a ser encontrados na busca.
