"""
ouvidoria_core.py — funções compartilhadas do projeto "Ouvidoria Inteligente".

Reúne tudo o que é usado tanto pelos notebooks quanto pelo app Streamlit:
carregamento da base, modelos de embedding, similaridade, detecção de
duplicatas, chunking e redução de dimensionalidade. Mantê-las num único
módulo garante que notebook e app produzam exatamente os mesmos números.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_MANIFESTACOES = PASTA_PROJETO / "manifestacoes.json"
ARQUIVO_GABARITO = PASTA_PROJETO / "duplicatas_gabarito.json"

# Modelos de embedding suportados.
# A família E5 foi treinada com prefixos: para tarefas simétricas
# (texto x texto, como duplicatas e busca entre manifestações) usa-se
# "query: " dos dois lados. Os modelos paraphrase-* não usam prefixo.
#
# "faixas" são os cortes de cor da busca (verde / amarelo / vermelho). O E5
# produz cossenos comprimidos entre ~0,80 e ~0,95 para qualquer par de textos
# em português (anisotropia), então as faixas 0,7/0,5 do enunciado deixariam
# tudo verde; para ele usamos cortes equivalentes calibrados na base.
# "limiar_duplicata" é o limiar calibrado no notebook deteccao_duplicatas.ipynb.
MODELOS = {
    "MiniLM multilíngue (paraphrase-multilingual-MiniLM-L12-v2)": {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "curto": "MiniLM",
        "prefixo": "",
        "dim": 384,
        "faixas": (0.70, 0.50),
        "limiar_duplicata": 0.70,
    },
    "MPNet multilíngue (paraphrase-multilingual-mpnet-base-v2)": {
        "id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "curto": "MPNet",
        "prefixo": "",
        "dim": 768,
        "faixas": (0.70, 0.50),
        "limiar_duplicata": 0.75,
    },
    "E5 small multilíngue (intfloat/multilingual-e5-small)": {
        "id": "intfloat/multilingual-e5-small",
        "curto": "E5-small",
        "prefixo": "query: ",
        "dim": 384,
        "faixas": (0.88, 0.85),
        "limiar_duplicata": 0.925,
    },
}
MODELO_PADRAO = next(iter(MODELOS))

# Stopwords do português (sem acento, pois o vetorizador remove acentos antes
# de comparar os tokens). Lista curta e explícita para o leitor saber o que sai.
# "não" e "sem" foram mantidos no vocabulário de propósito: numa reclamação
# eles carregam o sentido ("sem médico", "não passou").
STOPWORDS_PT = sorted(
    {
        "a", "ao", "aos", "as", "até", "com", "como", "da", "das", "de", "dela", "dele",
        "do", "dos", "e", "ela", "ele", "em", "entre", "era", "essa", "esse", "esta",
        "está", "estão", "este", "eu", "foi", "for", "há", "isso", "já", "la", "lhe",
        "mais", "mas", "me", "meu", "minha", "muito", "na", "nas", "nem", "no", "nos",
        "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos",
        "por", "pra", "que", "se", "ser", "seu", "sua", "são", "só", "também",
        "tem", "ter", "um", "uma", "à", "às", "é",
    }
)


def _sem_acento(palavra: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFKD", palavra) if not unicodedata.combining(c))


STOPWORDS_PT_SEM_ACENTO = sorted({_sem_acento(p) for p in STOPWORDS_PT})

# Paleta categórica em ordem fixa (validada para daltonismo em pares
# adjacentes). Com 5 categorias num gráfico de dispersão, a cor sozinha não
# basta — por isso cada categoria também tem um símbolo próprio.
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CORES_CATEGORIA = {
    "infraestrutura": PALETA[0],
    "saúde": PALETA[1],
    "segurança": PALETA[2],
    "educação": PALETA[3],
    "meio ambiente": PALETA[4],
}
SIMBOLOS_CATEGORIA = {  # nomes aceitos por matplotlib / plotly
    "infraestrutura": ("o", "circle"),
    "saúde": ("s", "square"),
    "segurança": ("^", "triangle-up"),
    "educação": ("D", "diamond"),
    "meio ambiente": ("P", "cross"),
}
# Rampa sequencial (azul, claro -> escuro) para heatmaps de similaridade.
RAMPA_SEQUENCIAL = ["#f7fbff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar_manifestacoes(caminho: str | Path = ARQUIVO_MANIFESTACOES) -> pd.DataFrame:
    """Lê o JSON de manifestações e devolve um DataFrame com coluna extra `n_caracteres`."""
    with open(caminho, encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    df["n_caracteres"] = df["texto"].str.len()
    return df


def carregar_gabarito(caminho: str | Path = ARQUIVO_GABARITO) -> dict:
    """Lê o gabarito com os pares de duplicatas reais e os 'falsos amigos'."""
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def pares_gabarito(gabarito: dict, chave: str = "duplicatas") -> set[frozenset]:
    """Converte a lista de pares do gabarito em um conjunto de pares não ordenados."""
    return {frozenset(item["par"]) for item in gabarito[chave]}


# ---------------------------------------------------------------------------
# Embeddings e similaridade
# ---------------------------------------------------------------------------
def carregar_modelo(nome: str = MODELO_PADRAO):
    """Carrega um SentenceTransformer pelo nome amigável (chave de MODELOS)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODELOS[nome]["id"])


def gerar_embeddings(modelo, textos: list[str], nome_modelo: str = MODELO_PADRAO) -> np.ndarray:
    """Gera embeddings L2-normalizados (assim, produto escalar == cosseno)."""
    prefixo = MODELOS[nome_modelo]["prefixo"]
    return modelo.encode(
        [prefixo + t for t in textos],
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def matriz_similaridade(vetores) -> np.ndarray:
    """Matriz de similaridade de cosseno n x n (aceita matrizes densas ou esparsas)."""
    return cosine_similarity(vetores)


def valores_fora_diagonal(sim: np.ndarray) -> np.ndarray:
    """Similaridades de todos os pares distintos (triângulo superior, sem a diagonal)."""
    i, j = np.triu_indices_from(sim, k=1)
    return sim[i, j]


def limiar_por_percentil(sim: np.ndarray, percentil: float = 90) -> float:
    """Limiar dinâmico: percentil da distribuição de similaridades entre pares distintos."""
    return float(np.percentile(valores_fora_diagonal(sim), percentil))


def detectar_duplicatas(
    textos: list[str],
    limiar: float = 0.85,
    ids: list[str] | None = None,
    modelo=None,
    nome_modelo: str = MODELO_PADRAO,
    embeddings: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Retorna todos os pares (i, j), i < j, cuja similaridade de cosseno entre
    embeddings é >= `limiar`, ordenados da maior para a menor similaridade.

    Parâmetros
    ----------
    textos : lista de manifestações.
    limiar : similaridade mínima para considerar duplicata.
    ids : identificadores opcionais (ex.: "M003"); por padrão usa o índice.
    modelo / nome_modelo : modelo usado para gerar os embeddings. Se `modelo`
        for None, o modelo `nome_modelo` é carregado.
    embeddings : embeddings já calculados (evita recalcular — útil no app e
        para testar vários limiares).
    """
    if ids is None:
        ids = [str(k) for k in range(len(textos))]
    if embeddings is None:
        modelo = modelo or carregar_modelo(nome_modelo)
        embeddings = gerar_embeddings(modelo, textos, nome_modelo)

    sim = matriz_similaridade(embeddings)
    i, j = np.triu_indices_from(sim, k=1)
    mascara = sim[i, j] >= limiar

    pares = pd.DataFrame(
        {
            "id_a": [ids[k] for k in i[mascara]],
            "id_b": [ids[k] for k in j[mascara]],
            "similaridade": sim[i, j][mascara],
            "texto_a": [textos[k] for k in i[mascara]],
            "texto_b": [textos[k] for k in j[mascara]],
        }
    )
    return pares.sort_values("similaridade", ascending=False, ignore_index=True)


def avaliar_deteccao(pares_detectados: set[frozenset], pares_reais: set[frozenset]) -> dict:
    """Precisão, revocação e F1 de um conjunto de pares detectados contra o gabarito."""
    vp = len(pares_detectados & pares_reais)
    fp = len(pares_detectados - pares_reais)
    fn = len(pares_reais - pares_detectados)
    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    return {"VP": vp, "FP": fp, "FN": fn, "precisão": precisao, "revocação": revocacao, "F1": f1}


def buscar_similares(
    consulta_emb: np.ndarray, base_emb: np.ndarray, top_k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Índices e scores das `top_k` linhas de `base_emb` mais similares à consulta."""
    scores = cosine_similarity(consulta_emb.reshape(1, -1), base_emb)[0]
    ordem = np.argsort(-scores)[:top_k]
    return ordem, scores[ordem]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
ESTRATEGIAS_CHUNKING = {
    "Recursiva (RecursiveCharacterTextSplitter)": "recursiva",
    "Tamanho fixo (CharacterTextSplitter, sem respeitar palavras)": "fixo",
    "Por sentenças (agrupa frases inteiras)": "sentencas",
}


def _dividir_sentencas(texto: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", texto) if s.strip()]


def dividir_em_chunks(
    texto: str,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    estrategia: str = "recursiva",
) -> list[str]:
    """
    Divide um texto em chunks.

    - "recursiva": RecursiveCharacterTextSplitter do LangChain; tenta quebrar
      primeiro em parágrafos, depois em frases, vírgulas, palavras e, só em
      último caso, caracteres.
    - "fixo": janelas de `chunk_size` caracteres, cortando no meio de palavras
      (linha de base "ingênua").
    - "sentencas": agrupa frases inteiras até `chunk_size` caracteres; o
      overlap é aproximado repetindo a(s) última(s) frase(s) que caibam nele.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap precisa ser menor que chunk_size.")

    if estrategia == "recursiva":
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            keep_separator="end",
        )
        return [c.strip() for c in splitter.split_text(texto)]

    if estrategia == "fixo":
        from langchain_text_splitters import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap, strip_whitespace=True
        )
        return splitter.split_text(texto)

    if estrategia == "sentencas":
        chunks, atual = [], []
        for frase in _dividir_sentencas(texto):
            if atual and len(" ".join(atual + [frase])) > chunk_size:
                chunks.append(" ".join(atual))
                # overlap: carrega as últimas frases que cabem em chunk_overlap
                carregadas = []
                for anterior in reversed(atual):
                    if len(" ".join([anterior] + carregadas)) > chunk_overlap:
                        break
                    carregadas.insert(0, anterior)
                atual = carregadas
            atual.append(frase)
        if atual:
            chunks.append(" ".join(atual))
        return chunks

    raise ValueError(f"Estratégia desconhecida: {estrategia}")


def chunks_consecutivos_similaridade(emb_chunks: np.ndarray) -> np.ndarray:
    """Cosseno entre cada chunk e o seguinte (mede a 'costura' semântica)."""
    if len(emb_chunks) < 2:
        return np.array([])
    return np.sum(emb_chunks[:-1] * emb_chunks[1:], axis=1)


# ---------------------------------------------------------------------------
# Estilo dos gráficos (matplotlib) — usado nos notebooks
# ---------------------------------------------------------------------------
def estilo_graficos() -> None:
    """Aplica um estilo limpo e consistente aos gráficos matplotlib."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 9.5,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#52514e",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": "#898781",
            "ytick.color": "#898781",
            "legend.frameon": False,
        }
    )


def cmap_sequencial():
    """Colormap sequencial de um só matiz (azul claro -> escuro)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("azul_seq", RAMPA_SEQUENCIAL)


# ---------------------------------------------------------------------------
# Redução de dimensionalidade
# ---------------------------------------------------------------------------
def reduzir_2d(vetores: np.ndarray, metodo: str = "PCA", perplexity: float = 8, semente: int = 42) -> np.ndarray:
    """Projeta os vetores em 2D com PCA ou t-SNE (métrica cosseno)."""
    if metodo.upper() == "PCA":
        return PCA(n_components=2, random_state=semente).fit_transform(vetores)
    # t-SNE exige perplexity < n_amostras
    perplexity = min(perplexity, max(2, len(vetores) - 1) / 3)
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="pca",
        random_state=semente,
    ).fit_transform(vetores)
