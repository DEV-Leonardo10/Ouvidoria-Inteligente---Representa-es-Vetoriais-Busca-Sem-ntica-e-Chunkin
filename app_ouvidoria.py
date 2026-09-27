"""
app_ouvidoria.py — Ouvidoria Inteligente: triagem semântica de manifestações.

Execute com:
    streamlit run app_ouvidoria.py

Abas
----
1. Busca Semântica  — o cidadão descreve o problema e vê as manifestações mais
                       parecidas (com aviso de possível duplicata).
2. Base Completa     — tabela filtrável + matriz de similaridade e pares duplicados.
3. Espaço Vetorial   — mapa 2D (PCA ou t-SNE) colorido por categoria ou cluster.
4. Chunking          — laboratório para dividir manifestações longas e inspecionar
                       os chunks e seus embeddings.

Toda a lógica de NLP (embeddings, similaridade, duplicatas, chunking, redução de
dimensionalidade) está em `ouvidoria_core.py`, compartilhada com os notebooks.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import ouvidoria_core as oc

st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

LIMITE_TEXTO_LONGO = 500  # caracteres; acima disso a manifestação é indexada também por chunks
COR_FAIXA = {"alta": "#0ca30c", "media": "#fab219", "baixa": "#d03b3b"}
EMOJI_FAIXA = {"alta": "🟢", "media": "🟡", "baixa": "🔴"}
EXEMPLOS_BUSCA = [
    "Tem uma cratera no asfalto perto da padaria da avenida",
    "Não tem médico no posto de saúde do meu bairro",
    "Som alto do bar não deixa ninguém dormir",
    "Rua escura, a lâmpada do poste está queimada",
    "Estacionamento da UPA sem luz e sem segurança à noite",
]


# ---------------------------------------------------------------------------
# Cache: modelo (recurso compartilhado) e dados/embeddings (valores)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo de embedding…")
def obter_modelo(nome_modelo: str):
    return oc.carregar_modelo(nome_modelo)


@st.cache_data
def obter_base() -> pd.DataFrame:
    return oc.carregar_manifestacoes()


@st.cache_data(show_spinner="Gerando embeddings…")
def embeddings_de(nome_modelo: str, textos: tuple[str, ...]) -> np.ndarray:
    return oc.gerar_embeddings(obter_modelo(nome_modelo), list(textos), nome_modelo)


@st.cache_data(show_spinner="Indexando manifestações…")
def indice_busca(nome_modelo: str, textos: tuple[str, ...], usar_chunks: bool, chunk_size: int, overlap: int):
    """
    Monta o índice de busca. Cada manifestação vira uma ou mais "unidades":
    o texto inteiro sempre, e, se for longa e `usar_chunks`, também cada chunk.
    O score de uma manifestação é o máximo entre as suas unidades.
    """
    unidades, dono = [], []
    for k, texto in enumerate(textos):
        unidades.append(texto)
        dono.append(k)
        if usar_chunks and len(texto) > LIMITE_TEXTO_LONGO:
            for chunk in oc.dividir_em_chunks(texto, chunk_size, overlap, "recursiva"):
                unidades.append(chunk)
                dono.append(k)
    return unidades, np.array(dono), embeddings_de(nome_modelo, tuple(unidades))


@st.cache_data(show_spinner="Projetando em 2D…")
def projecao_2d(vetores: np.ndarray, metodo: str, perplexity: float) -> np.ndarray:
    return oc.reduzir_2d(vetores, metodo, perplexity)


# ---------------------------------------------------------------------------
# Utilidades de interface
# ---------------------------------------------------------------------------
def faixa_do_score(score: float, faixas: tuple[float, float]) -> str:
    alta, media = faixas
    return "alta" if score > alta else "media" if score > media else "baixa"


def texto_com_destaque(texto: str, trecho: str | None) -> str:
    """HTML do texto com o `trecho` (chunk que casou com a busca) destacado."""
    if not trecho or trecho == texto or trecho not in texto:
        return html.escape(texto)
    ini = texto.index(trecho)
    fim = ini + len(trecho)
    return (html.escape(texto[:ini]) + "<mark style='background:#fde8a6;padding:0 2px;border-radius:3px'>"
            + html.escape(trecho) + "</mark>" + html.escape(texto[fim:]))


def cartao_resultado(posicao: int, linha: pd.Series, score: float, faixa: str, trecho: str | None) -> None:
    cor_cat = oc.CORES_CATEGORIA.get(linha.categoria_oficial, "#898781")
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        c1.markdown(
            f"**{posicao}. {linha.id}** &nbsp;·&nbsp; "
            f"<span style='color:{cor_cat};font-weight:600'>● {linha.categoria_oficial}</span>"
            f" &nbsp;·&nbsp; <span style='opacity:.7'>{linha.data}</span>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<div style='text-align:right;font-size:1.05rem'>{EMOJI_FAIXA[faixa]} "
            f"<b style='font-variant-numeric:tabular-nums'>{score:.3f}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='border-left:4px solid {COR_FAIXA[faixa]};padding-left:.7rem;line-height:1.5'>"
            f"{texto_com_destaque(linha.texto, trecho)}</div>",
            unsafe_allow_html=True,
        )
        if trecho and trecho != linha.texto:
            st.caption("Trecho destacado: o chunk desta manifestação longa que mais se aproximou da sua descrição.")


def layout_plotly(fig: go.Figure, altura: int = 480) -> go.Figure:
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
df = obter_base()
TEXTOS = tuple(df["texto"])
IDS = df["id"].tolist()

with st.sidebar:
    st.title("🏛️ Ouvidoria Inteligente")
    st.caption("Triagem semântica de manifestações cidadãs")

    nome_modelo = st.selectbox("Modelo de embedding", list(oc.MODELOS), index=0)
    cfg_modelo = oc.MODELOS[nome_modelo]
    top_k = st.slider("Top-k resultados da busca", 1, 15, 5)

    with st.expander("Faixas de cor da busca", expanded=False):
        faixa_alta = st.slider("🟢 acima de", 0.0, 1.0, cfg_modelo["faixas"][0], 0.01, key=f"alta_{nome_modelo}")
        faixa_media = st.slider("🟡 acima de", 0.0, 1.0, cfg_modelo["faixas"][1], 0.01, key=f"media_{nome_modelo}")
        if cfg_modelo["curto"] == "E5-small":
            st.caption("O E5 comprime todos os cossenos entre ~0,80 e ~0,95, então as faixas foram recalibradas "
                       "(0,88/0,85). Mesmo assim ele separa mal por valor absoluto; veja a Entrega 1.")
        else:
            st.caption("Padrão do enunciado: 🟢 > 0,70 · 🟡 > 0,50 · 🔴 demais.")
    FAIXAS = (faixa_alta, faixa_media)

    with st.expander("Indexação de textos longos", expanded=False):
        usar_chunks = st.toggle("Indexar manifestações longas por chunks", value=True,
                                help="O MiniLM e o MPNet leem só 128 tokens: sem chunks, o final das manifestações "
                                     "longas é ignorado na busca.")
        chunk_size_busca = st.slider("chunk_size (busca)", 80, 400, 120, 10, disabled=not usar_chunks)
        overlap_busca = st.slider("chunk_overlap (busca)", 0, 150, 40, 10, disabled=not usar_chunks)
        if overlap_busca >= chunk_size_busca:
            overlap_busca = chunk_size_busca // 3
            st.caption(f"Overlap ajustado para {overlap_busca}: precisa ser menor que o chunk_size.")

    st.divider()
    st.caption(f"**Modelo:** `{cfg_modelo['id']}`  \n"
               f"**Dimensão:** {cfg_modelo['dim']}  \n"
               f"**Limiar de duplicata calibrado:** {cfg_modelo['limiar_duplicata']:.3f}  \n"
               f"**Base:** {len(df)} manifestações")

modelo = obter_modelo(nome_modelo)
EMB_BASE = embeddings_de(nome_modelo, TEXTOS)

aba_busca, aba_base, aba_espaco, aba_chunk = st.tabs(
    ["🔎 Busca Semântica", "📋 Base Completa", "🗺️ Espaço Vetorial", "✂️ Chunking"]
)

# ---------------------------------------------------------------------------
# Aba 1 — Busca semântica
# ---------------------------------------------------------------------------
with aba_busca:
    st.subheader("Descreva o problema com suas palavras")
    st.caption("O sistema compara o **significado** da sua descrição com as manifestações registradas, "
               "e não apenas as palavras.")

    def usar_exemplo():
        if st.session_state.get("exemplo"):
            st.session_state["consulta"] = st.session_state["exemplo"]

    st.pills("Exemplos", EXEMPLOS_BUSCA, selection_mode="single", label_visibility="collapsed",
             key="exemplo", on_change=usar_exemplo)
    consulta = st.text_area("Descrição", key="consulta", height=100,
                            placeholder="Ex.: a rua da minha casa está toda esburacada e os carros quebram a suspensão…")

    if consulta.strip():
        st.session_state["ultima_busca"] = consulta.strip()
        unidades, dono, emb_unidades = indice_busca(nome_modelo, TEXTOS, usar_chunks, chunk_size_busca, overlap_busca)
        q = oc.gerar_embeddings(modelo, [consulta.strip()], nome_modelo)[0]
        scores_unidades = emb_unidades @ q

        # score da manifestação = máximo entre suas unidades (texto inteiro + chunks)
        melhor_score = np.full(len(df), -np.inf)
        melhor_unidade = np.zeros(len(df), dtype=int)
        for u, (k, s) in enumerate(zip(dono, scores_unidades)):
            if s > melhor_score[k]:
                melhor_score[k], melhor_unidade[k] = s, u
        ordem = np.argsort(-melhor_score)[:top_k]

        principal = melhor_score[ordem[0]]
        if principal >= cfg_modelo["limiar_duplicata"]:
            st.warning(f"⚠️ **Possível duplicata:** sua descrição é muito parecida com **{IDS[ordem[0]]}** "
                       f"(score {principal:.3f} ≥ limiar calibrado {cfg_modelo['limiar_duplicata']:.3f}). "
                       "Talvez esse problema já tenha sido registrado.")

        col_res, col_graf = st.columns([3, 2], gap="large")
        with col_res:
            for pos, k in enumerate(ordem, start=1):
                trecho = unidades[melhor_unidade[k]]
                cartao_resultado(pos, df.iloc[k], float(melhor_score[k]), faixa_do_score(melhor_score[k], FAIXAS), trecho)
        with col_graf:
            resultado = pd.DataFrame({
                "manifestação": [IDS[k] for k in ordem],
                "score": melhor_score[ordem],
                "faixa": [EMOJI_FAIXA[faixa_do_score(s, FAIXAS)] for s in melhor_score[ordem]],
            })
            fig = px.bar(resultado.iloc[::-1], x="score", y="manifestação", orientation="h", text="faixa",
                         color="faixa", color_discrete_map={EMOJI_FAIXA[f]: COR_FAIXA[f] for f in COR_FAIXA})
            fig.add_vline(x=faixa_alta, line_dash="dot", line_color="#898781")
            fig.add_vline(x=faixa_media, line_dash="dot", line_color="#898781")
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_layout(showlegend=False, title="Score de similaridade (cosseno)")
            st.plotly_chart(layout_plotly(fig, 60 + 45 * len(ordem)))

            todas = np.sort(melhor_score)[::-1]
            fig2 = px.histogram(x=todas, nbins=20, labels={"x": "score"}, color_discrete_sequence=["#9ec5f4"])
            fig2.add_vline(x=todas[min(top_k, len(todas)) - 1], line_color="#0b0b0b",
                           annotation_text=f"corte do top-{top_k}", annotation_position="top")
            fig2.update_layout(title="Distribuição dos scores nas 40 manifestações", yaxis_title="manifestações")
            st.plotly_chart(layout_plotly(fig2, 260))
            st.caption("O histograma mostra se os primeiros resultados realmente se destacam do resto da base "
                       "ou se estão \"no meio da multidão\".")
    else:
        st.info("Digite uma descrição ou escolha um exemplo acima para buscar.")

# ---------------------------------------------------------------------------
# Aba 2 — Base completa + matriz de similaridade
# ---------------------------------------------------------------------------
with aba_base:
    st.subheader("Base de manifestações")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Manifestações", len(df))
    m2.metric("Categorias", df["categoria_oficial"].nunique())
    m3.metric(f"Textos longos (> {LIMITE_TEXTO_LONGO} car.)", int((df["n_caracteres"] > LIMITE_TEXTO_LONGO).sum()))
    m4.metric("Período", f"{df['data'].min()[5:]} → {df['data'].max()[5:]}".replace("-", "/"))

    f1, f2 = st.columns([2, 3])
    cats = f1.multiselect("Categorias", sorted(df["categoria_oficial"].unique()), placeholder="todas")
    termo = f2.text_input("Filtrar por texto (palavra-chave)", placeholder="ex.: buraco")
    visao = df.copy()
    if cats:
        visao = visao[visao["categoria_oficial"].isin(cats)]
    if termo:
        visao = visao[visao["texto"].str.contains(termo, case=False, regex=False)]
    st.dataframe(
        visao[["id", "data", "categoria_oficial", "n_caracteres", "texto"]],
        hide_index=True, height=320,
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "data": st.column_config.TextColumn("Data", width="small"),
            "categoria_oficial": st.column_config.TextColumn("Categoria", width="small"),
            "n_caracteres": st.column_config.ProgressColumn("Tamanho", format="%d car.", min_value=0,
                                                            max_value=int(df["n_caracteres"].max())),
            "texto": st.column_config.TextColumn("Texto", width="large"),
        },
    )
    if termo:
        st.caption(f"A busca por palavra-chave \"{termo}\" encontrou {len(visao)} manifestação(ões). "
                   "Compare com a aba de Busca Semântica, que também encontra sinônimos e paráfrases.")

    st.divider()
    st.subheader("Matriz de similaridade")
    if st.button("🧮 Gerar matriz de similaridade", type="primary"):
        st.session_state["matriz_modelo"] = nome_modelo

    if st.session_state.get("matriz_modelo") == nome_modelo:
        S = oc.matriz_similaridade(EMB_BASE)
        ordem_cat = df.sort_values(["categoria_oficial", "id"]).index.to_numpy()
        M = S[np.ix_(ordem_cat, ordem_cat)].copy()
        np.fill_diagonal(M, np.nan)
        ids_ord = [IDS[k] for k in ordem_cat]
        fora = oc.valores_fora_diagonal(S)
        fig = px.imshow(M, x=ids_ord, y=ids_ord, color_continuous_scale=oc.RAMPA_SEQUENCIAL,
                        zmin=float(np.percentile(fora, 1)), zmax=float(fora.max()), aspect="equal",
                        labels=dict(color="cosseno"))
        fig.update_traces(hovertemplate="%{y} × %{x}<br>cosseno = %{z:.3f}<extra></extra>")
        fig.update_xaxes(tickfont_size=9, tickangle=90)
        fig.update_yaxes(tickfont_size=9)
        fig.update_layout(title=f"Cosseno entre as 40 manifestações ({cfg_modelo['curto']}), ordenadas por categoria")
        c_mat, c_pares = st.columns([3, 2], gap="large")
        with c_mat:
            st.plotly_chart(layout_plotly(fig, 720))
            st.caption("Escala de cor ajustada entre o percentil 1 e o máximo dos pares, para não esconder o contraste "
                       "em modelos de escala comprimida como o E5.")
        with c_pares:
            limiar = st.slider("Limiar para duplicatas", float(np.round(fora.min(), 2)), 1.0,
                               float(cfg_modelo["limiar_duplicata"]), 0.005,
                               help="Valor inicial = limiar calibrado para este modelo no notebook de duplicatas.")
            pares = oc.detectar_duplicatas(list(TEXTOS), limiar=limiar, ids=IDS,
                                           nome_modelo=nome_modelo, embeddings=EMB_BASE)
            reais = oc.pares_gabarito(oc.carregar_gabarito())
            pares["gabarito"] = ["✅ duplicata real" if frozenset((a, b)) in reais else "❌ não é duplicata"
                                 for a, b in zip(pares.id_a, pares.id_b)]
            av = oc.avaliar_deteccao({frozenset((a, b)) for a, b in zip(pares.id_a, pares.id_b)}, reais)
            k1, k2, k3 = st.columns(3)
            k1.metric("Pares marcados", len(pares))
            k2.metric("Precisão", f"{av['precisão']:.0%}")
            k3.metric("Revocação", f"{av['revocação']:.0%}")
            st.dataframe(pares[["id_a", "id_b", "similaridade", "gabarito", "texto_a", "texto_b"]],
                         hide_index=True, height=440,
                         column_config={"similaridade": st.column_config.NumberColumn(format="%.3f")})
            st.download_button("⬇️ Baixar pares (CSV)", pares.to_csv(index=False).encode("utf-8"),
                               file_name=f"duplicatas_{cfg_modelo['curto']}_{limiar:.3f}.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Aba 3 — Espaço vetorial
# ---------------------------------------------------------------------------
with aba_espaco:
    st.subheader("Mapa semântico das manifestações")
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.4, 1.6])
    metodo = c1.radio("Projeção", ["PCA", "t-SNE"], horizontal=True)
    perplexity = c2.slider("Perplexity (t-SNE)", 3, 20, 8, disabled=metodo != "t-SNE")
    colorir = c3.radio("Colorir por", ["Categoria oficial", "Cluster (K-Means)"], horizontal=True)
    k_clusters = c4.slider("Nº de clusters", 2, 8, 5, disabled=colorir == "Categoria oficial")

    o1, o2 = st.columns(2)
    ligar_dups = o1.toggle("Ligar as duplicatas do gabarito", value=True)
    mostrar_busca = o2.toggle("Mostrar a última busca no mapa", value=bool(st.session_state.get("ultima_busca")),
                              disabled=not st.session_state.get("ultima_busca"))

    vetores = EMB_BASE
    if mostrar_busca and st.session_state.get("ultima_busca"):
        q = oc.gerar_embeddings(modelo, [st.session_state["ultima_busca"]], nome_modelo)
        vetores = np.vstack([EMB_BASE, q])
    P = projecao_2d(vetores, metodo, perplexity)

    clusters = KMeans(n_clusters=k_clusters, n_init=20, random_state=42).fit_predict(EMB_BASE)
    mapa = df[["id", "categoria_oficial", "texto"]].copy()
    mapa["x"], mapa["y"] = P[: len(df), 0], P[: len(df), 1]
    mapa["cluster"] = [f"cluster {c + 1}" for c in clusters]
    mapa["texto_curto"] = mapa["texto"].str.slice(0, 110) + "…"

    if colorir == "Categoria oficial":
        fig = px.scatter(mapa, x="x", y="y", color="categoria_oficial", symbol="categoria_oficial",
                         color_discrete_map=oc.CORES_CATEGORIA,
                         symbol_map={c: s[1] for c, s in oc.SIMBOLOS_CATEGORIA.items()},
                         hover_name="id", hover_data={"x": False, "y": False, "texto_curto": True, "categoria_oficial": True})
    else:
        fig = px.scatter(mapa, x="x", y="y", color="cluster", symbol="categoria_oficial",
                         color_discrete_sequence=oc.PALETA,
                         symbol_map={c: s[1] for c, s in oc.SIMBOLOS_CATEGORIA.items()},
                         hover_name="id", hover_data={"x": False, "y": False, "texto_curto": True, "categoria_oficial": True})
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="#fcfcfb")))
    if ligar_dups:
        pos_id = {mid: k for k, mid in enumerate(IDS)}
        for par in oc.carregar_gabarito()["duplicatas"]:
            a, b = (pos_id[m] for m in par["par"])
            fig.add_trace(go.Scatter(x=P[[a, b], 0], y=P[[a, b], 1], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color="#52514e", width=1.2, dash="dot")))
    for _, r in mapa.iterrows():
        fig.add_annotation(x=r.x, y=r.y, text=r.id, showarrow=False, yshift=11, font=dict(size=9, color="#898781"))
    if len(vetores) > len(df):
        fig.add_trace(go.Scatter(x=[P[-1, 0]], y=[P[-1, 1]], mode="markers+text", name="sua busca",
                                 text=["sua busca"], textposition="bottom center",
                                 marker=dict(symbol="star", size=22, color="#0b0b0b", line=dict(width=1, color="#fcfcfb"))))
    fig.update_xaxes(showticklabels=False, title="")
    fig.update_yaxes(showticklabels=False, title="")
    fig.update_layout(title=f"{metodo} dos embeddings ({cfg_modelo['curto']})")

    col_mapa, col_metr = st.columns([3, 1.3], gap="large")
    with col_mapa:
        st.plotly_chart(layout_plotly(fig, 620))
        st.caption("Linhas pontilhadas ligam os pares de duplicatas do gabarito. A cor mostra a categoria (ou o "
                   "cluster) e o símbolo mostra sempre a categoria oficial, para não depender só de cor.")
    with col_metr:
        rotulos = df["categoria_oficial"].to_numpy()
        ari = adjusted_rand_score(rotulos, clusters)
        sil = silhouette_score(EMB_BASE, rotulos, metric="cosine")
        st.metric("Concordância cluster × categoria (ARI)", f"{ari:.2f}",
                  help="Adjusted Rand Index: 1 = clusters idênticos às categorias; 0 = acaso.")
        st.metric("Silhueta das categorias oficiais", f"{sil:.2f}",
                  help="Entre −1 e 1. Perto de 0 = categorias sobrepostas no espaço semântico.")
        st.markdown("**Clusters × categorias**")
        tabela = pd.crosstab(mapa["cluster"], mapa["categoria_oficial"])
        st.dataframe(tabela.style.background_gradient(cmap=oc.cmap_sequencial(), axis=None),
                     )

    with st.expander("💬 Comentário: os clusters semânticos coincidem com as categorias oficiais?", expanded=True):
        st.markdown(f"""
**Só em parte, e a parte que não coincide é informativa.** Com K-Means em 5 clusters, a concordância com as categorias
é moderada (ARI = **{ari:.2f}** com o {cfg_modelo['curto']}; nos três modelos testados fica em torno de 0,4).

* **Educação e saúde formam clusters quase puros.** O vocabulário é específico (escola, merenda, professor,
  transporte escolar; posto, médico, vacina, UPA) e não se confunde com o resto.
* **Infraestrutura, segurança e meio ambiente se misturam.** O cidadão descreve o problema pela **consequência** que
  sente, e não pela secretaria responsável: poste apagado (infraestrutura) é contado como medo de assalto (segurança);
  semáforo desligado e carros em alta velocidade (segurança) ficam ao lado de buracos e obras (infraestrutura); o córrego
  que transborda (meio ambiente) fica perto do buraco que piora com a chuva (infraestrutura).
* **Manifestações longas ficam "no meio do caminho"** entre grupos, porque o embedding de um texto com vários assuntos
  é uma média deles (veja a aba Chunking).

**Leitura para a gestão:** as categorias oficiais são **administrativas** (qual secretaria resolve) e os clusters
semânticos são **temáticos** (o que o cidadão viveu). Discordar não é erro do modelo. Os clusters revelam demandas
transversais, como "iluminação + segurança", que pedem ação conjunta de mais de uma secretaria. Para rotear, o ideal é
um classificador supervisionado treinado sobre os embeddings. Para descobrir padrões novos, os clusters.
""")

# ---------------------------------------------------------------------------
# Aba 4 — Chunking
# ---------------------------------------------------------------------------
with aba_chunk:
    st.subheader("Laboratório de chunking")
    st.caption("Cole uma manifestação longa, escolha a estratégia e os parâmetros e veja como o texto é dividido e "
               "onde cada chunk cai no espaço semântico.")

    longas = df[df["n_caracteres"] > LIMITE_TEXTO_LONGO].sort_values("id")
    opcoes = ["(texto livre)"] + [f"{r.id} — {r.categoria_oficial} ({r.n_caracteres} car.)" for r in longas.itertuples()]
    escolha = st.selectbox("Carregar uma manifestação longa da base", opcoes, index=1)
    if escolha != "(texto livre)" and st.session_state.get("_ultima_escolha") != escolha:
        st.session_state["texto_chunk"] = longas.set_index("id").loc[escolha.split(" — ")[0], "texto"]
    st.session_state["_ultima_escolha"] = escolha
    texto_longo = st.text_area("Manifestação", key="texto_chunk", height=170)

    p1, p2, p3 = st.columns([2, 1.3, 1.3])
    estrategia_nome = p1.selectbox("Estratégia", list(oc.ESTRATEGIAS_CHUNKING))
    chunk_size = p2.slider("chunk_size (caracteres)", 50, 800, 300, 10)
    chunk_overlap = p3.slider("chunk_overlap (caracteres)", 0, 300, 80, 10)

    if not texto_longo.strip():
        st.info("Cole um texto ou carregue uma manifestação da base.")
    elif chunk_overlap >= chunk_size:
        st.error("O chunk_overlap precisa ser menor que o chunk_size.")
    else:
        chunks = oc.dividir_em_chunks(texto_longo.strip(), chunk_size, chunk_overlap, oc.ESTRATEGIAS_CHUNKING[estrategia_nome])
        emb_chunks = oc.gerar_embeddings(modelo, chunks, nome_modelo)
        emb_doc = oc.gerar_embeddings(modelo, [texto_longo.strip()], nome_modelo)[0]
        coesao = oc.chunks_consecutivos_similaridade(emb_chunks)
        n_tokens = [len(modelo.tokenizer(c)["input_ids"]) for c in chunks]
        tokens_doc = len(modelo.tokenizer(texto_longo.strip())["input_ids"])
        limite = modelo.max_seq_length

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Chunks", len(chunks))
        k2.metric("Tamanho médio", f"{np.mean([len(c) for c in chunks]):.0f} car.")
        k3.metric("Coesão média entre vizinhos", f"{coesao.mean():.2f}" if len(coesao) else "—",
                  help="Cosseno médio entre o chunk i e o chunk i+1.")
        k4.metric("Maior chunk", f"{max(n_tokens)} tokens", delta=f"limite {limite}",
                  delta_color="off" if max(n_tokens) <= limite else "inverse")
        if tokens_doc > limite:
            st.warning(f"O texto inteiro tem **{tokens_doc} tokens**, mas o modelo lê só **{limite}**. Sem chunking, "
                       "o final da manifestação seria ignorado na busca.")
        if max(n_tokens) > limite:
            st.error("Há chunks maiores que o limite do modelo: eles também serão truncados. Reduza o chunk_size.")

        col_txt, col_viz = st.columns([1.1, 1], gap="large")
        with col_txt:
            st.markdown("**Chunks gerados** (trecho repetido do chunk anterior em destaque)")
            vizinhos = emb_chunks @ EMB_BASE.T
            if texto_longo.strip() in TEXTOS:          # não comparar o chunk com a própria manifestação de origem
                vizinhos[:, TEXTOS.index(texto_longo.strip())] = -np.inf
            for i, chunk in enumerate(chunks):
                repetido = 0
                if i > 0:
                    anterior = chunks[i - 1]
                    for n in range(min(len(anterior), len(chunk)), 0, -1):
                        if anterior.endswith(chunk[:n]):
                            repetido = n
                            break
                j = int(np.argmax(vizinhos[i]))
                corpo = (f"<mark style='background:#cde2fb;padding:0 2px;border-radius:3px'>{html.escape(chunk[:repetido])}</mark>"
                         if repetido else "") + html.escape(chunk[repetido:])
                with st.container(border=True):
                    st.markdown(
                        f"<div style='font-size:.8rem;opacity:.75'>chunk {i} · {len(chunk)} car. · {n_tokens[i]} tokens"
                        + (f" · coesão com o anterior {coesao[i - 1]:.2f}" if i > 0 else "")
                        + f" · mais parecido na base: <b>{IDS[j]}</b> ({df.iloc[j].categoria_oficial}, {vizinhos[i, j]:.2f})</div>"
                        f"<div style='line-height:1.5;margin-top:.2rem'>{corpo}</div>",
                        unsafe_allow_html=True,
                    )

        with col_viz:
            # Projeção 2D: base (cinza) + chunks (ligados em sequência) + documento inteiro (estrela)
            X = np.vstack([EMB_BASE, emb_chunks, emb_doc[None, :]])
            metodo_chunk = st.radio("Projeção dos chunks", ["PCA", "t-SNE"], horizontal=True, key="proj_chunk")
            P = oc.reduzir_2d(X, metodo_chunk, perplexity=8)
            nb_ = len(df)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=P[:nb_, 0], y=P[:nb_, 1], mode="markers", name="manifestações da base",
                marker=dict(size=9, color=[oc.CORES_CATEGORIA[c] for c in df["categoria_oficial"]], opacity=0.35),
                text=[f"{r.id} · {r.categoria_oficial}<br>{r.texto[:90]}…" for r in df.itertuples()],
                hovertemplate="%{text}<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=P[nb_:nb_ + len(chunks), 0], y=P[nb_:nb_ + len(chunks), 1], mode="lines+markers+text",
                name="chunks", text=[str(i) for i in range(len(chunks))], textposition="top center",
                line=dict(color="#0b0b0b", width=1), marker=dict(size=13, color="#0b0b0b", line=dict(width=1, color="#fcfcfb")),
                customdata=[c[:120] + "…" for c in chunks], hovertemplate="chunk %{text}<br>%{customdata}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[P[-1, 0]], y=[P[-1, 1]], mode="markers", name="texto inteiro",
                                     marker=dict(symbol="star", size=22, color="#eb6834", line=dict(width=1, color="#0b0b0b")),
                                     hovertemplate="texto inteiro<extra></extra>"))
            fig.update_xaxes(showticklabels=False, title="")
            fig.update_yaxes(showticklabels=False, title="")
            fig.update_layout(title=f"Chunks no espaço semântico ({metodo_chunk})",
                              legend=dict(orientation="h", y=-0.05))
            st.plotly_chart(layout_plotly(fig, 440))
            st.caption("Pontos coloridos claros = manifestações da base, na cor da categoria. Se os chunks se "
                       "espalham em direção a categorias diferentes, a manifestação contém mais de uma demanda.")

            if len(chunks) > 1:
                Sc = emb_chunks @ emb_chunks.T
                figc = px.imshow(Sc, x=[f"c{i}" for i in range(len(chunks))], y=[f"c{i}" for i in range(len(chunks))],
                                 color_continuous_scale=oc.RAMPA_SEQUENCIAL, zmin=float(Sc.min()), zmax=1.0,
                                 text_auto=".2f", labels=dict(color="cosseno"))
                figc.update_layout(title="Similaridade entre chunks")
                st.plotly_chart(layout_plotly(figc, 340))

        with st.expander(f"🔢 Embeddings dos chunks ({emb_chunks.shape[1]} dimensões)"):
            n_dims = 12
            tabela_emb = pd.DataFrame(emb_chunks[:, :n_dims], columns=[f"d{i}" for i in range(n_dims)])
            tabela_emb.insert(0, "chunk", [f"c{i}" for i in range(len(chunks))])
            tabela_emb["norma"] = np.linalg.norm(emb_chunks, axis=1)
            st.caption(f"Primeiras {n_dims} de {emb_chunks.shape[1]} dimensões. Os vetores são normalizados "
                       "(norma 1), então o produto escalar entre eles é o cosseno.")
            st.dataframe(tabela_emb.style.format(precision=3).background_gradient(
                cmap="RdBu_r", subset=[f"d{i}" for i in range(n_dims)], vmin=-0.15, vmax=0.15),
                hide_index=True)
            fig_e = px.imshow(emb_chunks, aspect="auto", color_continuous_scale="RdBu_r", zmin=-0.15, zmax=0.15,
                              labels=dict(x="dimensão", y="chunk", color="valor"),
                              y=[f"c{i}" for i in range(len(chunks))])
            fig_e.update_layout(title="Todas as dimensões (chunks parecidos têm \"impressões digitais\" parecidas)")
            st.plotly_chart(layout_plotly(fig_e, 110 + 28 * len(chunks)))
            csv = pd.DataFrame(emb_chunks, index=[f"c{i}" for i in range(len(chunks))]).assign(texto=chunks)
            st.download_button("⬇️ Baixar chunks + embeddings (CSV)", csv.to_csv().encode("utf-8"),
                               file_name="chunks_embeddings.csv", mime="text/csv")
