import subprocess
import sys
import streamlit as st
import os
from dotenv import load_dotenv
from src.embedder import charger_modele, charger_index
from src.agents import RetrieverAgent, GeneratorAgent

st.set_page_config(
    page_title="Assistant Médicaments — BDPM",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY manquante dans le fichier .env")
    st.stop()

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Reset Streamlit ──────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 4rem 2.5rem !important;
    max-width: 820px !important;
}

/* ─── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #9ca3af !important;
}
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] h3 {
    color: #f3f4f6 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1f2937 !important;
    margin: 0.6rem 0;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: #1f2937;
    color: #d1d5db;
    border: 1px solid #374151;
    border-radius: 7px;
    font-size: 0.81rem;
    padding: 7px 12px;
    text-align: left;
    transition: all 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #374151;
    color: #f9fafb;
    border-color: #4b5563;
}
[data-testid="stSidebar"] .stToggle label p {
    color: #d1d5db !important;
    font-size: 0.82rem !important;
}

/* ─── En-tête principal ────────────────────────────────────────────────── */
.app-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0c1e42;
    letter-spacing: -0.4px;
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.app-subtitle {
    font-size: 0.77rem;
    color: #6b7280;
    margin: 3px 0 0 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.app-divider {
    border: none;
    border-top: 2px solid #0c1e42;
    margin: 0.7rem 0 0 0;
}

/* ─── Bulles de chat ───────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 6px 0 !important;
    gap: 10px !important;
}

/* Message utilisateur : bulle droite */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
    align-items: flex-end;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] {
    background: #dbeafe;
    border-radius: 18px 4px 18px 18px;
    padding: 11px 15px;
    max-width: 78%;
    margin-left: auto;
    margin-right: 0;
    border: 1px solid #bfdbfe;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] p {
    color: #1e3a8a;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.5;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="chatAvatarIcon-user"] {
    margin-left: 8px;
    margin-right: 0;
}

/* Message assistant : propre, sans fond */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] {
    background: transparent;
    padding: 4px 0;
    max-width: 92%;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] p {
    color: #111827;
    font-size: 0.9rem;
    line-height: 1.68;
}

/* ─── Barre de saisie (style Claude — fond noir, bordure fine) ─────────── */

/* Zone fixe en bas : fond sombre qui tranche proprement avec le chat */
section[data-testid="stBottom"] {
    background: #0d0d0d !important;
    padding: 0 !important;
}
section[data-testid="stBottom"] > div {
    background: #0d0d0d !important;
    padding: 14px 0 22px 0 !important;
    border-top: 1px solid #262626 !important;
}

/* Reset des conteneurs intermédiaires */
[data-testid="stChatInput"] {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* La barre elle-même */
[data-testid="stChatInput"] > div > div {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    background: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 16px !important;
    padding: 6px 8px 6px 18px !important;
    transition: border-color 0.2s !important;
    box-shadow: none !important;
    overflow: visible !important;
}

/* Bordure légèrement plus visible au focus */
[data-testid="stChatInput"] > div > div:focus-within {
    border-color: #555555 !important;
    background: #1f1f1f !important;
}

/* Textarea : texte blanc, fond transparent */
[data-testid="stChatInput"] textarea {
    flex: 1 1 auto !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 10px 4px !important;
    margin: 0 !important;
    font-size: 0.92rem !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: #e5e7eb !important;
    line-height: 1.55 !important;
    resize: none !important;
    min-height: 42px !important;
    max-height: 180px !important;
    caret-color: #e5e7eb !important;
}
[data-testid="stChatInput"] textarea:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #555555 !important;
    font-size: 0.91rem !important;
}

/* Bouton envoi : cercle blanc discret */
[data-testid="stChatInput"] button {
    flex-shrink: 0 !important;
    background: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button:hover:not(:disabled) {
    background: #e5e7eb !important;
}
[data-testid="stChatInput"] button:disabled {
    background: #2a2a2a !important;
    cursor: default !important;
}
[data-testid="stChatInput"] button svg {
    fill: #0d0d0d !important;
    width: 15px !important;
    height: 15px !important;
    display: block !important;
}
[data-testid="stChatInput"] button:disabled svg {
    fill: #444444 !important;
}

/* ─── Bandeau reformulation ────────────────────────────────────────────── */
.reformulation-notice {
    font-size: 0.78rem;
    color: #4b5563;
    font-style: italic;
    background: #f0f7ff;
    border-left: 3px solid #3b82f6;
    padding: 5px 12px;
    border-radius: 0 5px 5px 0;
    margin: 2px 0 10px 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ─── Cartes sources ───────────────────────────────────────────────────── */
.source-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.source-card-name {
    font-weight: 600;
    font-size: 0.84rem;
    color: #0c1e42;
}
.source-card-badge {
    display: inline-block;
    font-size: 0.7rem;
    background: #ede9fe;
    color: #5b21b6;
    padding: 2px 9px;
    border-radius: 10px;
    margin: 4px 0;
    font-weight: 500;
}
.source-card-scores {
    font-size: 0.73rem;
    color: #9ca3af;
    margin: 1px 0 5px 0;
}
.source-card-body {
    font-size: 0.81rem;
    color: #374151;
    line-height: 1.55;
    border-top: 1px solid #f3f4f6;
    padding-top: 8px;
    margin-top: 4px;
    white-space: pre-wrap;
}

/* ─── Expander sources ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    background: transparent;
    margin-top: 8px;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: #6b7280 !important;
    font-weight: 500;
    padding: 8px 12px;
}
[data-testid="stExpander"] summary:hover {
    color: #374151 !important;
}

/* ─── Écran d'accueil ──────────────────────────────────────────────────── */
.welcome-card {
    background: linear-gradient(140deg, #f0f7ff 0%, #f8fafc 100%);
    border: 1px solid #dbeafe;
    border-radius: 12px;
    padding: 26px 30px;
    margin: 24px 0 16px 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.welcome-heading {
    font-size: 1rem;
    font-weight: 600;
    color: #1d4ed8;
    margin-bottom: 8px;
}
.welcome-desc {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.65;
    margin-bottom: 16px;
}
.welcome-label {
    font-size: 0.73rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94a3b8;
    margin-bottom: 8px;
}
.welcome-pill {
    display: inline-block;
    background: #fff;
    border: 1px solid #dbeafe;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    color: #1d4ed8;
    margin: 3px 4px 3px 0;
}

/* ─── Hors périmètre ───────────────────────────────────────────────────── */
.out-of-scope {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.86rem;
    line-height: 1.65;
    color: #92400e;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ─── Stats sidebar ────────────────────────────────────────────────────── */
.sb-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #1f2937;
    font-size: 0.8rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.sb-stat-lbl { color: #6b7280; }
.sb-stat-val { font-weight: 600; color: #60a5fa; }
.sb-section {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #374151;
    margin: 1.1rem 0 0.4rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.sb-footer {
    font-size: 0.69rem;
    color: #374151;
    line-height: 1.75;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── Initialisation du système ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialisation du système RAG...")
def init_rag_system():
    try:
        modele = charger_modele()
        index, chunks = charger_index()
        agent_retriever = RetrieverAgent(
            modele=modele,
            index=index,
            chunks=chunks,
            avec_reformulation=True,
        )
        agent_generateur = GeneratorAgent()
        return agent_retriever, agent_generateur, chunks
    except FileNotFoundError as e:
        st.error(f"{e} — Lancez d'abord : python indexer.py")
        st.stop()


agent_retriever, agent_generateur, chunks_index = init_rag_system()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Assistant Médicaments**")
    st.markdown('<div class="sb-section">Base de connaissances</div>', unsafe_allow_html=True)

    nb_meds   = len(set(c["metadata"]["code_cis"] for c in chunks_index))
    nb_chunks = len(chunks_index)
    nb_types  = len(set(c["metadata"].get("type_chunk") for c in chunks_index))

    st.markdown(f"""
    <div class="sb-stat">
        <span class="sb-stat-lbl">Médicaments</span>
        <span class="sb-stat-val">{nb_meds}</span>
    </div>
    <div class="sb-stat">
        <span class="sb-stat-lbl">Chunks indexés</span>
        <span class="sb-stat-val">{nb_chunks}</span>
    </div>
    <div class="sb-stat">
        <span class="sb-stat-lbl">Sections couvertes</span>
        <span class="sb-stat-val">{nb_types}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Paramètres</div>', unsafe_allow_html=True)
    show_debug = st.toggle("Afficher les scores de pertinence", value=False)

    st.markdown('<div class="sb-section">Actions</div>', unsafe_allow_html=True)

    if st.button("Ré-indexer la base"):
        with st.spinner("Indexation en cours..."):
            result = subprocess.run(
                [sys.executable, "indexer.py"],
                capture_output=True, text=True, encoding="utf-8",
            )
        if result.returncode == 0:
            st.success("Index mis à jour.")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error(f"Erreur :\n{result.stderr[-500:]}")

    if st.button("Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="sb-footer">
        Source &nbsp;&nbsp;&nbsp; BDPM — ANSM<br>
        LLM &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GPT-OSS-120B (Groq)<br>
        Reranking &nbsp; Cross-Encoder mMiniLM<br>
        Vecteurs &nbsp;&nbsp; multilingual-MiniLM-L12
    </div>
    """, unsafe_allow_html=True)


# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">Assistant Médicaments</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Système d\'aide à la décision — '
    'Base de Données Publique des Médicaments (BDPM) — ANSM</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="app-divider">', unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_historique():
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]


def afficher_sources(chunks: list, show_scores: bool):
    label = f"Sources consultées — {len(chunks)} extrait{'s' if len(chunks) > 1 else ''}"
    with st.expander(label, expanded=False):
        for i, chunk in enumerate(chunks, 1):
            nom     = chunk["metadata"].get("denomination", "Inconnu")
            type_c  = chunk["metadata"].get("type_chunk", "—")
            s_init  = chunk["score"]
            s_rr    = chunk.get("score_rerank")

            scores_html = ""
            if show_scores:
                if s_rr is not None:
                    scores_html = (
                        f'<div class="source-card-scores">'
                        f'Similarité : {s_init:.3f} &nbsp;|&nbsp; Rerank : {s_rr:.3f}'
                        f'</div>'
                    )
                else:
                    scores_html = (
                        f'<div class="source-card-scores">Similarité : {s_init:.3f}</div>'
                    )

            st.markdown(f"""
            <div class="source-card">
                <div class="source-card-name">{i}. {nom}</div>
                <span class="source-card-badge">{type_c}</span>
                {scores_html}
                <div class="source-card-body">{chunk["contenu"]}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Écran d'accueil ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-heading">Bienvenue</div>
        <div class="welcome-desc">
            Posez vos questions sur les médicaments de la BDPM : effets indésirables,
            posologie, contre-indications, interactions, grossesse. Le système s'appuie
            sur les notices officielles de l'ANSM et vous indique systématiquement
            ses sources.
        </div>
        <div class="welcome-label">Exemples de questions</div>
        <span class="welcome-pill">Quels sont les effets secondaires de l'ibuprofène ?</span>
        <span class="welcome-pill">Puis-je prendre du Doliprane pendant la grossesse ?</span>
        <span class="welcome-pill">Quelle est la posologie de l'amoxicilline adulte ?</span>
        <span class="welcome-pill">Y a-t-il des interactions entre aspirine et ibuprofène ?</span>
    </div>
    """, unsafe_allow_html=True)


# ── Historique des messages ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("reformulee") and msg.get("question_recherche"):
            st.markdown(
                f'<div class="reformulation-notice">'
                f'Recherche optimisée : {msg["question_recherche"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        if msg.get("sources"):
            afficher_sources(msg["sources"], show_debug)


# ── Saisie utilisateur ────────────────────────────────────────────────────────
if prompt := st.chat_input("Posez votre question sur un médicament..."):

    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Recherche dans la base de connaissances..."):
            historique = get_historique()
            resultat   = agent_retriever.run(prompt, historique=historique)

        # Bandeau reformulation
        if resultat["reformulee"]:
            st.markdown(
                f'<div class="reformulation-notice">'
                f'Recherche optimisée : {resultat["question_recherche"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Hors périmètre
        if not resultat["confiance_ok"]:
            reponse = (
                "**Question hors périmètre**\n\n"
                "Cette question dépasse le périmètre des médicaments indexés. "
                "Je ne dispose pas d'informations suffisamment pertinentes pour y répondre "
                "de manière fiable.\n\n"
                "**Médicaments couverts :** Doliprane, Dafalgan, Efferalgan, Advil, Nurofen, "
                "Aspirin, Aspégic, Amoxicilline, Augmentin, Smecta, Imodium, Ventoline, "
                "Bécotide, Oméprazole, Inexium, Metformine, Glucophage.\n\n"
                "*Ces informations ne remplacent pas l'avis d'un professionnel de santé. "
                "En cas de doute, consultez votre médecin ou votre pharmacien.*"
            )
            st.markdown(reponse)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": reponse})

        else:
            chunks_pertinents = resultat["resultats"]

            with st.spinner("Génération de la réponse..."):
                reponse = agent_generateur.run(
                    prompt, chunks_pertinents, historique=historique
                )

            st.markdown(reponse)
            afficher_sources(chunks_pertinents, show_debug)

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant",
                "content": reponse,
                "sources": chunks_pertinents,
                "reformulee": resultat["reformulee"],
                "question_recherche": resultat["question_recherche"],
            })
