import streamlit as st
import os
from dotenv import load_dotenv
from src.embedder import charger_modele, charger_index
from src.agents import RetrieverAgent, GeneratorAgent

# ── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistant Médicaments RAG",
    page_icon="💊",
    layout="centered"
)

# ── Chargement des variables d'environnement ────────────────────────────────
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("ERREUR : GROQ_API_KEY manquante dans le fichier .env")
    st.stop()

# ── Fonctions de mise en cache ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement de la base de données...")
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
        return agent_retriever, agent_generateur
    except FileNotFoundError as e:
        st.error(f"Erreur: {e}. Avez-vous lancé 'python indexer.py' ?")
        st.stop()

agent_retriever, agent_generateur = init_rag_system()

# ── Interface UI ────────────────────────────────────────────────────────────
st.title("💊 Assistant Médicaments")
st.markdown("""
Bienvenue dans l'assistant d'aide à la décision médicale basé sur la Base de Données Publique des Médicaments (BDPM).
Posez vos questions cliniques ou de prescription !
""")

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fonction pour formater l'historique pour l'Agent 2
def get_historique_pour_agent():
    # L'Agent 2 attend une liste de dictionnaires {"role": "user"|"assistant", "content": ...}
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]

# Affichage des messages existants
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Voir les sources"):
                for i, chunk in enumerate(msg["sources"], 1):
                    nom = chunk["metadata"].get("denomination", "Inconnu")
                    type_c = chunk["metadata"].get("type_chunk", "?")
                    score = chunk["score"]
                    st.write(f"**{i}. [{type_c}] {nom}** (Pertinence : {score:.1%})")
                    st.caption(chunk["contenu"][:300] + "..." if len(chunk["contenu"]) > 300 else chunk["contenu"])

# Champ de saisie utilisateur
if prompt := st.chat_input("Votre question... (ex: C'est quoi le paracétamol ?)"):
    # Affichage du message utilisateur
    st.chat_message("user").markdown(prompt)
    
    # ── Traitement de la requête ──────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Recherche et réflexion en cours..."):
            historique_agent = get_historique_pour_agent()
            
            # Agent 1 : Retriever
            resultat_retriever = agent_retriever.run(prompt, historique=historique_agent)
            
            # Affichage de la reformulation si elle a eu lieu
            if resultat_retriever["reformulee"]:
                st.caption(f"🔄 *Question reformulée : {resultat_retriever['question_recherche']}*")
            
            # Vérification de la confiance
            if not resultat_retriever["confiance_ok"]:
                reponse = "Je n'ai pas trouvé d'information directement pertinente dans ma base de données pour éclairer cette décision.\n\n[!] Cet outil d'aide à la décision est fourni à titre informatif. Il ne se substitue en aucun cas à l'évaluation clinique et à la responsabilité du prescripteur."
                st.markdown(reponse)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": reponse})
            else:
                chunks_pertinents = resultat_retriever["resultats"]
                
                # Agent 2 : Generator
                reponse = agent_generateur.run(prompt, chunks_pertinents, historique=historique_agent)
                st.markdown(reponse)
                
                # Affichage des sources dans un expander
                with st.expander("Voir les sources"):
                    for i, chunk in enumerate(chunks_pertinents, 1):
                        nom = chunk["metadata"].get("denomination", "Inconnu")
                        type_c = chunk["metadata"].get("type_chunk", "?")
                        score = chunk["score"]
                        st.write(f"**{i}. [{type_c}] {nom}** (Pertinence : {score:.1%})")
                        st.caption(chunk["contenu"][:300] + "..." if len(chunk["contenu"]) > 300 else chunk["contenu"])
                
                # Sauvegarde dans l'historique
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": reponse, 
                    "sources": chunks_pertinents
                })
