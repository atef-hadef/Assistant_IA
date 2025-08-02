import streamlit as st
from assistant import extract_text_from_pdf, extract_text_from_csv, split_into_chunks, create_vectorstore, ask_with_groq
import datetime
from typing import List, Dict

# Configuration de la page
st.set_page_config(
    page_title="Assistant IA", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

# CSS personnalisé pour une interface moderne
st.markdown("""
<style>
    /* Variables CSS */
    :root {
        --primary-color: #0066cc;
        --secondary-color: #f0f8ff;
        --accent-color: #4CAF50;
        --text-color: #333333;
        --border-radius: 12px;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Styling général */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header personnalisé */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
        color: white;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
        border: 1px solid #e0e0e0;
    }
    
    /* Messages de chat */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: var(--shadow);
    }
    
    .bot-message {
        background: #f8f9fa;
        color: var(--text-color);
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border-left: 4px solid var(--accent-color);
        box-shadow: var(--shadow);
    }
    
    .message-time {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
        border: 1px solid #e0e0e0;
    }
    
    /* Document info */
    .doc-info {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-ready {
        background: #d4edda;
        color: #155724;
    }
    
    .status-processing {
        background: #fff3cd;
        color: #856404;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* Animation pour les messages */
    @keyframes slideIn {
        from {
            transform: translateX(30px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .slide-in {
        animation: slideIn 0.3s ease-out;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.7rem 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Hide default streamlit elements */
    .stDeployButton {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = None

def add_message_to_history(role: str, content: str, sources: List = None):
    """Ajouter un message à l'historique de chat"""
    timestamp = datetime.datetime.now().strftime("%H:%M")
    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "sources": sources or []
    }
    st.session_state.chat_history.append(message)

def display_chat_history():
    """Afficher l'historique des conversations"""
    if st.session_state.chat_history:
        st.markdown("### 💬 Historique des conversations")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message slide-in">
                    <strong>Vous:</strong> {message["content"]}
                    <div class="message-time">🕒 {message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message slide-in">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                    <div class="message-time">🕒 {message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Afficher les sources si disponibles
                if message.get("sources"):
                    with st.expander(f"🔍 Sources utilisées ({len(message['sources'])} documents)", expanded=False):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**📄 Source {i+1}:**")
                            st.text(source[:300] + "..." if len(source) > 300 else source)
                            st.markdown("---")

def clear_chat_history():
    """Vider l'historique des conversations"""
    st.session_state.chat_history = []
    st.success("✅ Historique effacé!")

# Header principal
st.markdown("""
<div class="header-container">
    <div class="header-title">🤖 Assistant IA </div>
    <div class="header-subtitle">Analyse intelligente de vos documents</div>
</div>
""", unsafe_allow_html=True)

# Layout principal avec colonnes
col1, col2 = st.columns([1, 2])

# Sidebar pour le chargement de documents
with col1:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📂 Gestion des documents")
    
    uploaded_file = st.file_uploader(
        "Chargez votre document",
        type=["pdf", "csv"],
        help="Formats supportés: PDF, CSV"
    )
    
    if uploaded_file:
        if not st.session_state.document_loaded or st.session_state.document_name != uploaded_file.name:
            with st.spinner("📖 Traitement du document..."):
                try:
                    # Extraction du texte
                    if uploaded_file.name.endswith(".pdf"):
                        text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.name.endswith(".csv"):
                        text = extract_text_from_csv(uploaded_file)
                    else:
                        st.error("❌ Format non supporté.")
                        st.stop()
                    
                    # Création du vectorstore
                    chunks = split_into_chunks(text)
                    st.session_state.vectorstore = create_vectorstore(chunks)
                    st.session_state.document_loaded = True
                    st.session_state.document_name = uploaded_file.name
                    
                    st.success("✅ Document traité avec succès!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement: {str(e)}")
        
        # Affichage des informations du document
        if st.session_state.document_loaded:
            st.markdown(f"""
            <div class="doc-info">
                <strong>📄 Document chargé:</strong><br>
                {st.session_state.document_name}
                <br><br>
                <span class="status-badge status-ready">✅ Prêt pour les questions</span>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("📤 Veuillez charger un document pour commencer")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section de contrôle de l'historique
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🗂️ Historique")
    
    col_export, col_clear = st.columns(2)
    
    with col_export:
        if st.session_state.chat_history:
            # Bouton d'export (fonctionnalité future)
            st.button("📥 Exporter", disabled=True, help="Fonctionnalité à venir")
    
    with col_clear:
        if st.session_state.chat_history:
            if st.button("🗑️ Effacer", help="Effacer tout l'historique"):
                clear_chat_history()
                st.rerun()
    
    # Statistiques
    if st.session_state.chat_history:
        total_messages = len(st.session_state.chat_history)
        user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        st.markdown(f"""
        **📊 Statistiques:**
        - Messages total: {total_messages}
        - Questions posées: {user_messages}
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Zone principale de chat
with col2:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Affichage de l'historique
    display_chat_history()
    
    # Zone de saisie pour nouvelles questions
    if st.session_state.document_loaded:
        st.markdown("### ✍️ Posez votre question")
        
        # Formulaire pour éviter le rechargement
        with st.form(key="question_form", clear_on_submit=True):
            query = st.text_input(
                "💭 Votre question:",
                placeholder="Tapez votre question ici...",
                label_visibility="collapsed"
            )
            
            col_submit, col_examples = st.columns([1, 2])
            
            with col_submit:
                submitted = st.form_submit_button("🚀 Envoyer", use_container_width=True)
            
            with col_examples:
                st.markdown("*💡 Suggestions: Résume le document, Quels sont les points clés?, Explique...*")
        
        # Traitement de la question
        if submitted and query.strip():
            # Ajouter la question à l'historique
            add_message_to_history("user", query)
            
            with st.spinner("🤔 Réflexion en cours..."):
                try:
                    # Recherche dans le vectorstore
                    docs = st.session_state.vectorstore.similarity_search(query, k=3)
                    
                    # Génération de la réponse
                    response = ask_with_groq(query, docs)
                    
                    # Extraction des sources
                    sources = [doc.page_content for doc in docs]
                    
                    # Ajouter la réponse à l'historique
                    add_message_to_history("assistant", response, sources)
                    
                    # Recharger pour afficher la nouvelle conversation
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération de la réponse: {str(e)}")
                    add_message_to_history("assistant", f"Désolé, une erreur s'est produite: {str(e)}")
        
        elif submitted and not query.strip():
            st.warning("⚠️ Veuillez saisir une question avant d'envoyer.")
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h3>🤖 Assistant IA prêt à vous aider!</h3>
            <p>Chargez un document dans la sidebar pour commencer à poser vos questions.</p>
            <br>
            <p><strong>Fonctionnalités:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>📄 Support PDF et CSV</li>
                <li>💬 Historique des conversations</li>
                <li>🔍 Sources détaillées</li>
                <li>⚡ Réponses rapides avec GROQ</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; border-top: 1px solid #e0e0e0; margin-top: 2rem;">
    <p>🚀 Propulsé par <strong>Streamlit</strong> | 
    Développé avec Atef Ben Hadef pour une expérience utilisateur optimale</p>
</div>
""", unsafe_allow_html=True)