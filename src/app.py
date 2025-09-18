
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import PyPDF2
import pandas as pd
import io
import hashlib
from typing import List, Dict
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit page config
st.set_page_config(
    page_title="Chat with GPT + RAG",
    page_icon="📚",
    layout="wide"
)

# Document processing functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du PDF: {str(e)}")
        return ""

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        return str(txt_file.read(), "utf-8")
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du TXT: {str(e)}")
        return ""

def extract_text_from_csv(csv_file):
    """Extract text from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        return df.to_string()
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du CSV: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def create_embeddings(chunks: List[str]) -> List[Dict]:
    """Create embeddings for text chunks using OpenAI"""
    try:
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            embedded_chunks.append({
                "text": chunk,
                "embedding": response.data[0].embedding,
                "chunk_id": i
            })
        return embedded_chunks
    except Exception as e:
        st.error(f"Erreur lors de la création des embeddings: {str(e)}")
        return []

def find_relevant_chunks(query: str, embedded_chunks: List[Dict], top_k: int = 3) -> List[str]:
    """Find most relevant chunks for a query"""
    try:
        # Get query embedding
        query_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = query_response.data[0].embedding
        
        # Calculate cosine similarity
        import numpy as np
        
        similarities = []
        for chunk in embedded_chunks:
            similarity = np.dot(query_embedding, chunk["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk["embedding"])
            )
            similarities.append((similarity, chunk["text"]))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk_text for _, chunk_text in similarities[:top_k]]
        
    except Exception as e:
        st.error(f"Erreur lors de la recherche de chunks pertinents: {str(e)}")
        return []

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks = []

# Main title and description
st.title("📚 Chat with GPT + RAG")
st.write("Uploadez vos documents et chattez avec GPT en utilisant le contenu de vos fichiers !")

# Create two columns: chat and document upload
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📁 Gestion des Documents")
    
    # File uploader with drag & drop
    uploaded_files = st.file_uploader(
        "Glissez-déposez vos documents ici",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'csv'],
        help="Formats supportés: PDF, DOCX, TXT, CSV"
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
            uploaded_file.seek(0)  # Reset file pointer
            
            if file_hash not in st.session_state.documents:
                with st.spinner(f"Traitement de {uploaded_file.name}..."):
                    # Extract text based on file type
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_type == 'pdf':
                        text = extract_text_from_pdf(uploaded_file)
                    elif file_type == 'docx':
                        text = extract_text_from_docx(uploaded_file)
                    elif file_type == 'txt':
                        text = extract_text_from_txt(uploaded_file)
                    elif file_type == 'csv':
                        text = extract_text_from_csv(uploaded_file)
                    else:
                        st.error(f"Type de fichier non supporté: {file_type}")
                        continue
                    
                    if text.strip():
                        # Store document info
                        st.session_state.documents[file_hash] = {
                            "name": uploaded_file.name,
                            "text": text,
                            "type": file_type
                        }
                        
                        # Create chunks and embeddings
                        chunks = chunk_text(text)
                        if chunks:
                            embeddings = create_embeddings(chunks)
                            st.session_state.embedded_chunks.extend(embeddings)
                            st.success(f"✅ {uploaded_file.name} traité avec succès!")
                        else:
                            st.warning(f"⚠️ Aucun texte extrait de {uploaded_file.name}")
                    else:
                        st.error(f"❌ Impossible d'extraire le texte de {uploaded_file.name}")
    
    # Display loaded documents
    if st.session_state.documents:
        st.subheader("📄 Documents chargés")
        for doc_hash, doc_info in st.session_state.documents.items():
            col_name, col_remove = st.columns([3, 1])
            with col_name:
                st.write(f"• {doc_info['name']}")
            with col_remove:
                if st.button("🗑️", key=f"remove_{doc_hash}"):
                    # Remove document and its embeddings
                    del st.session_state.documents[doc_hash]
                    # Note: In a production app, you'd want to properly remove embeddings too
                    st.rerun()
        
        # Clear all documents button
        if st.button("🗑️ Supprimer tous les documents"):
            st.session_state.documents = {}
            st.session_state.embedded_chunks = []
            st.rerun()
    
    # RAG settings
    st.subheader("⚙️ Paramètres RAG")
    use_rag = st.checkbox("Utiliser les documents uploadés", value=True)
    if use_rag:
        top_k_chunks = st.slider("Nombre de chunks à utiliser", 1, 10, 3)

with col1:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Show sources if available
                if "sources" in message:
                    with st.expander("📑 Sources utilisées"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**Source {i}:**")
                            st.write(source[:200] + "..." if len(source) > 200 else source)

    # Chat input
    if prompt := st.chat_input("Votre message..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("Réflexion..."):
                    # Prepare conversation history for API
                    api_messages = []
                    
                    # Add system message with RAG context if enabled
                    if use_rag and st.session_state.embedded_chunks:
                        relevant_chunks = find_relevant_chunks(prompt, st.session_state.embedded_chunks, top_k_chunks)
                        if relevant_chunks:
                            context = "\n\n".join(relevant_chunks)
                            system_message = f"""Vous êtes un assistant IA qui répond aux questions en utilisant le contexte fourni des documents uploadés. 
                            
Contexte des documents:
{context}

Répondez en vous basant sur ce contexte. Si la réponse n'est pas dans le contexte, indiquez-le clairement."""
                            api_messages.append({"role": "system", "content": system_message})
                    
                    # Add conversation history (last 10 messages to avoid token limits)
                    recent_messages = st.session_state.messages[-10:]
                    for message in recent_messages:
                        api_messages.append({
                            "role": message["role"], 
                            "content": message["content"]
                        })
                    
                    # Call OpenAI API
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=api_messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    
                    assistant_reply = response.choices[0].message.content
                    message_placeholder.markdown(assistant_reply)
                    
                    # Add assistant response to chat history with sources if RAG was used
                    message_data = {
                        "role": "assistant",
                        "content": assistant_reply
                    }
                    
                    if use_rag and st.session_state.embedded_chunks:
                        relevant_chunks = find_relevant_chunks(prompt, st.session_state.embedded_chunks, top_k_chunks)
                        if relevant_chunks:
                            message_data["sources"] = relevant_chunks
                    
                    st.session_state.messages.append(message_data)
                    
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
                st.error("Veuillez vérifier votre clé API OpenAI et votre connexion internet.")

# Sidebar with controls
with st.sidebar:
    st.subheader("🎛️ Contrôles du Chat")
    
    # Clear chat button
    if st.button("🗑️ Effacer le Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Model selection
    model_option = st.selectbox(
        "Sélectionner le Modèle:",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature (créativité):",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Des valeurs plus élevées rendent la sortie plus créative mais moins focalisée"
    )
    
    st.subheader("📊 Statistiques")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Documents", len(st.session_state.documents))
    st.metric("Chunks embeddings", len(st.session_state.embedded_chunks))
    
    # Token usage estimate
    total_chars = sum(len(m["content"]) for m in st.session_state.messages)
    estimated_tokens = total_chars // 4
    st.metric("Tokens estimés", f"{estimated_tokens:,}")

# Tips section
if st.expander("💡 Conseils pour de meilleures conversations"):
    st.markdown("""
    **Avec RAG:**
    - **Uploadez des documents pertinents**: Plus vos documents sont liés à vos questions, meilleures seront les réponses
    - **Posez des questions spécifiques**: "Que dit le document sur X?" plutôt que des questions générales
    - **Vérifiez les sources**: Consultez les sources utilisées dans chaque réponse
    
    **Questions générales:**
    - **Soyez spécifique**: Fournissez un contexte et des détails clairs
    - **Posez des questions de suivi**: Construisez sur les réponses précédentes
    - **Utilisez des exemples**: Montrez ce que vous recherchez
    """)

# Requirements note
st.markdown("---")
st.markdown("**📦 Dépendances requises:** `pip install streamlit openai python-dotenv PyPDF2 python-docx pandas numpy`")