from typing import TypedDict, List
import base64
from langgraph.graph import StateGraph, START, END
from ollama import chat
import workspace as ws

# ==========================================
# 1. DÉFINITION DE L'ÉTAT DU GRAPHE
# ==========================================
class RAGState(TypedDict):
    query: str                  # La question de l'utilisateur
    image_paths: List[str]      # Les chemins des pages (images)
    raw_texts: List[str]        # Le texte extrait (pour l'Agent 2)
    
    # Sorties de l'Agent 1
    draft_answer: str           # La réponse brouillon (s'il a assez d'infos)
    draft_reasoning: str
    new_queries: List[str]      # Les nouvelles requêtes (s'il manque d'infos)
    
    # Sorties de l'Agent 2
    final_answer: str           # La réponse vérifiée
    is_consistent: bool         # Le statut de la vérification

# Fonction utilitaire pour encoder les images
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ==========================================
# 2. AGENT 1 : VISION ET RAISONNEMENT
# ==========================================
def agent_1_vision(state: RAGState):
    print("👀 Agent 1 (Vision) analyse les documents...")
    reasoning = ""
    # Préparation des images
    images_base64 = [encode_image(path) for path in state["image_paths"]]
    
    # system_prompt = """You are an expert document analyst. Look at the provided document pages to answer the user's query.
    # - If the documents contain enough information to answer, provide the answer enclosed in <ANSWER>...</ANSWER> tags.
    # - If the documents DO NOT contain enough information, generate 1 to 3 new specific search queries to find the missing info. Enclose them in <QUERIES>...</QUERIES> tags, separated by '|'.
    # Choose ONLY ONE of these two options."""

    system_prompt = """You are an expert pharmaceutical document analyst. Your task is to provide a detailed and synthesized answer to the user's query based ONLY on the provided document images.

        ### STRICT INSTRUCTIONS:
        You must ALWAYS start your response with a <REASONING>...</REASONING> block to explain your thought process, identify what information is present, and note any missing gaps.

        After your reasoning, you must choose EXACTLY ONE of the following two paths:

        - PATH 1: The document contains complete OR partial information.
        Provide a DETAILED, COMPREHENSIVE, and FACTUAL synthesis enclosed in <ANSWER>...</ANSWER> tags. 
        - Do not just extract a single word; explain the context found in the document.
        - If the information is partial, provide the most detailed answer possible based on what is available and mention what is lacking.
        - DO NOT add conversational filler outside the tags.

        - PATH 2: The document contains NO relevant information at all.
        Generate 1 to 3 specific search queries to find the missing info. Enclose them in <QUERIES>...</QUERIES> tags, separated by '|'.

        ### EXAMPLES:
        - Good Path 1: 
        <REASONING>The document details the clinical trial results. I found the recommended dosage and administration method, but contraindications are missing.</REASONING><ANSWER>According to the clinical trial protocol, the recommended dosage is 50mg administered orally twice a day. Note that the provided document does not specify any contraindications.</ANSWER>

        - Good Path 2: 
        <REASONING>The provided document discusses marketing strategies for the drug, but contains no medical or dosage information required to answer the user's query.</REASONING><QUERIES>recommended dosage for drug X | clinical trial results drug X</QUERIES>

        ### CRITICAL FORMATTING RULES:
        - NEVER output any text outside of the XML tags. 
        - Your response must strictly follow this structure: <REASONING>...</REASONING> followed immediately by EITHER <ANSWER>...</ANSWER> OR <QUERIES>...</QUERIES>."""

    # Appel au VLM (ex: qwen2.5-vl ou llama3.2-vision)
    response = chat(
        model=ws.model_name, # Modifiez avec votre modèle Vision
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": f"Query: {state['query']}",
                "images": images_base64 # API native Ollama pour les images
            }
        ],
        options={"temperature": 0.0}
    )
    
    content = response['message']['content']
    
    # Parsing basique de la réponse
    draft = ""
    queries = []
    if "<REASONING>" in content:
        reasoning = content.split("<REASONING>")[1].split("</REASONING>")[0].strip()
        print(f"🧠 Raisonement de l'Agent 1 : {reasoning}")
    if "<ANSWER>" in content:
        draft = content.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        #print(f"📝 Brouillon de réponse de l'Agent 1 : {draft}")
    elif "<QUERIES>" in content:
        queries_str = content.split("<QUERIES>")[1].split("</QUERIES>")[0].strip()
        queries = [q.strip() for q in queries_str.split('|') if q.strip()]
        
    return {"draft_answer": draft, "new_queries": queries, "draft_reasoning": reasoning}

# ==========================================
# 3. AGENT 2 : VÉRIFICATION DE CONSISTANCE
# ==========================================
# ==========================================
# 3. AGENT 2 : VÉRIFICATION DE CONSISTANCE
# ==========================================
def agent_2_verification(state: RAGState):
    print("🔎 Agent 2 (Vérification) contrôle les faits...")
    
    draft = state["draft_answer"]
    texts = "\n---\n".join(state["raw_texts"])
    
    system_prompt = """You are a highly skeptical and objective fact-checking AI. You are reviewing the work of another AI agent.

    You will be provided with:
    1. The Original Source Text.
    2. The Agent's Reasoning.
    3. The Agent's Draft Answer.

    ### INSTRUCTIONS:
    1. READ THE SOURCE TEXT FIRST. This is your only ground truth.
    2. Read the Agent's Reasoning, but DO NOT TRUST IT implicitly. Verify if every claim in their reasoning actually exists in the Source Text.
    3. Evaluate the Draft Answer for factual consistency.
    4. Assign an INCONSISTENCY SCORE from 0 to 10:
       - Score 0-2: Perfect match or valid logical deduction supported by the text.
       - Score 3-4: Minor phrasing differences, but core facts are safe.
       - Score 5-10: Hallucinations, flawed logic in the reasoning, or claims not found in the source.
    5. Make a final decision:
       - Score <= 4: Output PASS
       - Score >= 5: Output FAIL | <corrected concise answer based ONLY on the source>
    6. Answer only in plain text following the exact format below. DO NOT add anything else.

    ### REQUIRED FORMAT:
    SCORE: [Your Score]/10
    JUDGMENT: [PASS] OR [FAIL | <corrected answer>]"""

    user_prompt = f"""
    --- ORIGINAL SOURCE TEXT ---
    {texts}

    --- AGENT'S REASONING ---
    {state['draft_reasoning']}

    --- AGENT'S DRAFT ANSWER ---
    {state['draft_answer']}
    """

    # Appel au LLM
    response = chat(
        model=ws.model_name, 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.0}
    )
    
    content = response['message']['content']
    print(content)
    final_answer = draft
    is_consistent = False
    score = -1

    # ==========================================
    # CORRECTION ICI : Le parsing doit être DANS la fonction
    # ==========================================
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        
        # Extraction du score
        if line.startswith("SCORE:"):
            try:
                score = int(line.split("SCORE:")[1].split("/")[0].strip())
                print(f"   [Note d'incohérence accordée : {score}/10]") # Le print est bien ici
            except:
                pass
                
        # Extraction du jugement
        if line.startswith("JUDGMENT:"):
            judgment_str = line.split("JUDGMENT:")[1].strip()
            
            if "PASS" in judgment_str:
                is_consistent = True
                final_answer = draft
            elif "FAIL" in judgment_str:
                is_consistent = False
                parts = judgment_str.split("|")
                if len(parts) > 1:
                    final_answer = parts[1].strip()
                else:
                    final_answer = "Erreur : L'agent n'a pas fourni de correction."

    # Ce return doit impérativement être indenté dans la fonction
    return {"final_answer": final_answer, "is_consistent": is_consistent}
    
# ==========================================
# 4. ROUTAGE CONDITIONNEL
# ==========================================
def should_verify(state: RAGState):
    # Si l'Agent 1 a généré de nouvelles requêtes, on s'arrête là (ou on boucle vers le retriever)
    if len(state["new_queries"]) > 0:
        print("🔄 L'Agent 1 demande plus d'informations. Fin de ce cycle.")
        return "end_process"
    
    # S'il y a un brouillon, on passe à la vérification
    if state["draft_answer"]:
        return "verify_draft"
        
    return "end_process"


# ==========================================
# 5. CONSTRUCTION DU GRAPHE
# ==========================================
workflow = StateGraph(RAGState)

# Ajout des nœuds
workflow.add_node("Agent_Vision", agent_1_vision)
workflow.add_node("Agent_Verification", agent_2_verification)

# Définition des chemins
workflow.add_edge(START, "Agent_Vision")

# Ajout du routeur conditionnel après l'Agent 1
workflow.add_conditional_edges(
    "Agent_Vision",
    should_verify,
    {
        "verify_draft": "Agent_Verification",
        "end_process": END
    }
)

# Fin logique après l'Agent 2
workflow.add_edge("Agent_Verification", END)

# Compilation
pipeline = workflow.compile()

if __name__ == "__main__":
    from datasets import load_dataset
    import os
    
    print("📥 Chargement du dataset ViDoRe Pharmaceuticals...")
    # Les datasets de type BEIR/ViDoRe sont souvent divisés en 3 parties :
    # 1. queries (les questions)
    # 2. corpus (les documents/images)
    # 3. qrels (les liens entre les questions et les bons documents)
    
    # Remarque : si le dataset charge tout dans un seul objet plat sans configs,
    # il faudra adapter légèrement les clés du dictionnaire.
    queries_ds = load_dataset("vidore/vidore_v3_pharmaceuticals", "queries", split="test")
    corpus_ds = load_dataset("vidore/vidore_v3_pharmaceuticals", "corpus", split="test")
    qrels_ds = load_dataset("vidore/vidore_v3_pharmaceuticals", "qrels", split="test")

    print(f"✅ Dataset chargé : {len(queries_ds)} requêtes trouvées.")

    # ---------------------------------------------------------
    # ÉTAPE 1 : Récupérer les données via les qrels
    # ---------------------------------------------------------
    # Prenons la toute première relation (qrel) du dataset pour notre test
    test_qrel = qrels_ds[0]
    query_id = test_qrel["query_id"]
    corpus_id = test_qrel["corpus_id"]

    # Recherche de la question correspondante
    query_row = next(row for row in queries_ds if row["query_id"] == query_id)
    user_query = query_row["query"]

    # Recherche du document correspondant
    doc_row = next(row for row in corpus_ds if row["corpus_id"] == corpus_id)
    pil_image = doc_row["image"]
    
    # Certains datasets ont une colonne "text" ou "markdown", sinon on simule un texte vide.
    raw_text = doc_row.get("text", doc_row.get("markdown", "Texte extrait non disponible dans le dataset."))

    # ---------------------------------------------------------
    # ÉTAPE 2 : Préparation pour les Agents
    # ---------------------------------------------------------
    print(f"\n--- 🧪 TEST DE LA PIPELINE MULTI-AGENT ---")
    print(f"Question ciblée : {user_query}")
    print(f"Document source ID : {corpus_id}")

    # Sauvegarde temporaire de l'image (car notre Agent 1 attend un chemin de fichier)
    temp_image_path = "temp_pharmaceutical_doc.jpg"
    pil_image.save(temp_image_path)

    # Initialisation de l'état pour LangGraph
    initial_state = {
        "query": user_query,
        "image_paths": [temp_image_path],
        "raw_texts": [raw_text],
        "draft_answer": "",
        "new_queries": [],
        "final_answer": "",
        "is_consistent": False
    }

    # ---------------------------------------------------------
    # ÉTAPE 3 : Exécution de la Pipeline
    # ---------------------------------------------------------
    print("\n🚀 Lancement des agents...\n")
    import time
    start_time = time.time()
    # 'pipeline' est la variable compilée à la fin de la construction de votre graphe
    result = pipeline.invoke(initial_state)

    # ---------------------------------------------------------
    # RÉSULTATS
    # ---------------------------------------------------------
    print("\n=============================================")
    print("📊 RÉSULTAT FINAL")
    print("=============================================")
    if result.get("new_queries"):
        print("⚠️ L'Agent 1 a estimé qu'il manquait des informations. Il suggère de chercher :")
        for q in result["new_queries"]:
            print(f"  - {q}")
    else:
        print("✅ Réponse finale (après vérification) :")
        print(result["final_answer"])
        
        print(f"\nConsistant avec le texte brut ? : {'Oui (PASS)' if result['is_consistent'] else 'Non (Correction appliquée)'}")
        
        if not result['is_consistent']:
            print("\n❌ L'Agent 1 avait halluciné la réponse suivante (brouillon) :")
            print(result["draft_answer"])
    end_time = time.time()
    print(f"\n⏱️ Temps total d'exécution de la pipeline : {end_time - start_time:.2f} secondes")
    # Nettoyage de l'image temporaire
    if os.path.exists(temp_image_path):
        pass
        os.remove(temp_image_path)