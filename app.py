import os
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

# --- CONFIGURATION ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("="*80)
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please set the environment variable to your Google Gemini API key.")
    print("="*80)
    exit()

# Configure the generative model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 8192,
}
# Use the correct model name identified from your API key's list
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",  # <<< THE FINAL FIX IS HERE
    generation_config=generation_config,
)

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
CORS(app)

# --- GLOBAL VARIABLES ---
text_chunks = []
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- HELPER FUNCTIONS ---
def read_pdf(file_stream):
    """Reads a PDF file from a stream and extracts text."""
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def chunk_text(text):
    """Splits a long text into smaller, overlapping chunks."""
    global text_chunks
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + CHUNK_SIZE
        chunks.append(text[start_index:end_index])
        start_index += CHUNK_SIZE - CHUNK_OVERLAP
    text_chunks = chunks
    print(f"Text successfully split into {len(text_chunks)} chunks.")
    return True

def find_relevant_chunks(query, top_k=5):
    """Finds the most relevant text chunks based on a simple keyword match."""
    if not text_chunks:
        return []
    query_words = set(query.lower().split())
    scored_chunks = []
    for i, chunk in enumerate(text_chunks):
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append({"score": score, "content": chunk})
    
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return [chunk["content"] for chunk in scored_chunks[:top_k]]

# --- API ENDPOINTS ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, processes the text, and chunks it."""
    global text_chunks
    text_chunks = [] # Reset chunks on new upload

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        if file.filename.endswith('.pdf'):
            text = read_pdf(file)
        elif file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return jsonify({"error": "Unsupported file type"}), 400
        
        if "Error reading PDF" in text:
             return jsonify({"error": text}), 500

        chunk_text(text)
        return jsonify({
            "message": f"Successfully processed '{file.filename}'",
            "total_chunks": len(text_chunks)
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Receives a question, finds relevant context, and asks the Gemini model."""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Invalid request, 'question' is required."}), 400
    
    question = data['question']
    if not text_chunks:
        return jsonify({"error": "Please upload a document before asking a question."}), 400

    try:
        context = "\n---\n".join(find_relevant_chunks(question))
        
        prompt = f"""You are an expert teaching assistant. Your goal is to answer questions based ONLY on the provided text from a textbook. Do not use any outside knowledge. If the answer cannot be found in the provided text, you MUST say "I could not find the answer in the provided textbook content."

Here is the relevant text from the textbook:
---
{context}
---

Based on the text above, please answer the following question:
Question: {question}

Answer:"""
        
        response = model.generate_content(prompt)
        return jsonify({"answer": response.text})

    except Exception as e:
        print(f"AN UNEXPECTED ERROR OCCURRED: {e}") 
        return jsonify({"error": f"Error calling Gemini API: {e}"}), 500

if __name__ == '__main__':
    # Runs the Flask app. Debug mode should be OFF in production.
    app.run(host='0.0.0.0', port=5000, debug=True)

