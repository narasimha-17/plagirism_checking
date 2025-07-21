from flask import Flask, request, render_template, redirect, url_for, flash
import os
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pypdf import PdfReader
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey' # Needed for flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global model variable to avoid reloading it on every request
model = None

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def split_into_sentences(text):
    """
    Splits a given text into a list of sentences.
    A basic regex is used, which might not be perfect for all edge cases
    (e.g., abbreviations, complex punctuation).
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n', text)
    return [s.strip() for s in sentences if s.strip()]

def read_pdf(file_path):
    """
    Reads text content from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content from the PDF, or an empty string if an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return ""
    
    text_content = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text: # Ensure text was actually extracted from the page
                    text_content += extracted_text + "\n"
        print(f"Successfully read text from {file_path}")
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        text_content = ""
    return text_content

def check_plagiarism(suspicious_text, source_text, similarity_threshold=0.75):
    """
    Performs a plagiarism check between a suspicious text and a source text
    using semantic similarity (Sentence Embeddings and Cosine Similarity).
    """
    global model # Use the global model instance

    if model is None:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer model loaded successfully for Flask app.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            return None

    suspicious_sentences = split_into_sentences(suspicious_text)
    source_sentences = split_into_sentences(source_text)

    if not suspicious_sentences or not source_sentences:
        return {
            'percentage': 0.0,
            'matched_sentences': [],
            'total_suspicious_sentences': 0,
            'plagiarized_sentence_count': 0
        }

    print(f"Processing {len(suspicious_sentences)} suspicious sentences and {len(source_sentences)} source sentences...")

    suspicious_embeddings = model.encode(suspicious_sentences, convert_to_tensor=True)
    source_embeddings = model.encode(source_sentences, convert_to_tensor=True)

    plagiarized_sentence_count = 0
    matched_sentences = []

    for i, susp_sent in enumerate(suspicious_sentences):
        is_plagiarized = False
        best_match = {'sentence': '', 'similarity': 0.0}

        similarities = util.cos_sim(suspicious_embeddings[i], source_embeddings)[0]

        for j, src_sent in enumerate(source_sentences):
            similarity = similarities[j].item()

            if similarity > similarity_threshold:
                is_plagiarized = True
                if similarity > best_match['similarity']:
                    best_match = {'sentence': src_sent, 'similarity': similarity}

        if is_plagiarized:
            plagiarized_sentence_count += 1
            matched_sentences.append({
                'suspicious': susp_sent,
                'source_match': best_match['sentence'],
                'similarity': best_match['similarity']
            })

    total_suspicious_sentences = len(suspicious_sentences)
    plagiarism_percentage = (plagiarized_sentence_count / total_suspicious_sentences) * 100 if total_suspicious_sentences > 0 else 0.0

    return {
        'percentage': round(plagiarism_percentage, 2),
        'matched_sentences': matched_sentences,
        'total_suspicious_sentences': total_suspicious_sentences,
        'plagiarized_sentence_count': plagiarized_sentence_count
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles file uploads and displays results."""
    results = None
    suspicious_doc_text = ""
    source_doc_text = ""

    if request.method == 'POST':
        # Check if suspicious file is present
        if 'suspicious_file' not in request.files:
            flash('No suspicious file part')
            return redirect(request.url)
        suspicious_file = request.files['suspicious_file']

        # Check if source file is present
        if 'source_file' not in request.files:
            flash('No source file part')
            return redirect(request.url)
        source_file = request.files['source_file']

        if suspicious_file.filename == '' or source_file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if suspicious_file and allowed_file(suspicious_file.filename) and \
           source_file and allowed_file(source_file.filename):
            suspicious_filename = secure_filename(suspicious_file.filename)
            source_filename = secure_filename(source_file.filename)

            suspicious_filepath = os.path.join(app.config['UPLOAD_FOLDER'], suspicious_filename)
            source_filepath = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)

            suspicious_file.save(suspicious_filepath)
            source_file.save(source_filepath)

            # Read text from uploaded PDFs
            suspicious_doc_text = read_pdf(suspicious_filepath)
            source_doc_text = read_pdf(source_filepath)

            if suspicious_doc_text and source_doc_text:
                # Perform plagiarism check
                results = check_plagiarism(suspicious_doc_text, source_doc_text)
                if results is None:
                    flash("Error: ML model could not be loaded. Please check server logs.")
            else:
                flash("Error: Could not extract text from one or both PDF files.")

            # Clean up uploaded files after processing
            os.remove(suspicious_filepath)
            os.remove(source_filepath)
        else:
            flash('Allowed file types are PDF')
    
    # Render the template with results (if any)
    return render_template('index.html', results=results, 
                           suspicious_text_original=suspicious_doc_text,
                           source_text_original=source_doc_text)


if __name__ == '__main__':
    app.run(debug=True)
