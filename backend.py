import os
import torch
from pypdf import PdfReader
from transformers import BertTokenizerFast
from compact_model_loader import load_compact_model


# ============================================================
# MODEL PATH
# ============================================================

# All model files are in the same folder as backend.py
MODEL_PATH = "."


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading compact tokenizer...")

tokenizer = BertTokenizerFast(
    tokenizer_file=os.path.join(MODEL_PATH, "tokenizer.json")
)

print("Tokenizer loaded successfully.")


# ============================================================
# LOAD COMPACT INT8 QA MODEL
# ============================================================

print("Loading compact INT8 QA model...")

model = load_compact_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Using device:", device)


# ============================================================
# READ PDF
# ============================================================

def read_pdf(pdf_path):
    """
    Extract text from every page of the PDF.
    Returns a list containing the text of each page.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            text = text.strip()

            if text:
                pages.append(text)

    return pages


# ============================================================
# CREATE DOCUMENT CHUNKS
# ============================================================

def create_chunks(pages, words_per_chunk=250):
    """
    Split each PDF page into smaller chunks.

    Each chunk stores:
    - text
    - original page number
    """

    chunks = []

    for page_number, page_text in enumerate(pages):

        words = page_text.split()

        for i in range(0, len(words), words_per_chunk):

            chunk_words = words[i:i + words_per_chunk]

            if chunk_words:

                chunks.append({
                    "text": " ".join(chunk_words),
                    "page": page_number + 1
                })

    return chunks


# ============================================================
# ANSWER QUESTION FROM ONE CHUNK
# ============================================================

def answer_from_chunk(question, context):
    """
    Run the trained BERT QA model on one document chunk.
    """

    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        padding=False,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    # Character offsets are needed to convert
    # token positions back into actual text.
    offset_mapping = inputs["offset_mapping"][0]

    # Identify which tokens belong to the context.
    sequence_ids = inputs.sequence_ids(0)

    # Remove offset_mapping before sending inputs to the model.
    model_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key != "offset_mapping"
    }

    # Run inference
    with torch.no_grad():

        outputs = model(**model_inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Find tokens belonging to the document context.
    context_positions = [
        i
        for i, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
    ]

    if not context_positions:
        return "", 0.0

    # Get start and end scores only from context tokens.
    start_scores = start_logits[context_positions]
    end_scores = end_logits[context_positions]

    # Find highest scoring start and end positions.
    best_start_index = torch.argmax(start_scores).item()
    best_end_index = torch.argmax(end_scores).item()

    start_position = context_positions[best_start_index]
    end_position = context_positions[best_end_index]

    # Invalid span
    if end_position < start_position:
        return "", 0.0

    # Limit maximum answer length.
    if end_position - start_position > 30:
        end_position = start_position + 30

    # Convert token positions to character positions.
    start_char = offset_mapping[start_position][0].item()
    end_char = offset_mapping[end_position][1].item()

    if start_char >= end_char:
        return "", 0.0

    # Extract answer from the original context.
    answer = context[start_char:end_char].strip()

    # Calculate confidence.
    start_probability = torch.softmax(
        start_logits[context_positions],
        dim=0
    )[best_start_index].item()

    end_probability = torch.softmax(
        end_logits[context_positions],
        dim=0
    )[best_end_index].item()

    confidence = (
        start_probability + end_probability
    ) / 2

    return answer, confidence


# ============================================================
# PROCESS COMPLETE DOCUMENT
# ============================================================

def process_document(pdf_path, question):
    """
    Read the PDF, split it into chunks,
    run QA on every chunk,
    and return the best answer.
    """

    print("\nReading PDF...")

    pages = read_pdf(pdf_path)

    if not pages:
        return "", 0.0, None

    print("Pages extracted:", len(pages))

    # Create chunks
    chunks = create_chunks(pages)

    print("Document chunks:", len(chunks))

    best_answer = ""
    best_confidence = 0.0
    best_page = None

    # Process every chunk
    for i, chunk_data in enumerate(chunks):

        answer, confidence = answer_from_chunk(
            question,
            chunk_data["text"]
        )

        # Keep the answer with the highest confidence.
        if answer and confidence > best_confidence:

            best_answer = answer
            best_confidence = confidence
            best_page = chunk_data["page"]

        print(
            f"Processed chunk "
            f"{i + 1}/{len(chunks)}"
        )

    return (
        best_answer,
        best_confidence,
        best_page
    )

@app.route("/ask", methods=["POST"])
def ask_question():

    if "pdf" not in request.files:
        return jsonify({
            "error": "No PDF file uploaded."
        }), 400

    pdf_file = request.files["pdf"]

    question = request.form.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "No question provided."
        }), 400

    if pdf_file.filename == "":
        return jsonify({
            "error": "No PDF selected."
        }), 400

    # Save uploaded PDF temporarily
    pdf_path = "uploaded_document.pdf"
    pdf_file.save(pdf_path)

    try:

        answer, confidence, page = process_document(
            pdf_path,
            question
        )

        return jsonify({
            "answer": answer if answer else "No relevant answer found.",
            "confidence": round(confidence, 4),
            "page": page
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        # Delete temporary PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)@app.route("/ask", methods=["POST"])
def ask_question():

    if "pdf" not in request.files:
        return jsonify({
            "error": "No PDF file uploaded."
        }), 400

    pdf_file = request.files["pdf"]

    question = request.form.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "No question provided."
        }), 400

    if pdf_file.filename == "":
        return jsonify({
            "error": "No PDF selected."
        }), 400

    # Save uploaded PDF temporarily
    pdf_path = "uploaded_document.pdf"
    pdf_file.save(pdf_path)

    try:

        answer, confidence, page = process_document(
            pdf_path,
            question
        )

        return jsonify({
            "answer": answer if answer else "No relevant answer found.",
            "confidence": round(confidence, 4),
            "page": page
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        # Delete temporary PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Test PDF
    PDF_PATH = "test_legal_document.pdf"

    # Ask user for a question
    question = input(
        "\nEnter your question: "
    )

    # Check whether PDF exists
    if not os.path.exists(PDF_PATH):

        print("\nERROR:")
        print("PDF file not found.")
        print(
            "Make sure test_legal_document.pdf "
            "is in the same folder as backend.py."
        )

    else:

        # Process document
        answer, confidence, page = process_document(
            PDF_PATH,
            question
        )

        # Display results
        print("\n" + "=" * 60)

        print("QUESTION:")
        print(question)

        print("\nANSWER:")

        if answer:
            print(answer)
        else:
            print("No relevant answer found.")

        print("\nCONFIDENCE:")
        print(f"{confidence:.4f}")

        if page:
            print("\nPAGE:")
            print(page)

        print("=" * 60)
