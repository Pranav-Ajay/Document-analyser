import os
import tempfile

import torch
from pypdf import PdfReader
from transformers import BertTokenizerFast

from flask import Flask, request, jsonify
from flask_cors import CORS

from compact_model_loader import load_compact_model


# ============================================================
# CONFIGURATION
# ============================================================

# All model files are in the same folder as this backend.py
MODEL_PATH = "."


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Allow the HTML frontend to communicate with this Python server
CORS(app)


# Limit uploaded PDF size to 20 MB
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading compact tokenizer...")

try:

    tokenizer = BertTokenizerFast(
        tokenizer_file=os.path.join(
            MODEL_PATH,
            "tokenizer.json"
        )
    )

    print("Tokenizer loaded successfully.")

except Exception as e:

    print("ERROR: Could not load tokenizer.")
    print(e)
    raise


# ============================================================
# LOAD COMPACT INT8 QA MODEL
# ============================================================

print("Loading compact INT8 QA model...")

try:

    model = load_compact_model(MODEL_PATH)

    print("Model loaded successfully.")

except Exception as e:

    print("ERROR: Could not load compact QA model.")
    print(e)
    raise


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


try:

    model.to(device)

except Exception as e:

    print("WARNING: Could not move model to selected device.")
    print(e)
    print("Using the model in its current device configuration.")


model.eval()


# ============================================================
# READ PDF
# ============================================================

def read_pdf(pdf_path):
    """
    Extract text from every page of a PDF.

    Returns:
        List of page texts.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages):

        try:

            text = page.extract_text()

        except Exception as e:

            print(
                f"Warning: Could not extract "
                f"text from page {page_number + 1}: {e}"
            )

            continue

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
    Split PDF text into smaller chunks.

    Each chunk contains:
        text
        page number
    """

    chunks = []

    for page_number, page_text in enumerate(pages):

        words = page_text.split()

        for i in range(
            0,
            len(words),
            words_per_chunk
        ):

            chunk_words = words[
                i:i + words_per_chunk
            ]

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

    Returns:
        answer
        confidence
    """

    # --------------------------------------------------------
    # Tokenize question + document context
    # --------------------------------------------------------

    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        padding=False,
        return_tensors="pt",
        return_offsets_mapping=True
    )


    # --------------------------------------------------------
    # Character offsets
    # --------------------------------------------------------

    offset_mapping = inputs[
        "offset_mapping"
    ][0]


    # --------------------------------------------------------
    # Identify question/context tokens
    # --------------------------------------------------------

    sequence_ids = inputs.sequence_ids(0)


    # --------------------------------------------------------
    # Prepare model inputs
    # --------------------------------------------------------

    model_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key != "offset_mapping"
    }


    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **model_inputs
        )


    # --------------------------------------------------------
    # Get start/end logits
    # --------------------------------------------------------

    start_logits = outputs.start_logits[0]

    end_logits = outputs.end_logits[0]


    # --------------------------------------------------------
    # Find context token positions
    # --------------------------------------------------------

    context_positions = [
        i
        for i, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
    ]


    if not context_positions:

        return "", 0.0


    # --------------------------------------------------------
    # Get scores only for context
    # --------------------------------------------------------

    start_scores = start_logits[
        context_positions
    ]

    end_scores = end_logits[
        context_positions
    ]


    # --------------------------------------------------------
    # Find highest scoring start/end tokens
    # --------------------------------------------------------

    best_start_index = torch.argmax(
        start_scores
    ).item()

    best_end_index = torch.argmax(
        end_scores
    ).item()


    start_position = context_positions[
        best_start_index
    ]

    end_position = context_positions[
        best_end_index
    ]


    # --------------------------------------------------------
    # Check answer span
    # --------------------------------------------------------

    if end_position < start_position:

        return "", 0.0


    # Prevent excessively long answers
    if end_position - start_position > 30:

        end_position = start_position + 30


    # Make sure end position is valid
    if end_position >= len(offset_mapping):

        return "", 0.0


    # --------------------------------------------------------
    # Convert token positions to character positions
    # --------------------------------------------------------

    start_char = offset_mapping[
        start_position
    ][0].item()

    end_char = offset_mapping[
        end_position
    ][1].item()


    if start_char >= end_char:

        return "", 0.0


    # --------------------------------------------------------
    # Extract answer from context
    # --------------------------------------------------------

    answer = context[
        start_char:end_char
    ].strip()


    if not answer:

        return "", 0.0


    # --------------------------------------------------------
    # Calculate confidence
    # --------------------------------------------------------

    start_probability = torch.softmax(
        start_logits[context_positions],
        dim=0
    )[best_start_index].item()


    end_probability = torch.softmax(
        end_logits[context_positions],
        dim=0
    )[best_end_index].item()


    confidence = (
        start_probability +
        end_probability
    ) / 2


    return answer, confidence


# ============================================================
# PROCESS COMPLETE DOCUMENT
# ============================================================

def process_document(pdf_path, question):
    """
    Read PDF, split into chunks, run BERT QA
    on every chunk, and return the best answer.
    """

    print("\nReading PDF...")

    pages = read_pdf(pdf_path)


    if not pages:

        return "", 0.0, None


    print(
        "Pages extracted:",
        len(pages)
    )


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        pages,
        words_per_chunk=250
    )


    print(
        "Document chunks:",
        len(chunks)
    )


    if not chunks:

        return "", 0.0, None


    # --------------------------------------------------------
    # Store best answer
    # --------------------------------------------------------

    best_answer = ""

    best_confidence = 0.0

    best_page = None


    # --------------------------------------------------------
    # Process every chunk
    # --------------------------------------------------------

    for i, chunk_data in enumerate(chunks):

        print(
            f"Processing chunk "
            f"{i + 1}/{len(chunks)}..."
        )


        try:

            answer, confidence = answer_from_chunk(
                question,
                chunk_data["text"]
            )

        except Exception as e:

            print(
                f"Error processing chunk "
                f"{i + 1}: {e}"
            )

            continue


        # ----------------------------------------------------
        # Keep highest-confidence answer
        # ----------------------------------------------------

        if (
            answer
            and
            confidence > best_confidence
        ):

            best_answer = answer

            best_confidence = confidence

            best_page = chunk_data["page"]


    return (
        best_answer,
        best_confidence,
        best_page
    )


# ============================================================
# API: ASK QUESTION
# ============================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask_question():

    print("\n" + "=" * 60)

    print("Received request from frontend.")


    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if "pdf" not in request.files:

        return jsonify({
            "error": "No PDF file uploaded."
        }), 400


    pdf_file = request.files["pdf"]


    # --------------------------------------------------------
    # Check question
    # --------------------------------------------------------

    question = request.form.get(
        "question",
        ""
    ).strip()


    if not question:

        return jsonify({
            "error": "No question provided."
        }), 400


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if not pdf_file.filename:

        return jsonify({
            "error": "No PDF selected."
        }), 400


    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

    filename = pdf_file.filename.lower()


    if not filename.endswith(".pdf"):

        return jsonify({
            "error": "Only PDF files are supported."
        }), 400


    # --------------------------------------------------------
    # Create temporary PDF
    # --------------------------------------------------------

    temporary_file = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:

            pdf_file.save(
                temp.name
            )

            temporary_file = temp.name


        print(
            "PDF received:",
            pdf_file.filename
        )

        print(
            "Question:",
            question
        )


        # ----------------------------------------------------
        # Run document QA
        # ----------------------------------------------------

        answer, confidence, page = process_document(
            temporary_file,
            question
        )


        # ----------------------------------------------------
        # Prepare response
        # ----------------------------------------------------

        if not answer:

            answer = "No relevant answer found."


        response_data = {
            "answer": answer,
            "confidence": round(
                confidence,
                4
            ),
            "page": page
        }


        print("\nAnswer:", answer)

        print(
            "Confidence:",
            round(confidence, 4)
        )

        print(
            "Page:",
            page
        )


        print("=" * 60)


        # ----------------------------------------------------
        # Send JSON back to HTML
        # ----------------------------------------------------

        return jsonify(response_data)


    except Exception as e:

        print("\nERROR while processing document:")

        print(e)

        print("=" * 60)


        return jsonify({
            "error": str(e)
        }), 500


    finally:

        # ----------------------------------------------------
        # Delete temporary PDF
        # ----------------------------------------------------

        if (
            temporary_file
            and
            os.path.exists(temporary_file)
        ):

            try:

                os.remove(
                    temporary_file
                )

            except Exception as e:

                print(
                    "Warning: Could not delete "
                    "temporary PDF:",
                    e
                )


# ============================================================
# ERROR HANDLER: FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error":
        "The uploaded PDF is too large. "
        "Maximum size is 20 MB."
    }), 413


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print("DocuGuard Backend")

    print("=" * 60)

    print("Server starting...")

    print(
        "Frontend should connect to:"
    )

    print(
        "http://127.0.0.1:5000/ask"
    )

    print("=" * 60 + "\n")


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
