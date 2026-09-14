import os
import torch
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# ============================================================
# 1. MODEL LOCATION
# ============================================================

MODEL_PATH = "./trained_model"


# ============================================================
# 2. LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True
)

print("Tokenizer loaded successfully.")


# ============================================================
# 3. LOAD TRAINED BERT QA MODEL
# ============================================================

print("Loading trained BERT model...")

model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 4. SELECT DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

print("Using device:", device)


# ============================================================
# 5. EXTRACT TEXT FROM PDF
# ============================================================

def read_pdf(pdf_path):

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
# 6. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

def create_chunks(pages, words_per_chunk=250):

    chunks = []

    for page_number, page_text in enumerate(pages):

        words = page_text.split()

        for i in range(0, len(words), words_per_chunk):

            chunk_words = words[i:i + words_per_chunk]

            if chunk_words:

                chunk = " ".join(chunk_words)

                chunks.append({
                    "text": chunk,
                    "page": page_number + 1
                })

    return chunks


# ============================================================
# 7. ANSWER QUESTION FROM ONE CHUNK
# ============================================================

def answer_from_chunk(question, context):

    inputs = tokenizer(
        question,
        context,
        max_length=512,
        truncation="only_second",
        padding=False,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    # offset_mapping is only needed for finding the text answer.
    offset_mapping = inputs["offset_mapping"][0]

    # Find which tokens belong to the context.
    sequence_ids = inputs.sequence_ids(0)

    model_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key != "offset_mapping"
    }

    with torch.no_grad():

        outputs = model(**model_inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]


    # ========================================================
    # Only allow answer positions inside the document context
    # ========================================================

    context_positions = []

    for i, sequence_id in enumerate(sequence_ids):

        if sequence_id == 1:
            context_positions.append(i)


    if not context_positions:
        return "", 0.0


    # Get the best start and end positions
    start_scores = start_logits[context_positions]
    end_scores = end_logits[context_positions]

    best_start_index = torch.argmax(start_scores).item()
    best_end_index = torch.argmax(end_scores).item()

    start_position = context_positions[best_start_index]
    end_position = context_positions[best_end_index]


    # ========================================================
    # Make sure end comes after start
    # ========================================================

    if end_position < start_position:

        return "", 0.0


    # Avoid extremely long answers
    if end_position - start_position > 30:

        end_position = start_position + 30


    # ========================================================
    # Convert token positions back to original text positions
    # ========================================================

    start_char = offset_mapping[start_position][0].item()
    end_char = offset_mapping[end_position][1].item()


    if start_char >= end_char:

        return "", 0.0


    answer = context[start_char:end_char].strip()


    # ========================================================
    # Calculate confidence
    # ========================================================

    start_probability = torch.softmax(
        start_logits[context_positions],
        dim=0
    )[best_start_index].item()

    end_probability = torch.softmax(
        end_logits[context_positions],
        dim=0
    )[best_end_index].item()

    confidence = (start_probability + end_probability) / 2


    return answer, confidence


# ============================================================
# 8. PROCESS COMPLETE DOCUMENT
# ============================================================

def process_document(pdf_path, question):

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


    # ========================================================
    # Ask BERT about every chunk
    # ========================================================

    for i, chunk_data in enumerate(chunks):

        context = chunk_data["text"]
        page_number = chunk_data["page"]


        answer, confidence = answer_from_chunk(
            question,
            context
        )


        if answer and confidence > best_confidence:

            best_answer = answer
            best_confidence = confidence
            best_page = page_number


        print(
            f"Processed chunk {i + 1}/{len(chunks)}"
        )


    return best_answer, best_confidence, best_page


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Put your legal PDF filename here
    # --------------------------------------------------------

    PDF_PATH = "test_legal_document.pdf"


    # --------------------------------------------------------
    # Ask the question
    # --------------------------------------------------------

    question = input("\nEnter your question: ")


    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not os.path.exists(PDF_PATH):

        print("\nERROR:")
        print("PDF file not found.")
        print("Make sure the PDF is in the same folder as this Python file.")

    else:

        answer, confidence, page = process_document(
            PDF_PATH,
            question
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

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
