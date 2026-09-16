import os
import torch
from pypdf import PdfReader
from transformers import BertTokenizerFast
from compact_model_loader import load_compact_model

MODEL_PATH = "."

print("Loading compact tokenizer...")
tokenizer = BertTokenizerFast(
    tokenizer_file=os.path.join(MODEL_PATH, "tokenizer.json")
)
print("Tokenizer loaded successfully.")

print("Loading compact INT8 QA model...")
model = load_compact_model(MODEL_PATH)
print("Model loaded successfully.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Using device:", device)


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


def create_chunks(pages, words_per_chunk=250):
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


def answer_from_chunk(question, context):
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        padding=False,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    offset_mapping = inputs["offset_mapping"][0]
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

    context_positions = [
        i for i, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
    ]

    if not context_positions:
        return "", 0.0

    start_scores = start_logits[context_positions]
    end_scores = end_logits[context_positions]

    best_start_index = torch.argmax(start_scores).item()
    best_end_index = torch.argmax(end_scores).item()

    start_position = context_positions[best_start_index]
    end_position = context_positions[best_end_index]

    if end_position < start_position:
        return "", 0.0

    if end_position - start_position > 30:
        end_position = start_position + 30

    start_char = offset_mapping[start_position][0].item()
    end_char = offset_mapping[end_position][1].item()

    if start_char >= end_char:
        return "", 0.0

    answer = context[start_char:end_char].strip()

    start_probability = torch.softmax(
        start_logits[context_positions], dim=0
    )[best_start_index].item()

    end_probability = torch.softmax(
        end_logits[context_positions], dim=0
    )[best_end_index].item()

    confidence = (start_probability + end_probability) / 2

    return answer, confidence


def process_document(pdf_path, question):
    print("\nReading PDF...")
    pages = read_pdf(pdf_path)

    if not pages:
        return "", 0.0, None

    print("Pages extracted:", len(pages))

    chunks = create_chunks(pages)
    print("Document chunks:", len(chunks))

    best_answer = ""
    best_confidence = 0.0
    best_page = None

    for i, chunk_data in enumerate(chunks):
        answer, confidence = answer_from_chunk(
            question,
            chunk_data["text"]
        )

        if answer and confidence > best_confidence:
            best_answer = answer
            best_confidence = confidence
            best_page = chunk_data["page"]

        print(f"Processed chunk {i + 1}/{len(chunks)}")

    return best_answer, best_confidence, best_page


if __name__ == "__main__":
    PDF_PATH = "test_legal_document.pdf"
    question = input("\nEnter your question: ")

    if not os.path.exists(PDF_PATH):
        print("\nERROR:")
        print("PDF file not found.")
        print("Make sure the PDF is in the same folder as this Python file.")
    else:
        answer, confidence, page = process_document(
            PDF_PATH,
            question
        )

        print("\n" + "=" * 60)
        print("QUESTION:")
        print(question)

        print("\nANSWER:")
        print(answer if answer else "No relevant answer found.")

        print("\nCONFIDENCE:")
        print(f"{confidence:.4f}")

        if page:
            print("\nPAGE:")
            print(page)

        print("=" * 60)
