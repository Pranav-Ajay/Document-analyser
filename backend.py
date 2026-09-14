import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pypdf import PdfReader

# -----------------------------
# Load your trained CUAD model
# -----------------------------

MODEL_PATH = "./trained_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# -----------------------------
# Read PDF
# -----------------------------

def read_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    lines = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            page_lines = text.splitlines()

            for line in page_lines:
                line = line.strip()

                if line:
                    lines.append(line)

    return lines


# -----------------------------
# Create text chunks
# -----------------------------

def create_chunks(lines, lines_per_chunk=15):

    chunks = []

    for i in range(0, len(lines), lines_per_chunk):

        chunk = " ".join(lines[i:i + lines_per_chunk])

        chunks.append(chunk)

    return chunks


# -----------------------------
# Ask the trained BERT model
# -----------------------------

def answer_question(question, context):

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_position = torch.argmax(start_logits, dim=1).item()
    end_position = torch.argmax(end_logits, dim=1).item()

    # Invalid span
    if end_position < start_position:
        return "", 0

    answer_tokens = inputs["input_ids"][0][
        start_position:end_position + 1
    ]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    )

    # Confidence
    start_confidence = torch.softmax(start_logits, dim=1)[
        0, start_position
    ].item()

    end_confidence = torch.softmax(end_logits, dim=1)[
        0, end_position
    ].item()

    confidence = (start_confidence + end_confidence) / 2

    return answer, confidence


# -----------------------------
# Process entire document
# -----------------------------

def process_document(pdf_path, question):

    lines = read_pdf(pdf_path)

    print("Number of lines:", len(lines))

    chunks = create_chunks(lines)

    print("Number of chunks:", len(chunks))

    best_answer = ""
    best_confidence = 0

    for chunk in chunks:

        answer, confidence = answer_question(
            question,
            chunk
        )

        if answer.strip() and confidence > best_confidence:

            best_answer = answer
            best_confidence = confidence

    return best_answer, best_confidence


# -----------------------------
# Example
# -----------------------------

pdf_path = "test_legal_document.pdf"

question = "What is the termination date of the agreement?"

answer, confidence = process_document(
    pdf_path,
    question
)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)

print("\nCONFIDENCE:")
print(confidence)
