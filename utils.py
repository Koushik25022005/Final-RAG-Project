import io
import warnings
import PyPDF2
from PIL import Image
import pytesseract
import numpy as np

# optional, more robust PDF extractor
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf_fileobj(fileobj):
    """
    Try pdfplumber first (more tolerant), fall back to PyPDF2.
    fileobj: file-like opened in binary mode, must be seekable.
    """
    fileobj.seek(0)
    text_pieces = []

    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(fileobj) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_pieces.append(t)
            if text_pieces:
                return "\n".join(text_pieces)
            # else fall through to PyPDF2
        except Exception:
            # if pdfplumber fails, fallback to PyPDF2
            pass

    # Fallback: PyPDF2 (may warn on weird PDFs)
    try:
        fileobj.seek(0)
        # suppress PyPDF2 warnings printed to stderr by using warnings.catch_warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pdf_reader = PyPDF2.PdfReader(fileobj)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pieces.append(page_text)
        return "\n".join(text_pieces)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")


def process_file(file):
    """
    Accepts a file-like object with an optional 'type' attribute (e.g. "application/pdf" or "image/png").
    Returns list of text chunks.
    """
    # Extract text
    text = ""

    # Detect file type robustly
    ftype = getattr(file, "type", None)
    if ftype and ftype.lower() == "application/pdf":
        text = extract_text_from_pdf_fileobj(file)

    elif ftype and ftype.lower().startswith("image"):
        file.seek(0)
        image = Image.open(file)
        text = pytesseract.image_to_string(image)

    else:
        # If unknown type, try to sniff PDF or image; else try to read as bytes->utf8 text
        file.seek(0)
        header = file.read(1024)
        file.seek(0)
        if header.startswith(b"%PDF"):
            text = extract_text_from_pdf_fileobj(file)
        else:
            try:
                # attempt to read as text (utf-8)
                raw = file.read()
                if isinstance(raw, bytes):
                    text = raw.decode("utf-8", errors="ignore")
                else:
                    text = str(raw)
            except Exception:
                raise ValueError("Unsupported file type and could not decode as text.")

    if not text or not text.strip():
        raise ValueError("No text could be extracted from the file.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    return chunks


def generate_response(query, chunks, top_k=3,
                      embed_model_name="all-MiniLM-L6-v2",
                      lm_model_name="gpt2", device="cpu"):
    """
    - Uses SentenceTransformer for embeddings (on CPU).
    - Uses sklearn NearestNeighbors to find top_k relevant chunks.
    - Uses transformers AutoTokenizer + AutoModelForCausalLM with model.generate(...)
      and explicit generation args (no conflicting max_length/max_new_tokens).
    """

    if not chunks:
        raise ValueError("No chunks provided")

    # 1) Embeddings (sentence-transformers materializes weights on cpu by default)
    embed_device = "cpu"  # keep embeddings on CPU
    embedder = SentenceTransformer(embed_model_name, device=embed_device)
    chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

    # 2) Similarity search (sklearn)
    nbrs = NearestNeighbors(n_neighbors=min(top_k, len(chunks)), metric="cosine").fit(chunk_embeddings)
    query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
    distances, indices = nbrs.kneighbors([query_embedding])
    indices = indices[0].tolist()
    top_chunks = [chunks[i] for i in indices]
    context = "\n\n---\n\n".join(top_chunks)

    # 3) LLM generation
    # device handling: "cpu" or "cuda" supported. Map to torch device
    torch_device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    model = AutoModelForCausalLM.from_pretrained(lm_model_name)
    model.to(torch_device)

    # Ensure we have a pad token (gpt2 doesn't have pad_token by default)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Tokenize prompt with explicit truncation to silence tokenizer truncation warning
    # We choose a reasonable max length for input prompt; adapt if needed
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=1024, return_tensors="pt")
    input_ids = input_ids.to(torch_device)

    # Generation args (explicit)
    gen_kwargs = dict(
        max_new_tokens=256,   # generates up to 256 new tokens
        do_sample=True,       # use sampling
        temperature=0.8,      # sampling temperature
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id  # avoid warnings about pad token
    )

    # Now generate (single source of truth for generation params)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            **gen_kwargs,
            attention_mask=None  # model.generate will handle masks if needed
        )

    generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # Extract only the portion after the "Answer:" marker
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        # fallback: return generated text minus the original prompt (if present)
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):].strip()
        else:
            answer = generated_text.strip()

    return {
        "answer": answer,
        "context_chunks": top_chunks,
        "indices": indices,
        "distances": distances[0].tolist()
    }


# Example usage:
# with open("document.pdf","rb") as f:
#     f.type = "application/pdf"
#     chunks = process_file(f)
#     resp = generate_response("What's the summary?", chunks, device="cpu")
#     print(resp["answer"])
