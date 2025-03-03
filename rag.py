import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from tqdm import tqdm
import unicodedata

# Load RAG prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

# Initialize models
llm = OllamaLLM(model="deepseek-r1:14b")
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# Context file:
CONTEXT_FILE_PATH = "context_files/ScientificAdvertising.txt"

# Persistent FAISS storage
FAISS_DB_PATH = "faiss_index"
PROGRESS_FILE = "progress.txt"
MAX_LINES = 20000  # ⚡ Limit for testing


def count_lines(filename):
    """Counts the total number of lines in the file for tqdm transparency."""
    with open(filename, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def get_last_processed_line():
    """Reads the last processed line number from progress file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def save_last_processed_line(line_number):
    """Saves the last processed line number to resume later."""
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(line_number))


def clean_text(text):
    """Removes unwanted Unicode characters like U+200E (Left-To-Right Mark)."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf").strip()


def parse_book_text(filename, batch_size=5, start_line=0):
    """Parses book text into meaningful chunks, preserving paragraph integrity."""
    batch = []
    chunk = []  # Stores multiple paragraphs for a single chunk
    processed_lines = 0
    line_number = start_line  # Initialize with start_line to avoid unbound errors

    with open(filename, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):  # Track line number
            if line_number < start_line:
                continue  # Skip already processed lines

            if processed_lines >= MAX_LINES:
                return

            line = clean_text(line.strip())  # Remove unwanted Unicode chars

            if line:
                chunk.append(line)
            else:
                # Empty line indicates a paragraph break
                if chunk:
                    batch.append(" ".join(chunk))
                    chunk = []  # Reset for new paragraph

            # If batch reaches batch size, yield
            if len(batch) >= batch_size or processed_lines >= MAX_LINES:
                yield batch, line_number
                batch = []

            processed_lines += 1

    # Store the last paragraph if any
    if chunk:
        batch.append(" ".join(chunk))

    if batch and processed_lines < MAX_LINES:
        yield batch, line_number


def embed_and_store():
    """Embeds book text and stores it in a FAISS vector database."""
    last_processed_line = get_last_processed_line()
    total_lines = min(count_lines(CONTEXT_FILE_PATH), MAX_LINES)

    vector_store = None

    if os.path.exists(FAISS_DB_PATH) and os.listdir(FAISS_DB_PATH):
        try:
            print("🟢 Attempting to load FAISS index in embed_and_store()...")
            vector_store = FAISS.load_local(
                FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True
            )
            print(
                f"✅ FAISS index loaded successfully! Contains {vector_store.index.ntotal} entries."
            )
        except Exception as e:
            print(f"❌ Error loading FAISS index in embed_and_store(): {e}")
            os.system(f"rm -rf {FAISS_DB_PATH}")  # Delete corrupted FAISS index
            vector_store = None
    else:
        print("❌ FAISS index does not exist or is empty. Creating a new index...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    with tqdm(
        total=total_lines,
        desc=f"Processing first {MAX_LINES} lines",
        unit=" lines",
        initial=last_processed_line,
    ) as pbar:
        for batch, line_number in parse_book_text(
            CONTEXT_FILE_PATH, start_line=last_processed_line
        ):
            docs = [
                Document(page_content=text, metadata={"source": f"batch_{hash(text)}"})
                for text in batch
            ]
            chunks = text_splitter.split_documents(docs)

            if vector_store:
                vector_store.add_documents(chunks)
                print(f"➕ {len(chunks)} new documents added to FAISS.")
            else:
                print("🛠 Creating a new FAISS index...")
                vector_store = FAISS.from_documents(chunks, embedding_model)
                print(
                    f"✅ New FAISS index created with {vector_store.index.ntotal} entries."
                )

            print(
                f"🧐 Checking FAISS size... Current vector count: {vector_store.index.ntotal}"
            )

            if vector_store and vector_store.index.ntotal > 0:
                vector_store.save_local(FAISS_DB_PATH)
                print(
                    f"💾 FAISS index saved at line {line_number}. Contains {vector_store.index.ntotal} entries."
                )
            else:
                print("⚠️ Warning: FAISS index is empty after processing!")

            save_last_processed_line(line_number)
            pbar.update(len(batch))

    if vector_store and vector_store.index.ntotal > 0:
        vector_store.save_local(FAISS_DB_PATH)
        print(
            f"✅ FAISS index successfully saved! Final size: {vector_store.index.ntotal} entries."
        )
    else:
        print("❌ FAISS index was NOT saved. Check embeddings.")

    return vector_store


def query_chat(query):
    """Retrieves relevant book context and generates an AI response."""
    try:
        print("🟢 Attempting to load FAISS index in query_chat()...")
        vector_store = FAISS.load_local(
            FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True
        )
        print("✅ FAISS index loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading FAISS index in query_chat(): {e}")
        return "Error: Could not load FAISS index."

    similar_docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in similar_docs])

    print("\n🔍 Retrieved Context from FAISS:\n", context)
    formatted_prompt = (
        prompt.invoke({"context": context, "question": query}).to_messages()[0].content
    )
    print("\n📝 Full Prompt to LLM:\n", formatted_prompt)
    response = llm.invoke(formatted_prompt)
    return response


def main():
    print(f"Processing book text (LIMITED to {MAX_LINES} lines for testing)...")
    embed_and_store()

    while True:
        user_query = input("Ask a question about the book (or type 'exit'): ")
        if user_query.lower() in ["exit", "quit"]:
            break
        response = query_chat(user_query)
        print("\nAI Response:", response, "\n")


if __name__ == "__main__":
    main()
