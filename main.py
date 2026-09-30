from openai import OpenAI
import chromadb
import pymupdf
import re
import json

client_openai = OpenAI()
client_chroma = chromadb.Client()

collection = client_chroma.get_or_create_collection("resume")


# Load PDF
def load_pdf(filename):
    return pymupdf.open(filename)


# Extract text
def extract_text(pdf):
    pages = []

    for page in pdf:
        page_text = " ".join(page.get_text().split())
        pages.append(page_text)

    return pages


# Create chunks
def create_chunks(pages, chunk_size=100, overlap=1):
    chunks = []

    for page_num, page_text in enumerate(pages, start=1):
        sentences = re.split(r'(?<=[.!?])\s+', page_text)
        current_chunks = []

        for sentence in sentences:
            new_length = len(
                " ".join(current_chunks + [sentence]).split()
            )

            if new_length <= chunk_size:
                current_chunks.append(sentence)
            else:
                chunks.append({
                    "text": " ".join(current_chunks),
                    "page": page_num
                })

                current_chunks = current_chunks[-overlap:]
                current_chunks.append(sentence)

        if current_chunks:
            chunks.append({
                "text": " ".join(current_chunks),
                "page": page_num
            })

    return chunks


# Create embeddings
def create_embeddings(texts):
    response = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [
        embed.embedding
        for embed in response.data
    ]

    return embeddings


# Store documents
def store_documents(collection, ids, documents, embeddings, metadatas):
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


# Retrieve documents
def retrieve_documents(query, n_results=4):
    response = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    q_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []

    for i, document in enumerate(documents):
        output.append({
            "text": document,
            "page": metadatas[i]["page"],
            "distance": distances[i]
        })

    for i,single_output in enumerate(output,start =1):
        print(f""" === Vector Retrieval ===
        Chunk {i} : {single_output["text"]}
        Page : {single_output["page"]}
        Distance : {single_output["distance"]}
         """)


    return output


# Rerank documents
def rerank_documents(query, documents):
    scores = []

    for doc in documents:
        prompt = f"""
        Context: {doc["text"]}

        Question: {query}
        """

        response = client_openai.responses.create(
            model="gpt-5-mini",
            input=prompt,
            instructions=(

             "Evaluate how directly the context answers the question. "

             "Return a relevance score from 0 to 1. "

              "Use 1.0 for a context that directly answers the question, "

              "0.7 to 0.9 for highly relevant supporting information, "

              "0.4 to 0.6 for somewhat relevant information, "

              "0.1 to 0.3 for weakly related information, and "

               "0.0 for completely irrelevant information."),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "relevance_score",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number"
                            }
                        },
                        "required": ["score"],
                        "additionalProperties": False
                    }
                }
            }
        )

        score = json.loads(response.output_text)["score"]
        scores.append(score)
    print(scores)
    return scores


# Add scores
def combine_scores(documents, scores):
    for i, doc in enumerate(documents):
        doc["score"] = scores[i]
        
        print(f"""
        Chunk {i+1}
        Distance {doc["distance"]}
        Page {doc["page"]}
        Score {doc["score"]}
        """)

    return documents


# Sort documents
def sort_documents(documents, top_k=3):
    sorted_documents = sorted(
        documents,
        key=lambda doc: doc["score"],
        reverse=True
    )

    return sorted_documents[:top_k]


# Build context
def build_context(documents):
    context = ""

    for doc in documents:
        context += f"[Page {doc['page']}]\n"
        context += doc["text"]
        context += "\n\n"

    return context


# Generate answer
def answer_question(query, context):
    prompt = f"""
    Context:
    {context}

    Question:
    {query}
    """

    response = client_openai.responses.create(
        model="gpt-5-mini",
        instructions=(
            "Answer the question using only the provided context. "
            "If the answer cannot be found in the context, say that "
            "you don't have enough information."
        ),
        input=prompt
    )

    return response.output_text


# Prepare documents
pdf = load_pdf("resume.pdf")
pages = extract_text(pdf)
chunks = create_chunks(pages)

texts = [chunk["text"] for chunk in chunks]
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas = [{"page": chunk["page"]} for chunk in chunks]

embeddings = create_embeddings(texts)

store_documents(
    collection,
    ids,
    texts,
    embeddings,
    metadatas
)


# Run RAG
query = input("Ask your question: ")

results = retrieve_documents(query)

scores = rerank_documents(
    query,
    results
)

reranked = combine_scores(
    results,
    scores
)

sorted_docs = sort_documents(
    reranked,
    top_k=3
)

context = build_context(sorted_docs)

answer = answer_question(
    query,
    context
)

print(f"\nAnswer: {answer}")