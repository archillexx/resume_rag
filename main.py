from openai import OpenAI
import chromadb
import pymupdf


client_openai = OpenAI()
client_choma = chromadb.Client()

collection = client_choma.create_collection("my_documents")

pdf = pymupdf.open("resume.pdf")

headers = [
    "Summary",
    "Technical Skills",
    "Education",
    "Experience",
    "Projects",
    "Professional Reference"
]

pages=[]

for page in pdf:
    extracted_text = page.get_text()
    pages.append(extracted_text)
    

chunks = []
ids=[]
metadatas= []
current_chunk = []
current_header = None

for page_num, page in enumerate(pages, start=1):

    lines = page.splitlines()
    current_page=page_num


    for line_num, line in enumerate(lines, start=1):
        if line.strip() in headers:
            if current_chunk:
                chunks.append(current_chunk)
                ids.append(str(len(chunks)- 1))
                metadatas.append({
                "source":"resume.pdf",
                "page": current_page
                })
            current_header = line.strip()
            current_chunk = []
            current_chunk.append(current_header)
        else:
            current_chunk.append(line)
            
    

if current_chunk:
    chunks.append(current_chunk)
    ids.append(str(len(chunks)- 1))
    metadatas.append({
        "source":"resume.pdf",
        "page": current_page
        })

text_chunks = ["\n".join(chunk) for chunk in chunks]

response = client_openai.embeddings.create(
    model="text-embedding-3-small",
    input=text_chunks
)

embeddings = [embed.embedding for embed in response.data]

collection.add(
    ids=ids,
    metadatas=metadatas,
    embeddings=embeddings,
    documents=text_chunks
)

query=input("Question you want to ask?")


response_query = client_openai.embeddings.create(
    model = "text-embedding-3-small",
    input=query
)

q_embedding = response_query.data[0].embedding

results = collection.query(
    query_embeddings=[q_embedding],
    n_results=3
)

context = "\n\n".join(results["documents"][0])

prompt = f"""

    Context: {context}

    Question: {query}

    """

response = client_openai.responses.create(
    model="gpt-5-mini",
    instructions="Answer the question using only the provided context",
    input = prompt
)

print(response.output_text)

