import fitz
import numpy as np
import time
import faiss
import json
from pathlib import Path
from litellm import completion
from benchmark import benchmark

pdf_start = time.time()

BASE_DIR = Path(__file__).parent

PDF_PATH = BASE_DIR / "examples"/"documents"/"attention-residuals.pdf"

RESULTS_DIR = Path("final_results")
RESULTS_DIR.mkdir(parents=True, exist_ok = True)

doc = fitz.open(str(PDF_PATH))

text = ""
page_offsets=[]

for page_num, page in enumerate(doc):
    page_offsets.append({
        "page" : page_num+1,
        "start" : len(text)
    })

    text += page.get_text()

print("Characters:", len(text))

pdf_end=time.time()
print(f"pdf extraction time: {pdf_end - pdf_start:.2f}sec ")

text = text.replace("\n\n\n", "\n")

from langchain_text_splitters import RecursiveCharacterTextSplitter
chunk_start = time.time()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

print("\nFIRST CHUNK:\n")
print(chunks[0])

print("\nSECOND CHUNK:\n")
print(chunks[1])

chunk_end = time.time()

print(f"Chunking Time: {chunk_end-chunk_start:.2f} sec")

import ollama

print("\nGenerating embeddings...")

chunk_embeddings = []

embedding_start = time.time()



for i, chunk in enumerate(chunks):
    embedding = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )["embeddings"][0]

    chunk_embeddings.append(embedding)

    if i % 20 == 0:
        print(f"Processed {i}/{len(chunks)} chunks")

print("Embedding generation complete!")

embedding_end = time.time()
print(f"Embedding Time: {embedding_end-embedding_start:.2f} sec")

index_start = time.time()

embeddings_np = np.array(chunk_embeddings).astype("float32")

faiss.normalize_L2(embeddings_np)

dimension = embeddings_np.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings_np)

index_end = time.time()

print(f"FAISS Index Time: {index_end-index_start:.4f} sec")

for qid,item in benchmark.items():

    query = item["question"]

    print("\n"+ "="*80)
    print(f"running {qid}")
    print(query)
    print("="*80)
    
    total_start = time.time()

    query_embedding = np.array(
        ollama.embed(
            model="nomic-embed-text",
            input=query
        )["embeddings"][0],
        dtype="float32"
    ).reshape(1, -1)

    faiss.normalize_L2(query_embedding)


    top_k=8
    retrieval_start = time.time()

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_items = indices[0].tolist()
    print("retrieved items",retrieved_items)

    top_k=8
    print("\nTOP CHUNKS\n")

    for rank in range(top_k):

        idx = indices[0][rank]
        score = distances[0][rank]

        print("=" * 60)
        print(f"Rank {rank+1}")
        print(f"chunk id : {idx}")
        print(f"Score: {score:.4f}")
        print(chunks[idx][:500])

    context = "\n\n".join(
        [chunks[idx] for idx in indices[0]]
    )

    retrieval_end = time.time()

    retrieval_time = retrieval_end - retrieval_start


    prompt = f"""
    provide only the answer 

    you are answering questions about a research paper .

    use only the provided context below . 

    if the answer is not present, say " Not found in the retrieved pages " . 

    context 
    {context}

    question 
    {query}

    give concise answer within 5-8 bullet points .
    """

    generation_start = time.time()
    response = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options = {
            "temperature" : 0
        }
    )
    generation_end = time.time()

    generation_time = generation_end-generation_start

    print("\n" + "="*80)
    print("FINAL ANSWER")
    print("="*80)

    answer = response["message"]["content"]

    print(answer)

    total_end = time.time()
    total_time = total_end-total_start

    print("\n" + "="*60)
    print("timings")
    print("="*60)

    print(f"retrieval time : {retrieval_time:.3f} sec")
    print(f"generation time : {generation_time:.3f} sec ")
    print(f"total time : {total_time:.3f} sec ")


    results = {
        "system": "RAG",

        "question": query,
    
        "top_k": top_k,

        "retrieved_items" : retrieved_items,

        "retrieved_context" : context,

        "answer" : answer,
    
        "retrieval_time" : retrieval_time,

        "generation_time" : generation_time,

        "pipeline_time" : total_time

    }
    output_path = RESULTS_DIR / f"RAG_result_{qid}.json"

    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"results saved to {output_path}")



