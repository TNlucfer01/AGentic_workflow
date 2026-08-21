import os
import getpass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.documents import Document
import pypdf

# setting the documents
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
# seeing the doccuments
for i in range(len(documents)):
    print(f"{i}", documents[i].page_content)

# setting the embedding model for our agent
file_path = "./nke-10k-2023.pdf"
if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# genrate embedding
vector_1 = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
).embed_query(documents[0].page_content)
vector_2 = embeddings.embed_query(documents[1].page_content)


assert len(vector_1) == len(vector_2)

# printing the vector
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:3000])

# qdrant getting  for storing he vectors
from qdrant_client.models import Distance, VectorParams

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# just add t api key here
client = QdrantClient(":memory:")
# for a  mention size of eah vector
vector_size = len(embeddings.embed_query("sample text"))


if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)
import pypdf
from langchain_core.documents import Document


# Below is a minimal helper for demonstration purposes.
def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]

#loadeed the document successfully
docs = load_pdf_pages(file_path)
print(docs[20])

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=300, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
ids=vector_store.add_documents(documents=all_splits[1:100])
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])
print("hello wthis ia after the results \n\n\n\n results")
