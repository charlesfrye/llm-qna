import modal

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "langchain~=0.0.98",
    "openai~=0.26.3",
    "pinecone-client",
    "pymongo[srv]",
    "tiktoken",
)

stub = modal.Stub(
    name="llm-qna",
    image=image,
    secrets=[
        modal.Secret.from_name("pinecone-api-key-personal"),
        modal.Secret.from_name("openai-api-key-fsdl"),
        modal.Secret.from_name("mongodb"),
    ],
)

PINECONE_INDEX = "openai-ada-llm-qna"
MONGO_COLLECTION = "llm-papers"

# Terminal codes for pretty-printing.
START, END = "\033[1;38;5;214m", "\033[0m"


def qanda_langchain(query: str) -> str:
    import os

    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Pinecone
    import openai
    import pinecone

    pretty_log(f"running on query: {query}")

    ###
    # Set up embedding model and llm
    ###
    openai.api_key = os.environ["OPENAI_API_KEY"]
    base_embeddings = OpenAIEmbeddings()
    llm = OpenAI(model_name="text-davinci-003", temperature=0.)

    ###
    # Connect to VectorDB
    ###
    pretty_log("connecting to Pinecone")
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pinecone.init(api_key=pinecone_api_key, environment="us-east1-gcp")
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=base_embeddings)

    ###
    # Run docsearch
    ###
    pretty_log(f"selecting sources by similarity to query")
    docs = docsearch.similarity_search(query, k=4)

    print(*[doc.page_content for doc in docs], sep="\n\n---\n\n")

    ###
    # Run chain
    ###
    pretty_log("running query against Q&A chain")
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    answer = result["output_text"]

    return answer


def pretty_log(str):
    print(f"{START}🥞: {str}{END}")


def connect_to_doc_db():
    import os

    import pymongo

    mongodb_password = os.environ["MONGODB_PASSWORD"]
    mongodb_uri = os.environ["MONGODB_URI"]
    connection_string = f"mongodb+srv://fsdl:{mongodb_password}@{mongodb_uri}/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(connection_string)

    return client


@stub.function(
    image=image,
    timeout=1000,
)
def sync_vector_db_to_doc_db():
    import os

    from langchain.embeddings import OpenAIEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import Pinecone
    import openai
    import pinecone
    import pymongo

    ###
    # Connect to Document DB
    ###
    client = connect_to_doc_db()
    pretty_log("connected to document DB")

    ###
    # Connect to VectorDB
    ###
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pinecone.init(api_key=pinecone_api_key, environment="us-east1-gcp")
    pretty_log("connected to vector store")

    try:  # sync the vector index onto the document store
        index = pinecone.Index(PINECONE_INDEX)
        index.delete(delete_all=True)
        pretty_log("existing index wiped")
    except (pinecone.core.client.exceptions.NotFoundException, pinecone.core.exceptions.PineconeProtocolError):
        pretty_log("creating vector index")
        pinecone.create_index(name=PINECONE_INDEX, dimension=1536, metric="cosine", pod_type="p1.x1")
        pretty_log("vector index created")


    ###
    # Spin up EmbeddingEngine
    ###
    openai.api_key = os.environ["OPENAI_API_KEY"]
    base_embeddings = OpenAIEmbeddings()

    ###
    # Retrieve Documents
    ###
    db = client.get_database("fsdl")
    collection = db.get_collection(MONGO_COLLECTION)

    pretty_log(f"pulling documents from {collection.full_name}")
    docs = collection.find({"metadata.ignore": False})

    ###
    # Chunk Documents and Spread Sources
    ###
    pretty_log("splitting into bite-size chunks")

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100,
        separator="\n",
    )
    ids, texts, metadatas = [], [], []
    for document in docs:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata["sha256"]] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas

    ###
    # Upsert to VectorDB
    ###
    pretty_log(f"sending to vectorDB {PINECONE_INDEX}")
    Pinecone.from_texts(
        texts, base_embeddings, metadatas=metadatas, ids=ids, index_name=PINECONE_INDEX
    )


@stub.function(image=image)
@modal.web_endpoint(method="GET", label="llm-qna-hook")
def web(query: str, request_id=None):
    pretty_log(f"handling request with client-provided id: {request_id}") if request_id else None
    answer = qanda_langchain(query)
    return {
        "answer": answer,
    }


# Add a debugging access point on Modal
@stub.function(
    image=image,
    interactive=True
    )
def debug():
    import IPython
    IPython.embed()
