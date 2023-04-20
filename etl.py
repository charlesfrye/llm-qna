import modal

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "langchain~=0.0.98",
    "pypdf~=3.8",
    "pymongo[srv]==3.11",
)

stub = modal.Stub(
    name="llm-papers",
    image=image,
    secrets=[
        modal.Secret.from_name("mongodb"),
    ],
)

MONGO_COLLECTION = "llm-papers"


@stub.local_entrypoint()
def main(json_path):
    import json

    with open(json_path) as f:
        pdf_infos = json.load(f)

    pdf_urls = [pdf["url"] for pdf in pdf_infos]

    results = list(extract_pdf.map(pdf_urls, return_exceptions=True))
    add_to_document_db.call(results)


@stub.function(image=image)
def flush_doc_db():
    client = connect_to_doc_db()

    db = client.get_database("fsdl")
    collection = db.get_collection(MONGO_COLLECTION)
    collection.drop()


@stub.function(image=image)
def extract_pdf(pdf_url):
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_url)
    pages = loader.load_and_split()

    for page in pages:
        page.metadata["source"] = pdf_url

    pages = enrich_metadata(pages)

    return [page.json() for page in pages]


@stub.function(image=image)
def add_to_document_db(all_pages_jsons):
    from langchain.docstore.document import Document
    from pymongo import InsertOne

    client = connect_to_doc_db()

    db = client.get_database("fsdl")
    collection = db.get_collection(MONGO_COLLECTION)

    all_pages = []
    for pages_json in all_pages_jsons:
        pages = [Document.parse_raw(page) for page in pages_json]
        if len(pages) >= 75:
            continue
        all_pages += pages

    requesting, CHUNK_SIZE = [], 250

    for page in all_pages:
        metadata = page.metadata
        document = {"text": page.page_content, "metadata": metadata}
        requesting.append(InsertOne(document))

        if len(requesting) >= CHUNK_SIZE:
            collection.bulk_write(requesting)
            requesting = []

    if requesting:
        collection.bulk_write(requesting)


def annotate_endmatter(pages, min_pages=6):
    out, after_references = [], False
    for idx, page in enumerate(pages):
        content = page.page_content.lower()
        if idx >= min_pages and ("references" in content or "bibliography" in content):
            after_references = True
        page.metadata["is_endmatter"] = after_references
        out.append(page)
    return out


def enrich_metadata(pages):
    import hashlib

    pages = annotate_endmatter(pages)
    for page in pages:
        m = hashlib.sha256()
        m.update(page.page_content.encode("utf-8"))
        page.metadata["sha256"] = m.hexdigest()
        if page.metadata.get("is_endmatter"):
            page.metadata["ignore"] = True
        else:
            page.metadata["ignore"] = False
    return pages


def connect_to_doc_db():
    import os

    import pymongo

    mongodb_password = os.environ["MONGODB_PASSWORD"]
    mongodb_uri = os.environ["MONGODB_URI"]
    connection_string = f"mongodb+srv://fsdl:{mongodb_password}@{mongodb_uri}/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(connection_string)

    return client
