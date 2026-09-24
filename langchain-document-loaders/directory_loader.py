from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

dir_loader = DirectoryLoader(
    path="pdfs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = dir_loader.lazy_load()

for document in docs:
    print(document.metadata)