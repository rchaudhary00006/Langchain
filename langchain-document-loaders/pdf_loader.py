from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader("pdfs/RFM.pdf")

pdf_docs = pdf_loader.load()

print(len(pdf_docs))
