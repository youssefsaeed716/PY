from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings


class PDFToChromaETL:
    def __init__(
        self,
        persist_dir="./chroma_db",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.embeddings = None

    def get_embeddings(self):
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return self.embeddings

    def run(self, pdf_path):
        documents = PyPDFLoader(str(pdf_path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        database = Chroma(
            collection_name="pdf-context",
            embedding_function=self.get_embeddings(),
            persist_directory=self.persist_dir,
        )
        database.add_documents(chunks)
        return database


etl_service = PDFToChromaETL()