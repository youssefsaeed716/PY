from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class LocalRAGService:
    def __init__(self, persist_dir="./chroma_db"):
        # 1. تعريف الـ Embeddings (نفس اللي استخدمناها في الـ ETL)
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # 2. ربط قاعدة بيانات الـ Chroma المحلية
        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # 3. تشغيل نموذج Llama محلياً عبر Ollama
        self.llm = ChatOllama(
            model="llama3.2:1b",
            temperature=0.3
        )

        # 4. بناء الـ Prompt
        self.template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        
    def answer_query(self, question: str):
        rag_chain = (
            {"context": self.retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
                "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)

ai_service = LocalRAGService()