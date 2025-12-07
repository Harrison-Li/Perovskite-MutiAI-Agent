from sam_agent.tools.rag.embedding import OpenAIEmbedding_model
from sam_agent.tools.rag.vector_storage import VectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os



class RetrievalQA:
    def __init__(self,llm = 'chatgpt',path = 'PSC_parameters_vector_updated'):
        self.data = VectorStore()
        self.path = path
        self.embedding_model = OpenAIEmbedding_model(model_type="text-embedding-3-large",api_key=os.environ['OPENAI_API_KEY'])
        if llm == 'deepseek':
            self.llm = ChatOpenAI(model_name="deepseek-chat",
                                  temperature= 0,
                                  base_url= "https://api.deepseek.com",
                                  max_tokens=None,
                                  api_key=os.environ['DeepSeek_API_KEY'])
        elif llm == 'chatgpt':
            self.llm = ChatOpenAI(model_name="gpt-4.1-mini",
                                  base_url="https://api.chatanywhere.tech/v1",
                                  temperature= 0,
                                  max_tokens=None,
                                  api_key=os.environ['OPENAI_API_KEY'])
        else:
            raise ValueError("Invalid LLM name. Choose either 'deepseek' or 'chatgpt'.")

    def run(self, query: str, top_k: int = 1) -> str:
        # Retrieve relevant documents
        if query:
            self.data.load_vector(self.path)
        context = ""
        for string in self.data.query(query, self.embedding_model, k= top_k):
            context += string + '\n'
        # Prepare the prompt
        messages = [
            SystemMessage(content="You are a helpful assistant. You will be given a section of an article and a question. Please answer the question based on the provided article section and provide the references."),
            HumanMessage(f'\n\narticle section:\n"""\n{context}\n"""'),
            HumanMessage(content=query),
        ]
            
        # Get the response from the LLM
        response = self.llm(messages)
        return response.content