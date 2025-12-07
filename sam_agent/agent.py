import logging
# Set the logging level for httpx to WARNING to suppress INFO messages
logging.getLogger("httpx").setLevel(logging.WARNING)

from langchain_core.prompts import ChatPromptTemplate
from sam_agent.tools.generator.sam_generator import generator_tool
from sam_agent.tools.predictor.prop_predictor import Predictor
from sam_agent.tools.molecular_tool.price import Molinfo
from sam_agent.tools.rag.retrieval import RetrievalQA
from langchain.agents import AgentExecutor, create_structured_chat_agent, Tool
from langchain_experimental.utilities import PythonREPL
from langchain.tools.base import StructuredTool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from sam_agent.tools.molecular_tool.mol_agent import Mol_agent
from sam_agent.tools.data_mining.data_extractor_cat import DataExtractor
from sam_agent.tools.synthesis_advisor.advisor import plan_and_print
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List,Optional,Union

###################################################
class GeneratorToolInput(BaseModel):
    gen_size: int
    scaf_condition: List[str]
    anchoring_group: str
    
class PredictorToolInput(BaseModel):
    smiles: Optional[List[str]] = None
    generated: bool
    HOMO: bool
    LUMO: bool
    DM: bool

class MolinfoToolInput(BaseModel):
    generated: bool
    mol_list: Optional[List[str]] = None
    
class Mol_agentInput(BaseModel):
    input_message: str
    
class Data_extractorInput(BaseModel):
    path: Union[str, dict]
    pdf: bool = False
    text: bool = False
    mode: str
    
class RetrievalQAInput(BaseModel):
    query: str
    top_k: int
    
class SynthesisInput(BaseModel):
    smiles:str
###################################################
    


class PerovskiteMutiAIAgent:
    def __init__(self,open_ai_key,deepseek_key ,tavily_key=None,verbose=True,llm_model='deepseek'):
        self.open_ai_key=open_ai_key
        self.tavily_key=tavily_key
        self.deepseek_key = deepseek_key
        self.verbose = verbose
        self.input=None
        self.llm_model = llm_model
        self.prop_predictor = Predictor()
        # Instantiate tools here
        self.molinfo_tool = Molinfo()
        self.mol_agent_tool = Mol_agent(self.open_ai_key, verbose=True)
        self.data_extractor_tool = DataExtractor(self.open_ai_key, self.deepseek_key, 'collect_data.csv', llm_model='deepseek', mutimodal=False, verbose=True)
        self.retrieval_qa_tool = RetrievalQA(path='sam_agent/psc_vectors', llm='chatgpt') # Or pass self.llm() if needed later
        self.python_repl = PythonREPL()
        self.chat_history = None
        self.memory = InMemoryChatMessageHistory(session_id="test-session")


    
    # Call the LLM model 
    def llm(self):
        engine = self.llm_model.lower()
        if engine == 'chatgpt':
            llm_agent = ChatOpenAI(model_name="gpt-4.1",
                                    temperature=0.3,
                                    max_tokens=None,
                                    timeout=None,
                                    base_url="https://api.chatanywhere.tech/v1",
                                    max_retries= 8,
                                    api_key=self.open_ai_key)
            return llm_agent
        elif engine == 'deepseek':
            llm_agent = ChatOpenAI(model_name="deepseek-chat",
                                    temperature=0.3,
                                    max_tokens=None,
                                    timeout=None,
                                    base_url="https://api.deepseek.com",
                                    max_retries= 8,
                                    api_key=self.deepseek_key)
            return llm_agent
        else:
            raise ValueError("Unsupported power model. Please choose 'chatgpt' or 'deepseek'.")
    
    # def debug_generator_tool(*args, **kwargs):
    #     print("generator_tool called with args:", args)
    #     print("generator_tool called with kwargs:", kwargs)
    #     # Validate input
    #     if not kwargs:
    #         raise ValueError("Tool input must be a dictionary with the required keys.")
    #     if "gen_size" not in kwargs or "scaf_condition" not in kwargs or "anchoring_group" not in kwargs:
    #         raise ValueError("Missing one or more required keys: 'gen_size', 'scaf_condition', 'anchoring_group'.")
    #     return generator_tool(*args, **kwargs)
    
    
    def bind_tools(self):
        generator = StructuredTool(
        name="generator",
        description=(
            "This tool is strictly for generating new molecules."
            "Both scaffold and anchoring group are SMILES, if they are molecule name, please using Molecular_agent to convert chemical name to SMILES first."
            "Do not use this tool for general questions or explanations."),
        args_schema=GeneratorToolInput,
        func=generator_tool,
        )
        
        prop_predictor= StructuredTool(
            name='property predictor',
            description=(
                "This tool is strictly for predicting the property (HOMO, LUMO, and Dipole moment) of molecules."
                "The data is already stored in the dataframe"
                "Do not use this tool for general questions or explanations."
                ),
            args_schema=PredictorToolInput,
            func=self.prop_predictor.prop_pred
        )

            
        repl_tool = Tool(
            name="python_executor",
            description=("A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."
                         "When you find a task that you can not solve based on provided functions, you can write some codes to solve tasks like complex calculation, creating a machine learning models."),
            func=self.python_repl.run,
        )
        
        search = TavilySearchResults(max_results=5, 
                                     tavily_api_key=self.tavily_key,
                                     search_depth = "advanced",
                                     include_answer= True,
                                     description="A search engine optimized for comprehensive, accurate, and trusted results."
                                     "When the task is beyond your capability. You should use this tool to help you find final solution."
                                     "Input should be a search query.")
        
        
        Mol_info=StructuredTool(
            name="Supplier_info",
            description=("This tool is for search for the molecular commercial informations, including its price, supplier,purity ...."
                         "Use this to if you need some commercial information about the molecules."
                         "The input should be SMILES or molecule name, if you received none of them, the smiles input should be None and you should use generated dataset."
                         ),
            args_schema=MolinfoToolInput,
            func=self.molinfo_tool.collect_info

                   
        )
        
    
        Molecular_agent=StructuredTool(
            name="Molecular_agent",
            description=("This tool is strictly for answering the questions about the molecules"
                         "It can convert the SMILES to IUPAC name, chemical name to SMILES and also it can use SMILES for visualizing the molecule, draw chemical reactions."
                         "Just use format Products>SAScore>Intermediate products|Intermediate products>SAScore>Precursors or direcltly the Synthesis_advisor ouput(eg. C1=CC=C2C(=C1)C3=CC=CC=C3N2CCP(=O)(O)O>0.0010>c1ccc2c(c1)[nH]c1ccccc12.O=P(O)(O)CCCl), if you want to draw the reaction routes."
                         "Do not use this tool for general questions or explanations."),
            args_schema=Mol_agentInput,
            func=self.mol_agent_tool.invoke
            )

        Data_mining_agent=StructuredTool(
            name="Data_extractor",
            description=("This tool is used for extract parameters or data from the literatures"
                         "The accepted type are txt, pdf, XML."
                         "You need to specify the mode to be one of the following: 'material preparation', 'material property', 'device performance' or 'device fabrication'."
                         "Do not use this tool for general questions or explanations."
                ),
            args_schema=Data_extractorInput,
            func=self.data_extractor_tool.data_collection
            )
        
        # RetrievalQA by RAG
        RetrievalQA_agent = StructuredTool(
            name="RetrievalQA",
            description=("This tool is used for RAG retrieval of relevant data for Self-assembled molecules (SAM) in PSC."
                         "Such as SAM synthesis, SAM properties, device fabrication, device performance"
                         "The top_k is the number of the relevant documents you want to retrieve."
                         "You should give all references to the user in the final response."
                         "The input is a query, and you need to modfiy the query standard to better matching the vector database."
            ),
            args_schema=RetrievalQAInput,
            func=self.retrieval_qa_tool.run,
        )
        
        # Synthesis advisor
        synthesis_advisor = StructuredTool(
            name="Synthesis_advisor",
            description=("This tool is used for synthesis planning of the molecules."
                         "The input should be SMILES of the target molecule."
                         "If you got nothing, you can try RetrievalQA_agent."
            ),
            args_schema=SynthesisInput,
            func=plan_and_print,
        )
        
        
        return [generator,prop_predictor,repl_tool,Mol_info,
                search,Molecular_agent,Data_mining_agent,
                RetrievalQA_agent,synthesis_advisor]
    
    
    
    def prompt(self):
        system_prompt = """You are a materials science assitant in photovoltatic area, If you are not sure about answer to the user's request, use your tools to gather the relevant 
        information and you need to give detail information for subagent to execute: do NOT guess or make up an answer.
        If you are uncertain about the request you can ask user again. Also, for your final response, you should give user some suggestion based on your capability.
        You have access to the following tools:
        
        {tools}

        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

        Valid "action" values: "Final Answer" or {tool_names}

        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{
        "action": $TOOL_NAME,
        "action_input": $INPUT
        }}
        ```

        Follow this format:

        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        ... (repeat Thought/Action/Observation N times)
        Thought: I know what to respond
        Action:
        ```
        {{
        "action": "Final Answer",
        "action_input": "Final response to human"
        }}

        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Thought, Action:```$JSON_BLOB```then Observation"""
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("human", "{agent_scratchpad}"),
            ]
        )
        
        
        return prompt
    

    def agent(self):
        structured_agent = create_structured_chat_agent(llm=self.llm(),
                                                        tools = self.bind_tools(),
                                                        prompt = self.prompt()
                                                        )
        agent_executor = AgentExecutor(
            agent = structured_agent,
            tools= self.bind_tools(),
            verbose = True, # some intermediate steps will be printed if True
            max_iterations=15,
            max_execution_time= 100,
            handle_parsing_errors=True,
            )
    
        return agent_executor


    def invoke(self, input_message: str) -> str:
        """
        Invoke the agent with an input message.

        Args:
            input_message (str): The input message for the agent.

        Returns:
            str: The output from the agent.
        """
        agent_with_chat_history = RunnableWithMessageHistory(
            self.agent(),
            # This is needed because in most real world scenarios, a session id is needed
            # It isn't really used here because we are using a simple in memory ChatMessageHistory
            lambda session_id: self.memory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        config = {"configurable": {"session_id": "test-session"}}
        response = agent_with_chat_history.invoke({"input": input_message}, config)
            
        return response.get("output")
        

