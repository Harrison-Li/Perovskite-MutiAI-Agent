from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_structured_chat_agent
from sam_agent.tools.molecular_tool.mol_tool import Mol_tool
from langchain.tools.base import StructuredTool
from pydantic import BaseModel
from typing import List,Optional


class tools_Input(BaseModel):
    data_list: List[str]
    generated: bool = False
    
class image_tool_input(BaseModel):
    smiles_list: List[str]  
    generated: bool = False
    molsPerRow: Optional[int] = 5
    display_limit: Optional[int] = None

class reaction_image_input(BaseModel):
    reaction_routes_str: str
    


class Mol_agent:
    def __init__(self,open_ai_key,verbose=True):
        self.open_ai_key=open_ai_key
        self.verbose = verbose
        
    # Call the mol agent 
    def llm(self):
        model=ChatOpenAI(model_name="gpt-4.1-mini",
                         temperature=0.3,
                         max_tokens=None,
                         timeout=None,
                         base_url="https://api.chatanywhere.tech/v1",
                         max_retries=3,
                         api_key=self.open_ai_key
                         )
        return model
    
    def mol_tools(self):
        Name_SMILES=StructuredTool(
            name='IUPAC name to SMILES',
            description=(
                "This tool is used for converting IUPAC name or chemical names to SMILES"
            ),
            args_schema=tools_Input,
            func=Mol_tool().Name2Smiles
        )
        
        SMILES_Name=StructuredTool(
            name='SMILES to IUPAC name',
            description=(
                "This tool is used for converting SMILES to IUPAC name"
            ),
            args_schema=tools_Input,
            func=Mol_tool().Smiles2Name
        )
            
        Mol_Image=StructuredTool(
            name='Mol_visualization',
            description=(
                "This tool is used for display the molecular structures by image."
                "The data parameter is SMILES."
            ),
            args_schema=image_tool_input,
            func=Mol_tool().Mol2Image
        )
        
        Reaction_Image = StructuredTool(
            name='reaction_visualization',
            description=(
                "This tool is used for display the reaction routes by image."
                "The data parameter is string of reaction routes."
                "The format of the data should follow this: Products>SAScore>Intermediate products|Intermediate products>SAScore>Precursors."
                "Example:'O=C(OCCCCCCCCCCCCCP(=O)(O)O)c1ccccc1>0.8935>CCOP(=O)(CCCCCCCCCCCCCOC(=O)c1ccccc1)OCC|CCOP(=O)(CCCCCCCCCCCCCOC(=O)c1ccccc1)OCC>0.8282>O=C(OCCCCCCCCCCCCCBr)c1ccccc1.CCOP(OCC)OCC'."
            ),
            args_schema=reaction_image_input,
            func=Mol_tool().reaction2image
        )
        return [Name_SMILES,SMILES_Name,Mol_Image,Reaction_Image]
        
        
        
    
    def prompt(self):
        system = '''Respond to the human as helpfully and accurately as possible. You have access to the following tools:

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

        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Thought, Action:```$JSON_BLOB```then Observation'''

        human = '''{input}

        {agent_scratchpad}

        (reminder to respond in a JSON blob no matter what)'''

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", human),
            ]
        )
        return prompt
    
    def agent(self):
        mol_structured_agent = create_structured_chat_agent(llm=self.llm(),
                                                        tools = self.mol_tools(),
                                                        prompt = self.prompt()
                                                        )
        agent_executor = AgentExecutor(
            agent = mol_structured_agent,
            tools= self.mol_tools(),
            verbose = True, # some intermediate steps will be printed if True
            max_iterations=5,
            max_execution_time= 40,
            handle_parsing_errors=True,
            )
    
        return agent_executor

    def invoke(self, input_message: str):
        """
        Invoke the agent with an input message.

        Args:
            input_message (str): The input message for the agent.

        Returns:
            str: The output from the agent.
        """
        agent_executor=self.agent()
        response = agent_executor.invoke({"input": input_message})
            
        return response.get("output")