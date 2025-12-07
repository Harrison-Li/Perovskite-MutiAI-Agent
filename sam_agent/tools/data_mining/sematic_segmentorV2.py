from langchain_openai import ChatOpenAI
import pandas as pd
from langchain.output_parsers.json import SimpleJsonOutputParser
from sam_agent.tools.data_mining.utils import load_txt, get_pdf_image, extract_xml_structure, extract_pdf_structure
from sam_agent.tools.data_mining.data_miner import DataMiner
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder,ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.messages import AIMessage
import re
import os

class SemanticSegmentor:
    def __init__(self,open_ai_key,csv_path:str,verbose=True):
        self.api_key=open_ai_key
        self.verbose=verbose
        self.csv_path = csv_path
        
        
    # Call the parameter extractor agent 
    def llm(self): 
        model=ChatOpenAI(model_name="gpt-4o-mini",
                         temperature= 0,
                         max_tokens=None,
                         timeout=None,
                         base_url="https://api.chatanywhere.tech/v1",
                         max_retries=1,
                         api_key=self.api_key
                         )
        return model

    def get_content(self, path: str, pdf: bool = False, txt: bool = False, xml:bool = False):
        if not (pdf ^ txt ^ xml): 
            raise ValueError("Either 'pdf','txt' or 'xml' must be True, not both or neither.")
        
        try:
            if pdf:
                content = extract_pdf_structure(path)
            elif txt:
                content= load_txt(path)
            elif xml:
                content = extract_xml_structure(path)
            
            if not content:
                raise ValueError("File content is empty or invalid.") 
            return content
        
        except Exception as e:
            print('Error in file:', path, 'Error message:', str(e))
            return "An error occurred during processing."
        
    def prompt(self,text):     
        examples = [
            {
                "input": "Employment of SAM as an HSL in inverted PSCs offers multiple unique properties. Firstly, SAM molecules can be easily modified by altering functional groups",
                "output": [0]
            },
            {
                "input": ".The HOMO energy levels for PY-series molecules are in the order of CBZ-B-PY(−5.75 eV) < POZ-PY (−5.39 eV) < DPA-B-PY (−5.32 eV)，with resepct PCE 22.3%, 21%, 25%",
                "output": [2,4]
            }
        ] 
        
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )   
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful literature paragraph classifier in the field of photovoltaic areas, specifically focusing on self-assembled monolayer (SAM) based perovskite solar cells. Please classify the content of the paper into the following categories:" 
                 "1. SAMs Preparation: Only classify under this category if the text provides explicit and complete synthesis details of SAM molecules. This must include the identification of precursors and a description of the reaction conditions (such as solvents, temperature, time, catalysts, etc.). Merely mentioning SAM names, their use, or general effects on device performance should not qualify."
                 "2. SAMs Structure and Property: Explicit SAM names with molecular chemical property values e.g., HOMO/LUMO(Conduction band/Valence band), dipole moment, Binding energy, etc. are provided. Merely mentioning SAM names are not qualified"
                 "3. Device Fabrication: Device fabrication parameters, device structure (layer1/layer2/layer3), annealing time and temperature, "
                    "device area or related procedural details. "
                 "4. Device Performance: Performance outcomes such as device efficiency, fill factor, stability tests, or other performance metrics."
                 " If the information does not fit into categories 1-4, return [0] for other information."
                 "If one part can be classified into multi-groups, return a list of integers with no extra characters e.g [1,3] that corresponds to the relevant category."),
                few_shot_prompt,
                ("human", "Now you have got the request and example, here is paper content:{input}"),
            ]
        )
        
        content=final_prompt.format(input=text)
        return content
    
    def outputparser(self, ai_message: AIMessage) -> str:
        return ai_message.content.strip()
    
    def call(self,path: str, pdf: bool = False, txt: bool = False, xml:bool = False, batch: bool=False):
        content=self.get_content(path,pdf,txt,xml)
        Title, DOI = content.get('Title'), content.get('DOI')
        label_data = {'Title':Title, 'DOI':DOI}
        list0, list1, list2, list3, list4 = [], [], [], [], []
        agent = self.llm()
        # Checking which sections contain nested data
        for section_key, section_values in content.items():
            if 'Abstract' in section_key:
                input = self.prompt(text = section_values)
                output = self.outputparser(agent.invoke(input))
                if '0' in map(str, output):
                    list0.append(section_key)
                if '1' in map(str, output):
                    list1.append(section_key)
                if '2' in map(str, output):
                    list2.append(section_key)
                if '3' in map(str, output):
                    list3.append(section_key)
                if '4' in map(str, output):
                    list4.append(section_key)
            elif isinstance(section_values, dict):  # Check if the section contains nested data
                for block_key, block_value in section_values.items():
                    if isinstance(block_value, dict):  # Proper way to check if it's a dictionary
                        for sub_block_key, sub_block_value in block_value.items():
                            input = self.prompt(text = sub_block_value)
                            output = self.outputparser(agent.invoke(input))
                            if '0' in map(str, output):
                                list0.append(f'{section_key}{[block_key]}{[sub_block_key]}')
                            if '1' in map(str, output):
                                list1.append(f'{section_key}{[block_key]}{[sub_block_key]}')
                            if '2' in map(str, output):
                                list2.append(f'{section_key}{[block_key]}{[sub_block_key]}')
                            if '3' in map(str, output):
                                list3.append(f'{section_key}{[block_key]}{[sub_block_key]}')
                            if '4' in map(str, output):
                                list4.append(f'{section_key}{[block_key]}{[sub_block_key]}')
                    else:
                        input = self.prompt(text = block_value)
                        output = self.outputparser(agent.invoke(input))
                        if '0' in map(str, output):
                            list0.append(f'{section_key}{[block_key]}')
                        if '1' in map(str, output):
                            list1.append(f'{section_key}{[block_key]}')
                        if '2' in map(str, output):
                            list2.append(f'{section_key}{[block_key]}')
                        if '3' in map(str, output):
                            list3.append(f'{section_key}{[block_key]}')
                        if '4' in map(str, output):
                            list4.append(f'{section_key}{[block_key]}')
        
        label_data['Inrelevant'] = list0                
        label_data['SAMs Preparation'] = list1
        label_data['SAMs Structure and Property'] = list2
        label_data['Device Fabrication'] = list3
        label_data['Device Performance'] = list4           
                    
        if batch:
            if not isinstance(label_data, dict):
                raise ValueError('Parsed data is not a dictionary.')
            else:
                df = pd.DataFrame([label_data])
                csv_path = self.csv_path
                # Check the existed file and if it is empty
                if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                    dataset = pd.read_csv(csv_path)
                    combined_df = pd.concat([dataset, df], ignore_index=True)
                else: # For empty cases
                    combined_df = df
                combined_df.to_csv(csv_path,index=False)
            return print('finished')
            
        else:
            return label_data