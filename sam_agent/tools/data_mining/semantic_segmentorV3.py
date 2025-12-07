from typing import Optional
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain.output_parsers.json import SimpleJsonOutputParser
from sam_agent.tools.data_mining.utils import load_txt, get_pdf_image, extract_xml_structure, extract_pdf_structure
from sam_agent.tools.data_mining.data_miner import DataMiner
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
import os

class SemanticSegmentor:
    def __init__(self,open_ai_key = None, deepseek_key = None, llm_model: str = 'deepseek',
                 csv_path:str = 'classification_data',verbose=True):
        self.open_ai_key=open_ai_key
        self.deepseek_key = deepseek_key  # Assuming no deepseek key is used
        self.verbose=verbose
        self.csv_path = csv_path if csv_path else print('Try to provide csv path to save the results if you want to save the results.')
        self.llm_model = llm_model
        
    # Call the LLM model 
    def llm(self):
        engine = self.llm_model.lower()
        if engine == 'chatgpt':
            llm_agent = ChatOpenAI(model_name="gpt-4o-mini",
                                    temperature=0,
                                    max_tokens=None,
                                    timeout=None,
                                    base_url="https://api.chatanywhere.tech/v1",
                                    max_retries=3,
                                    api_key=self.open_ai_key)
            return llm_agent
        elif engine == 'deepseek':
            llm_agent = ChatOpenAI(model_name="deepseek-chat",
                                    temperature=0,
                                    max_tokens=None,
                                    timeout=None,
                                    base_url="https://api.deepseek.com",
                                    max_retries=3,
                                    api_key=self.deepseek_key)
            return llm_agent
        else:
            raise ValueError("Unsupported power model. Please choose 'chatgpt' or 'deepseek'.")
        
    def get_content(self, text_input: Optional[str], path: Optional[str], pdf: bool = False, text: bool = False, xml: bool = False):
        content = {}
        if not (pdf ^ text ^ xml): 
            raise ValueError("Either 'pdf','txt' or 'xml' must be True, not both or neither.")
        
        try:
            if pdf and path and not text_input:
                content = DataMiner().pdf_to_dict(path)
            elif text and text_input:
                content= text_input
            elif xml and path and not text_input:
                content = DataMiner().xml_to_dict(path)
            
            if not content:
                raise ValueError("File content is empty or invalid.") 
            return content
        
        except Exception as e:
            print('Error in file:', path, 'Error content message:', str(e))
            return "An error occurred during processing."
        
    def prompt(self,text):     
        # Few-shot examples for the classifier

        examples = [
            {
                "input": {
                    "Abstract": " In particular, after 30 days of monitoring for PSC systems, the PCEs of CNph@POZ-BT-PY-modified devices main-tain 87% and 91% of their original eﬃciencies for perovskites with bandgaps of 1.55 eV and 1.73 eV, respectively.",
                    "Figures": {'Fig. 1': 'Figure 1. a) structure of PSCs we fabricated. b) molecular structure of BA-CF 3 , BA-CH 3 and NA-CF 3 . c) HOMO and LUMO of BA-CF 3 . d) BA-CF 3 SAMs mechanism of regulated SnO 2 /MAPbI 3 interface.', "Fig. 2": "..."}
                },
                "output": {
                    "Materials Structure and Property": ["Abstract", "Figures['Fig. 1']"],
                    "Device Performance": ["Abstract"]
                }
            },
            {
                "input": {
                    "batch0": {
                    "block224":"Table S1. Experimentally measured and theoretically calculated energy levels of Br-2EPO.",
                    "block225": "(eV)",
                    "block226": "LUMOe",
                    "block227": "(eV)",
                    "block228": "HOMO",
                    "block229": "(eV)",
                    "block230": "(eV) \nBr-2EPO  \n-5.5 \n-2.7 \n3.5 \n-5.2 \n-1.88 \n3.49 \nBr-2EPT \n-5.6 \n-2.41 \n- \n-5.56",
                    }
                },
                "output":{
                    "Materials Structure and Property": ["batch0['block224']","batch0['block225']","batch0['block225']","batch0['block226']","batch0['block227']",
                                                         "batch0['block228']","batch0['block229']","batch0['block230']"],
                }
                    
            },
            {
                "input": {
                    "1. Introduction": {
                        "block0": "Inverted PSCs were fabricated with a device architecture of MgF2/glass/ITO/SAM/perovskite/PCBM/BCP/Ag to evaluate the improvement upon employing the SAM@pseudo-planar monolayer strategy for enhancing photovoltaic performance.",
                        "block1": "synthetic route to zwitter ions salts. a)Pd(OAc)2, xantphos, NaOtBu, toluene, 110°C,16h. b)Pd(PPH3)2Cl2, K2CO3,toluene/EtOH/H2O,110°C,16h.",
                        "block2": "Particularly note worthy is the in-corporation of these sulfonate-based D–𝜋–A SAMs, coated atop CNphSAMviaatwo-stepapproach."
                    }
                },
                "output": {
                    "Materials Preparation": ["1. Introduction['block1']"],
                    "Device Fabrication": ["1. Introduction['block0']","Figures['Fig. 4']"]
                }
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
                ("system", "You are a helpful literature paragraph classifier in the field of photovoltaic areas, "
                    "specifically focusing on self-assembled monolayer (SAM) based perovskite solar cells. "
                    "Please classify the content of the paper into the following 4 categories following the attached creteria: "
                    "1. Materials Preparation: Explicit and complete synthesis details of SAM molecules are provided. Details must include "
                    "the identification of precursors and a description of the reaction conditions (such as solvents, temperature, time, catalysts, etc.). "
                    "Merely mentioning SAM names, their use, etc. without details are not qualified. "
                    "2. Materials Structure and Property: Explicit SAM names with molecular chemical property values and Perovskite material properties e.g., HOMO/LUMO(Conduction band/Valence band), dipole moment, Binding energy, etc. are provided. Merely mentioning SAM names are not qualified"
                    "3. Device Fabrication: Device fabrication parameters, device structure (layer1/layer2/layer3), layer compostion,thickness, annealing time and temperature, etc. "
                    "related procedural details. "
                    "4. Device Performance: Performance outcomes such as photovoltaic_parameters like device efficiency, fill factor and stability tests, or other performance metrics. "
                    "If no relevant information is found for the above categories, just skip it. Do not add any catagories."
                    "**Pay more attention to table and figure caption, it only occupy one block, follwing blocks might contain important information**"
                    "If part contains information with repect to each catagories. Return results by referencing the corresponding keys from the sections, tables, or figure captions provided in the paper content.  The output format is valid JSON "
                    "using double quotes."
                    "Do Not create new catagories, just based on above four categories. "),
                few_shot_prompt,
                ("human", "Now you have got the request and example, here is paper content:{input}")
            ]
        )
        
        content=final_prompt.format(input=text)
        return content
    
    
    def call(self,text_input: Optional[str], path: Optional[str], pdf: bool = False, text: bool = False, xml:bool = False, dataset: bool=False):
        content=self.get_content(text_input,path,pdf,text,xml)
        if not isinstance(content, dict):
            raise ValueError("Content is not a dictionary.")
        if path and (pdf or xml):
            Title, DOI = content.get('Title'), content.get('DOI')
            label_data = {"Materials Preparation": [], "Materials Structure and Property": [], "Device Fabrication": [], "Device Performance": [], "Irrelevant": [],"DOI":DOI,"Title":Title }
        elif text_input and text:
            label_data = {"Materials Preparation": [], "Materials Structure and Property": [], "Device Fabrication": [], "Device Performance": [], "Irrelevant": [], "path": path}
        else:
            raise ValueError("Invalid input. Please provide either a file path or text input.")
        json_parser = SimpleJsonOutputParser()
        agent = self.llm()
        # Checking which sections contain nested data
        for section_key, section_values in content.items():
            if 'Abstract' in section_key:
                input = self.prompt(text = {section_key: section_values})
                output = agent.invoke(input)
                output = json_parser.invoke(output)
                for output_key, output_value in output.items(): 
                    # Initialize the key if it doesn't exist
                    if output_key in label_data:
                        label_data[output_key] += output_value
                    else:
                        label_data[output_key] = output_value
            elif isinstance(section_values, dict): # Check if the section contains nested data
                if not section_values:
                    label_data["Irrelevant"] += [section_key]
                    
                else:    
                    input = self.prompt(text = {section_key: section_values})
                    output = agent.invoke(input)
                    output = json_parser.invoke(output)
                    for output_key, output_value in output.items(): 
                        # Initialize the key if it doesn't exist
                        if output_key in label_data:
                            label_data[output_key] += output_value
                        else:
                            label_data[output_key] = output_value
                
                    
        if dataset:
            if not isinstance(output, dict):
                raise ValueError('Parsed data is not a dictionary.')
            else:
                df = pd.DataFrame([label_data])
                csv_path = self.csv_path
                # Check the existed file and if it is empty
                if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                    dataset_new = pd.read_csv(csv_path)
                    combined_df = pd.concat([dataset_new, df], ignore_index=True)
                else: # For empty cases
                    combined_df = df
                combined_df.to_csv(csv_path,index=False)
            return print('finished')
            
        else:
            return label_data