import re
from typing import Optional
from xmlrpc.client import Boolean
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain.output_parsers.json import SimpleJsonOutputParser
from sam_agent.tools.data_mining.data_miner import DataMiner
from sam_agent.tools.data_mining.utils import convert_pii_link, load_txt, get_pdf_image, extract_xml_structure
from sam_agent.tools.data_mining.prompt import preparation_prompt,property_prompt,performance_prompt,fabrication_prompt
import os
import base64
import httpx




class DataExtractor:
    def __init__(self,open_ai_key = None, deepseek_key = None, stored_path = 'collect_data.csv', llm_model:str = 'chatgpt',mutimodal:bool= False,verbose=True):
        self.open_ai_key=open_ai_key
        self.deepseek_key= deepseek_key
        self.verbose=verbose
        self.csv_path = stored_path
        self.llm_model = llm_model
        self.mutimodal = mutimodal

        
        
    # Call the LLM model 
    def llm(self):
        engine = self.llm_model.lower()
        if engine == 'chatgpt':
            llm_agent = ChatOpenAI(model_name="gpt-4.1-mini",  #"gpt-4o",
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
        
    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    
    def get_content(self,text_input: Optional[str], path: Optional[str], pdf: bool = False, text: bool = False, xml:bool = False):
        if not (pdf ^ text ^ xml): 
            raise ValueError("Either 'pdf','txt' or 'xml' must be True, not both or neither.")
        
        content = None
        try:
            if pdf:
                content = DataMiner().pdf_to_dict(path)
            elif text:
                content= text_input
            elif xml:
                content = DataMiner().xml_to_dict(path)
            
            if not content:
                raise ValueError("File content is empty or invalid.") 
            return content
        
        except Exception as e:
            print('Error in file:', path, 'Error content message:', str(e))
            return "An error occurred during processing."
    
    def get_images(self, path: str, refs: str):
        if not (path.endswith('.pdf') or path.endswith('.xml')):
            raise ValueError("Only 'pdf' and 'xml' can be True.")
        try:
            if path.endswith('.pdf') and refs:
                images_path = get_pdf_image(path,output_dir='extracted_images')
                image_list = []
                for i in images_path:
                    for ref in refs:
                        if ref in i:
                            image_list.append(i)

                base64_images = [self.encode_image(i) for i in image_list]
                return base64_images

                            
                
            elif path.endswith('.xml') and refs:
                paper = extract_xml_structure(path)
                pii_links = []
                for ref in refs:
                    pii = paper['Figures'][ref]
                    pii_links.append(pii)
                    
                image_urls = [convert_pii_link(i) for i in pii_links] # Here ref is pii link
                base64_images = [base64.b64encode(httpx.get(i).content).decode("utf-8") for i in image_urls]
                return base64_images
            else:
                print('Please give which figure you want to deal with')
            
        
            
                                
        except Exception as e:
            print('Error in file:', path, 'Error image message:', str(e))
            return "An error occurred during processing."
                     
                  
    
    
    # A function that can deal with batches of files in one time and collect a organized dataset (etc. csv, xlsx)
    def data_collection(self, path: Optional[str], text_input = None, figures_ref = None, dataset: bool = False,  
                        pdf: bool = False, text: bool = False, xml:bool = False, mode: str = 'preparation'):
        print('Current power model is:', self.llm_model)
        print("Model initition finished, please choose the catagory you want to extract in data_collection(mode=''): \n 1. 'material preparation' \n 2. 'material property' \n 3. 'device performance' \n 4. 'device fabrication' ")
        
        if self.mutimodal:
            print('Mutimodal is on, DataExtractor can process with text and images')
        else:
            print('Multimodal is off, DataExtractor only process with text')
            
        data_tb = {}
        content=self.get_content(text_input,path,pdf,text,xml)
        if self.mutimodal:  
            if figures_ref is None:
                raise ValueError("Please provide the references of the figures.")
            if path is None:
                raise ValueError("Please provide the path of the file.")
            encoded_figures = self.get_images(path=path, refs=figures_ref)
            if not encoded_figures:
                raise ValueError("No images found for the provided references.")
        else:
            encoded_figures = None
        
        json_parser = SimpleJsonOutputParser()
            
        agent = self.llm()
        if mode == 'material preparation':
            input = preparation_prompt(content,image_list = encoded_figures)
        elif mode == 'material property':
            input = property_prompt(content, encoded_image_list = encoded_figures)
        elif mode == 'device performance':
            input = performance_prompt(content, image_list = encoded_figures)
        elif mode == 'device fabrication':
            input = fabrication_prompt(content, image_list = encoded_figures)
        else:
            raise ValueError("Invalid mode. Please choose 'material preparation', 'material property', 'device performance' or 'device fabrication'.")

        data_tb = agent.invoke(input)
        data_tb = json_parser.invoke(data_tb)
        # print("Raw output from agent:", data_tb)
            
        # Ensure data_tb is a list of dictionaries
        if not isinstance(data_tb, dict):
            raise ValueError("Parsed data is not a dictionaries.")
        
        df = pd.DataFrame([data_tb])
        
        if dataset:
            # Check the existed file and if it is empty
            csv_path = self.csv_path
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                dataset_new = pd.read_csv(csv_path)
                combined_df = pd.concat([dataset_new, df], ignore_index=True)
            else:
                # If file does not exist or empty, create a new one
                combined_df = df
            # Write the combined DataFrame to a CSV file
            combined_df.to_csv(csv_path,index=False)
            return print('finished')
        else:
            return df