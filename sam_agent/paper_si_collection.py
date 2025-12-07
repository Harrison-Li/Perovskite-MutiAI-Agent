from sam_agent.tools.data_mining.data_extractor_cat import DataExtractor
import os
from tqdm import tqdm
from sam_agent.tools.data_mining.utils import extract_XML_blocks, extract_pdf_structure, extract_xml_structure
import pandas as pd
import ast
import fitz
import re

def read_files_list(path):
    file_list = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)  # Create the full path
        if os.path.isfile(file_path):
            if file_name.lower().endswith('.txt'):
                file_list.append(file_path)
            elif file_name.lower().endswith('.pdf'):
                file_list.append(file_path)
            elif file_name.lower().endswith('.xml'):
                file_list.append(file_path)
    return file_list

def convert_doi_to_doc(doi):
    if not pd.isna(doi):
        doi = str(doi).strip()
        if 'doi.org' in doi:
            doi = doi.replace('doi.org/', '').replace('/','_').replace("']",'').replace("['",'')
        elif '[' and ']' in doi:
            doi = doi.replace('/','_').replace("']",'').replace("['",'')
        else:
            doi = doi.replace('/','_') # for xml style
        return doi
    else:
        return None
    
def extract_block_pdf(pdf_path):
    # Define a list of regular expression patterns for references
    doc = fitz.open(pdf_path)
    batchs = {}
    start = True
    block_idx = 0
    page_num = 0
    
    for page in doc:
        batch_idx = page_num // 4
        
        # Initialize this batch's block_content if it doesn't exist
        if f"batch{batch_idx}" not in batchs:
            batchs[f"batch{batch_idx}"] = {}
            
        blocks = page.get_text("blocks")
        for block in blocks:
            block_text = block[4].strip()
            if re.fullmatch(r"references", block_text, re.IGNORECASE) and page_num > 3:
                start = False
            # Check if the block contains text
            if isinstance(block_text, str) and block_text.strip() and start:
                batchs[f"batch{batch_idx}"][f"block{block_idx}"] = block_text.strip()
                block_idx += 1
        
        page_num += 1
    return batchs

def get_info(raw_string,content):
    aggregated_content = {}
    figures = {}
    input = ast.literal_eval(raw_string)
    for idx, key in enumerate(input):
        if 'block' in key or 'Tables' in key:
        # Split the string at the ['
            part = key.split("['")
            if len(part) == 2:
                section = part[0]
                # Remove the trailing ']
                identifier = part[1].replace("']", "")
                aggregated_content[f'Info {idx}'] = content[section][identifier]
            elif len(part) == 3:
                section = part[0]
                identifier = part[1].replace("']", "")
                sub_identifier = part[2].replace("']", "")
                aggregated_content[f'Info {idx}'] = content[section][identifier][sub_identifier]
        elif 'Figures' in key:
            part = key.split("['")
            if len(part) == 2:
                section = part[0]
                identifier = part[1].replace("']", "")
                number = identifier.replace("Fig. ",'')
                figures[f'Figure{number}'] = content[section][identifier]
            else:
                print ('Wrong format for figure key:', key)
            
        else:
            aggregated_content[f'Info {idx}'] = content[key]
    return aggregated_content, figures


if __name__ == "__main__":
    pdf_list = (
        read_files_list('AI_agents/Elsevier')
    )
    print(len(pdf_list), 'pdf', "in the folder")

    os.environ['OPENAI_API_KEY'] = "key"
    os.environ['DeepSeek_API_KEY'] = "key"
    extractor = DataExtractor(open_ai_key=os.environ['OPENAI_API_KEY'], deepseek_key=os.environ['DeepSeek_API_KEY'],
                              verbose=True, llm_model='deepseek',mutimodal= False, stored_path="Elsevier_performance_521.csv")
    csv_path = extractor.csv_path

    classification = pd.read_csv('/data2/b_li/Test codes/agent_dataset/Elsevier_merged_classification.csv')
    si_material_preparation = classification['Materials Preparation_si']
    si_material_property = classification['Materials Structure and Property_si']
    si_device_fabrication = classification['Device Fabrication_si']
    si_device_performance = classification['Device Performance_si']
    paper_material_preparation = classification['Materials Preparation_paper']
    paper_material_property = classification['Materials Structure and Property_paper']
    paper_device_fabrication = classification['Device Fabrication_paper']
    paper_device_performance = classification['Device Performance_paper']
    DOI = classification['DOI']
    
    path_list = [i.replace('/','_') for i in DOI if pd.notna(i)]
    for idx, doc_path in tqdm(enumerate(path_list)):
        content = {}
        # Find the actual file paths in the scanned directories
        si_path = None
        paper_path = None
        for pdf_path in pdf_list:
            if doc_path + '_si.pdf' in pdf_path or doc_path + '/si.pdf' in pdf_path:
                si_path = pdf_path
            elif doc_path + '.xml' in pdf_path and not (doc_path + '_si.pdf' in pdf_path or doc_path + '/si.pdf' in pdf_path):
                paper_path = pdf_path
        file_path = ""
        try:
            if si_path in pdf_list:
                si = extract_block_pdf(si_path)
                si_result, si_figures= get_info(si_device_performance[idx], si)
                content["Support Information"] = si_result
                file_path += si_path
            if paper_path in pdf_list:
                paper = extract_XML_blocks(paper_path)
                paper_result, paper_figures= get_info(paper_device_performance[idx], paper)
                content["Paper content"] = paper_result
                file_path += paper_path
            # Combine results
            df = extractor.data_collection(text_input = content , path = None ,dataset= False ,pdf= False, text=True,xml=False,mode = 'device performance')
            df['DOI'] = doc_path if doc_path else None
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                dataset = pd.read_csv(csv_path)
                combined_df = pd.concat([dataset, df], ignore_index=True)
            else:
                combined_df = df
                        
            combined_df.to_csv(csv_path,index=False)
            print(f"Processed file {file_path} for DOI {doc_path}")
                
        except Exception as e:
                print(f"Error processing file {file_path} for DOI {doc_path}: {e}")
                continue
            