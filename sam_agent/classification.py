from sam_agent.tools.data_mining.utils import extract_XML_blocks, extract_pdf_structure
from sam_agent.tools.data_mining.semantic_segmentorV3 import SemanticSegmentor
import os
from tqdm import tqdm
import fitz  # PyMuPDF
import re

def read_files_list(path):
    file_list = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)  # Create the full path
        if os.path.isfile(file_path):
            if file_name.lower().endswith('txt'):
                file_list.append(file_path)
            elif file_name.lower().endswith('pdf'):
                file_list.append(file_path)
            elif file_name.lower().endswith('xml'):
                file_list.append(file_path)
    return file_list


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



def execute():
    #pdf_list = read_files_list('AI_agents/Elsevier')
    #xml_list=read_files_list('AI_agents/Elsevier')
    file_list = read_files_list('AI_agents/Elsevier')
    #print('Total number of files:','xmls',len(xml_list), "pdfs", len(pdf_list))
    #file_list=pdf_list # + xml_list
    print('Total number of files to process:', len(file_list))
    os.environ['OPENAI_API_KEY']="key"
    os.environ['DeepSeek_API_KEY'] = "key"
    classifier=SemanticSegmentor(open_ai_key=os.environ['OPENAI_API_KEY'],deepseek_key=os.environ['DeepSeek_API_KEY'], llm_model='deepseek',verbose=True,csv_path="paper_classification_Elsevier_XML.csv")
    error_list = []
    processed_list = []
    for file in tqdm(file_list):
        '''
        if file.lower().endswith('_si.pdf'):
            try:
                # Extract text from the PDF file
                batchs = extract_block_pdf(file)
                classifier.call(text_input=batchs, path=file, pdf=False, text=True, dataset=True)
                processed_list.append(file)
            except Exception as e:
                error_list.append(file)
                print(f"Error reading file {file}: {e}")

        '''
        if file.lower().endswith(".pdf"):
            continue
            '''
            try:
                classifier.call(path=file, pdf=True, text=False, xml=False,dataset=True)
                processed_list.append(file)
            except Exception as e:
                error_list.append(file)
                print(f"Error reading file {file}: {e}")
                '''
        elif file.lower().endswith('.xml'):
            try:
                                # Extract text from the PDF file
                batchs = extract_XML_blocks(file)
                classifier.call(text_input= batchs,path = file, pdf = False, text=True,dataset = True)
                processed_list.append(file)
            except Exception as e:
                error_list.append(file)
                print(f"Error reading file {file}: {e}")

    print('Processed files:', len(processed_list))
                
if __name__ == "__main__":
    execute()