from sam_agent.tools.data_mining.utils import *

class DataMiner:
    def __init__(self):
        self.pdf_extractor = pdf_to_text
        self.xml_extractor = xml_to_text
        self.structure_pdf_extractor = extract_pdf_structure
        self.structure_xml_extractor = extract_xml_structure
        self.cleaner = remove_ref
        self.token_counter= count_tokens
        
######################################################################################        
    def pdf_to_content(self,pdf_path):
        # Extract text from the PDF and filter the reference part.
        content= self.pdf_extractor(pdf_path)
        clean_content= self.cleaner(content)
        total_tokens = self.token_counter(clean_content)
        return clean_content, total_tokens
    
    def content_txt(self,content, txt_path):
        # convert clearn text to X.txt
        with open(txt_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return txt_path
    
    def xml_to_content(self, xml_path):
        content = self.xml_extractor(xml_path)
        total_tokens = self.token_counter(content)
        return content, total_tokens
    
    def pdf_to_txt(self, pdf_path, txt_path):
        # Process the PDF and store the output in a text file
        clean_content, total_tokens = self.pdf_to_content(pdf_path)
        txt_path = self.content_txt(clean_content, txt_path)
        return txt_path
    
######################################################################################        
    # Structure output
    def pdf_to_dict(self, pdf_path):
        # Process the PDF and store the output in a dictionary
        content= self.structure_pdf_extractor(pdf_path)
        return content
    def xml_to_dict(self, xml_path):
        # Process the XML and store the output in a dictionary
        content = self.structure_xml_extractor(xml_path)
        return content
