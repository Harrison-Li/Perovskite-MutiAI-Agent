import requests
import PyPDF2
import pymupdf
import pandas as pd
import numpy as np
import tiktoken
import re
import xml.etree.ElementTree as ET
import os




############################################################################################################
# Count how many tokens cost for the input
def count_tokens(text):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


# Remove the reference part to save cost
def remove_ref(pdf_text):
    """This function removes reference section from a given PDF text. It uses regular expressions to find the index of the words to be filtered out."""
    # Regular expression pattern for the words to be filtered out
    pattern = r'(REFERENCES|BIBLIOGRAPHY|Acknowledgements|ACKNOWLEDGMENT)'
    match = re.search(pattern, pdf_text,re.IGNORECASE) # Makes it case-insensitive (References is treated as same to REFERENCES)

    if match:
        # If a match is found, remove everything after the match
        start_index = match.start()
        clean_text = pdf_text[:start_index].strip()
    else:
        # Define a list of regular expression patterns for references
        reference_patterns = [
            '\[[\d\w]{1,3}\].+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5};','\([\d\w]{1,3}\).+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5},',
            '\([\d\w]{1,3}\).+?[\d]{3,5},','\[[\d\w]{1,3}\].+?[\d]{3,5}','[\d\w]{1,3}\).+?[\d]{3,5}\.','[\d\w]{1,3}\).+?[\d]{3,5}',
            '\([\d\w]{1,3}\).+?[\d]{3,5}','^[\w\d,\.– ;)-]+$',
        ]

        # Find and remove matches with the first eight patterns
        for pattern in reference_patterns[:8]:
            matches = re.findall(pattern, pdf_text, flags=re.S)
            pdf_text = re.sub(pattern, '', pdf_text) if len(matches) > 500 and matches.count('.') < 2 and matches.count(',') < 2 and not matches[-1].isdigit() else pdf_text

        # Split the text into lines
        lines = pdf_text.split('\n')

        # Strip each line and remove matches with the last two patterns
        for i, line in enumerate(lines):
            lines[i] = line.strip()
            for pattern in reference_patterns[7:]:
                matches = re.findall(pattern, lines[i])
                lines[i] = re.sub(pattern, '', lines[i]) if len(matches) > 500 and len(re.findall('\d', matches)) < 8 and len(set(matches)) > 10 and matches.count(',') < 2 and len(matches) > 20 else lines[i]

        # Join the lines back together, excluding any empty lines
        clean_text = '\n'.join([line for line in lines if line])

    return clean_text

#################################################################
# Section for extract text from the XML file

def extract_text(element, full_text=""):
    if element is not None:
        if element.text:
            if 'label' in element.tag:
                full_text += '\n' + element.text.strip() + '.'
            elif 'section-title' in element.tag:
                full_text += element.text.strip() + '\n'
            else:
                full_text += element.text.strip()
        
    # Iterate over all child elements of this element
    for child in element:
        full_text = extract_text(child, full_text)  # Recursively process each child
        if child.tail:
            full_text += child.tail.strip()
    
    return full_text

# Extract whole sections from the XML file and concatenate them.
def extract_text2(sections, full_text=""):
    namespaces = {'ce': "http://www.elsevier.com/xml/common/dtd"}
    content = {}
    section_list = [section for section in sections]
    
    for section in section_list:
        section_title = ""
        text = ""
        subsections_content = {}
        
        if section is not None:
            section_title_element = section.find('.//ce:section-title', namespaces)
            if section_title_element is not None and section_title_element.text:
                section_title = section_title_element.text.strip()
            
            para_elements = section.findall('.//ce:para', namespaces)
            for para in para_elements:
                text += " ".join([s.strip() for s in para.itertext() if s])
            
            subsection_list = section.findall('.//ce:section', namespaces)
            for subsection in subsection_list:
                subsection_title = ""
                subsection_text = ""
                subsection_title_element = subsection.find('.//ce:section-title', namespaces)
                if subsection_title_element is not None and subsection_title_element.text:
                    subsection_title = subsection_title_element.text.strip()
                
                for para in  subsection.iterfind('.//ce:para', namespaces):
                    subsection_text += " ".join([s.strip() for s in para.itertext() if s])
                
                if subsection_title:
                    subsections_content[subsection_title] = subsection_text
            
            if section_title:
                if subsections_content:
                    content[section_title] = subsections_content
                else:
                    content[section_title] = text
    
    return content

# Extract sections from the XML file and seperate them into blocks.
def extract_XML_blocks(xml_path):
    # Load the XML file 
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Get different parts of the XML file
    title = root.find('.//{http://purl.org/dc/elements/1.1/}title').text
    abstract = root.find('.//{http://purl.org/dc/elements/1.1/}description').text
    # article = root.find('.//{http://www.elsevier.com/xml/ja/dtd}article')
    doi = root.find('.//{http://www.elsevier.com/xml/xocs/dtd}doi').text
    # body = article.find('.//{http://www.elsevier.com/xml/ja/dtd}body')
    sections= root.find('.//{http://www.elsevier.com/xml/common/dtd}sections') # 4 subsections inside section
    floats = root.find('.//{http://www.elsevier.com/xml/common/dtd}floats')
    table = floats.findall('.//{http://www.elsevier.com/xml/common/dtd}table')
    figures = floats.findall('.//{http://www.elsevier.com/xml/common/dtd}figure')
    
    namespaces = {'ce': "http://www.elsevier.com/xml/common/dtd"}
    content = {}
    content['Abstract'] = abstract.strip()
    content['Tables'] = extract_table(table)
    section_list = [section for section in sections]
    
    for section in section_list:
        section_title = ""
        blocks={}
        subsections_content = {}
        
        if section is not None:
            section_title_element = section.find('.//ce:section-title', namespaces)
            if section_title_element is not None and section_title_element.text:
                section_title = section_title_element.text.strip()
            
            para_elements = section.findall('.//ce:para', namespaces)
            for index, para in enumerate(para_elements):
                blocks[f"block{index}"] = "".join([s.strip() for s in para.itertext() if s])
            
            subsection_list = section.findall('.//ce:section', namespaces)
            for subsection in subsection_list:
                sub_blocks={}
                subsection_title = ""
                subsection_title_element = subsection.find('.//ce:section-title', namespaces)
                if subsection_title_element is not None and subsection_title_element.text:
                    subsection_title = subsection_title_element.text.strip()
                
                for sub_index, para in  enumerate(subsection.iterfind('.//ce:para', namespaces)):
                    sub_blocks[f"block{sub_index}"] = "".join([s.strip() for s in para.itertext() if s])
                
                if subsection_title:
                    subsections_content[subsection_title] = sub_blocks
            
            if section_title:
                if subsections_content:
                    content[section_title] = subsections_content
                else:
                    content[section_title] = blocks
    
    return content







####################################################################################################
# Section for extract tables and figures from the XML file

def extract_table(tables):
    namespaces = {'ce':"http://www.elsevier.com/xml/common/dtd",'cals':'http://www.elsevier.com/xml/common/cals/dtd'}
    total_data_tb = {}
    for table in tables:
        if table is not None:
            table_content = ''
            for element in table:
                if 'label' in element.tag:
                    label = element.text
                elif 'caption' in element.tag:
                    caption = "".join(element.itertext()).strip()
                elif 'tgroup' in element.tag:
                    for part in element:
                        # Extract the columns of the table
                        if 'thead' in part.tag and part.iterfind('.//ce:entry', namespaces):
                            for entry in part.iterfind('.//ce:entry', namespaces):
                                if len(list(entry.iter())) > 1:
                                        table_content +="".join([s.strip() for s in entry.itertext() if s])
                                        table_content+= ' '                                                          
                                else:
                                    if entry.text:
                                        table_content += entry.text.strip()
                                        table_content+= ' '
                            table_content+= '\n'
                        elif 'tbody' in part.tag:
                            for row in part.iterfind('.//cals:row', namespaces):
                                for entry in row.iterfind('.//ce:entry', namespaces):
                                    # Extract the rows of the table
                                    if len(list(entry.iter())) > 1:
                                        table_content +="".join([s.strip() for s in entry.itertext() if s]) + '\n'
                                        #table_content+= ''                                                          
                                    else:
                                        if entry.text:
                                            table_content += entry.text.strip() + '\n'
                                            #table_content+= ''

                        
        full_table = f"{label.strip()}\n{caption}\n{table_content.strip()}"
        total_data_tb[label] = full_table
    return total_data_tb


def extract_figures(figures):
    namespaces = {'ce':"http://www.elsevier.com/xml/common/dtd", 'xlink':"http://www.w3.org/1999/xlink"}
    fig_dict = {}
    for fig in figures:
        label_text = ""
        caption_text = ""
        link_url = ""
        for label in fig.iterfind('.//ce:label', namespaces):
            label_text += label.text.strip()
        for caption in fig.iterfind('.//ce:caption', namespaces):
            caption_text += "".join([s.strip() for s in caption.itertext() if s])
        fig_cap = label_text + caption_text
        
        for link in fig.iterfind('.//ce:link', namespaces):
            if link is not None:
                image_url = link.get('{http://www.w3.org/1999/xlink}href')
            else:
                image_url = None
                
        
        fig_dict[label_text] = {'caption':caption_text, 'link':image_url}
        
    return fig_dict










        
    

############################################################################################################

# Extract texts from the PDF file
def pdf_to_text(pdf_path):
    """
    Extract text from a PDF file
    :param pdf_path: path to the PDF file
    :return: text extracted from the PDF file
    """
    text = ""  # Initialize the text variable
    with open(pdf_path, 'rb') as pdf_literature:  # open it in binary mode "rb"
        pdf_reader = PyPDF2.PdfReader(pdf_literature)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        text_clean = remove_ref(text)  # Remove the reference part
    return text_clean

def get_pdf_image(path, output_dir, scale_factor=2.5):
    figure_caption_pattern = re.compile(r'^(Figure|Fig\.|图)\s?\d*[\.:]?', re.IGNORECASE)
    doc = pymupdf.open(path)  
    fig_num = 0  # record the number of figures in file
    os.makedirs(output_dir, exist_ok=True)
    # Remove any existing images in the output directory (renew images)
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):  # Delete files
            os.remove(file_path)
            
    # Use the scale_factor parameter to get higer resolution image
    zoom = pymupdf.Matrix(scale_factor, scale_factor)
     
    saved_images = []    
    for page_num, page in enumerate(doc):
        dict = page.get_text("dict")
        width, height = dict['width'], dict['height']  # record the size of the paper
        blocks = dict["blocks"]
        fig_bbox = None  # store the last found figure caption block's bounding box
        
        for block in blocks:
            upper_found = False
            lower_found = False
            fig_block = False
            
            if 'lines' not in block:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    text = span['text']
                    font_size = span['size']
                    if 4.9 <= font_size < 8.1 and figure_caption_pattern.match(text):
                        fig_block = True
                        fig_num += 1  # increase figure number if a figure caption appears
                        break
                if fig_block:
                    break
                                    
            if fig_block:
                x0, y0, x1, y1 = block['bbox']
                fig_bbox = block['bbox']  # save the figure caption block's bbox
                # Set flags based on vertical position
                if y1 > 4 * height / 5:
                    lower_found = True
                else:
                    upper_found = True
            
            # If a figure caption was found, determine cropping regions and extract figures
            if fig_bbox and (upper_found or lower_found):
                x0, y0, x1, y1 = fig_bbox
                
                # If the caption is in the upper part
                if upper_found:
                    # this if loop try to minimize the influence of the caption length on the figure
                    if x1 < width/2:
                        new_crop = pymupdf.Rect(x0, height * 1/12, x1, y1)
                    elif x1 > width * 3/4:
                        new_crop = pymupdf.Rect(x0, height * 1/12, x1, y1)
                    else:
                        new_crop = pymupdf.Rect(x0, height * 1/12, width, y1)
                    pix = page.get_pixmap(matrix=zoom, clip=new_crop)
                    out_path = os.path.join(output_dir, f"Figure{fig_num}.jpg")
                    saved_images.append(out_path)
                    pix.save(out_path)
                # If the caption is in the lower part
                elif lower_found:
                    if x1 < width/2:
                        new_crop = pymupdf.Rect(x0, height * 1/12, x1, y1)
                    elif x1 > width * 3/4:
                        new_crop = pymupdf.Rect(x0, height * 1/12, x1, y1)
                    else:
                        new_crop = pymupdf.Rect(x0, height * 1/12, width, y1)
                    new_crop = pymupdf.Rect(x0, y1 - height * 1/2, x1, y1)
                    pix = page.get_pixmap(matrix=zoom, clip=new_crop)
                    out_path = os.path.join(output_dir, f"Figure{fig_num}.jpg")
                    saved_images.append(out_path)
                    pix.save(out_path)
    return saved_images



# Not work for agent, pass encoded image will be my final choice, this function is just for backup and display image.
def get_XML_image(xml_path, label, api_key):
    # Load XML file
    header = {"X-ELS-APIKey": api_key}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Find the url of the image
    object = root.findall('.//{http://www.elsevier.com/xml/svapi/article/dtd}object')
    for obj in object:
        if 'image/jpeg' in obj.get('mimetype') and obj.get('category') == ('standard' or 'high') and obj.get('ref') == label:
            url = obj.text.strip() if obj.text else None
            try:
                response = requests.get(url, headers = header)
                if response.status_code == 200:
                    image_data = response.url
                    return image_data
                else:
                    print(f"Error: Failed to get. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error: {e} for {label}")
                
            
            
            
    
    





# Directly load the the text for txt if txt is aviailable, useless function just for backup.
def load_txt(file_path):
    """
    Load the text from a txt file.
    
    file_path: Path to the txt file to be read (default is "data.txt").
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
def xml_to_text(xml_path):
    # Load the XML file 
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Get different parts of the XML file
    title = root.find('.//{http://purl.org/dc/elements/1.1/}title').text
    abstract = root.find('.//{http://purl.org/dc/elements/1.1/}description').text
    # serial_item = root.find('.//{http://www.elsevier.com/xml/xocs/dtd}serial-item')
    # article = root.find('.//{http://www.elsevier.com/xml/ja/dtd}article')
    doi = root.find('.//{http://www.elsevier.com/xml/xocs/dtd}doi').text
    # body = article.find('.//{http://www.elsevier.com/xml/ja/dtd}body')
    sections= root.find('.//{http://www.elsevier.com/xml/common/dtd}sections') # 4 subsections inside section
    table = root.findall('.//{http://www.elsevier.com/xml/common/dtd}table')
    
    # Combine different parts of the paper
    title = 'Title'+ '\n'+title.strip() + '\n'
    doi = 'DOI' + '\n' + doi.strip() + '\n'
    abstract = 'Abstract' + '\n' + abstract.strip() 
    full_text = extract_text(sections) +'\n'
    tables = extract_table(table)
    
    full_content = title + doi + abstract + full_text
    for table in tables:
        full_content += table + '\n'
    
    return full_content
    
########################################################################################################################
# Extract content from XML -------> Only Elsevier publisher !!!!!!
def extract_xml_structure(xml_path):
    # Load the XML file 
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Get different parts of the XML file
    title = root.find('.//{http://purl.org/dc/elements/1.1/}title').text
    abstract = root.find('.//{http://purl.org/dc/elements/1.1/}description').text
    # article = root.find('.//{http://www.elsevier.com/xml/ja/dtd}article')
    doi = root.find('.//{http://www.elsevier.com/xml/xocs/dtd}doi').text
    # body = article.find('.//{http://www.elsevier.com/xml/ja/dtd}body')
    sections= root.find('.//{http://www.elsevier.com/xml/common/dtd}sections') # 4 subsections inside section
    floats = root.find('.//{http://www.elsevier.com/xml/common/dtd}floats')
    table = floats.findall('.//{http://www.elsevier.com/xml/common/dtd}table')
    figures = floats.findall('.//{http://www.elsevier.com/xml/common/dtd}figure')
    # Combine different parts of the paper
    paper = {
        "Title": title.strip(),
        "DOI": doi.strip(),
        "Abstract": abstract.strip(),
        **extract_blocks(sections),  # unpack sections into dict
        "Tables": extract_table(table),
        "Figures": extract_figures(figures)
    }

    
    return paper


# Extract content from PDF -------> Only WEILY publisher !!!!!!
def extract_pdf_structure(pdf_path):
    doc = pymupdf.open(pdf_path)
    paper_title = None
    abstract = None
    sections = {}
    DOI = ""
    current_section = None
    exclude_section = ['Supporting Information', 'Acknowledgements', 'Conflict of Interest', 'Data Availability Statement', 'Keywords',
                       'Author Contributions','Conﬂict of Interest']
    figure_captions = {}  # Global dictionary to keep all figures
    table = {}
    global_fig_label = 0  # Global counter for figure captions
    global_tab_label = 0

    # Patterns
    heading_pattern = re.compile(r"^(?:\d+\.|[A-Z][a-z]+).*")  # Matches numbered or capitalized headings
    doi_pattern = re.compile(r"(?:DOI:\s*|https?://doi\.org/|doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")  # DOI regex pattern
    url_pattern = re.compile(
        r"((https?://)?"                     # Optional http:// or https://
        r"([a-zA-Z0-9\-]+\.)+"               # Domain name (e.g., www.example.)
        r"[a-zA-Z]{2,6}"                     # Domain extension (.com, .edu, etc.)
        r"(/[a-zA-Z0-9\-._~:/?#@!$&'()*+,;=%]*)?)"  # Optional path
    )
    author_pattern = re.compile(r"^[A-Z][a-zA-Z\.\-\s]*,[\sA-Z][a-zA-Z\.\-\s]*")  # Matches potential author names
    figure_caption_pattern = re.compile(r'^(Figure|Fig\.|图)\s?\d*[\.:]?', re.IGNORECASE)
    table_caption_pattern = re.compile(r"^Table(?:\s+\d+[:\.]?)?.*", re.IGNORECASE)
    
    
    # Process each page
    exclude_content_block = False
    for page in doc:
        tab_content = ""
        blocks = page.get_text("dict")["blocks"] # blocks (like paragraph) in same page
        tab_content_block = False
        
        for block in blocks: # Skip foot of the page
            if "lines" not in block:
                continue
            
            title_block = False
            abstract_block = False
            section_title_block = False
            DOI_block = False
            content_block = False
            fig_block = False
            block_text = ""    
            for line in block["lines"]:
                for span in line["spans"]:
                    font_name = span["font"]
                    text = span["text"].strip()
                    font_size = span["size"]

                    if not text:
                        continue
                    
                    # Collect text for this block
                    block_text += text + " "
                    
                    # Check if it is paper title
                    if 15.8 <= font_size <= 20.1: #and ("Bold" in font_name or "Black" in font_name):
                        title_block = True   
                    # Check if this is abstract text (within the font size range)
                    elif 9.0 <= font_size <= 9.8:
                        abstract_block = True
                    elif 10.5 <=font_size <= 11.7 and heading_pattern.match(text):
                        if text in exclude_section:
                            exclude_content_block = True  
                        else:
                            section_title_block = True
                            i = 0

                    if 7.9 <= font_size <=9.1:
                        if doi_pattern.findall(text) and not exclude_content_block: # and ("Bold" in font_name or "Black" in font_name):
                            DOI_block = True
                        elif not (table_caption_pattern.search(block_text) or figure_caption_pattern.search(block_text) or exclude_content_block or ('ScalaSansPro-Regular' in font_name) or 'ScalaSansPro-Bold' in font_name):
                            # Remove page footer, figure_caption, table_caption, and 'Supporting Information', 'Acknowledgements',etc. irrelevant sections.
                            content_block = True # Start to extract content block (Paragraph)

                    if 4.9 <= font_size < 8.1:
                        if table_caption_pattern.search(block_text):
                            tab_content_block = True
                        elif figure_caption_pattern.search(text):
                            fig_block = True
                          

                    
            
            # After processing the entire block, if it was an abstract block, add all text to abstract
            if not paper_title and title_block and block_text:
                paper_title = block_text.strip()
            if not abstract and abstract_block and block_text:
                abstract = block_text.strip()   
                
            if section_title_block and block_text:
                if current_section:
                    sections[current_section["section_title"]] = current_section["blocks"]
                # Start a new section.
                current_section = {
                    "section_title": block_text.strip(),
                    "blocks": {}
                    }
            if DOI_block and block_text:
                DOI = doi_pattern.findall(block_text)
            if content_block and block_text and not url_pattern.search(block_text):
                if current_section:
                    current_section['blocks'][f"block{i}"] = block_text.strip()
                    i += 1
            if fig_block and block_text:
                global_fig_label += 1
                figure_captions[f"Fig. {global_fig_label}"] = block_text.strip()
            if tab_content_block and block_text and not figure_caption_pattern.search(block_text) and not content_block:
                tab_content += block_text.strip() + '\n'
 
        if tab_content and tab_content_block:
            global_tab_label += 1                 
            table[f"Table {global_tab_label}"] = tab_content     

        # Append current section content if available (optional: you might want to only do this at the end)
        if current_section:
            sections[current_section["section_title"]] = current_section['blocks']

    return {
        "Title": paper_title.strip() if paper_title else "Untitled",
        "DOI": DOI,
        "Abstract": abstract,
        **sections,
        "Figures": figure_captions,
        "Tables": table
    }
    

# Convert XML image pii link to accessible URL.    
def convert_pii_link(link):
    if link.startswith("pii:"):
        link = link[4:]
    # Split using '/'
    parts = link.split('/')
    if len(parts) != 2:
        raise ValueError('Split failed, check pii link format')
    pii_number = parts[0]
    ref = parts[1]
    api_key = "6ca1c8129abdb24dafa5af7d754b0a6a"
    base_url = "https://api.elsevier.com/content/object/eid"
    url = f"{base_url}/1-s2.0-{pii_number}-{ref}.jpg?httpAccept=*/*&apiKey={api_key}"
    return url