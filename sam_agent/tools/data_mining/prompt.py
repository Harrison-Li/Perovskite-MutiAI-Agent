from langchain_core.prompts import PromptTemplate, MessagesPlaceholder,ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

def preparation_prompt(text, image_list = None):
    if image_list:
        image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        for image_data in image_list
        ]
    else:  
        image_messages = []  
    
    examples = [
            {
                "input": "These are the examples: Extract the parameters relevant to synthesis of SAM molecules from the paper",
                "output": {
                    "synthesized_molecules": [
                        {
                            "molecule_name": "(2-(3,7-Dibromo-10H-phenoxazin-10-yl)ethyl)phosphonic acid (Br-2EPO)",
                            "total_synthesis_step": 5,
                            "steps": [
                                {
                                    "step_name": "Step 1",
                                    "reactants_name": "10H-phenoxazine (1a), bromoacetyl bromide",
                                    "reactants_SMILES": "N1C2=C(OC3=C1C=CC=C3)C=CC=C2, O=C(Br)CBr",
                                    "products_name": "2-Bromo-1-(10H-phenoxazin-10-yl)ethan-1-one (2a)",
                                    "products_SMILES": "BrCC(=O)N1C2=C(OC3=C1C=CC=C3)C=CC=C2",
                                    "reagents": "bromoacetyl bromide",
                                    "reaction_condition": "80°C for 12 h under nitrogen",
                                    "catalyst": "N/A",
                                    "solvent": "dry toluene",
                                    "dilution_washing_solvent": "ethyl acetate, sodium bicarbonate, water, brine",
                                    "Purification": "flash column chromatography",
                                    "eluent": "hexane/ethyl acetate (100/0 to 80/20)",
                                    "products_appearance": "yellowish liquid",
                                    "yield": "85%",
                                    "characterization": {
                                        "1H_NMR": {
                                            "condition": "400 MHz, 298 K, CDCl3",
                                            "data": "δ 7.62 (d, J = 7.9 Hz, 2H) 7.27-7.22 (m, 2H),  7.19-7.14 (m, 4H), 4.13 (s, 2H)"
                                            },
                                        "13C_NMR": {
                                            "condition": "101 MHz, 298 K, CDCl3",
                                            "data": "δ 165.8, 151.2, 129.0, 127.8, 124.5, 123.9, 117.3, 27.1"
                                            },
                                        "31P_NMR": {
                                            "condition": "N/A",
                                            "data": "N/A"
                                            },
                                        "HRMS": {
                                            "condition": "N/A",
                                            "data": "m/z calcd for C14H11BrNO2 [M + H]+ 303.9968, found 303.9970"
                                            }
                                    }
                                }
                            ]
                            }
                    ]
                    }
            }       
        ]
    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
        )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # A prompt template to format the example
        example_prompt=example_prompt,
        examples=examples,
        )
        
    final_prompt = ChatPromptTemplate.from_messages(
        [("system","You are helpful data extractor, relevant to the **chemical synthesis** details of the **self-assembled molecules** themselves."
          "Before extraction, check whether the parameters describe the **process of creating the molecule** through chemical reactions. If the parameters describe device fabrication steps (like deposition, coating, annealing of layers, etc.) or are related to the assembly of molecules into a device rather than their creation, then do not extract them and return 'N/A'."
          "I will give you examples, you should follow the exact same format as provided examples"
          "Ensure that parameter names match exactly as in the example. Do not add additional or omit parameters in your result."
          "If any parameter is not explicitly mentioned in the text, return 'N/A' for that parameter."
          "**Return the result as valid JSON using double quotes (\"), not single quotes ('), with correct key-value pairs.**"),
         few_shot_prompt,
        ("user", [
            {
                "type": "text",
                "text": "Above are instructions and examples, strictly following the format of examples, extract parameters from next:{input}"
            },
            *image_messages
        ])
    ])
    
    prompt=final_prompt.invoke({"input": text})
    return prompt


def property_prompt(text, encoded_image_list = None):
    if encoded_image_list:
        image_messages = [
        { "type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        for image_data in encoded_image_list
        ]
    else:  
        image_messages = [] 

    # This is a prompt template used to format each individual example.
    examples = [
        {
            "input": "These are the examples: The SAMs 4-(trifluoromethyl) benzoic acid and 2PACz are measured to have HOMO 5.2 eV and 5.1 eV, and LUMO -1.87 eV and -1.85 eV respectively ...",
            "output": {
                "self_assembled_molecule":{
                    "molecule_name": "4-(trifluoromethyl) benzoic acid, 2PACz",
                    "HOMO (eV)":
                        [
                        {
                            "value": "-5.26, -5.56",
                            "method type" : "Simulation",
                            "method": "DFT Calculation",
                            'details': "B3LYP/def2-TZVP level of theory in ethanol solvent based on density (SMD) implicit solvation model with the Gaussian 09"
                        },
                        {
                            "value": "-5.67, -5.76",
                            "method type": "Experimental",
                            "method": "UPS Measurement",
                            "details": "UPS and absorption spectra in solution state"
                        }
                        ],
                    "LUMO (eV)":
                        [
                        {
                            "value": "-1.82, -1.92",
                            "method type" : "Simulation",
                            "method": "DFT Calculation",
                            'details': "N/A"
                        }
                        ],
                    "Dipole_moment (Debye)":
                        [
                        {
                            "value": "3.8, 4.0",
                            "method type" : "Simulation",
                            "method": "DFT Calculation",
                            'details': "N/A"
                        }
                        ]
                    },
                "perovskite_layer": {
                    "composition": "N/A",
                    "bandgap": "N/A",
                    "Photoluminescence (PL)_peak_position": "N/A",
                    "valence_band_maximum": "-5.58 eV",
                    "conduction_band_minimum": "-4.12 eV"
                }
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
        # A prompt template to format the example
        example_prompt=example_prompt,
        examples=examples,
        )
    
    final_prompt = ChatPromptTemplate.from_messages(
        messages =[
            SystemMessage(content = "You are helpful data extractor, extract parameters relevant properties of self_assembled_molecule and perovskite_layer. I will give you examples, you should follow the exact same format as provided examples"
                          "Ensure that parameter names matched same as in the example and extract them. Do not add additional or omit parameters in your result."
                          "If any parameter is not explicitly mentioned in the text, return 'N/A' for that parameter."
                          "Do not extract parameters in examples in output, you just need to extract parameters from text."
                          "**Return the result as valid JSON using double quotes (\"), not single quotes ('), with correct key-value pairs.**"),
            few_shot_prompt,
            ("user", 
             [
                 {"type": "text", "text": "Above are instructions and examples, follow parameters key and format like examples, then extract them from next:{input}"},
                 *image_messages,
                 #{"type": "text", "text": "If you received no content given by me, just return 'N/A'."}
             ]
             )]
        )
    
    prompt=final_prompt.invoke({"input": text})
    return prompt

def fabrication_prompt(text, image_list = None):
    if image_list:
        image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        for image_data in image_list
        ]
    else:  
        image_messages = []
    
    examples= [
        {
        "input": "These are the examples: Here is the content related to the device fabrication .....",
        "output":
            {
            "device_fabrication_process":{
                "device_structure": "N/A",
                "device_type": "p-i-n",
                "substrate type": "Fluorine doped tin oxide (FTO); Indium doped tin oxide (ITO)",
                "Metal_oxide_layer": {
                    "composition": "N/A",
                    "additive": "N/A",
                    "additive_concentration": "N/A",
                    "preparation_method": "N/A",
                    "solvent_used": "N/A",
                    "precursor_concentration": "N/A",
                    "spin_coating_speed": "N/A",
                    "spin_coating_time": "N/A",
                    "annealing_time": "N/A",
                    "annealing_temperature": "N/A",
                    "thickness": "N/A"
                    },
                "self_assembled_monolayer": {
                    "composition": "Br-2EPO, Br-2EPT",
                    "additive": "N/A",
                    "additive_concentration": "N/A",
                    "preparation_method": "spin-coating",
                    "solvent_used": "ethanol",
                    "precursor_concentration": "1 mM",
                    "spin_coating_speed": "3000 rpm",
                    "spin_coating_time": "30 s",
                    "annealing_temperature": "100°C",
                    "annealing_time": "10 min",
                    "thickness": "N/A"
                    },
                "bottom_interface_layer": {
                    "composition": "N/A",
                    "additive": "N/A",
                    "additive_concentration": "N/A",
                    "preparation_method": "N/A",
                    "solvent_used": "N/A",
                    "precursor_concentration": "N/A",
                    "spin_coating_speed": "N/A",
                    "spin_coating_time": "N/A",
                    "annealing_temperature": "N/A",
                    "annealing_time": "N/A",
                    "thickness": "N/A"
                },
                "perovskite_layer": {
                    "composition": "Cs0.05(FA0.92MA0.08)Pb(I0.92Br0.08)3",
                    "additive": "N/A",
                    "additive_concentration": "N/A",
                    "preparation_method": "one-step spin coating",
                    "solvent_used": "DMF:DMSO",
                    "precursor_concentration": "N/A",
                    "spin_coating_speed": "2000 and 4000 rpm",
                    "spin_coating_time": "10 and 30 s",
                    "antisolvent": "chlorobenzene",
                    "annealing_time": "30 min",
                    "annealing_temperature": "100°C",
                    "thickness": "N/A"
                },
                "top_interface_layer": {
                    "composition": "N/A",
                    "additive": "N/A",
                    "additive_concentration": "N/A",
                    "preparation_method": "N/A",
                    "solvent_used": "N/A",
                    "precursor_concentration": "N/A",
                    "spin_coating_speed": "N/A",
                    "spin_coating_time": "N/A",
                    "annealing_time": "N/A",
                    "annealing_temperature": "N/A",
                    "thickness": "N/A"
                },
                "top_charge_transport_layer": {
                    "composition": "C60",
                    "additive": "N/A",
                    "additive_concentration": "N/A",
                    "preparation_method": "thermal evaporation",
                    "solvent_used": "N/A",
                    "precursor_concentration": "N/A",
                    "spin_coating_speed": "N/A",
                    "spin_coating_time": "N/A",
                    "annealing_time": "N/A",
                    "annealing_temperature": "N/A",
                    "thickness": "30 nm"
                },
                "charge_block_layer": {
                    "composition": "bathocuproine",
                    "additive": "N/A",
                    "additive_concentration": "N/A",
                    "preparation_method": "thermal evaporation",
                    "solvent_used": "N/A",
                    "precursor_concentration": "N/A",
                    "spin_coating_speed": "N/A",
                    "spin_coating_time": "N/A",
                    "annealing_time": "N/A",
                    "annealing_temperature": "N/A",
                    "thickness": "6 nm"
                },
                "metal_electrodes": {
                    "type": "Cu",
                    "preparation_method": "thermal evaporation",
                    "thickness": "90 nm"
                },
                "device_area": "0.0942 cm2"
                }   
            } 
        }
    ]
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{input}"),
            ("ai", "{output}"),
        ]
        )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # A prompt template to format the example
        example_prompt=example_prompt,
        examples=examples,
        )
    

    
    final_prompt = ChatPromptTemplate.from_messages(
        [("system","You are helpful data extractor, extract parameters relevant to device fabrication. I will give you examples, you should follow the exact same format as provided examples"
          "Before extraction, check whether the parameters are relevant to device fabrication or not. If not, please don't extract them and return 'N/A'."
          "Ensure that parameter names match exactly as in the example. Do not add addtional or omit parameters in your result."
          "If any parameter is not explicitly mentioned in the text, return 'N/A' for that parameter."
          "**Return the result as valid JSON using double quotes (\"), not single quotes ('), with correct key-value pairs.**"),
         few_shot_prompt,
         ("user", 
          [
              {"type": "text", "text": "Above are instructions and examples, strictly following the format of examples, then extract parameters from next:{input}"},
              *image_messages 
              ]
          )]
        )
    
    prompt=final_prompt.invoke({"input": text})
    return prompt

def performance_prompt(text, image_list = None):
    if image_list:
        image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        for image_data in image_list
        ]
    else:  
        image_messages = []    
    
    examples= [
        {
        "input": "These are the examples: Here is the content related to the device performance and measurement .....",
        "output":{
            "device_performance": [
                {
                    "device_name": "Br-2EPSe-based PSC",
                    "device_structure": "FTO/SAM/CsFAMA/C60/BCP/Cu",
                    "photovoltaic_parameters": {
                        "forward_scan": {
                            "open_circuit_voltage": "1.12 V",
                            "short_circuit_current_density": "24.49 mA/cm²",
                            "fill_factor": "82.86%",
                            "power_conversion_efficiency": "22.73%"
                            },
                        "reverse_scan": {
                            "open_circuit_voltage": "N/A",
                            "short_circuit_current_density": "N/A",
                            "fill_factor": "N/A",
                            "power_conversion_efficiency": "N/A"
                            },
                        "average_scan_values": {
                            "open_circuit_voltage": "0.522±0.053 V",
                            "short_circuit_current_density": "22.22±2.70 mA/cm²",
                            "fill_factor": "80±4.9%",
                            "power_conversion_efficiency": "20.01±0.89%"
                            },
                        "integrated_current_density_from_EQE": "24.46 mA/cm²"
                        },
                    "stability": {
                        "thermal_stability": {
                            "atmosphere": "N/A",
                            "condition": "N/A",
                            "aging_time": "N/A",
                            "degradation_percentage": "N/A"
                            },
                        "photostability": {
                            "atmosphere": "N/A",
                            "condition": "N/A",
                            "aging_time": "N/A",
                            "degradation_percentage": "N/A"
                            },
                        "operational_stability": {
                            "atmosphere": "Ambient air (15-25% RH)",
                            "condition": "Unencapsulated, MPP tracking",
                            "aging_time": "500 h",
                            "degradation_percentage": "4%"
                            },
                        "humidity_stability": {
                            "atmosphere": "N/A",
                            "condition": "N/A",
                            "aging_time": "N/A",
                            "degradation_percentage": "N/A"
                            },
                        "storage_stability": 
                            {
                                "atmosphere": "Glovebox (dark)",
                                "condition": "Shelf storage",
                                "aging_time": "1600 h",
                                "degradation_percentage": "≈10%"
                            }
                        }
                }
                ]
            }
        }]

    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
        )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # A prompt template to format the example
        example_prompt=example_prompt,
        examples=examples,
        )
        
    final_prompt = ChatPromptTemplate.from_messages(
        [("system","You are helpful data extractor, extract parameters relevant to device_performance. I will give you examples, you should follow the exact same format as provided examples"
          "Ensure that parameter names match exactly as in the example. Do not add addtional or omit parameters in your result."
          "If any parameter is not explicitly mentioned in the text, return 'N/A' for that parameter."
          "**Return the result as valid JSON using double quotes (\"), not single quotes ('), with correct key-value pairs.**"),
         few_shot_prompt,
         ("user", 
          [
              {"type": "text", "text": "Above are instrutions and examples, strictly following the format of examples, extract parameters from next:{input}"},
              *image_messages 
              ]
          )]
        )
    
    prompt=final_prompt.invoke({"input": text})
    return prompt