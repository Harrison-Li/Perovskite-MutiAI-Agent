from sam_agent.tools.molecular_tool.mol_utils import Smiles2MolPort_id,MolPort_info, canonical_smiles
import pandas as pd

class Molinfo:
    def __init__(self):
        self.api_key = "d44aff4f-faae-41aa-8f1f-ee740daff7fe"
        self.gen_smiles_list=[]
    
    # Decide which dataset to use and convert the SMILES to the canonical SMILES    
    def data_loader(self,smiles_list=None, generated=False):
        if generated:
            decided_list = self.gen_smiles_list
        else:
            if smiles_list is None:
                raise ValueError("smiles_list must be provided if generated is False.")
            elif isinstance(smiles_list, list):
                decided_list = smiles_list
            else:
                decided_list = [smiles_list]
            
        # Convert the list to Canonical SMILES
        # Can_SMILES_list=[canonical_smiles(i) for i in decided_list]
            
        return decided_list
        
        
        
        
    def find_ids(self, mol_list=None, generated=False):
        canonical_smiles = self.data_loader(mol_list, generated)
        ids = Smiles2MolPort_id(smiles_list=canonical_smiles, API_key=self.api_key)
        return ids
        
        
    def collect_info(self, mol_list=None, generated=False):
        #ids = self.find_ids()  # Ensure ids are obtained from find_ids method
        canonical_smiles = self.data_loader(mol_list, generated)
        info = MolPort_info(input_list=canonical_smiles, API_key=self.api_key)
        return info



    

    
    
    
    
    
    
    
    
