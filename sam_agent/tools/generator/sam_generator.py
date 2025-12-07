from tqdm import tqdm
import pandas as pd
from sam_agent.tools.generator.model import GPT, GPTConfig
from sam_agent.tools.generator.utils_generator import load_stoi, canonic_smiles, get_mol, sample
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
import torch
import re



class SAMGenerator:
    # init the paras of the model
    def __init__(self,scaf_condition,anchoring_group,gen_size):
        self.regex=re.compile("(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
        self.prop = None
        self.model_weight = '/home/b_li/Test codes/weights/scaffold_guacamol_all.pt'
        self.scaffold = True
        self.list = False
        self.scaf_condition = scaf_condition#['c1ccccc1', 'c1ccc2c(c1)[nH]c1ccccc12']
        self.lstm = False
        self.context = anchoring_group # "O=P(O)(O)"
        self.csv_name = 'test.csv'
        self.stoi_name = 'updated_vocabulary_stoi'
        self.stoi, self.itos = load_stoi(self.stoi_name)
        self.batch_size = gen_size if gen_size < 100 else 100
        self.gen_size = gen_size
        self.vocab_size = 143
        self.block_size = 202
        self.props = []
        self.num_props = len(self.props)
        self.n_layer = 8
        self.n_head = 8
        self.n_embd = 256
        self.lstm_layers = 2
        self.model = self.load_model()
        
        
    def load_model(self):
        scaffold_max_len = 123 if self.scaffold else 0
        num_props = len(self.props)
        config = GPTConfig(self.vocab_size, self.block_size, num_props = num_props,
                           n_layer=self.n_layer, n_head=self.n_head, n_embd = self.n_embd, scaffold = self.scaffold, scaffold_maxlen = scaffold_max_len,
                           lstm = self.lstm, lstm_layers = self.lstm_layers
                           )
        model = GPT(config)
        model.load_state_dict(torch.load(self.model_weight))
        model.to('cuda')
        print('Model loaded')
        return model
    
    
    # Convert the medel logits to the text form.
    def process_output(self, y, scaf):
        molecules = []
        invalid=0
        for gen_mol in y:
            completion = ''.join([self.itos[int(i)] for i in gen_mol]).replace('<', '')
            mol = get_mol(completion)
            if mol:
                smiles = Chem.MolToSmiles(mol)
                scaffold_smiles = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
                mol_dict = {
                    'smiles': smiles,
                    'scaffold_condition': scaf.replace('<', '') if scaf else None,
                    'scaffold_smiles': scaffold_smiles
                }
                molecules.append(mol_dict)
            else:
                invalid=1
        return molecules,invalid    
    
    
    def generate_with_scaffold(self):
        scaffold_max_len = 123 if self.scaffold else 0
        scaf_token = [ i + str('<')*(scaffold_max_len - len(self.regex.findall(i))) for i in self.scaf_condition]
        mol_dict = []
        total_invalid = 0
        total_valid = 0 
        while total_valid < self.gen_size:
            for scaf in scaf_token:
                
                x = torch.tensor([self.stoi[s] for s in self.regex.findall(self.context)], dtype=torch.long)[None,...].repeat(self.batch_size, 1).to('cuda')
                sca = torch.tensor([self.stoi[s] for s in self.regex.findall(scaf)], dtype=torch.long)[None,...].repeat(self.batch_size, 1).to('cuda')
                y = sample(
                    self.model,
                    x,
                    self.block_size,
                    temperature=1,
                    sample=True,
                    top_k=10,
                    prop=None,
                    scaffold=sca
                )
                # Get both valid mols and invalid numbers
                valid_mols, invalid_count = self.process_output(y, scaf)
                mol_dict.extend(valid_mols)
                total_invalid += invalid_count
                total_valid = len(mol_dict)
                

        results = pd.DataFrame(mol_dict)
        results = results.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True)
        results['scaffold_condition'] = results['scaffold_condition'].str.replace('<', '')
        return results
    
def generator_tool(gen_size,scaf_condition,anchoring_group:str = 'OP(O)(=O)'):
    scaf_condition = [canonic_smiles(i) for i in scaf_condition]
    if not scaf_condition:
        raise ValueError("Invalid scaffold condition provided.")
    anchoring_group = canonic_smiles(anchoring_group)
    if not anchoring_group:
        raise ValueError("Invalid anchoring group provided.")
    generator=SAMGenerator(gen_size=gen_size,scaf_condition=scaf_condition,anchoring_group=anchoring_group)
    results=generator.generate_with_scaffold()
    smiles_df = pd.DataFrame({'SMILES': results['smiles']})
    return results,smiles_df.to_csv('generated_data.csv',index=False)



