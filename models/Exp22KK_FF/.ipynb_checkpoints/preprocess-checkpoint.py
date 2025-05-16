import sys

#change the path into where the nfp folder is
sys.path.append('modules')

import pandas as pd
import numpy as np
import gzip

import warnings
from tqdm import tqdm

import gzip
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ForwardSDMolSupplier

from itertools import islice

from nfp.preprocessing import MolAPreprocessor, GraphSequence

shifts = pd.read_pickle('../../data/Exp22K_FF/NMR22K_FF_Shifts.pkl')

mols = []
with gzip.open('../../data/Exp22K_FF/NMR22K_FF.sdf.gz', 'r') as sdfile:
    mol_supplier = Chem.ForwardSDMolSupplier(sdfile, removeHs=False, sanitize=False)
    for mol in tqdm(mol_supplier):
        if mol:
            mols += [(int(mol.GetProp('_Name')), mol, mol.GetNumAtoms())]

mols = pd.DataFrame(mols, columns=['Mol_ID', 'Mol', 'n_atoms'])

df_Shift = mols.set_index('Mol_ID').join(shifts.set_index('Mol_ID'))

test = df_Shift.sample(n=2200, random_state=666)
valid = df_Shift[~df_Shift.index.isin(test.index)].sample(n=2200, random_state=666)
train = df_Shift[
    (~df_Shift.index.isin(test.index) & ~df_Shift.index.isin(valid.index))
              ]

test.to_pickle('test.pkl.gz', compression='gzip')
valid.to_pickle('valid.pkl.gz', compression='gzip')
train.to_pickle('train.pkl.gz', compression='gzip')

# Preprocess molecules
def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()
    
def Mol_iter(df):
    for index,r in df.iterrows():
        yield(r['Mol'], r['Atomic_Indices'])

preprocessor = MolAPreprocessor(
    n_neighbors=100, cutoff=5, atom_features=atomic_number_tokenizer)

inputs_train = preprocessor.fit(Mol_iter(train))
inputs_valid = preprocessor.predict(Mol_iter(valid))
inputs_test = preprocessor.predict(Mol_iter(test))

import pickle
with open('processed_inputs.p', 'wb') as file:        
    pickle.dump({
        'inputs_train': inputs_train,
        'inputs_valid': inputs_valid,
        'inputs_test': inputs_test,
        'preprocessor': preprocessor,
    }, file)