import pandas as pd
import numpy as np
import faiss
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def gpu_cluster_large_library(smiles_path, output_path, batch_size=100000, cutoff=0.7):
    # 1. Define dimension
    dim = 2048  # Must match Morgan fingerprint size
    
    # 2. Try using a simpler approach with a single GPU first
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # Use GPU 0
    
    # 3. Process data in batches
    reader = pd.read_csv(smiles_path, sep='\t', header=None,
                        names=['SMILES', 'ID', 'Value', 'Group'],
                        chunksize=batch_size)
    representatives = []
    
    for chunk_idx, chunk in enumerate(tqdm(reader)):
        # 4. Generate fingerprints
        mols = [Chem.MolFromSmiles(smi) for smi in chunk['SMILES']]
        valid_mols = [mol for mol in mols if mol is not None]
        valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
        
        # Skip if no valid molecules
        if not valid_mols:
            continue
            
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=dim) for mol in valid_mols]
        
        # 5. Convert to numpy arrays
        fp_arrays = np.zeros((len(fps), dim), dtype=np.float32)
        for i, fp in enumerate(fps):
            # Get binary representation
            bits = list(fp.GetOnBits())
            fp_arrays[i, bits] = 1.0
        
        faiss.normalize_L2(fp_arrays)
        
        # 6. Find similar molecules
        if gpu_index.ntotal > 0:
            D, I = gpu_index.search(fp_arrays, 1)
            mask = D[:, 0] > (1 - cutoff)  # Using distance threshold
            new_indices = [valid_indices[i] for i, m in enumerate(mask) if m]
            new_reps = [chunk.index[i] for i in new_indices]
        else:
            new_reps = [chunk.index[i] for i in valid_indices]
            mask = np.ones(len(valid_indices), dtype=bool)
        
        # 7. Update index with new representatives
        if len(new_reps) > 0:
            new_fps = fp_arrays[mask] if gpu_index.ntotal > 0 else fp_arrays
            gpu_index.add(new_fps)
            representatives.extend(new_reps)
        
        # 8. Save progress
        if chunk_idx % 10 == 0:
            pd.DataFrame({'index': representatives}).to_csv(
                f'{output_path}_temp', index=False)
    
    # 9. Final save
    df = pd.read_csv(smiles_path, sep='\t', header=None,
                    names=['SMILES', 'ID', 'Value', 'Group'])
    diverse_lib = df.iloc[representatives].drop_duplicates(subset=['SMILES'])
    diverse_lib.to_csv(output_path, sep='\t', index=False, header=False)

if __name__ == "__main__":
    gpu_cluster_large_library(
        smiles_path='Br_COO.tsv',
        output_path='Br_COO_diverse.tsv',
        batch_size=100000,
        cutoff=0.7
    )
