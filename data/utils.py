from rdkit.Chem import AllChem
from scipy import sparse
from e3fp.fingerprint.generate import fprints_dict_from_mol


def get_maccskey(mol):
    if mol:
        return AllChem.GetMACCSKeysFingerprint(mol)
    else:
        return None

def get_e3fp(mol):
    try:
        if mol:
            return fprints_dict_from_mol(mol, bits=1024)[5][0].to_rdkit()
        else:
            return None
    except:
        return None

def get_cfp(mol, nBits=1024):
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    else:
        return None

def tc_sim(fp1, fp2):
    if fp1 and fp2:
        csr1, csr2 = sparse.csr_matrix(list(fp1)), sparse.csr_matrix(list(fp2))
        sim = ((csr1 * csr2.T).data/(csr1.sum() + csr2.sum() - (csr1 * csr2.T).data)).tolist()
        return sim[0] if sim else 0.0  # for if fp1 and fp2 contain no same bit, sim will be [].
    else:
        return 0
