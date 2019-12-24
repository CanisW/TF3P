from rdkit.Chem import AllChem, ChemicalForceFields
import torch

# MMFF94
# type=index+1,   alpha-i,   N-i,   A-i,   G-i
ff_param = [
    [1.050, 2.490, 3.890, 1.282],
    [1.350, 2.490, 3.890, 1.282],
    [1.100, 2.490, 3.890, 1.282],
    [1.300, 2.490, 3.890, 1.282],
    [0.250, 0.800, 4.200, 1.209],
    [0.700, 3.150, 3.890, 1.282],
    [0.650, 3.150, 3.890, 1.282],
    [1.150, 2.820, 3.890, 1.282],
    [0.900, 2.820, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [0.350, 3.480, 3.890, 1.282],
    [2.300, 5.100, 3.320, 1.345],
    [3.400, 6.000, 3.190, 1.359],
    [5.500, 6.950, 3.080, 1.404],
    [3.000, 4.800, 3.320, 1.345],
    [3.900, 4.800, 3.320, 1.345],
    [2.700, 4.800, 3.320, 1.345],
    [2.100, 4.800, 3.320, 1.345],
    [4.500, 4.200, 3.320, 1.345],
    [1.050, 2.490, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [1.100, 2.490, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [0.150, 0.800, 4.200, 1.209],
    [1.600, 4.500, 3.320, 1.345],
    [3.600, 4.500, 3.320, 1.345],
    [0.150, 0.800, 4.200, 1.209],
    [0.150, 0.800, 4.200, 1.209],
    [0.150, 0.800, 4.200, 1.209],
    [1.350, 2.490, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [0.750, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [1.000, 2.820, 3.890, 1.282],
    [1.500, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [1.350, 2.490, 3.890, 1.282],
    [0.850, 2.820, 3.890, 1.282],
    [1.100, 2.820, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [1.100, 2.490, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [3.000, 4.800, 3.320, 1.345],
    [1.150, 2.820, 3.890, 1.282],
    [1.300, 2.820, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [1.200, 2.820, 3.890, 1.282],
    [1.000, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [0.400, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [1.000, 2.820, 3.890, 1.282],
    [1.300, 2.820, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [1.000, 2.490, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [0.650, 3.150, 3.890, 1.282],
    [1.800, 2.490, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [1.300, 2.820, 3.890, 1.282],
    [1.350, 2.490, 3.890, 1.282],
    [1.350, 2.490, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [0.750, 2.820, 3.890, 1.282],
    [0.950, 2.820, 3.890, 1.282],
    [0.900, 2.820, 3.890, 1.282],
    [0.950, 2.820, 3.890, 1.282],
    [0.870, 3.150, 3.890, 1.282],
    [0.150, 0.800, 4.200, 1.209],
    [4.000, 4.800, 3.320, 1.345],
    [3.000, 4.800, 3.320, 1.345],
    [3.000, 4.800, 3.320, 1.345],
    [4.000, 4.500, 3.320, 1.345],
    [1.200, 2.820, 3.890, 1.282],
    [1.500, 5.100, 3.320, 1.345],
    [1.350, 2.490, 3.890, 1.282],
    [1.000, 2.820, 3.890, 1.282],
    [1.000, 2.490, 3.890, 1.282],
    [0.800, 2.820, 3.890, 1.282],
    [0.950, 2.820, 3.890, 1.282],
    [0.450, 6.000, 4.000, 1.400],
    [0.550, 6.000, 4.000, 1.400],
    [1.400, 3.480, 3.890, 1.282],
    [4.500, 5.100, 3.320, 1.345],
    [6.000, 6.000, 3.190, 1.359],
    [0.150, 2.000, 4.000, 1.300],
    [0.400, 3.500, 4.000, 1.300],
    [1.000, 5.000, 4.000, 1.300],
    [0.430, 6.000, 4.000, 1.400],
    [0.900, 5.000, 4.000, 1.400],
    [0.350, 6.000, 4.000, 1.400],
    [0.400, 6.000, 4.000, 1.400],
    [0.350, 3.500, 4.000, 1.300],
    [0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000],
]

ff_param_const = {
    'power': 0.25,
    'B': 0.2,
    'Beta': 12.0,
    'DARAD': 0.8,
    'DAEPS': 0.5,
    'elec_const': 332.0716,
    'cut_off': 30.0
}

diameter = 20.0
grid_size = 64
resolution = diameter/grid_size


def from_mol_to_array(mol):
    """

    Parameters
    ----------
    mol : rdkit.Mol
            must have 3D conformer

    Returns
    gs_charge : list
    atom_type : list
    pos : list of list (N*3)
    -------

    """
    try:
        if mol:
            # initialization
            gs_charge, atom_type, pos = [], [], []
            mmff_prop = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
            AllChem.ComputeGasteigerCharges(mol)

            # get charge, atom type, 3D coordinates
            for i in range(mol.GetNumAtoms()):
                # get charge
                gs_charge_i = float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge'))
                # get atom type
                atom_type_i = mmff_prop.GetMMFFAtomType(i) - 1
                # get coordinate
                pos_i = mol.GetConformer().GetAtomPosition(i)
                pos_x_i, pos_y_i, pos_z_i = pos_i.x, pos_i.y, pos_i.z

                gs_charge.append(gs_charge_i)
                atom_type.append(atom_type_i)
                pos.append([pos_x_i, pos_y_i, pos_z_i])

            return gs_charge, atom_type, pos
        else:
            return None
    except:
        return None


def from_array_to_ff_batch(gs_charge, atom_type, pos, nums_atoms, device=None, grid_size=grid_size, group_size=4):
    '''
    batch processing
    Parameters
    ----------
    gs_charge
    atom_type
    pos
    nums_atoms
    device

    Returns
    -------

    '''
    # for detailed computing methods, see ref "Journal of Computational Chemistry, Vol. 17, Nos. 5 &6, 520-552 (1996)"

    from torch_scatter import scatter_add

    if isinstance(gs_charge, torch.Tensor):
        device = gs_charge.device
        atom_type.to(torch.long)
    else:
        if not device: device = torch.device('cuda:0')
        # calc prop
        gs_charge, atom_type, pos = torch.tensor(gs_charge, device=device, dtype=torch.float32), \
                                    torch.tensor(atom_type, device=device, dtype=torch.long), \
                                    torch.tensor(pos, device=device, dtype=torch.float32)
    # used in scatter_add
    mol_idx = torch.cat([torch.zeros(n, dtype=torch.long, device=device) + i for i, n in enumerate(nums_atoms)])

    nd_ff_param = torch.tensor(ff_param, device=device, dtype=torch.float32)

    # alpha-i, N-i, A-i, G-i
    atom_type = nd_ff_param[atom_type, :]

    # create grid
    x, y, z = [torch.arange(grid_size, device=device, dtype=torch.float32) \
               / (grid_size - 1) * diameter - diameter / 2
               for _ in range(3)]
    x, y, z = x.expand(1, 1, 1, -1).permute(0, 3, 1, 2), \
              y.expand(1, 1, 1, -1).permute(0, 1, 3, 2), \
              z.expand(1, 1, 1, -1)
    pos = pos.expand(1, 1, 1, -1, -1).permute(3, 4, 0, 1, 2)
    # resulted dims: x: 1*gs*1*1, y: 1*1*gs*1, z:1*1*1*gs, pos: N*3*1*1*1

    # distance between atoms and grid points, dim: N*gs*gs*gs
    r_ij = torch.sqrt(
        torch.pow(x - pos[:, 0, :, :, :], 2) + torch.pow(y - pos[:, 1, :, :, :], 2) + torch.pow(z - pos[:, 2, :, :, :],
                                                                                                2))

    # R_ij: vdW minimum-energy separation, e_ij: vdW well depth; two parameters in vdW equation
    R_i = ff_param[0][2] * (ff_param[0][0] ** ff_param_const['power'])
    R_j = atom_type[:, 2] * (atom_type[:, 0] ** ff_param_const['power'])
    gamma_ij = (R_i - R_j) / (R_i + R_j)
    f = ff_param_const['B'] * (1 - torch.exp(- ff_param_const['Beta'] * gamma_ij))
    R_ij = ff_param_const['DAEPS'] * (R_i + R_j) * (1.0 + f)
    e_ij = (181.16 *
            ff_param[0][3] * atom_type[:, 3] *
            ff_param[0][0] * atom_type[:, 0]) / \
           (((ff_param[0][0] / ff_param[0][1]) ** 0.5 +
             torch.sqrt(atom_type[:, 0] / atom_type[:, 1])) * (R_ij ** 6))
    e_ij, R_ij = e_ij.expand(1, 1, 1, -1).permute(3, 0, 1, 2), \
                 R_ij.expand(1, 1, 1, -1).permute(3, 0, 1, 2)
    # resulted dims: e_ij: N*1*1*1, R_ij: N*1*1*1

    # dim: N*gs*gs*gs
    E_vdw = e_ij * ((1.07 * R_ij / (r_ij + 0.07 * R_ij)) ** 7) * (1.12 * (R_ij ** 7) / (r_ij ** 7 + 0.12 * R_ij) - 2.0)
    ff_cutoff = torch.tensor([ff_param_const['cut_off']], device=device)
    E_vdw = torch.min(scatter_add(E_vdw, mol_idx, dim=0), ff_cutoff).reshape((-1, group_size, grid_size, grid_size, grid_size))
    # sumarize of E_vdW of all atoms at a point and cut off, dim: gs*gs*gs

    # similar to vdW
    gs_charge = gs_charge.expand(1, 1, 1, -1).permute(3, 0, 1, 2)
    E_ele = ff_param_const['elec_const'] * gs_charge / r_ij
    E_ele = torch.clamp(scatter_add(E_ele, mol_idx, dim=0), min=-ff_param_const['cut_off'],
                        max=ff_param_const['cut_off']).reshape((-1, group_size, grid_size, grid_size, grid_size))

    E = torch.stack((E_vdw, E_ele), 2)  # 0, vdw; 1, ele

    return E.squeeze()


def from_array_to_ff(gs_charge, atom_type, pos, device, grid_size=grid_size):
    '''
    single mol processing
    Parameters
    ----------
    gs_charge
    atom_type
    pos
    device
    grid_size

    Returns
    -------

    '''
    # for detailed computing methods, see ref "Journal of Computational Chemistry, Vol. 17, Nos. 5 &6, 520-552 (1996)"

    # calc prop
    gs_charge, atom_type, pos = torch.tensor(gs_charge, device=device,  dtype=torch.float32), \
                                torch.tensor(atom_type, device=device, dtype=torch.long), \
                                torch.tensor(pos, device=device, dtype=torch.float32)

    nd_ff_param = torch.tensor(ff_param, device=device, dtype=torch.float32)

    # alpha-i, N-i, A-i, G-i
    atom_type = nd_ff_param[atom_type, :]

    # create grid
    x, y, z = [torch.arange(grid_size, device=device, dtype=torch.float32) \
               / (grid_size - 1) * diameter - diameter / 2
               for _ in range(3)]
    x, y, z = x.expand(1, 1, 1, -1).permute(0, 3, 1, 2), \
              y.expand(1, 1, 1, -1).permute(0, 1, 3, 2), \
              z.expand(1, 1, 1, -1)
    pos = pos.expand(1, 1, 1, -1, -1).permute(3, 4, 0, 1, 2)

    # distance between atoms and grid points
    r_ij = torch.sqrt(torch.pow(x - pos[:, 0, :, :, :], 2) + torch.pow(y - pos[:, 1, :, :, :], 2) + torch.pow(z - pos[:, 2, :, :, :], 2))

    # R_ij: vdW minimum-energy separation, e_ij: vdW well depth
    R_i = ff_param[0][2] * (ff_param[0][0] ** ff_param_const['power'])
    R_j = atom_type[:, 2] * (atom_type[:, 0] ** ff_param_const['power'])
    gamma_ij = (R_i - R_j)/(R_i + R_j)
    f = ff_param_const['B']*(1 - torch.exp(- ff_param_const['Beta'] * gamma_ij))
    R_ij = ff_param_const['DAEPS'] * (R_i + R_j) * (1.0 + f)
    e_ij = (181.16 *
            ff_param[0][3] * atom_type[:, 3] *
            ff_param[0][0] * atom_type[:, 0])/\
           (((ff_param[0][0]/ff_param[0][1])**0.5 +
             torch.sqrt(atom_type[:, 0]/atom_type[:, 1]))*(R_ij ** 6))
    e_ij, R_ij = e_ij.expand(1, 1, 1, -1).permute(3, 0, 1, 2), \
                 R_ij.expand(1, 1, 1, -1).permute(3, 0, 1, 2)

    E_vdw = e_ij * ((1.07 * R_ij / (r_ij + 0.07 * R_ij)) ** 7) * (1.12 * (R_ij ** 7) / (r_ij**7 + 0.12 * R_ij) - 2.0)
    ff_cutoff = torch.tensor([ff_param_const['cut_off']], device=device)
    E_vdw = torch.min(torch.sum(E_vdw, 0), ff_cutoff)

    gs_charge = gs_charge.expand(1, 1, 1, -1).permute(3, 0, 1, 2)
    E_ele = ff_param_const['elec_const']*gs_charge/r_ij
    E_ele = torch.clamp(torch.sum(E_ele, 0), min=-ff_param_const['cut_off'], max=ff_param_const['cut_off'])

    E = torch.stack((E_vdw, E_ele), 0) # 0, vdw; 1, ele

    return E




