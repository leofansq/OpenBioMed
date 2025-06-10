from typing import Any, Dict, List, Optional

import itertools
import logging
import numpy as np
try:
    from openbabel import openbabel as ob
    ob.obErrorLog.SetOutputLevel(0)
except ImportError:
    logging.warning("OpenBabel is not installed. The MolCRAFT model will not be able to decode generated molecules.")
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum
from tqdm import tqdm

from open_biomed.data import Molecule, MoleculeConstructError, Pocket, estimate_ligand_atom_num
from open_biomed.models.task_models.structure_based_drug_design import StructureBasedDrugDesignModel
from open_biomed.utils.collator import PygCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurized, MoleculeFeaturizer, PocketFeaturizer
from open_biomed.utils.misc import safe_index

### Borrowed from https://github.com/AlgoMole/MolCRAFT

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}

# Featurizers
class MolCRAFTMoleculeFeaturizer(MoleculeFeaturizer):
    def __init__(self, pos_norm=1.0) -> None:
        super().__init__()
        self.pos_norm = pos_norm

    def __call__(self, molecule: Molecule) -> Dict[str, Any]:
        molecule._add_rdmol()
        rdmol = molecule.rdmol
        node_feat_list = []
        for atom in rdmol.GetAtoms():
            node_feat_list.append(safe_index(MAP_ATOM_TYPE_AROMATIC_TO_INDEX, (atom.GetAtomicNum(), atom.GetIsAromatic())))
        node_feat = F.one_hot(torch.LongTensor(node_feat_list), num_classes=len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)).float()
        pos = torch.tensor(molecule.conformer, dtype=torch.float32)
        pocket_center = pos.mean(dim=0)
        pos -= pocket_center
        pos /= self.pos_norm
        
        return Data(**{
            "atom_feature": node_feat,
            "pos": pos,
        })
    
    def reconstruct_from_generated(self, xyz: List[List[float]], atomic_nums: List[int], aromatic: List[bool], basic_mode: bool=True) -> Chem.RWMol:
        def fixup(atoms: List[Any], mol: ob.OBMol, indicators: List[bool]) -> List[Chem.Atom]:
            '''Set atom properties to match channel.  Keep doing this
            to beat openbabel over the head with what we want to happen.'''

            """
            for now, indicators only include 'is_aromatic'
            """
            mol.SetAromaticPerceived(True)  # avoid perception
            for i, atom in enumerate(atoms):
                if indicators is not None:
                    if indicators[i]:
                        atom.SetAromatic(True)
                        atom.SetHyb(2)
                    else:
                        atom.SetAromatic(False)

                if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():  # Nitrogen, Oxygen
                    # this is a little iffy, ommitting until there is more evidence it is a net positive
                    # we don't have aromatic types for nitrogen, but if it
                    # is in a ring with aromatic carbon mark it aromatic as well
                    acnt = 0
                    for nbr in ob.OBAtomAtomIter(atom):
                        if nbr.IsAromatic():
                            acnt += 1
                    if acnt > 1:
                        atom.SetAromatic(True)
        
        mol = ob.OBMol()
        mol.BeginModify()
        atoms = []
        for xyz, t in zip(xyz, atomic_nums):
            x, y, z = xyz
            # ch = struct.channels[t]
            atom = mol.NewAtom()
            atom.SetAtomicNum(t)
            atom.SetVector(x, y, z)
            atoms.append(atom)
        fixup(atoms, mol, aromatic)

        # Connect the dots
        '''Custom implementation of ConnectTheDots.  This is similar to
        OpenBabel's version, but is more willing to make long bonds 
        (up to maxbond long) to keep the molecule connected.  It also 
        attempts to respect atom type information from struct.
        atoms and struct need to correspond in their order
        Assumes no hydrogens or existing bonds.
        '''
        pt = AllChem.GetPeriodicTable()
        covalent_factor = 1.3

        mol.BeginModify()

        # just going to to do n^2 comparisons, can worry about efficiency later
        coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
        dists = squareform(pdist(coords))
        # types = [struct.channels[t].name for t in struct.c]

        for i, j in itertools.combinations(range(len(atoms)), 2):
            a = atoms[i]
            b = atoms[j]
            a_r = ob.GetCovalentRad(a.GetAtomicNum()) * covalent_factor
            b_r = ob.GetCovalentRad(b.GetAtomicNum()) * covalent_factor
            if dists[i, j] < a_r + b_r:
                flag = 0
                if aromatic[i] and aromatic[j]:
                    flag = ob.OB_AROMATIC_BOND
                mol.AddBond(a.GetIdx(), b.GetIdx(), 1, flag)

        atom_maxb = {}
        for (i, a) in enumerate(atoms):
            # set max valance to the smallest max allowed by openbabel or rdkit
            # since we want the molecule to be valid for both (rdkit is usually lower)
            maxb = min(ob.GetMaxBonds(a.GetAtomicNum()), pt.GetDefaultValence(a.GetAtomicNum()))

            nbrs_of_elem = 0
            for nbr in ob.OBAtomAtomIter(a):
                if nbr.GetAtomicNum() == 8:
                    nbrs_of_elem += 1
            if a.GetAtomicNum() == 16:  # sulfone check
                if nbrs_of_elem >= 2:
                    maxb = 6

            atom_maxb[a.GetIdx()] = maxb

        # remove any impossible bonds between halogens
        for bond in ob.OBMolBondIter(mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
                mol.DeleteBond(bond)

        def get_bond_info(biter):
            '''Return bonds sorted by their distortion'''
            bonds = [b for b in biter]
            binfo = []
            for bond in bonds:
                bdist = bond.GetLength()
                # compute how far away from optimal we are
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum())
                stretch = bdist / ideal
                binfo.append((stretch, bond))
            binfo.sort(reverse=True, key=lambda t: t[0])  # most stretched bonds first
            return binfo

        def reachable_r(a, b, seenbonds):
            '''Recursive helper.'''

            for nbr in ob.OBAtomAtomIter(a):
                bond = a.GetBond(nbr).GetIdx()
                if bond not in seenbonds:
                    seenbonds.add(bond)
                    if nbr == b:
                        return True
                    elif reachable_r(nbr, b, seenbonds):
                        return True
            return False
        
        def reachable(a, b):
            '''Return true if atom b is reachable from a without using the bond between them.'''
            if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
                return False  # this is the _only_ bond for one atom
            # otherwise do recursive traversal
            seenbonds = set([a.GetBond(b).GetIdx()])
            return reachable_r(a, b, seenbonds)


        def forms_small_angle(a, b, cutoff=60):
            '''Return true if bond between a and b is part of a small angle
            with a neighbor of a only.'''

            for nbr in ob.OBAtomAtomIter(a):
                if nbr != b:
                    degrees = b.GetAngle(a, nbr)
                    if degrees < cutoff:
                        return True
            return False

        binfo = get_bond_info(ob.OBMolBondIter(mol))
        # now eliminate geometrically poor bonds
        for stretch, bond in binfo:

            # can we remove this bond without disconnecting the molecule?
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            # as long as we aren't disconnecting, let's remove things
            # that are excessively far away (0.45 from ConnectTheDots)
            # get bonds to be less than max allowed
            # also remove tight angles, because that is what ConnectTheDots does
            if stretch > 1.2 or forms_small_angle(a1, a2) or forms_small_angle(a2, a1):
                # don't fragment the molecule
                if not reachable(a1, a2):
                    continue
                mol.DeleteBond(bond)

        # prioritize removing hypervalency causing bonds, do more valent
        # constrained atoms first since their bonds introduce the most problems
        # with reachability (e.g. oxygen)
        hypers = [(atom_maxb[a.GetIdx()], a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms]
        hypers = sorted(hypers, key=lambda aa: (aa[0], -aa[1]))
        for mb, diff, a in hypers:
            if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                continue
            binfo = get_bond_info(ob.OBAtomBondIter(a))
            for stretch, bond in binfo:

                if stretch < 0.9:  # the two atoms are too closed to remove the bond
                    continue
                # can we remove this bond without disconnecting the molecule?
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()

                # get right valence
                if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
                    # don't fragment the molecule
                    if not reachable(a1, a2):
                        continue
                    mol.DeleteBond(bond)
                    if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                        break  # let nbr atoms choose what bonds to throw out

        mol.EndModify()
        fixup(atoms, mol, aromatic)

        mol.AddPolarHydrogens()
        mol.PerceiveBondOrders()
        fixup(atoms, mol, aromatic)

        for (i, a) in enumerate(atoms):
            ob.OBAtomAssignTypicalImplicitHydrogens(a)
        fixup(atoms, mol, aromatic)

        mol.AddHydrogens()
        fixup(atoms, mol, aromatic)

        # make rings all aromatic if majority of carbons are aromatic
        for ring in ob.OBMolRingIter(mol):
            if 5 <= ring.Size() <= 6:
                carbon_cnt = 0
                aromatic_ccnt = 0
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    if a.GetAtomicNum() == 6:
                        carbon_cnt += 1
                        if a.IsAromatic():
                            aromatic_ccnt += 1
                if aromatic_ccnt >= carbon_cnt / 2 and aromatic_ccnt != ring.Size():
                    # set all ring atoms to be aromatic
                    for ai in ring._path:
                        a = mol.GetAtom(ai)
                        a.SetAromatic(True)

        # bonds must be marked aromatic for smiles to match
        for bond in ob.OBMolBondIter(mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.IsAromatic() and a2.IsAromatic():
                bond.SetAromatic(True)

        mol.PerceiveBondOrders()

        def calc_valence(rdatom):
            '''Can call GetExplicitValence before sanitize, but need to
            know this to fix up the molecule to prevent sanitization failures'''
            cnt = 0.0
            for bond in rdatom.GetBonds():
                cnt += bond.GetBondTypeAsDouble()
            return cnt
        
        def convert_ob_mol_to_rd_mol(ob_mol, struct=None):
            '''Convert OBMol to RDKit mol, fixing up issues'''
            ob_mol.DeleteHydrogens()
            n_atoms = ob_mol.NumAtoms()
            rd_mol = AllChem.RWMol()
            rd_conf = AllChem.Conformer(n_atoms)

            for ob_atom in ob.OBMolAtomIter(ob_mol):
                rd_atom = AllChem.Atom(ob_atom.GetAtomicNum())
                # TODO copy format charge
                if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
                    # don't commit to being aromatic unless rdkit will be okay with the ring status
                    # (this can happen if the atoms aren't fit well enough)
                    rd_atom.SetIsAromatic(True)
                i = rd_mol.AddAtom(rd_atom)
                ob_coords = ob_atom.GetVector()
                x = ob_coords.GetX()
                y = ob_coords.GetY()
                z = ob_coords.GetZ()
                rd_coords = Geometry.Point3D(x, y, z)
                rd_conf.SetAtomPosition(i, rd_coords)

            rd_mol.AddConformer(rd_conf)

            for ob_bond in ob.OBMolBondIter(ob_mol):
                i = ob_bond.GetBeginAtomIdx() - 1
                j = ob_bond.GetEndAtomIdx() - 1
                bond_order = ob_bond.GetBondOrder()
                if bond_order == 1:
                    rd_mol.AddBond(i, j, AllChem.BondType.SINGLE)
                elif bond_order == 2:
                    rd_mol.AddBond(i, j, AllChem.BondType.DOUBLE)
                elif bond_order == 3:
                    rd_mol.AddBond(i, j, AllChem.BondType.TRIPLE)
                else:
                    raise Exception('unknown bond order {}'.format(bond_order))

                if ob_bond.IsAromatic():
                    bond = rd_mol.GetBondBetweenAtoms(i, j)
                    bond.SetIsAromatic(True)

            rd_mol = AllChem.RemoveHs(rd_mol, sanitize=False)

            pt = AllChem.GetPeriodicTable()
            # if double/triple bonds are connected to hypervalent atoms, decrement the order

            # TODO: fix seg fault
            # if struct is not None:
            #     positions = struct
            positions = rd_mol.GetConformer().GetPositions()
            nonsingles = []
            for bond in rd_mol.GetBonds():
                if bond.GetBondType() == AllChem.BondType.DOUBLE or bond.GetBondType() == AllChem.BondType.TRIPLE:
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    # TODO: ugly fix
                    dist = np.linalg.norm(positions[i] - positions[j])
                    nonsingles.append((dist, bond))
            nonsingles.sort(reverse=True, key=lambda t: t[0])

            for (d, bond) in nonsingles:
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()

                if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
                        calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
                    btype = AllChem.BondType.SINGLE
                    if bond.GetBondType() == AllChem.BondType.TRIPLE:
                        btype = AllChem.BondType.DOUBLE
                    bond.SetBondType(btype)

            # fix up special cases
            for atom in rd_mol.GetAtoms():
                # set nitrogens with 4 neighbors to have a charge
                if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
                    atom.SetFormalCharge(1)

                # check if there are any carbon atoms with 2 double C-C bonds
                # if so, convert one to a single bond
                if atom.GetAtomicNum() == 6 and atom.GetDegree() == 4:
                    cnt = 0
                    i = atom.GetIdx()
                    for nbr in atom.GetNeighbors():
                        if nbr.GetAtomicNum() == 6:
                            j = nbr.GetIdx()
                            bond = rd_mol.GetBondBetweenAtoms(i, j)
                            if bond.GetBondType() == AllChem.BondType.DOUBLE:
                                cnt += 1
                    if cnt == 2:
                        for nbr in atom.GetNeighbors():
                            if nbr.GetAtomicNum() == 6:
                                j = nbr.GetIdx()
                                bond = rd_mol.GetBondBetweenAtoms(i, j)
                                if bond.GetBondType() == AllChem.BondType.DOUBLE:
                                    bond.SetBondType(AllChem.BondType.SINGLE)
                                    break

            rd_mol = AllChem.AddHs(rd_mol, addCoords=True)
            # TODO: fix seg fault
            positions = rd_mol.GetConformer().GetPositions()
            center = np.mean(positions[np.all(np.isfinite(positions), axis=1)], axis=0)
            for atom in rd_mol.GetAtoms():
                i = atom.GetIdx()
                pos = positions[i]
                if not np.all(np.isfinite(pos)):
                    # hydrogens on C fragment get set to nan (shouldn't, but they do)
                    rd_mol.GetConformer().SetAtomPosition(i, center)

            try:
                AllChem.SanitizeMol(rd_mol, AllChem.SANITIZE_ALL ^ AllChem.SANITIZE_KEKULIZE)
            except:
                raise MoleculeConstructError("Failed to construct molecule from generated coordinates and atoms.")

            # but at some point stop trying to enforce our aromaticity -
            # openbabel and rdkit have different aromaticity models so they
            # won't always agree.  Remove any aromatic bonds to non-aromatic atoms
            for bond in rd_mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                if bond.GetIsAromatic():
                    if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                        bond.SetIsAromatic(False)
                elif a1.GetIsAromatic() and a2.GetIsAromatic():
                    bond.SetIsAromatic(True)

            return rd_mol

        def postprocess_rd_mol_1(rdmol):
            UPGRADE_BOND_ORDER = {AllChem.BondType.SINGLE: AllChem.BondType.DOUBLE, AllChem.BondType.DOUBLE: AllChem.BondType.TRIPLE}
            rdmol = AllChem.RemoveHs(rdmol)

            # Construct bond nbh list
            nbh_list = {}
            for bond in rdmol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if begin not in nbh_list:
                    nbh_list[begin] = [end]
                else:
                    nbh_list[begin].append(end)

                if end not in nbh_list:
                    nbh_list[end] = [begin]
                else:
                    nbh_list[end].append(begin)

            # Fix missing bond-order
            for atom in rdmol.GetAtoms():
                idx = atom.GetIdx()
                num_radical = atom.GetNumRadicalElectrons()
                if num_radical > 0:
                    for j in nbh_list[idx]:
                        if j <= idx: continue
                        nb_atom = rdmol.GetAtomWithIdx(j)
                        nb_radical = nb_atom.GetNumRadicalElectrons()
                        if nb_radical > 0:
                            bond = rdmol.GetBondBetweenAtoms(idx, j)
                            bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                            nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                            num_radical -= 1
                    atom.SetNumRadicalElectrons(num_radical)

                num_radical = atom.GetNumRadicalElectrons()
                if num_radical > 0:
                    atom.SetNumRadicalElectrons(0)
                    num_hs = atom.GetNumExplicitHs()
                    atom.SetNumExplicitHs(num_hs + num_radical)

            return rdmol


        def postprocess_rd_mol_2(rdmol):
            rdmol_edit = AllChem.RWMol(rdmol)

            ring_info = rdmol.GetRingInfo()
            ring_info.AtomRings()
            rings = [set(r) for r in ring_info.AtomRings()]
            for i, ring_a in enumerate(rings):
                if len(ring_a) == 3:
                    non_carbon = []
                    atom_by_symb = {}
                    for atom_idx in ring_a:
                        symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                        if symb != 'C':
                            non_carbon.append(atom_idx)
                        if symb not in atom_by_symb:
                            atom_by_symb[symb] = [atom_idx]
                        else:
                            atom_by_symb[symb].append(atom_idx)
                    if len(non_carbon) == 2:
                        rdmol_edit.RemoveBond(*non_carbon)
                    if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                        rdmol_edit.RemoveBond(*atom_by_symb['O'])
                        rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                            rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                        )
                        rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                            rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                        )
            rdmol = rdmol_edit.GetMol()

            for atom in rdmol.GetAtoms():
                if atom.GetFormalCharge() > 0:
                    atom.SetFormalCharge(0)

            return rdmol

        rd_mol = convert_ob_mol_to_rd_mol(mol, struct=xyz)
        try:
            # Post-processing
            rd_mol = postprocess_rd_mol_1(rd_mol)
            rd_mol = postprocess_rd_mol_2(rd_mol)
        except:
            raise MoleculeConstructError("Failed to construct molecule from generated coordinates and atoms.")

        return rd_mol
    
    def decode(self, preds: Dict[str, torch.Tensor], pocket_center: Optional[List[float]]) -> Optional[Molecule]:
        pos = preds["pos"] * self.pos_norm
        if pocket_center is not None:
            pos += pocket_center
        
        preds["pos"] = pos.cpu().numpy().tolist()
        preds["is_aromatic"] = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[x][1] for x in preds["atom_type"].cpu().numpy()]
        preds["atom_type"] = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[x][0] for x in preds["atom_type"].cpu().numpy()]

        try:
            rdmol = self.reconstruct_from_generated(preds["pos"], preds["atom_type"], preds["is_aromatic"])
            molecule = Molecule.from_rdmol(rdmol)
        except Exception as e:
            logging.warn(e)
            molecule = None

        return molecule

class MolCRAFTPocketFeaturizer(PocketFeaturizer):
    def __init__(self, pos_norm: float=1.0) -> None:
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20
        self.pos_norm = pos_norm
        
    def __call__(self, pocket: Pocket) -> Dict[str, Any]:
        elements = torch.LongTensor([atom["atomic_number"] for atom in pocket.atoms])
        elements_one_hot = (elements.view(-1, 1) == self.atomic_numbers.view(1, -1)).long()
        aa_type = torch.LongTensor([atom["aa_type"] for atom in pocket.atoms])
        aa_one_hot = F.one_hot(aa_type, num_classes=self.max_num_aa)
        is_backbone = torch.LongTensor([atom["is_backbone"] for atom in pocket.atoms]).unsqueeze(-1)
        x = torch.cat([elements_one_hot, aa_one_hot, is_backbone], dim=-1).float()
        pos = torch.tensor(pocket.conformer, dtype=torch.float32)
        pocket_center = pos.mean(dim=0)
        pos -= pocket_center
        pos /= self.pos_norm

        return Data(**{
            "atom_feature": x,
            "pos": pos,
            "pocket_center": pocket_center.unsqueeze(0),
            "estimated_ligand_num_atoms": torch.tensor(estimate_ligand_atom_num(pocket)).unsqueeze(0),
        })

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    'silu': nn.SiLU()
}

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            # customized offset
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]

def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False, name='unio2net'):
        super().__init__()
        self.name = name
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.knn = knn
        self.ew_net_type = ew_net_type  # [r, m, none]

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.knn, batch=batch, flow='source_to_target')
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):
        all_x = [x]
        all_h = [h]

        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            src, dst = edge_index

            # edge type (dim: 4)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs

class TimeEmbedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = lambda x: x

    def forward(self, t):
        return self.time_emb(t)
    
class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class MolCRAFT(StructureBasedDrugDesignModel):
    def __init__(self, model_cfg: Config) -> None:
        super(MolCRAFT, self).__init__(model_cfg)
        self.config = model_cfg
        self.node_indicator = getattr(model_cfg, 'node_indicator', False)
        emb_dim = model_cfg.hidden_dim if not self.node_indicator else model_cfg.hidden_dim - 1

        self.time_emb_layer = TimeEmbedLayer()
        self.protein_atom_emb = nn.Linear(model_cfg.protein_atom_feature_dim, emb_dim)
        self.ligand_atom_emb = nn.Linear(model_cfg.ligand_atom_feature_dim + 1, emb_dim)
        self.unio2net = UniTransformerO2TwoUpdateGeneral(**model_cfg.unio2net.todict())
        self.v_inference = nn.Sequential(
            nn.Linear(model_cfg.hidden_dim, model_cfg.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(model_cfg.hidden_dim, model_cfg.ligand_atom_feature_dim),
        )  # [hidden to 13]
        self.sigma1_coord = torch.tensor(model_cfg.sigma1_coord, dtype=torch.float32)
        self.beta1 = torch.tensor(model_cfg.beta1, dtype=torch.float32)

        self.featurizers = {
            "molecule": MolCRAFTMoleculeFeaturizer(pos_norm=model_cfg.pos_norm),
            "pocket": MolCRAFTPocketFeaturizer(pos_norm=model_cfg.pos_norm),
        }
        self.collators = {
            "molecule": PygCollator(follow_batch=["mu_pos"]),
            "pocket": PygCollator(follow_batch=["pos"])
        }

        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

    def continuous_var_bayesian_update(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
        gamma = (1 - torch.pow(self.sigma1_coord, 2 * t))  # [B]
        mu = gamma * x + torch.sqrt(gamma* (1 - gamma)) * torch.randn_like(x)
        return mu, gamma

    def discrete_var_bayesian_update(self, t: torch.Tensor, x: torch.Tensor, K: int) -> torch.Tensor:
        # Eq.(182): β(t) = t**2 β(1)
        beta = (self.config.beta1 * (t**2))  # (B,)

        # Eq.(185): p_F(θ|x;t) = E_{N(y | β(t)(Ke_x−1), β(t)KI)} δ (θ − softmax(y))
        # can be sampled by first drawing y ~ N(y | β(t)(Ke_x−1), β(t)KI)
        # then setting θ = softmax(y)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        return theta

    def compose_context(self, h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
        # previous version has problems when ligand atom types are fixed
        # (due to sorting randomly in case of same element)

        batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
        sort_idx = torch.sort(batch_ctx, stable=True).indices

        mask_ligand = torch.cat([
            torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
            torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
        ], dim=0)[sort_idx]

        batch_ctx = batch_ctx[sort_idx]
        h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
        pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

        return h_ctx, pos_ctx, batch_ctx, mask_ligand

    def interdependency_modeling(
        self,
        time,
        protein_pos,  # transform from the orginal BFN codebase
        protein_v,  # transform from
        batch_protein,  # index for protein
        theta_h_t,
        mu_pos_t,
        batch_ligand,  # index for ligand
        gamma_coord,
        return_all=False,  # legacy from targetdiff
        fix_x=False,
    ):
        """
        Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits).
        Draw output_sample = x' ~ p_O (x' | θ; t).
            continuous x ~ δ(x - x_hat(θ, t))
            discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
        Args:
            time: [node_num x batch_size, 1] := [N_ligand, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [node_num x batch_size, protein_atom_feature_dim] := [N_protein, 27]
            batch_protein: [node_num x batch_size] := [N_protein]
            theta_h_t: [node_num x batch_size, atom_type] := [N_ligand, 13]
            mu_pos_t: [node_num x batch_size, 3] := [N_ligand, 3]
            batch_ligand: [node_num x batch_size] := [N_ligand]
            gamma_coord: [node_num x batch_size, 1] := [N_ligand, 1]
        """
        theta_h_t = 2 * theta_h_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]

        # ---------for targetdiff-----------
        init_ligand_v = theta_h_t
        # time embedding
        time_emb = self.time_emb_layer(time)
        input_ligand_feat = torch.cat([init_ligand_v, time_emb], -1)

        h_protein = self.protein_atom_emb(protein_v)  # [N_protein, self.hidden_dim - 1]
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # [N_ligand, self.hidden_dim - 1]

        if self.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim]
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim]

        h_all, pos_all, batch_all, mask_ligand = self.compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=mu_pos_t,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )
        # get the context for the protein and ligand, while the ligand is h is noisy (h_t)/ pos is also the noise version. (pos_t)

        # time = 2 * time - 1
        outputs = self.unio2net(
            h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x
        )
        final_pos, final_h = outputs["x"], outputs["h"]
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)  # [N_ligand, 13]

        # 1. for continuous, network outputs eps_hat(θ, t)
        # Eq.(84): x_hat(θ, t) = μ / γ(t) − \sqrt{(1 − γ(t)) / γ(t)} * eps_hat(θ, t)
        # 2. for discrete, network outputs Ψ(θ, t)
        # take softmax will do
        return final_ligand_pos, F.softmax(final_ligand_v, dim=-1), torch.zeros_like(mu_pos_t)

    def create_dummy_molecule(self, pocket: Featurized[Pocket]) -> Featurized[Molecule]:
        num_atoms = pocket["estimated_ligand_num_atoms"].cpu()
        return Data(**{
            "mu_pos": torch.zeros(num_atoms.sum().item(), 3),
            "theta_h": torch.ones(num_atoms.sum().item(), self.config.ligand_atom_feature_dim) / self.config.ligand_atom_feature_dim,
            "mu_pos_batch": torch.repeat_interleave(torch.arange(len(num_atoms)), num_atoms),
        }).to(pocket["atom_feature"].device)

    @torch.no_grad()
    def sample(self, molecule: Featurized[Molecule], pocket: Featurized[Pocket]) -> List[Molecule]:
        in_traj, out_traj = [], []
        device = molecule['mu_pos'].device
        num_atoms = molecule['mu_pos_batch'].shape[0]

        for step in tqdm(range(1, self.config.num_sample_steps + 1), desc="Sampling"):
            t = torch.ones((num_atoms, 1), dtype=torch.float, device=device) * (step - 1) / self.config.num_sample_steps
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)
            in_traj.append((molecule["mu_pos"].clone(), molecule["theta_h"].clone()))
            coord_pred, p0_h, _ = self.interdependency_modeling(
                time=t,
                protein_pos=pocket['pos'],
                protein_v=pocket['atom_feature'],
                batch_protein=pocket['pos_batch'],
                batch_ligand=molecule['mu_pos_batch'],
                theta_h_t=molecule['theta_h'],
                mu_pos_t=molecule['mu_pos'],
                gamma_coord=gamma_coord,
            )
            out_traj.append((coord_pred.detach().clone(), p0_h.detach().clone()))

            t = torch.ones((num_atoms, 1), dtype=torch.float, device=device) * step / self.config.num_sample_steps
            molecule['theta_h'] = self.discrete_var_bayesian_update(t, p0_h, self.config.ligand_atom_feature_dim)
            molecule['mu_pos'], _ = self.continuous_var_bayesian_update(t, coord_pred)

            """
            if (step - 1) % 10 == 0:
                print(molecule['mu_pos'][:10])
                print(molecule['theta_h'][:10])
            """

        # Compute final output distribution parameters for p_O (x' | θ; t)
        in_traj.append((molecule["mu_pos"].detach().clone(), molecule["theta_h"].detach().clone()))
        mu_pos_final, p0_h_final, _ = self.interdependency_modeling(
            time=torch.ones((num_atoms, 1)).to(device),
            protein_pos=pocket['pos'],
            protein_v=pocket['atom_feature'],
            batch_protein=pocket['pos_batch'],
            batch_ligand=molecule['mu_pos_batch'],
            theta_h_t=molecule['theta_h'],
            mu_pos_t=molecule['mu_pos'],
            gamma_coord=1 - self.sigma1_coord ** 2,  # γ(t) = 1 − (σ1**2) ** t
        )
        p0_h_final = torch.clamp(p0_h_final, min=1e-6)
        out_traj.append((mu_pos_final.detach().clone(), p0_h_final.detach().clone()))

        num_mols = molecule['mu_pos_batch'].max() + 1
        in_traj_split, out_traj_split = [], []
        out_molecules = []
        for i in range(num_mols):
            cur_molecule = {}
            idx = torch.where(molecule['mu_pos_batch'] == i)[0]
            in_traj_split.append({
                "pos": torch.stack([in_traj[j][0][idx] for j in range(len(in_traj))], dim=0),
                "atom_type": torch.stack([in_traj[j][1][idx] for j in range(len(in_traj))], dim=0),
            })
            out_traj_split.append({
                "pos": torch.stack([out_traj[j][0][idx] for j in range(len(out_traj))], dim=0),
                "atom_type": torch.stack([out_traj[j][1][idx] for j in range(len(out_traj))], dim=0),
            })
            cur_molecule = {
                "pos": out_traj_split[i]["pos"][-1],
                "atom_type": torch.argmax(out_traj_split[i]["atom_type"][-1], dim=-1),
            }
            out_molecules.append(self.featurizers["molecule"].decode(cur_molecule, pocket["pocket_center"][i]))
            """
            import pickle
            from datetime import datetime
            pickle.dump({
                "in_traj": in_traj_split[0],
                "out_traj": out_traj_split[0],
            }, open(f"./tmp/debug_traj_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", "wb"))
            """
        return out_molecules

    def forward_structure_based_drug_design(self, pocket: Featurized[Pocket], label: Featurized[Molecule]) -> Dict[str, torch.Tensor]:
        pass

    @torch.no_grad()
    def predict_structure_based_drug_design(self, pocket: Featurized[Pocket]) -> F.List[Molecule]:
       molecule = self.create_dummy_molecule(pocket)
       return self.sample(molecule, pocket)