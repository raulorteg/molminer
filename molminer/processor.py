import sys

sys.path.append("..")

import gc
import copy
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    DataStructs,
    MolSanitizeException,
    SanitizeMol,
    rdchem,
)
from rdkit.Chem.Draw import MolsToGridImage
from tqdm.auto import tqdm
from typing import Dict, Any, List, Optional, Tuple, Union
from rdkit.Chem.rdchem import Mol

# suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


class GraphProcessor(object):
    def __init__(self):
        pass

    def _find_clusters(self, molecule: Mol) -> List[List[int]]:
        """
        Find ring and bond-based clusters (grains) in the molecule.

        Parameters
        ----------
        molecule : rdkit.Chem.rdchem.Mol
            The molecule to analyze.

        Returns
        -------
        List[List[int]]
            A list of clusters, each as a list of atom indices.
        """
        # smallest set of simple rings
        clusters = [list(x) for x in Chem.rdmolops.GetSSSR(molecule)]
        for bond in molecule.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bond_in_existing_clusters = False
            for cluster in clusters:
                # if both atoms in same cluster then their bond is also there
                # then its not a new cluster
                if (a1 in cluster) and (a2 in cluster):
                    bond_in_existing_clusters = True
                    break

            if not bond_in_existing_clusters:
                clusters.append((a1, a2))

        return clusters

    def _get_all_possible_attachments(
        self, canonical_smiles: str, isring: bool
    ) -> List[List[int]]:
        """
        Compute all possible attachment points for a grain.

        Parameters
        ----------
        canonical_smiles : str
            Canonical SMILES of the grain.
        isring : bool
            True if the grain is a ring, allowing for double anchors.

        Returns
        -------
        List[List[int]]
            List of single or double atom index lists that can be used for attachment.
        """

        # list all possible single-anchor points for the molecule
        possible_anchors = []
        mol = Chem.MolFromSmiles(canonical_smiles)
        for i, atom in enumerate(mol.GetAtoms()):
            max_valence = max(
                rdchem.GetPeriodicTable().GetValenceList(
                    rdchem.GetPeriodicTable().GetElementSymbol(
                        rdchem.GetPeriodicTable().GetAtomicNumber(atom.GetSymbol())
                    )
                )
            )

            if max_valence > atom.GetExplicitValence():
                possible_anchors.append([i])

        # if grain is a "ring-type" grain we need to allow also for double anchors (ring-ring attachments)
        if isring:
            num_atoms = len(mol.GetAtoms())

            # mark the atoms that are full, and so cant be part of an attachment
            flagged_atoms = []
            for i, atom in enumerate(mol.GetAtoms()):
                max_valence = max(
                    rdchem.GetPeriodicTable().GetValenceList(
                        rdchem.GetPeriodicTable().GetElementSymbol(
                            rdchem.GetPeriodicTable().GetAtomicNumber(atom.GetSymbol())
                        )
                    )
                )

                if max_valence <= atom.GetExplicitValence():
                    flagged_atoms.append([i])

            # now append all double attachments, e.g (1,2), (2,3), ... (i, i+1)
            # Note: we also need to add the inverse (2,1), (3,2), (i+1,i) because the order does matter
            # if the atom types are not the same for both atoms in the double anchor
            for i in range(num_atoms):
                if (i not in flagged_atoms) and (
                    (i + 1) % num_atoms not in flagged_atoms
                ):
                    possible_anchors.append([i, (i + 1) % num_atoms])
                    possible_anchors.append([(i + 1) % num_atoms, i])

        return possible_anchors

    def _compute_possible_remaps(
        self, mol_reference: Mol, mol_other: Mol, radius: int = 2
    ) -> List[Dict[int, int]]:
        """
        Compute all valid remappings between atoms in two equivalent molecules.

        Parameters
        ----------
        mol_reference : rdkit.Chem.rdchem.Mol
            The reference molecule.
        mol_other : rdkit.Chem.rdchem.Mol
            The molecule to be remapped.
        radius : int
            Radius for the Morgan fingerprints.

        Returns
        -------
        List[Dict[int, int]]
            A list of index remapping dictionaries.
        """
        # generate atom-wise morgan fingerprints for the molecules
        atom_fps1 = [
            AllChem.GetMorganFingerprintAsBitVect(
                mol_reference,
                fromAtoms=[atom.GetIdx()],
                radius=radius,
                useBondTypes=True,
                useChirality=False,
            )
            for atom in mol_reference.GetAtoms()
        ]
        atom_fps2 = [
            AllChem.GetMorganFingerprintAsBitVect(
                mol_other,
                fromAtoms=[atom.GetIdx()],
                radius=radius,
                useBondTypes=True,
                useChirality=False,
            )
            for atom in mol_other.GetAtoms()
        ]

        similarity_matrix = torch.zeros(
            len(mol_reference.GetAtoms()), len(mol_other.GetAtoms())
        )

        # iterate over atoms in mol1 and find the corresponding atom in mol2
        for i, atom_fp1 in enumerate(atom_fps1):
            # Iterate over atoms in mol2 to find the best match
            for j, atom_fp2 in enumerate(atom_fps2):
                similarity = DataStructs.TanimotoSimilarity(atom_fp1, atom_fp2)
                similarity_matrix[i, j] = similarity

                if similarity == 1.0:
                    similarity_matrix[i, j] = 1.0

        possible_remaps = self._extract_patterns(matrix=similarity_matrix)

        # remove duplicated remaps
        # Note: this happens in bond-type grains, since the
        # similarity matrix is 2x2 the inverse and direct superdiagonals are counted
        # twice
        reduced_possible_remaps = []
        for remap in possible_remaps:
            if remap not in reduced_possible_remaps:
                reduced_possible_remaps.append(remap)

        return reduced_possible_remaps

    def _extract_patterns(self, matrix: torch.Tensor) -> List[Dict[int, int]]:
        """
        Extract index remapping patterns from the similarity matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Similarity matrix between atoms.

        Returns
        -------
        List[Dict[int, int]]
            List of circular remapping patterns.
        """

        num_nodes = matrix.shape[0]

        # the number of possible maps is 2*len(idxs)
        idxs = (matrix[0] == 1.0).nonzero().numpy()

        # initialize the empty list that will containing the mapping dictionaries
        possible_maps = []

        for idx in idxs:
            idx = idx[0]

            # extract direct superdiagonal starting in (0,idx)
            # we sum all elements across the clock direct superdiagonal
            # if the sum is != num_nodes, then the maps is not valid
            possible_map = {}
            count = 0
            for i in range(num_nodes):
                count += int(matrix[i, (i + idx) % num_nodes].item())
                possible_map[i] = (i + idx) % num_nodes

            if count == num_nodes:
                possible_maps.append(possible_map)

            # extract inverse superdiagonal starting in (0,idx)
            # we sum all elements across the clock inverse superdiagonal
            # if the sum is != num_nodes, then the maps is not valid
            possible_map = {}
            count = 0
            for i in range(num_nodes):
                count += int(matrix[i, (num_nodes - i + idx) % num_nodes].item())
                possible_map[i] = (num_nodes - i + idx) % num_nodes

            if count == num_nodes:
                possible_maps.append(possible_map)

        return possible_maps

    def _get_canonicalization_map(
        self, possible_maps: List[Dict[int, int]]
    ) -> Dict[int, int]:
        """
        Return a canonical remapping from the list of possible remaps.

        Parameters
        ----------
        possible_maps : List[Dict[int, int]]
            All valid index remapping dictionaries.

        Returns
        -------
        Dict[int, int]
            Canonical remapping using minimum index.
        """
        num_remaps = len(possible_maps)

        reference_map = {value: key for key, value in possible_maps[0].items()}
        combined_maps = defaultdict(list)
        for i in range(num_remaps):
            for key, value in reference_map.items():
                combined_maps[key].append(possible_maps[i][value])

        canonicalization_map = {
            key: min(values) for key, values in combined_maps.items()
        }
        return canonicalization_map

    def _coarsify(self, molecule: Mol, clusters: List[List[int]]) -> Dict[str, Any]:
        """
        Convert molecule into coarse-grained representation using clusters.

        Parameters
        ----------
        molecule : rdkit.Chem.rdchem.Mol
            The RDKit molecule.
        clusters : List[List[int]]
            List of atom index groups (grains).

        Returns
        -------
        Dict[str, Any]
            Dictionary with coarse information including SMILES, mappings, and attachments.
        """
        adjacency_matrix = np.zeros((len(clusters), len(clusters)))
        cluster_canonical_smiles = []
        cluster_possible_remaps = []
        possible_attachments_lists = []
        canon_attachments_lists = []
        cluster_canon_map = []
        cluster_original_attachments = {}

        for i, cluster in enumerate(clusters):
            editable_mol = Chem.RWMol()
            atom_index_map = {}

            # add the atoms of the grain to an empty molecule
            for atom_idx in cluster:
                original_atom = molecule.GetAtomWithIdx(atom_idx)
                new_atom_idx = editable_mol.AddAtom(original_atom)

                atom_index_map[atom_idx] = new_atom_idx

            # now add the bonds between atoms into the editable molecule
            for bond in molecule.GetBonds():
                start_atom_idx, end_atom_idx = (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                )

                if (start_atom_idx in cluster) and (end_atom_idx in cluster):
                    editable_mol.AddBond(
                        cluster.index(start_atom_idx),
                        cluster.index(end_atom_idx),
                        bond.GetBondType(),
                    )

            # clear aromatic flags from atoms
            for a in editable_mol.GetAtoms():
                if (not a.IsInRing()) and a.GetIsAromatic():
                    a.SetIsAromatic(False)

            # clear aromatic flags from bonds
            for b in editable_mol.GetBonds():
                if (not b.IsInRing()) and b.GetIsAromatic():
                    b.SetIsAromatic(False)

            # sanitize it
            Chem.SanitizeMol(editable_mol)

            # get the non-editable molecule object and the canonical smiles
            built_mol = editable_mol.GetMol()
            smiles_grain = Chem.MolToSmiles(built_mol, canonical=True)
            cluster_canonical_smiles.append(smiles_grain)
            # cluster_editable_molecules.append(editable_mol)

            possible_attachments = self._get_all_possible_attachments(
                canonical_smiles=smiles_grain,
                isring=built_mol.GetRingInfo().NumRings() > 0,
            )
            possible_attachments_lists.append(possible_attachments)

            # by making the grain into canonical smiles we lost the information
            # on the connectivity between this grain and the others, because this connectivity
            # is expressed in the old indexing, which has been reordered in the canonicalization
            # using this function we can retrieve all possible maps between the new and old atom indexes
            possible_indexes_remaps = self._compute_possible_remaps(
                mol_reference=built_mol,
                mol_other=Chem.MolFromSmiles(smiles_grain, sanitize=True),
            )
            new_possible_remaps = []

            for possible_remap in possible_indexes_remaps:
                dict_tmp = {}
                for key in atom_index_map.keys():
                    dict_tmp[key] = possible_remap[atom_index_map[key]]
                new_possible_remaps.append(dict_tmp)

            canon_map = self._get_canonicalization_map(
                possible_maps=new_possible_remaps
            )

            cluster_possible_remaps.append(new_possible_remaps[0])
            cluster_canon_map.append(canon_map)

            # finally lets get what were the atom indexes in the original molecule
            # that were part of an attachment
            for j, cluster_j in enumerate(clusters):
                if i != j:
                    # get indexes of shared atoms
                    shared_atoms = set(clusters[i]).intersection(clusters[j])

                    # only if there are shared atoms between grains they are connected
                    if len(shared_atoms) >= 1:
                        adjacency_matrix[i, j] = 1.0
                        if i not in cluster_original_attachments.keys():
                            cluster_original_attachments[i] = {}

                        cluster_original_attachments[i][j] = list(shared_atoms)

            canon_attachments = []
            canon_attachments_tmp = []
            for anchor in possible_attachments:
                tmp_buffer = []
                for item in anchor:
                    tmp_buffer.append(cluster_canon_map[i][item])

                if tmp_buffer not in canon_attachments_tmp:
                    canon_attachments.append(anchor)
                    canon_attachments_tmp.append(tmp_buffer)

            unique_canon_attachments = [
                list(t) for t in {tuple(lst) for lst in canon_attachments}
            ]
            canon_attachments_lists.append(unique_canon_attachments)

        local_original_attachments = copy.deepcopy(cluster_original_attachments)
        for i in local_original_attachments.keys():
            for j in local_original_attachments[i].keys():
                local_original_attachments[i][j] = [
                    cluster_possible_remaps[i][item]
                    for item in local_original_attachments[i][j]
                ]

        return {
            "nodes_vocab_smiles": cluster_canonical_smiles,  # list of strings ["c1cccc1", "CC", ...]
            "global_to_local": cluster_possible_remaps,  # list of dictionaries [{7:0, 6:1, 5:2, 4:3, 3:4, 2:5}, ....]
            "local_to_canon": cluster_canon_map,  # list of dictionaries [{0:0, 1:0, 2:0, 3:0, 4:0, 5:0}, ....]
            "original_attachments": cluster_original_attachments,  # nested dictionary used to index the atoms shared by two clusters {0: {1: [0,1]}, ....} (grain 0 and 1 have atoms [0,1] in their intersection)
            "local_original_attachments": local_original_attachments,
            "grains_to_atoms_correspondence": clusters,  # list of lists [[1,2], [1,2,3,4], ...]
            "possible_local_attachments": possible_attachments_lists,
            "canon_attachments": canon_attachments_lists,
            "adjacency_matrix": adjacency_matrix,
        }

    def show_molecule(self, molecule: Mol) -> bytes:
        """
        Create a visualization of the molecule with grains and attachments highlighted.

        Parameters
        ----------
        molecule : rdkit.Chem.rdchem.Mol
            The molecule to visualize.

        Returns
        -------
        bytes
            PNG image in bytes.
        """
        clusters = self._find_clusters(molecule=molecule)

        coarse_info = self._coarsify(molecule=molecule, clusters=clusters)

        highlightAtomLists = []
        molecules = []
        for i, cluster in enumerate(coarse_info["grains_to_atoms_correspondence"]):
            # cluster highlighted
            AllChem.Compute2DCoords(molecule)
            molecules.append(molecule)
            highlightAtomLists.append(cluster)

            # attachments highlighted
            molecules.append(molecule)

            buffer = []
            for j in coarse_info["original_attachments"][i].keys():
                buffer += coarse_info["original_attachments"][i][j]
            highlightAtomLists.append(buffer)

            fragment = Chem.MolFromSmiles(
                coarse_info["nodes_vocab_smiles"][i], sanitize=False
            )
            AllChem.Compute2DCoords(fragment)

            molecules.append(fragment)
            tmp = [int(coarse_info["global_to_local"][i][item]) for item in buffer]
            highlightAtomLists.append(tmp)

        # Grid image of highlighted grains
        img = MolsToGridImage(
            molecules,
            molsPerRow=3,
            highlightAtomLists=highlightAtomLists,
            subImgSize=(200, 150),
            returnPNG=True,
            maxMols=150,
        )
        return img

    def coarsify_molecule(self, molecule: Mol) -> Dict[str, Any]:
        """
        Coarsify a molecule into grains and compute associated mappings and attachments.

        Parameters
        ----------
        molecule : rdkit.Chem.rdchem.Mol
            The input molecule.

        Returns
        -------
        Dict[str, Any]
            Dictionary with coarse representation of the molecule.
        """
        clusters = self._find_clusters(molecule=molecule)

        coarse_info = self._coarsify(molecule=molecule, clusters=clusters)

        return coarse_info

    def get_conditions(
        self, coarse_info: Dict[str, Any], molfile: str
    ) -> Dict[str, Any]:
        """
        Extract numerical properties from a molfile and attach them to coarse_info.

        Parameters
        ----------
        coarse_info : dict
            The dictionary with coarse representation of the molecule.
        molfile : str
            Path to the molfile to read properties from.

        Returns
        -------
        Dict[str, Any]
            Updated coarse_info with 'properties' key added.
        """
        with open(molfile, "r") as f:
            for line in f.readlines():
                if "OpenBabel" in line:
                    properties = [float(item) for item in line.strip().split(",")[1:]]
                    properties = torch.tensor(properties)
                    break

        for property_ in properties:
            if np.isnan(property_):
                raise Exception("Nan detected in properties")
        coarse_info["properties"] = properties
        return coarse_info


class Architect:
    def __init__(
        self,
        vocab_fragments_df: pd.DataFrame,
        vocab_attachments_df: pd.DataFrame,
        vocab_anchors_df: pd.DataFrame,
        properties: torch.Tensor,
        sigma: float,
    ):
        """
        Architect class for assembling molecules from fragments.

        Parameters
        ----------
        vocab_fragments_df : pd.DataFrame
            DataFrame with canonical fragment SMILES and attachment metadata.
        vocab_attachments_df : pd.DataFrame
            DataFrame wiht fragments and their attachment points.
        vocab_anchors_df : pd.DataFrame
            DataFrame with anchor type definitions.
        properties : torch.Tensor
            Tensor of conditioning properties.
        sigma : float
            Sigma value for computing the distance bias in the attn mechanism.
        """
        self.vocab_fragments_df = vocab_fragments_df
        self.vocab_attachments_df = vocab_attachments_df
        self.vocab_anchors_df = vocab_anchors_df

        # this parameter controls the std used to make pairwise distances into attn_scores
        self.sigma = sigma

        self.properties = properties
        self.reset()

    def reset(self) -> None:
        """Reset the molecule to an empty initialization state."""
        self.molecule = {
            "nodes_vocab_smiles": [],  # SMILES strings of the fragments
            "nodes_vocab_index": [],  # int indexes representing the fragments in th vocabulary
            "local_to_global": [],  # list of dictionary used to map local atom index into the global index
            "rdkit_molecule": None,  # editable molecule
            "grains_to_anchors_correspondence": [
                []
            ],  # list of lists, correspondance between grains and anchors
            "grains_to_atoms_correspondence": [
                []
            ],  # list of lists, correspondance between grains and atoms
            "atomic_positions": [],  # atomic coordinates
            "fragment_positions": [],  # fragment's positions
            "queue": [],  # queue for the decoding of the molecule
            "nodes_saturation": [],  # list of list of three elements (used docks, free docks, total_docks)
        }

    def start(self, smiles: str) -> None:
        """
        Initialize molecule with a starting fragment.

        Parameters
        ----------
        smiles : str
            SMILES string of the initial fragment.
        """
        # A fragment was given as a SMILES, get the index it corresponds to in the
        # vocabulary of grains, and the possible anchors
        tmp_result = self.vocab_fragments_df.loc[
            self.vocab_fragments_df["canon_smiles"] == smiles, ["anchors"]
        ]
        smiles_idx, possible_anchors = (
            tmp_result.index.values[0],
            tmp_result["anchors"].values[0],
        )

        self.molecule["nodes_vocab_smiles"].append(smiles)
        self.molecule["nodes_vocab_index"].append(smiles_idx)
        self.molecule["nodes_saturation"].append(
            [len(possible_anchors), 0, len(possible_anchors)]
        )

        # construct the editable molecule object
        molecule = Chem.MolFromSmiles(smiles)
        grains_to_atoms_correspondence = []
        local_to_global_map = {}
        editable_molecule = Chem.RWMol()
        for atom in molecule.GetAtoms():
            local_atom_idx = editable_molecule.AddAtom(atom)
            global_atom_idx = atom.GetIdx()
            grains_to_atoms_correspondence.append(global_atom_idx)
            local_to_global_map[local_atom_idx] = global_atom_idx

        for bond in molecule.GetBonds():
            editable_molecule.AddBond(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                order=bond.GetBondType(),
            )

        self.molecule["local_to_global"] = [local_to_global_map]
        self.molecule["rdkit_molecule"] = editable_molecule
        self.molecule["grains_to_anchors_correspondence"] = []
        self.molecule["grains_to_atoms_correspondence"] = [
            grains_to_atoms_correspondence
        ]
        self._compute_positions(
            num_conformers=1
        )  # sets ["atomic_positions", "fragment_positions"]

        for cand_anchor in possible_anchors:
            self.molecule["queue"].append(
                {
                    "focal_fragment_order": 0,
                    "focal_fragment_idx": smiles_idx,
                    "focal_fragment_smiles": smiles,
                    "focal_fragment_local_anchor": cand_anchor,
                    "focal_fragment_global_anchor": [
                        self.molecule["local_to_global"][0][item]
                        for item in cand_anchor
                    ],
                }
            )
        random.shuffle(self.molecule["queue"])
        self._purge_queue()

    def _purge_queue(self) -> None:
        """Remove invalid docking points from the queue based on atom valence constraints."""
        molecule = self.molecule["rdkit_molecule"].GetMol()
        Chem.SanitizeMol(molecule, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)

        flagged_atoms = []
        for atom in molecule.GetAtoms():
            max_valence = max(
                rdchem.GetPeriodicTable().GetValenceList(
                    rdchem.GetPeriodicTable().GetElementSymbol(
                        rdchem.GetPeriodicTable().GetAtomicNumber(atom.GetSymbol())
                    )
                )
            )

            if atom.GetExplicitValence() >= max_valence:
                flagged_atoms.append(atom.GetIdx())

        purged_queue = []
        free_docks = [0 for _ in self.molecule["nodes_saturation"]]
        for item in self.molecule["queue"]:
            if not any(
                element in flagged_atoms
                for element in item["focal_fragment_global_anchor"]
            ):
                purged_queue.append(item)
                free_docks[item["focal_fragment_order"]] += 1

        for i in range(len(free_docks)):
            self.molecule["nodes_saturation"][i][0] = free_docks[i]

        purged_count = len(self.molecule["queue"]) - len(purged_queue)
        self.molecule["queue"] = purged_queue

    def can_dock(self, smiles: str, attachment: List[int]) -> Tuple[bool, str]:
        """
        Check if a fragment can be docked to the current molecule.

        Parameters
        ----------
        smiles : str
            SMILES of the fragment to attach.
        attachment : list of int
            Atom indices of the fragment's attachment point.

        Returns
        -------
        bool
            Whether docking is allowed.
        str
            Message indicating the result.
        """

        # exception, if smiles=="<empty>" its a cauterization, and that is always possible
        if smiles == "<empty>":
            return True, "Cauterization is valid."

        # simple check, if the number of docking points differs then its not possible
        if len(attachment) != len(
            self.molecule["queue"][0]["focal_fragment_global_anchor"]
        ):
            active_dock_points = len(
                self.molecule["queue"][0]["focal_fragment_global_anchor"]
            )
            return (
                False,
                f"[FAILED] Different number of docking points. Active dock expects {active_dock_points} and {len(attachment)} were given.",
            )

        molecule = Chem.MolFromSmiles(smiles)
        # simple check, validate that the atom types at docking points match
        if len(attachment) == 1:
            atom_origin = (
                self.molecule["rdkit_molecule"]
                .GetAtomWithIdx(
                    self.molecule["queue"][0]["focal_fragment_global_anchor"][0]
                )
                .GetSymbol()
                .lower()
            )
            atom_destination = (
                molecule.GetAtomWithIdx(attachment[0]).GetSymbol().lower()
            )

            if atom_origin != atom_destination:
                return (
                    False,
                    f"[Failed] Different atom types at single dock. {atom_origin}!={atom_destination}",
                )

        # case len(attachment) == 2:
        else:
            atom_origin_0 = (
                self.molecule["rdkit_molecule"]
                .GetAtomWithIdx(
                    self.molecule["queue"][0]["focal_fragment_global_anchor"][0]
                )
                .GetSymbol()
                .lower()
            )
            atom_destination_0 = (
                molecule.GetAtomWithIdx(attachment[0]).GetSymbol().lower()
            )

            atom_origin_1 = (
                self.molecule["rdkit_molecule"]
                .GetAtomWithIdx(
                    self.molecule["queue"][0]["focal_fragment_global_anchor"][1]
                )
                .GetSymbol()
                .lower()
            )
            atom_destination_1 = (
                molecule.GetAtomWithIdx(attachment[1]).GetSymbol().lower()
            )

            if (atom_origin_0 != atom_destination_0) or (
                atom_origin_1 != atom_destination_1
            ):
                return (
                    False,
                    f"[Failed] Different atom types at double dock. ({atom_origin_0},{atom_origin_1})!=({atom_destination_0},{atom_destination_1})",
                )

        # final check, we attempt to create the attachment and sanitize the resulting molecule
        # to verify valence rules and aromacity is correct.
        editable_molecule = copy.deepcopy(self.molecule["rdkit_molecule"])
        local_to_global = {
            k: v
            for k, v in zip(
                attachment, self.molecule["queue"][0]["focal_fragment_global_anchor"]
            )
        }
        for idx, atom in enumerate(molecule.GetAtoms()):
            # if not one of the atoms of the attachment then add it (otherwise they would be repeated)
            if idx not in attachment:
                new_atom = editable_molecule.AddAtom(atom)
                local_to_global[idx] = new_atom

        for bond in molecule.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # if the bond is between two atoms in the attachment dont add it, its already there
            if (begin_idx not in attachment) or (end_idx not in attachment):
                editable_molecule.AddBond(
                    local_to_global[begin_idx],
                    local_to_global[end_idx],
                    order=bond.GetBondType(),
                )
        mol_to_sanitize = editable_molecule.GetMol()

        try:
            SanitizeMol(mol_to_sanitize)
            return True, "Molecule is valid."
        except MolSanitizeException as e:
            return False, str(e)

    def dock(
        self, smiles: str, attachment: List[int], usethreads: bool = False
    ) -> None:
        """
        Attach a fragment to the current molecule.

        Parameters
        ----------
        smiles : str
            SMILES of the fragment.
        attachment : list of int
            Atom indices of the attachment.
        usethreads : bool, optional
            Whether to use multithreaded conformer optimization.
        """

        if smiles == "<empty>":
            self.molecule["queue"] = self.molecule["queue"][1:]
            random.shuffle(self.molecule["queue"])
            self._purge_queue()  # so that it updates the free_docks counters
            return

        # update the used_docks counter of focal fragment
        focal_idx = self.molecule["queue"][0]["focal_fragment_order"]

        self.molecule["nodes_saturation"][focal_idx][1] += 1
        self.molecule["nodes_saturation"][focal_idx][1] = min(
            self.molecule["nodes_saturation"][focal_idx][1],
            self.molecule["nodes_saturation"][focal_idx][2],
        )

        # if its valid then perform the attachment
        molecule = Chem.MolFromSmiles(smiles)
        editable_molecule = self.molecule["rdkit_molecule"]
        local_to_global = {
            k: v
            for k, v in zip(
                attachment, self.molecule["queue"][0]["focal_fragment_global_anchor"]
            )
        }
        for idx, atom in enumerate(molecule.GetAtoms()):
            # if not one of the atoms of the attachment then add it (otherwise they would be repeated)
            if idx not in attachment:
                new_atom = editable_molecule.AddAtom(atom)
                local_to_global[idx] = new_atom

        for bond in molecule.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # if the bond is between two atoms in the attachment dont add it, its already there
            if (begin_idx not in attachment) or (end_idx not in attachment):
                editable_molecule.AddBond(
                    local_to_global[begin_idx],
                    local_to_global[end_idx],
                    order=bond.GetBondType(),
                )

        tmp_result = self.vocab_fragments_df.loc[
            self.vocab_fragments_df["canon_smiles"] == smiles, ["anchors"]
        ]
        smiles_idx, possible_anchors = (
            tmp_result.index.values[0],
            tmp_result["anchors"].values[0],
        )

        self.molecule["nodes_vocab_smiles"].append(smiles)
        self.molecule["nodes_saturation"].append(
            [len(possible_anchors) - 1, 1, len(possible_anchors)]
        )
        self.molecule["nodes_vocab_index"].append(smiles_idx)
        self.molecule["local_to_global"].append(local_to_global)
        self.molecule["rdkit_molecule"] = editable_molecule
        self.molecule["grains_to_atoms_correspondence"].append(
            [k for k in local_to_global.values()]
        )
        self._compute_positions(
            usethreads=usethreads
        )  # sets ["atomic_positions", "fragment_positions"]

        curr_idx = len(self.molecule["nodes_vocab_smiles"]) - 1
        for cand_anchor in possible_anchors:
            self.molecule["queue"].append(
                {
                    "focal_fragment_order": curr_idx,
                    "focal_fragment_idx": smiles_idx,
                    "focal_fragment_smiles": smiles,
                    "focal_fragment_local_anchor": cand_anchor,
                    "focal_fragment_global_anchor": [
                        self.molecule["local_to_global"][curr_idx][item]
                        for item in cand_anchor
                    ],
                }
            )

        self.molecule["queue"] = self.molecule["queue"][1:]
        random.shuffle(self.molecule["queue"])
        self._purge_queue()
        return

    def create_query(self) -> Dict[str, Any]:
        """
        Create a query for the model to predict the next action.

        Returns
        -------
        dict
            Query dictionary with fragment indices, distances, properties, etc.
        """

        # get the atomic symbols of the atoms in the attachment
        atom_symbols = [
            self.molecule["rdkit_molecule"].GetAtomWithIdx(idx).GetSymbol().lower()
            for idx in self.molecule["queue"][0]["focal_fragment_global_anchor"]
        ]
        # attachment_type = self.attachment_vocab.index(atom_symbols)
        index = next(
            (
                i
                for i, lst in enumerate(self.vocab_anchors_df["anchors"])
                if lst == atom_symbols
            ),
            None,
        )
        attachment_type = index

        hit_location = (
            torch.tensor(
                np.mean(
                    self.molecule["atomic_positions"][
                        self.molecule["queue"][0]["focal_fragment_global_anchor"], :
                    ],
                    axis=0,
                )
            )
            .unsqueeze(0)
            .float()
        )
        distances_to_hit = torch.exp(
            -(torch.cdist(hit_location, self.molecule["fragment_positions"], p=2) ** 2)
            / (2 * self.sigma**2)
        )

        molecule = {
            "nodes_vocab_index": torch.tensor(self.molecule["nodes_vocab_index"]),
            "pairwise_distances": self.molecule["pairwise_distances"],
            "focal_fragment_idx": torch.tensor(
                self.molecule["queue"][0]["focal_fragment_idx"]
            ),
            "focal_fragment_order": torch.tensor(
                self.molecule["queue"][0]["focal_fragment_order"]
            ),
            "distances_to_hit": distances_to_hit,
            "properties": self.properties,
            "focal_attachment_type": torch.tensor(attachment_type),
            "focal_attachment_order": self.molecule["queue"][0]["focal_fragment_order"],
            "focal_attachment_local_anchor": self.molecule["queue"][0][
                "focal_fragment_local_anchor"
            ],
            "focal_attachment_symbols": atom_symbols,
            "nodes_saturation": self.molecule["nodes_saturation"],
        }
        return molecule

    def _compute_positions(
        self, num_conformers: int = 5, maxIters: int = 80, usethreads: bool = False
    ) -> None:
        """
        Compute atomic and fragment-level 3D positions from conformer optimization.

        Parameters
        ----------
        num_conformers : int
            Number of conformers to generate.
        maxIters : int
            Max iterations for UFF optimization.
        usethreads : bool
            Whether to use multithreading.
        """

        molecule = copy.deepcopy(self.molecule["rdkit_molecule"].GetMol())
        Chem.SanitizeMol(molecule, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
        molecule = Chem.AddHs(molecule)

        # create the conformers
        AllChem.EmbedMultipleConfs(molecule, num_conformers)

        # optimize the conformers. Note: numThreads=0 to use all available
        force_field = AllChem.UFFGetMoleculeForceField(molecule)
        if usethreads:
            numThreads = 0
        else:
            numThreads = 1
        results = AllChem.OptimizeMoleculeConfs(
            molecule, force_field, numThreads=numThreads, maxIters=maxIters
        )

        # find the lowest energy conformer
        conf_id = None
        min_energy = float("inf")
        for current_id, result in enumerate(results):
            if (result[1]) < min_energy:
                min_energy = result[1]
                conf_id = current_id

        # get the positions of the lowest energy conformer
        positions = np.array(molecule.GetConformer(conf_id).GetPositions())

        self.molecule["atomic_positions"] = positions

        # pool positions into fragments to compute fragments' positions
        fragment_positions = []
        for i in range(len(self.molecule["nodes_vocab_smiles"])):
            fragment_positions.append(
                np.mean(
                    self.molecule["atomic_positions"][
                        self.molecule["grains_to_atoms_correspondence"][i], :
                    ],
                    axis=0,
                ).tolist()
            )
        fragment_positions = torch.tensor(fragment_positions)
        self.molecule["fragment_positions"] = fragment_positions
        self.molecule["pairwise_distances"] = torch.exp(
            -(torch.cdist(fragment_positions, fragment_positions, p=2) ** 2)
            / (2 * self.sigma**2)
        )


class StoryAnotator:
    def __init__(
        self,
        coarse_info: Dict[str, Any],
        vocab_attachments_df: pd.DataFrame,
        vocab_fragments_df: pd.DataFrame,
    ):
        """
        Annotates a coarse-grained molecule rollout.

        Parameters
        ----------
        coarse_info : dict
            Coarse molecule representation including fragments and attachments.
        vocab_attachments_df : pd.DataFrame
            DataFrame of (fragment, anchor) pairs in the vocabulary.
        vocab_fragments_df : pd.DataFrame
            DataFrame of fragment canonical SMILES and attachment info.
        """
        self.vocab_attachments_df = vocab_attachments_df
        self.vocab_fragments_df = vocab_fragments_df
        self.coarse_info = coarse_info

        self.properties = self.coarse_info["properties"]
        self.order_index_map_new2old = {}
        self.order_index_map_old2new = {}

    def _get_canonical_attachment(
        self, smiles: str, anchor: List[int]
    ) -> Optional[List[int]]:
        """
        Match a candidate anchor to its canonical representation.

        Parameters
        ----------
        smiles : str
            Canonical SMILES string of the fragment.
        anchor : list of int
            Atom indices representing the anchor.

        Returns
        -------
        list of int or None
            Canonicalized anchor if found.
        """
        index = self.vocab_fragments_df[
            self.vocab_fragments_df["canon_smiles"] == smiles
        ].index.item()

        canon_anchors = self.vocab_fragments_df.loc[index]["canon_anchors"]
        canon_map = self.vocab_fragments_df.loc[index]["local_to_canon"]

        canon_rep_cand_anchor = [canon_map[item] for item in anchor]

        for canon_anchor in canon_anchors:
            canon_rep_ref_anchor = [canon_map[item] for item in canon_anchor]
            if canon_rep_cand_anchor == canon_rep_ref_anchor:
                return canon_anchor

        return None

    def start(self) -> str:
        """
        Randomly select a starting fragment for rollout reconstruction.

        Returns
        -------
        str
            SMILES string of the starting fragment.
        """
        order_index = random.randint(0, len(self.coarse_info["nodes_vocab_smiles"]) - 1)
        smiles = self.coarse_info["nodes_vocab_smiles"][order_index]

        self.order_index_map_new2old = {0: order_index}
        self.order_index_map_old2new = {order_index: 0}
        return smiles

    def process_query(
        self, query: Dict[str, Any]
    ) -> List[Union[str, List[int], Optional[List[int]]]]:
        """
        Given a query, determine the next fragment and its attachment.

        Parameters
        ----------
        query : dict
            Model query containing current fragment and attachment info.

        Returns
        -------
        list
            [next_smiles, next_attachment, global_attachment_atoms]
        """
        new_order_idx = query["focal_attachment_order"]
        old_order_idx = self.order_index_map_new2old[new_order_idx]

        local_attachments = query["focal_attachment_local_anchor"]
        old_order_attached = None
        other_attachment = None
        for k, v in self.coarse_info["local_original_attachments"][
            old_order_idx
        ].items():
            if (v == local_attachments) and (
                k not in self.order_index_map_old2new.keys()
            ):
                old_order_attached = k
                other_attachment = self.coarse_info["local_original_attachments"][k][
                    old_order_idx
                ]
                break

        if old_order_attached is not None:
            if old_order_attached not in self.order_index_map_old2new.keys():
                new_order_attached = max(self.order_index_map_new2old.keys()) + 1
                self.order_index_map_new2old[new_order_attached] = old_order_attached
                self.order_index_map_old2new[old_order_attached] = new_order_attached

            next_smiles = self.coarse_info["nodes_vocab_smiles"][old_order_attached]
            next_attachment = self._get_canonical_attachment(
                smiles=next_smiles, anchor=other_attachment
            )
            index = self.vocab_attachments_df[
                (self.vocab_attachments_df["smiles"] == next_smiles)
                & (
                    self.vocab_attachments_df["anchors"].apply(
                        lambda x: x == next_attachment
                    )
                )
            ].index
            return [
                next_smiles,
                next_attachment,
                [int(item) for item in other_attachment],
            ]

        else:
            return ["<empty>", [], None]


def create_story(
    coarse_info: Dict[str, Any],
    architect: Any,
    vocab_fragments_df: pd.DataFrame,
    vocab_attachments_df: pd.DataFrame,
    return_snapshots: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Reconstruct a molecular rollout/story from coarse information.

    Parameters
    ----------
    coarse_info : dict
        Coarsified molecule with fragment information.
    architect : Architect
        Architect object to handle molecule construction.
    vocab_fragments_df : pd.DataFrame
        DataFrame of fragments and metadata.
    vocab_attachments_df : pd.DataFrame
        DataFrame of (fragment, anchor) combinations.
    return_snapshots : bool, optional
        Whether to return intermediate molecule states.

    Returns
    -------
    story : list of dict
        List of actions taken to reconstruct the molecule.
    snapshots : list of dict, optional
        List of molecule states after each action, if `return_snapshots` is True.
    """
    story = []
    snapshots = []
    anotator = StoryAnotator(
        coarse_info=coarse_info,
        vocab_attachments_df=vocab_attachments_df,
        vocab_fragments_df=vocab_fragments_df,
    )
    architect.reset()
    architect.properties = anotator.properties

    smiles = anotator.start()
    architect.start(smiles=smiles)

    if return_snapshots:
        snapshots.append(copy.deepcopy(architect.molecule))

    while len(architect.molecule["queue"]) > 0:
        query = architect.create_query()
        (
            next_fragment,
            next_highlightAtoms,
            other_highlightatoms,
        ) = anotator.process_query(query=query)
        nodes_saturation = copy.deepcopy(query["nodes_saturation"])
        query_ = copy.deepcopy(query)

        bool_flag, reason = architect.can_dock(next_fragment, other_highlightatoms)
        if bool_flag:
            architect.dock(next_fragment, other_highlightatoms)
            if return_snapshots:
                snapshots.append(copy.deepcopy(architect.molecule))

        query_["next_token"] = vocab_attachments_df[
            (vocab_attachments_df["smiles"] == next_fragment)
            & (
                vocab_attachments_df["anchors"].apply(
                    lambda x: x == next_highlightAtoms
                )
            )
        ].index.to_list()[0]

        query_["nodes_saturation"] = nodes_saturation

        story.append(query_)

    del architect, vocab_fragments_df, vocab_attachments_df
    gc.collect()

    if return_snapshots:
        return story, snapshots
    else:
        return story
