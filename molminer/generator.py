import sys
from typing import Union

sys.path.append("..")

import ast

import joblib
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from molminer.models import FragmentStarter, MoleculeTransformer
from molminer.processor import Architect
from molminer.scalers import PropertyScaler
from scipy.stats import multivariate_normal
from molminer.utils import (
    collate_fn,
    load_molminer,
    load_starter,
    top_k_sample,
)
from tqdm import tqdm
import pathlib


class MolecularGenerator(object):
    """
    Molecule generator for the MolMiner pipeline.

    Combines:
    - A Gaussian Mixture Model (GMM) for property vector sampling
    - A FragmentStarter model for initializing fragment rollouts
    - An autoregressive MolMiner transformer for fragment-by-fragment generation

    Used for both conditional and unconditional generation of 3D-aware molecules.
    """

    def __init__(
        self,
        ckpt_molminer: Union[str, pathlib.Path],
        ckpt_starter: Union[str, pathlib.Path],
        ckpt_gmm: Union[str, pathlib.Path],
        stats_path: Union[str, pathlib.Path],
        vocab_fragments: Union[str, pathlib.Path],
        vocab_attachments: Union[str, pathlib.Path],
        vocab_anchors: Union[str, pathlib.Path],
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Initialize and load pretrained components: MolMiner, FragmentStarter, GMM,
        property scaler, and vocabularies.

        Parameters
        ----------
        ckpt_molminer, ckpt_starter, ckpt_gmm : str or Path
            Checkpoint files for the trained models.
        stats_path : str or Path
            Path to JSON file with property statistics for scaling.
        vocab_fragments, vocab_attachments, vocab_anchors : str or Path
            Vocabulary CSVs generated during preprocessing.
        device : str or torch.device, default="cuda"
            Device to run the models on.
        """

        # load the model checkpoints
        self.molminer, _, _ = load_molminer(
            checkpoint_path=ckpt_molminer,
            model_class=MoleculeTransformer,
            device=device,
        )
        self.starter, _, _ = load_starter(
            checkpoint_path=ckpt_starter, model_class=FragmentStarter, device=device
        )
        self.gmm = joblib.load(ckpt_gmm)
        self.molminer.eval()
        self.starter.eval()

        # load the scaler and vocabularies
        self.scaler = PropertyScaler(stats_path=stats_path)

        self.vocab_fragments = pd.read_csv(vocab_fragments, index_col=0)
        for col in ["anchors", "canon_anchors", "local_to_canon"]:
            self.vocab_fragments[col] = self.vocab_fragments[col].map(ast.literal_eval)
        self.fragments_smiles = self.vocab_fragments["canon_smiles"].to_list()

        self.vocab_attachments = pd.read_csv(vocab_attachments, index_col=0)
        for col in ["anchors"]:
            self.vocab_attachments[col] = self.vocab_attachments[col].map(
                ast.literal_eval
            )

        self.vocab_anchors = pd.read_csv(vocab_anchors, index_col=0)
        for col in ["anchors"]:
            self.vocab_anchors[col] = self.vocab_anchors[col].map(ast.literal_eval)

        self.device = device
        self.prop_order = [
            "logP",
            "qed",
            "SAS",
            "FractionCSP3",
            "molWt",
            "TPSA",
            "MR",
            "hbd",
            "hba",
            "num_rings",
            "num_rotable_bonds",
            "num_quiral_centers",
        ]
        self.num_dims = len(self.prop_order)

    def _conditional_sample_gmm(
        self, x: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample missing property values from the GMM, conditioned on known properties.

        Parameters
        ----------
        x : np.ndarray
            Full property vector in scaled space.
        mask : np.ndarray
            Binary mask: 0 for observed, 1 for unknown dimensions.

        Returns
        -------
        tuple
            Original vector and the same vector with missing values filled in.
        """

        mixture_weights = self.gmm.weights_  # pi_k, shape: (num_components, )
        mixture_means = self.gmm.means_  # mu_k, shape: (num_components, d)
        mixture_covariances = (
            self.gmm.covariances_
        )  # Sigma_k, shape: (num_components, d, d)
        mixture_components = self.gmm.n_components  # scalar num_components
        d = mixture_means.shape[1]

        # cast sample, and mask into np.array
        x = np.asarray(x)
        mask = np.atleast_1d(mask)

        idx_obs = np.where(mask == 0)[0]  # known elements in the sample
        idx_target = np.where(mask == 1)[0]  # unknown elements to sample

        x_obs = x[idx_obs]

        mu_xtar_given_xobs = []
        Sigma_xtar_given_xobs = []
        likelihoods_obs = []

        for k in range(mixture_components):
            # partition the mixtures mean matrix
            mu_x_obs = mixture_means[k, idx_obs]  # (d_obs, )
            mu_x_target = mixture_means[k, idx_target]  # (d_target, )

            # partition the mixtures covariance matrix
            Sigma_obsobs = mixture_covariances[k][
                np.ix_(idx_obs, idx_obs)
            ]  # Shape (d_obs, d_obs)
            Sigma_tartar = mixture_covariances[k][
                np.ix_(idx_target, idx_target)
            ]  # Shape (d_target, d_target)
            Sigma_tarobs = mixture_covariances[k][
                np.ix_(idx_target, idx_obs)
            ]  # Shape (d_target, d_obs)
            Sigma_obstar = mixture_covariances[k][
                np.ix_(idx_obs, idx_target)
            ]  # Shape (d_obs, d_target)

            # compute conditional mean and variance
            Sigma_obsobs += 1e-6 * np.eye(Sigma_obsobs.shape[0])  # to avoid singular
            mu_xtar_given_xobs_k = mu_x_target + Sigma_tarobs @ np.linalg.inv(
                Sigma_obsobs
            ) @ (x_obs - mu_x_obs)  # Shape (d_target, )
            Sigma_xtar_given_xobs_k = (
                Sigma_tartar - Sigma_tarobs @ np.linalg.inv(Sigma_obsobs) @ Sigma_obstar
            )  # Shape (d_target, d_target)

            mu_xtar_given_xobs.append(mu_xtar_given_xobs_k.flatten())
            Sigma_xtar_given_xobs.append(Sigma_xtar_given_xobs_k)
            likelihoods_obs.append(
                multivariate_normal.pdf(x_obs, mean=mu_x_obs, cov=Sigma_obsobs)
            )

        # compute the 'mixture weights' of the conditional gaussian mixture
        likelihoods_obs = np.array(likelihoods_obs)
        w_k = (mixture_weights * likelihoods_obs) / np.sum(
            mixture_weights * likelihoods_obs
        )

        # Sample a k component to use the gaussian from (weighted by w_k)
        k_selected = np.random.choice(mixture_components, p=w_k)

        # then based on the k selected sample from their gaussian mixture
        xa = np.random.multivariate_normal(
            mu_xtar_given_xobs[k_selected], Sigma_xtar_given_xobs[k_selected]
        )

        # then lets complete the original sample, and return both original and 'reconstructed'
        x_recon = x.copy()
        x_recon[idx_target] = xa

        return (x, x_recon)

    def sample(
        self,
        *,
        logP: Union[float, None] = None,
        qed: Union[float, None] = None,
        SAS: Union[float, None] = None,
        FractionCSP3: Union[float, None] = None,
        molWt: Union[float, None] = None,
        TPSA: Union[float, None] = None,
        MR: Union[float, None] = None,
        hbd: Union[float, None] = None,
        hba: Union[float, None] = None,
        num_rings: Union[float, None] = None,
        num_rotable_bonds: Union[float, None] = None,
        num_quiral_centers: Union[float, None] = None,
        num_samples: int = 1,
        topk: int = 5,
        weighted: bool = True,
        greedy: bool = False,
        max_tries: int = 10,
    ) -> list[str]:
        """
        Generate one or more molecules, optionally conditioned on property values.

        Parameters
        ----------
        logP, qed, ..., num_quiral_centers : float or None
            Optional property constraints (unscaled). Leave as None for unconditional.
        num_samples : int
            Number of molecules to generate.
        topk : int
            Number of initial fragments to consider for seed selection.
        weighted : bool
            Use top-k logits as weights (if False, uniform distribution).
        greedy : bool
            If True, use deterministic argmax sampling in the transformer.
        max_tries : int
            Max attempts per attachment point before skipping.

        Returns
        -------
        list of str
            SMILES strings of successfully generated molecules.
        """

        prompted = {
            "logP": logP,
            "qed": qed,
            "SAS": SAS,
            "FractionCSP3": FractionCSP3,
            "molWt": molWt,
            "TPSA": TPSA,
            "MR": MR,
            "hbd": hbd,
            "hba": hba,
            "num_rings": num_rings,
            "num_rotable_bonds": num_rotable_bonds,
            "num_quiral_centers": num_quiral_centers,
        }

        generated_samples = []

        # Template/mask for a sample
        proto_x = np.zeros(self.num_dims, dtype=float)  # placeholder
        proto_msk = np.ones(self.num_dims, dtype=int)  # 1 = unknown

        for i, prop in enumerate(self.prop_order):
            v = prompted[prop]
            if v is not None:
                proto_x[i] = self.scaler.scale(v, property_name=prop)  # scaled
                proto_msk[i] = 0  # observed

        # if everything is unknown -> unconditional GMM
        if proto_msk.sum() == self.num_dims:
            scaled_conditions, _ = self.gmm.sample(n_samples=num_samples)
        else:
            # Conditional sampling
            samples = []
            for _ in range(num_samples):
                _, x_recon = self._conditional_sample_gmm(proto_x, proto_msk)
                samples.append(x_recon)
            scaled_conditions = np.vstack(samples)

        for scaled_condition in tqdm(scaled_conditions):
            # generate a molecule using the conditions
            smiles, failedFlag, failedMsg = self._sample(
                c=torch.tensor(scaled_condition, dtype=torch.float).to(self.device),
                topk=topk,
                weighted=weighted,
                greedy=greedy,
                max_tries=max_tries,
            )

            if not failedFlag:
                generated_samples.append(smiles)

        return generated_samples

    def _sample(self, c, topk, weighted, greedy, max_tries):
        """
        Core sampling routine: seed selection + autoregressive fragment generation.

        Called internally during `sample()` for each property vector.
        """
        # predict the seed/initial grain
        with torch.no_grad():
            logits = self.starter(x=c.unsqueeze(0)).squeeze(0).cpu()

        fragment_idx = top_k_sample(logits, k=topk, weighted=weighted)
        smiles = self.fragments_smiles[fragment_idx]

        architect = Architect(
            vocab_fragments_df=self.vocab_fragments,
            vocab_attachments_df=self.vocab_attachments,
            vocab_anchors_df=self.vocab_anchors,
            properties=c,
            sigma=2.0,
        )

        ErrorRaised = False
        architect.start(smiles=smiles)

        while (len(architect.molecule["queue"]) > 0) and (not ErrorRaised):
            # create the input to the model
            batch = collate_fn([architect.create_query()], inference_mode=True)

            # make a prediction
            with torch.no_grad():
                logits = self.molminer(
                    atom_ids=batch["nodes_vocab_index"].to(self.device),
                    attn_mask=batch["pairwise_distances"].to(self.device),
                    focal_attachment_order=batch["focal_attachment_order"].to(
                        self.device
                    ),
                    nodes_saturation=batch["nodes_saturation"].to(self.device),
                    focal_attachment_type=batch["focal_attachment_type"].to(
                        self.device
                    ),
                    properties=batch["properties"].to(self.device),
                    attn_mask_readout=batch["distances_to_hit"].to(self.device),
                )

            # cast to float64 to prevent precision errors (probs must sum up to 1)
            logits = logits.squeeze(0).double()
            probs = torch.softmax(logits, dim=0).cpu()

            dock_accepted = False
            num_tries = 0
            while not dock_accepted:
                if num_tries > max_tries:
                    ErrorRaised = True
                    fail_reason = f"Max. number of docking attempts reached (tries {num_tries}/{max_tries})"
                    break

                probs_ = probs / sum(probs)

                if greedy:
                    i = int(torch.argmax(probs_, dim=-1))
                else:
                    i = np.random.choice(np.arange(0, len(probs)), p=probs_.numpy())

                tmp = self.vocab_attachments.iloc[i]
                next_fragment, next_highlightAtoms = (
                    tmp["smiles"],
                    tmp["anchors"],
                )

                bool_flag, fail_reason = architect.can_dock(
                    next_fragment, next_highlightAtoms
                )
                if bool_flag:
                    try:
                        architect.dock(
                            next_fragment, next_highlightAtoms, usethreads=True
                        )
                        dock_accepted = True
                    except:
                        ErrorRaised = True
                        fail_reason = "Exception in docking, can_dock didnt catch it"
                        break
                else:
                    num_tries += 1
                    probs[i] *= 0

        if (len(architect.molecule["nodes_vocab_index"]) > 1) and (not ErrorRaised):
            mol = architect.molecule["rdkit_molecule"].GetMol()
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)

            return smiles, False, ""

        else:
            return None, True, fail_reason
