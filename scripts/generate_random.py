from __future__ import annotations

import argparse
import pathlib
from typing import List

from rdkit.Chem import QED, RDConfig
import sys
import os

sys.path.append("..")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

from rdkit import Chem
from rdkit.Chem import (
    QED,
    Crippen,
    Descriptors,
    Lipinski,
)
import sascorer

from molminer.generator import MolecularGenerator


def rdkit_props(mol: Chem.Mol) -> List[float]:
    """Return the 12-element property vector used in the paper."""
    return [
        Crippen.MolLogP(mol),
        QED.qed(mol),
        sascorer.calculateScore(mol),
        Chem.rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.MolWt(mol),
        Chem.rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolMR(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Chem.rdMolDescriptors.CalcNumRings(mol),
        Descriptors.NumRotatableBonds(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    ]


def main() -> None:
    p = argparse.ArgumentParser("MolMiner calibration sweep")
    p.add_argument(
        "--samples", type=int, default=30, help="Number of molecules per prompt value"
    )
    p.add_argument("--ckpt_molminer", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_starter", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_gmm", required=True, type=pathlib.Path)
    p.add_argument("--stats_path", required=True, type=pathlib.Path)
    p.add_argument("--vocab_fragments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_attachments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_anchors", required=True, type=pathlib.Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--weighted", action="store_true")
    p.add_argument("--max_tries", type=int, default=10)
    p.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("../data"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen = MolecularGenerator(
        ckpt_molminer=args.ckpt_molminer,
        ckpt_starter=args.ckpt_starter,
        ckpt_gmm=args.ckpt_gmm,
        stats_path=args.stats_path,
        vocab_fragments=args.vocab_fragments,
        vocab_attachments=args.vocab_attachments,
        vocab_anchors=args.vocab_anchors,
        device=args.device,
    )

    outfile = args.out_dir / f"generated.txt"
    errfile = args.out_dir / f"generated.err"

    smileses = gen.sample(
        num_samples=args.samples,
        topk=args.k,
        weighted=args.weighted,
        greedy=args.greedy,
        max_tries=args.max_tries,
    )

    # Log every successful molecule
    for smi in smileses:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            errfile.write_text(f"Bad SMILES: {smi}\n", append=True)
            continue

        with outfile.open("a+") as f:
            row = ",".join([smi])
            f.write(row + "\n")


if __name__ == "__main__":
    main()
