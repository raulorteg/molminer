from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
from typing import List

from rdkit.Chem import RDConfig

sys.path.append("..")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import (
    QED,
    Crippen,
    Descriptors,
    Lipinski,
)
import sascorer

from molminer.generator import MolecularGenerator


PROP_ORDER_INTERNAL = [
    "logP", "qed", "SAS", "FractionCSP3", "molWt", "TPSA", "MR",
    "hbd", "hba", "num_rings", "num_rotable_bonds", "num_quiral_centers",
]

PROP_ORDER_OUTPUT = [
    "logP", "qed", "SAS", "FractionCSP3", "molWt", "TPSA", "MR",
    "hbd", "hba", "num_rings", "num_rotable_bonds", "num_chiral_centers",
]

# integer counts on a real molecule, so rounded before comparison
COUNT_PROPS = {
    "hbd", "hba", "num_rings", "num_rotable_bonds", "num_quiral_centers",
}

_ALIASES = {"num_chiral_centers": "num_quiral_centers"}

_OPS = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RANGE_RE = re.compile(rf"^({_NUM})\s*(<=|<)\s*(\w+)\s*(<=|<)\s*({_NUM})$")
_SIMPLE_RE = re.compile(rf"^(\w+)\s*(<=|>=|==|<|>)\s*({_NUM})$")


def canonical_prop(name: str) -> str:
    """Map a user-supplied property name onto its internal name."""
    name = _ALIASES.get(name, name)
    if name not in PROP_ORDER_INTERNAL:
        raise argparse.ArgumentTypeError(
            f"unknown property '{name}'. Choose from: {', '.join(PROP_ORDER_OUTPUT)}"
        )
    return name


class Constraint:
    """A single inequality on one property."""

    def __init__(self, prop: str, tests: list, text: str):
        self.prop = prop
        self.tests = tests
        self.text = text

    def holds(self, values: dict) -> bool:
        v = values[self.prop]
        if self.prop in COUNT_PROPS:
            v = round(v)
        return all(op(v, thr) for op, thr in self.tests)


def parse_constraint(spec: str) -> Constraint:
    """Parse 'molWt<=500', '1<=logP<=5' or 'num_chiral_centers==0'."""
    s = spec.replace(" ", "")

    m = _RANGE_RE.match(s)
    if m:
        lo, lo_op, prop, hi_op, hi = m.groups()
        # 'lo <= prop' is 'prop >= lo', so flip the operator
        flipped = {"<=": ">=", "<": ">"}[lo_op]
        return Constraint(
            canonical_prop(prop),
            [(_OPS[flipped], float(lo)), (_OPS[hi_op], float(hi))],
            spec.strip(),
        )

    m = _SIMPLE_RE.match(s)
    if m:
        prop, op, thr = m.groups()
        return Constraint(
            canonical_prop(prop), [(_OPS[op], float(thr))], spec.strip()
        )

    raise argparse.ArgumentTypeError(
        f"cannot parse constraint '{spec}'. Expected forms: 'molWt<=500', "
        f"'1<=logP<=5', 'num_chiral_centers==0'."
    )


class Window:
    """A conjunction of constraints; an empty window accepts everything."""

    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints

    def __contains__(self, values: dict) -> bool:
        return all(c.holds(values) for c in self.constraints)

    def describe(self) -> str:
        if not self.constraints:
            return "  (no constraints, sampling the GMM prior)"
        return "\n".join(f"  {c.text}" for c in self.constraints)

    @property
    def properties(self) -> List[str]:
        return sorted({c.prop for c in self.constraints})


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


def unscale_vector(scaled_vec: np.ndarray, scaler) -> dict:
    """Convert a scaled property vector to a dict of unscaled values."""
    return {
        p: float(scaled_vec[i] * scaler.get("std", p) + scaler.get("mean", p))
        for i, p in enumerate(PROP_ORDER_INTERNAL)
    }


def props_to_dict(props: List[float]) -> dict:
    """Key a 12-element property list by the internal property order."""
    return {p: props[i] for i, p in enumerate(PROP_ORDER_INTERNAL)}


def format_props(props: List[float]) -> List[str]:
    """Cast count properties to int, keep the rest as floats."""
    return [
        str(int(v)) if PROP_ORDER_INTERNAL[j] in COUNT_PROPS else f"{v:.6g}"
        for j, v in enumerate(props)
    ]


def collect_window_vectors(
    gen,
    window: Window,
    n_target: int,
    batch_size: int = 10000,
    max_total_draws: int = 10_000_000,
):
    """Rejection-sample the GMM until n_target in-window vectors are collected."""
    accepted = []
    n_drawn = 0
    while len(accepted) < n_target and n_drawn < max_total_draws:
        batch_scaled, _ = gen.gmm.sample(n_samples=batch_size)
        n_drawn += batch_size
        for scaled_vec in batch_scaled:
            if unscale_vector(scaled_vec, gen.scaler) in window:
                accepted.append(scaled_vec)
                if len(accepted) >= n_target:
                    break
    if len(accepted) < n_target:
        raise RuntimeError(
            f"Only {len(accepted)}/{n_target} vectors accepted after {n_drawn} "
            f"draws ({len(accepted) / n_drawn:.2%}). The window is too tight for "
            f"the GMM, loosen a constraint or raise --max_total_draws."
        )
    return np.array(accepted), n_drawn


def main() -> None:
    p = argparse.ArgumentParser("MolMiner window-constrained generation")
    p.add_argument(
        "--constraint", action="append", default=[], metavar="EXPR",
        help="Window constraint, e.g. 'molWt<=500' or '1<=logP<=5'. Repeatable",
    )
    p.add_argument("--n_target", type=int, default=100,
                   help="Number of molecules to generate")
    p.add_argument("--ckpt_molminer", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_starter", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_gmm", required=True, type=pathlib.Path)
    p.add_argument("--stats_path", required=True, type=pathlib.Path)
    p.add_argument("--vocab_fragments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_attachments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_anchors", required=True, type=pathlib.Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--max_tries", type=int, default=10)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--weighted", action="store_true")
    p.add_argument("--batch_size", type=int, default=10000,
                   help="GMM draws per rejection-sampling batch")
    p.add_argument("--max_total_draws", type=int, default=10_000_000,
                   help="Give up if the window is not filled within this many draws")
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("../data/generated/window.txt"))
    args = p.parse_args()

    window = Window([parse_constraint(c) for c in args.constraint])

    print("Window (evaluated on unscaled properties):")
    print(window.describe())
    free = [q for q in PROP_ORDER_OUTPUT
            if canonical_prop(q) not in window.properties]
    if free:
        print(f"Free (sampled from the GMM prior): {', '.join(free)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    outfile = args.out
    passfile = args.out.with_name(f"{args.out.stem}_passing{args.out.suffix}")
    errfile = args.out.with_suffix(".err")

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

    if gen.prop_order != PROP_ORDER_INTERNAL:
        raise RuntimeError(
            f"prop_order mismatch: generator has {gen.prop_order}, "
            f"expected {PROP_ORDER_INTERNAL}"
        )

    accepted, n_drawn = collect_window_vectors(
        gen, window, n_target=args.n_target,
        batch_size=args.batch_size, max_total_draws=args.max_total_draws,
    )
    print(f"Prompt acceptance rate: {args.n_target / n_drawn:.2%} "
          f"({args.n_target} accepted / {n_drawn} drawn)")

    header = "smiles," + ",".join(PROP_ORDER_OUTPUT) + "\n"
    n_ok = n_failed = n_hits = 0

    with open(outfile, "w") as f_all, open(passfile, "w") as f_pass, \
            open(errfile, "w") as f_err:
        f_all.write(header)
        f_pass.write(header)

        # Log every successful molecule, and the in-window ones twice
        for i, scaled_vec in enumerate(accepted, start=1):
            c = torch.tensor(scaled_vec, dtype=torch.float).to(gen.device)
            smiles, failed, msg = gen._sample(
                c=c, topk=args.topk, weighted=args.weighted,
                greedy=args.greedy, max_tries=args.max_tries,
            )
            if failed or smiles is None:
                n_failed += 1
                f_err.write(f"{i}: generation failed: {msg}\n")
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                n_failed += 1
                f_err.write(f"{i}: Bad SMILES: {smiles}\n")
                continue

            props = rdkit_props(mol)
            row = ",".join([smiles] + format_props(props)) + "\n"
            f_all.write(row)
            n_ok += 1

            # hits are counted on the product, not on the prompt
            if props_to_dict(props) in window:
                f_pass.write(row)
                n_hits += 1

            if i % 10 == 0:
                print(f"  {i}/{args.n_target}  valid={n_ok} failed={n_failed} "
                      f"hits={n_hits}")

    print(f"\nGenerated {n_ok}/{args.n_target} valid molecules ({n_failed} failed).")
    if n_ok:
        print(f"In-window hits: {n_hits}/{n_ok} ({n_hits / n_ok:.1%})")


if __name__ == "__main__":
    main()
