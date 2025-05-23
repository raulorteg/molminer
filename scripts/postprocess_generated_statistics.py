import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import rdkit.Chem as Chem
import seaborn as sns
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, RDConfig
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import itertools
from typing import Dict

import pandas as pd
import sascorer
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import wasserstein_distance


def compute_generation_stats(
    gen_file: str, ref_file: str, sep: str = ","
) -> Dict[str, float]:
    """
    Compute generation quality metrics for a set of generated SMILES strings compared to a reference set.

    Metrics computed:
        - Validity: Percentage of valid SMILES in the generated set.
        - Uniqueness: Percentage of unique connectivity keys in the generated set.
        - Novelty: Percentage of unique keys not found in the reference set.
        - Diversity: Average pairwise Tanimoto distance among unique valid molecules.

    Parameters:
    -----------
    gen_file : str
        Path to a file containing generated SMILES strings with a 'smiles' column.

    ref_file : str
        Path to a file containing reference SMILES strings with a 'smiles' column.

    sep : str, optional
        Separator used in the input files (default is a comma).

    Returns:
    --------
    Dict[str, float]
        A dictionary with keys: 'validity', 'uniqueness', 'novelty', 'diversity'.
    """

    def conn_key(mol):
        # only the first InChIKey block (connectivity)
        return Chem.MolToInchiKey(mol).split("-")[0]

    # load the sets of generated and reference SMILES molecules
    gen_df = pd.read_csv(gen_file, sep=sep, dtype=str)
    ref_df = pd.read_csv(ref_file, sep=sep, dtype=str)
    gen_smiles = gen_df["smiles"].dropna().tolist()
    ref_smiles = ref_df["smiles"].dropna().tolist()

    # Validity of generated molecules (sanity check)
    valid_mols = []
    for smi in gen_smiles:
        mol = Chem.MolFromSmiles(smi)  # this includes sanitization
        if mol is not None:
            valid_mols.append(mol)
    validity = 100.0 * len(valid_mols) / len(gen_smiles) if gen_smiles else 0.0

    # Uniqueness of the generated SMILES (% of unique connectivities)
    gen_keys = [conn_key(m) for m in valid_mols]
    unique_keys = set(gen_keys)
    uniqueness = 100.0 * len(unique_keys) / len(gen_smiles) if gen_smiles else 0.0

    # Novelty of the generated SMILES (% of connectivities not in the dataset)
    ref_keys = {
        conn_key(m) for smi in ref_smiles if (m := Chem.MolFromSmiles(smi)) is not None
    }
    novel_keys = unique_keys - ref_keys
    novelty = 100.0 * len(novel_keys) / len(unique_keys) if unique_keys else 0.0

    # Diversity among unique valid generated molecules
    # Morgan fingerprints (radius=2, 2048 bits)
    key_to_mol = {}
    for mol, key in zip(valid_mols, gen_keys):
        key_to_mol.setdefault(key, mol)

    mols_for_div = list(key_to_mol.values())
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        for m in mols_for_div
    ]
    total_dist = 0.0
    count = 0
    for fp1, fp2 in itertools.combinations(fps, 2):
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)

        # Tanimoto distance is 1-tanimoto_similarity
        total_dist += 1.0 - sim

        count += 1
    diversity = total_dist / count

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "diversity": diversity,
    }


def compute_properties(smiles: str):
    """
    Compute a set of 12 molecular properties for a given SMILES string.

    Properties returned:
        - logP: Octanol-water partition coefficient
        - qed: Quantitative Estimate of Drug-likeness
        - SAS: Synthetic Accessibility Score
        - FractionCSP3: Fraction of sp3-hybridized carbons
        - molWt: Molecular weight
        - TPSA: Topological Polar Surface Area
        - MR: Molar refractivity
        - hbd: Number of hydrogen bond donors
        - hba: Number of hydrogen bond acceptors
        - num_rings: Number of rings
        - num_rotable_bonds: Number of rotatable bonds
        - num_quiral_centers: Number of chiral centers

    Parameters:
    -----------
    smiles : str
        SMILES string of a molecule.

    Returns:
    --------
    List
        A list of 12 property values. Returns [None] * 12 if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 12

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


def compute_wasserstein_distances(
    file1: str, file2: str, sep: str = ","
) -> Dict[str, float]:
    """
    Compute the 1D Wasserstein distance for each property between two datasets.

    Parameters:
    -----------
    file1 : str
        Path to the first CSV or TXT file (should contain 'smiles' and property columns).

    file2 : str
        Path to the second CSV or TXT file.

    sep : str, optional
        Field separator used in both files (default is comma).

    Returns:
    --------
    Dict[str, float]
        A dictionary mapping each shared property name to its Wasserstein distance.
    """
    # load the sets of generated and reference SMILES molecules
    df1 = pd.read_csv(file1, sep=sep)
    df2 = pd.read_csv(file2, sep=sep)

    # Identify common property columns (exclude 'smiles')
    props = [c for c in df1.columns if c != "smiles" and c in df2.columns]

    distances = {}
    for prop in props:
        # drop NaNs just in case
        a = df1[prop].dropna().values
        b = df2[prop].dropna().values

        # compute 1D wasserstein
        distances[prop] = wasserstein_distance(a, b)

    return distances


def annotate_smiles_w_props(infile: str, outfile: str):
    """
    Annotate a list of SMILES strings with molecular properties and save the results to a CSV file.

    Parameters:
    -----------
    infile : str
        Path to a input file containing SMILES strings (one per line, no header).

    outfile : str
        Path to the output CSV file where the annotated results will be saved.

    Notes:
    ------
    - Uses the `compute_properties` function to calculate 12 properties per SMILES.
    - The output file will contain the original SMILES and the corresponding properties.
    - Invalid SMILES will have None in all property columns.
    """
    # load the data containing a single smiles column (no header)
    df = pd.read_csv(infile, header=None, names=["smiles"], sep="\t", dtype=str)
    df["smiles"] = df["smiles"].str.strip()

    prop_names = [
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

    # compute all properties for all smiles
    all_props = []
    for smi in tqdm(df["smiles"], desc="Computing properties"):
        all_props.append(compute_properties(smi))

    # collect the results
    props_df = pd.DataFrame(all_props, columns=prop_names)
    out_df = pd.concat([df, props_df], axis=1)

    # save the resulting annotated SMILES with properties in a comma separated file
    out_df.to_csv(outfile, sep=",", index=False)
    print(f"-> Saved annotated properties to {outfile}")


def plot_property_kde(
    file_paths,
    names,
    outfile,
    fig_width=7,
    fig_height=9,
    label_fontsize=9,
    tick_fontsize=8,
    legend_fontsize=8,
):
    """
    Generate KDE and histogram plots for molecular properties across multiple datasets.

    Parameters:
    -----------
    file_paths : List[str]
        List of file paths containing annotated property data (with 'smiles' and property columns).

    names : List[str]
        List of names corresponding to each dataset (used for labeling in the legend).

    outfile : str
        Path to save the resulting multi-panel PNG figure.

    fig_width : float
        Width of the figure in inches (default: 7).

    fig_height : float
        Height of the figure in inches (default: 9).

    label_fontsize : int
        Font size for axis labels.

    tick_fontsize : int
        Font size for axis ticks.

    legend_fontsize : int
        Font size for the legend text.

    Notes:
    ------
    - Continuous properties are plotted as KDE curves.
    - Discrete properties are shown as normalized histograms.
    - The plot includes a shared legend and adjusts for unused subplots.
    """
    properties = [
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

    discrete_properties = [
        "hbd",
        "hba",
        "num_rings",
        "num_rotable_bonds",
        "num_quiral_centers",
    ]

    dataframes = []
    for path in file_paths:
        df = pd.read_csv(path)
        df = df[properties]
        dataframes.append(df)

    fig, axes = plt.subplots(4, 3, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    # Collect labels once for global legend
    handles = []
    labels = []

    for i, prop in enumerate(properties):
        ax = axes[i]
        if prop not in discrete_properties:
            for df, name in zip(dataframes, names):
                if name == "Dataset":
                    plot = sns.kdeplot(
                        df[prop],
                        ax=ax,
                        label=name,
                        linewidth=2,
                        fill=True,
                        alpha=0.3,
                        color="gray",
                    )
                else:
                    plot = sns.kdeplot(
                        df[prop], ax=ax, label=name, linewidth=2, fill=False, alpha=0.8
                    )
        else:
            width = 0.8 / len(dataframes)
            all_values = sorted(
                set().union(*[df[prop].dropna().unique() for df in dataframes])
            )
            for j, (df, name) in enumerate(zip(dataframes, names)):
                counts = (
                    df[prop]
                    .value_counts(normalize=True)
                    .reindex(all_values, fill_value=0)
                    .sort_index()
                )
                bars = ax.bar(
                    [x + j * width for x in range(len(all_values))],
                    counts.values,
                    width=width,
                    label=name,
                    color="gray" if name == "Dataset" else None,
                    edgecolor="black" if name == "Dataset" else None,
                )
                if i == 0 and j == 0:
                    # Save only one set of bars for the legend
                    handles.append(bars)
                    labels.append(name)

            ax.set_xticks(
                [x + width * (len(dataframes) - 1) / 2 for x in range(len(all_values))]
            )
            ax.set_xticklabels(all_values, fontsize=tick_fontsize)
            max_bar = {
                "hbd": 5,
                "hba": 8.5,
                "num_rings": 5.5,
                "num_rotable_bonds": 9.5,
                "num_quiral_centers": 4.5,
            }
            ax.set_xlim(-0.5, max_bar[prop])

        ax.set_ylabel(
            "Density" if prop not in discrete_properties else "Frequency",
            fontsize=label_fontsize,
        )
        ax.set_xlabel(prop, fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    # Turn off any unused subplots
    for j in range(len(properties), len(axes)):
        axes[j].axis("off")

    # Global legend outside the grid
    fig.legend(
        names,
        fontsize=legend_fontsize,
        loc="lower center",
        ncol=len(names),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(outfile, bbox_inches="tight", dpi=800)
    plt.close()


def compute_all_stats(gen_file: str, gen_wprops_file: str, ref_dataset_file: str):
    """
    Perform full evaluation of a generated molecular dataset against a reference dataset.

    Steps:
    ------
    1. Annotate the generated SMILES with molecular properties.
    2. Compute 1D Wasserstein distances for each property.
    3. Compute generation quality metrics: validity, uniqueness, novelty, and diversity.

    Parameters:
    -----------
    gen_file : str
        Path to the file containing generated SMILES (one per line).

    gen_wprops_file : str
        Path to save the annotated version of the generated file with computed properties.

    ref_dataset_file : str
        Path to the annotated reference dataset with computed properties.

    Outputs:
    --------
    - Prints Wasserstein distances for each property.
    - Prints validity, uniqueness, novelty, and diversity scores.
    """

    print(f"Processing file {gen_file}")

    annotate_smiles_w_props(infile=gen_file, outfile=gen_wprops_file)

    # compute the 1-d Wassersteing distances to the reference dataset for each property
    dists = compute_wasserstein_distances(gen_wprops_file, ref_dataset_file, sep=",")
    for prop, w in dists.items():
        print(f"{prop:20s}: {w:.4f}")

    # compute novelty, validity, uniqueness and diversity
    stats = compute_generation_stats(gen_wprops_file, ref_dataset_file, sep=",")
    print(f"Uniqueness: {stats['uniqueness']:.1f}%")
    print(f"Novelty   : {stats['novelty']:.1f}%")
    print(f"Diversity : {stats['diversity']:.4f}")
    print(f"Validity  : {stats['validity']:.1f}%")
    print()


if __name__ == "__main__":
    ref_dataset_file = "../data/zinc/augmented_zinc250k.txt"
    outfile_kde_plot = "../figures/benchmark_dists.png"

    gen_files = ["../data/generated.txt"]
    gen_wprops_files = [
        "../data/generated/generated_annotated.txt",
    ]

    for gen_file, gen_wprops_file in zip(gen_files, gen_wprops_files):
        compute_all_stats(
            gen_file=gen_file,
            gen_wprops_file=gen_wprops_file,
            ref_dataset_file=ref_dataset_file,
        )

    file_list = [
        ref_dataset_file,
        "../data/generated/generated_annotated.txt",
    ]
    names = ["Dataset", "MolMiner"]
    plot_property_kde(
        file_list,
        names,
        outfile=outfile_kde_plot,
        fig_width=7.5,
        fig_height=9.5,
        label_fontsize=9,
        tick_fontsize=8,
        legend_fontsize=12,
    )
