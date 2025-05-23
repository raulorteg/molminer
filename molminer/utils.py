import copy
import torch
import rdkit.Chem as Chem
from torch.utils.data import DataLoader
import multiprocessing
from functools import partial
from typing import Optional, Generator, Any, Tuple, List, Dict, Union, Type
from torch.nn import Module
from torch.optim import Optimizer
from threading import Lock
import numpy as np
import torch

import random
import pandas as pd
import os
import pickle
import gc
import itertools

import sys

sys.path.append(".")
sys.path.append("..")

from molminer.processor import create_story


def top_k_sample(logits: torch.Tensor, k: int, weighted: bool) -> int:
    """
    Sample an index from the top-k logits.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits from a model (1D tensor).
    k : int
        Number of top logits to consider.
    weighted : bool
        If True, sample using softmax weights; else uniform.

    Returns
    -------
    int
        Index of selected token from original logits.
    """

    # Get the top-k indices and corresponding logits
    topk_logits, topk_indices = torch.topk(logits, k)

    # Weighted sampling
    probs = (
        torch.softmax(topk_logits, dim=0) if weighted else torch.ones_like(topk_logits)
    )

    # Sample from the top-k options
    sampled_index = torch.multinomial(probs, 1)

    return topk_indices[sampled_index].item()  # Return the chosen label index


def read_smiles_generator(
    file_path: str,
    chunk_size: int,
    scaler: Any,
    cutoff: Optional[int] = None,
) -> Generator[Tuple, None, None]:
    """
    Lazily yield SMILES and scaled properties from CSV.

    Parameters
    ----------
    file_path : str
        Path to the SMILES CSV.
    chunk_size : int
        Number of rows per chunk.
    scaler : Any
        Scaler with `.scale(value, property_name)` method.
    cutoff : int, optional
        Max rows to yield.

    Yields
    ------
    tuple
        (smiles, scaled_property1, ..., scaled_propertyN)
    """

    # instead of loading the whole thing into memory we do it as an iterator
    # to be more memory efficient
    global_i = 0
    for chunk in pd.read_csv(file_path, sep=",", chunksize=chunk_size):
        for _, row in chunk.iterrows():
            if cutoff is not None:
                if global_i >= cutoff:
                    break
            global_i += 1
            yield (
                row["smiles"],
                *(
                    scaler.scale(value=row[col], property_name=col)
                    for col in row.index
                    if col != "smiles"
                ),
            )


def pickle_in_slices(story: List, N: int, file_prefix: str = "slice") -> None:
    """
    Save a list into N pickle slices.

    Parameters
    ----------
    story : list
        List to split and save.
    N : int
        Number of parts to split into.
    file_prefix : str
        Filename prefix (default: "slice").

    Returns
    -------
    None
    """

    len_story = len(story)
    print("Lenght story:", len_story)
    slice_size = len_story // N
    for i in range(N):
        # Create the slice
        start_index = i * slice_size
        if i == N - 1:  # Last slice takes the rest of the elements
            end_index = len_story
        else:
            end_index = (i + 1) * slice_size

        story_slice = story[start_index:end_index]

        # Pickle the slice into a file
        with open(f"{file_prefix}_{i}.pkl", "wb") as f:
            pickle.dump(story_slice, f)
        print(f"Pickled slice {i + 1}/{N}")


def collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, list]]], inference_mode: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Batch and pad graph data for MolMiner.

    Parameters
    ----------
    batch : list of dict
        Graph data points.
    inference_mode : bool
        If True, omit 'next_token' from output.

    Returns
    -------
    dict of torch.Tensor
        Batched graph tensors.
    """

    def _to_tensor(x, *, dtype, device=None):
        """Cheap and warning-free conversion."""
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)  # no copy if dtype matches
        return torch.as_tensor(x, dtype=dtype, device=device)

    pairwise_distances = [entry["pairwise_distances"] for entry in batch]

    # Compute the maximum number of nodes for padding
    max_nodes = max(dist.shape[0] for dist in pairwise_distances)

    focal_attachment_types = [entry["focal_attachment_type"] for entry in batch]
    focal_attachment_orders = [entry["focal_attachment_order"] for entry in batch]
    nodes_vocab_indices = [entry["nodes_vocab_index"] for entry in batch]
    if not inference_mode:
        next_tokens = [entry["next_token"] for entry in batch]
    nodes_saturations = [entry["nodes_saturation"] for entry in batch]
    properties = [entry["properties"] for entry in batch]
    distances_to_hit = [entry["distances_to_hit"] for entry in batch]

    padded_distances = torch.full(
        (len(pairwise_distances), max_nodes, max_nodes),
        -1e9,
        dtype=torch.float32,
    )
    for i, dist in enumerate(pairwise_distances):
        padded_distances[i, : dist.shape[0], : dist.shape[1]] = _to_tensor(
            dist, dtype=torch.float32
        )

    # focal distances
    padded_distances_to_hit = torch.full(
        (len(distances_to_hit), 1, max_nodes),
        -1e9,
        dtype=torch.float32,
    )
    for i, dist in enumerate(distances_to_hit):
        padded_distances_to_hit[i, 0, : dist.shape[1]] = _to_tensor(
            dist[0],
            dtype=torch.float32,
        )

    # vocab indices
    padded_vocab_indices = torch.full(
        (len(nodes_vocab_indices), max_nodes),
        0,
        dtype=torch.long,
    )
    for i, vocab_index in enumerate(nodes_vocab_indices):
        padded_vocab_indices[i, : vocab_index.shape[0]] = _to_tensor(
            vocab_index, dtype=torch.long
        )

    # saturations
    processed_saturations = []
    for sat_list in nodes_saturations:
        x_sat = _to_tensor(sat_list, dtype=torch.float32)
        counts = x_sat[:, -1] - x_sat[:, :-1].sum(dim=-1)
        x_sat = torch.cat([counts.unsqueeze(1), x_sat], dim=1)
        x_sat = x_sat / x_sat[:, -1].unsqueeze(1)
        processed_saturations.append(x_sat[:, :-1])

    padded_saturations = torch.zeros(
        len(processed_saturations),
        max_nodes,
        processed_saturations[0].shape[1],
        dtype=torch.float32,
    )
    for i, sat in enumerate(processed_saturations):
        padded_saturations[i, : sat.shape[0], :] = sat

    # properties  (shape:  (B, C) ->  (B, max_nodes, C))
    properties_t = torch.stack(
        [_to_tensor(p, dtype=torch.float32) for p in properties]  # (B, C)
    ).unsqueeze(1)  # -> (B, 1, C)

    # focal attachment types  (scalar -> (B, 1))
    focal_attachment_types = torch.stack(
        [_to_tensor(p, dtype=torch.long) for p in focal_attachment_types]
    ).unsqueeze(-1)

    # attachment orders  (list of ints -> tensor)
    focal_attachment_orders = _to_tensor(
        focal_attachment_orders, dtype=torch.long
    ).unsqueeze(-1)

    if not inference_mode:
        # Stack next_tokens into a tensor Shape: (batch_size, 1)
        next_tokens = torch.tensor(next_tokens).unsqueeze(-1)

        # Return batched data as a dictionary
        return {
            "pairwise_distances": padded_distances,  # (batch_size, max_nodes, max_nodes)
            "focal_attachment_type": focal_attachment_types,  # (batch_size, 1)
            "focal_attachment_order": focal_attachment_orders,  # (batch_size, 1)
            "nodes_vocab_index": padded_vocab_indices,  # (batch_size, max_nodes)
            "next_token": next_tokens,  # (batch_size, 1)
            "nodes_saturation": padded_saturations,  # (batch_size, max_nodes, num_sat=4)
            "properties": properties_t,  # (batch_size, max_nodes, num_props=3)
            "distances_to_hit": padded_distances_to_hit,  # (batch_size, max_nodes)
        }
    else:
        # Return batched data as a dictionary
        return {
            "pairwise_distances": padded_distances,  # (batch_size, max_nodes, max_nodes)
            "focal_attachment_type": focal_attachment_types,  # (batch_size, 1)
            "focal_attachment_order": focal_attachment_orders,  # (batch_size, 1)
            "nodes_vocab_index": padded_vocab_indices,  # (batch_size, max_nodes)
            "nodes_saturation": padded_saturations,  # (batch_size, max_nodes, num_sat=4)
            "properties": properties_t,  # (batch_size, max_nodes, num_props=3)
            "distances_to_hit": padded_distances_to_hit,  # (batch_size, max_nodes)
        }


def collate_fn_starter(
    batch: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate (a, b) tuples of NumPy arrays.

    Parameters
    ----------
    batch : list of tuple
        Each tuple contains two arrays of shape (N, 3).

    Returns
    -------
    tuple of torch.Tensor
        Batches stacked into (B, N, 3) tensors.
    """
    a_batch = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    b_batch = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)
    return a_batch, b_batch


def load_molminer(
    checkpoint_path: str, model_class: Type[Module], device: str = "cpu"
) -> Tuple[Module, Dict, Dict[str, int]]:
    """
    Load a MolMiner model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.
    model_class : type
        Class used to instantiate the model.
    device : str
        Device to map the model ("cpu" or "cuda").

    Returns
    -------
    model : torch.nn.Module
        Loaded model.
    optimizer_state_dict : dict
        Optimizer state from checkpoint.
    checkpoint_info : dict
        Extra info (e.g., epoch, seen_batches).
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters and instantiate the model
    hyperparams = checkpoint["model_hyperparams"]
    model = model_class(
        fragment_vocab_size=hyperparams["fragment_vocab_size"],
        attachment_vocab_size=hyperparams["attachment_vocab_size"],
        anchor_vocab_size=hyperparams["anchor_vocab_size"],
        d_model=hyperparams["d_model"],
        n_heads=hyperparams["n_heads"],
        num_layers=hyperparams["num_layers"],
        d_ff=hyperparams["d_ff"],
        num_properties=hyperparams["num_properties"],
        dropout=hyperparams["dropout"],
        d_anchor=hyperparams["d_anchor"],
        context_hidden_dim=hyperparams["context_hidden_dim"],
        context_n_layers=hyperparams["context_n_layers"],
        geom_requires_grad=hyperparams["geom_requires_grad"],
        geom_init_value=hyperparams["geom_init_value"],
    )

    # Load the model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Extract optimizer state dict
    optimizer_state_dict = checkpoint["optimizer_state_dict"]

    # Extract additional checkpoint information
    checkpoint_info = {
        "epoch": checkpoint["epoch"],
        "seen_batches": checkpoint["seen_batches"],
    }

    return model, optimizer_state_dict, checkpoint_info


def load_starter(
    checkpoint_path: str, model_class: Type[Module], device: str = "cpu"
) -> Tuple[Module, Dict, Dict[str, int]]:
    """
    Load a FragmentStarter model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.
    model_class : type
        Class used to instantiate the model.
    device : str
        Device to map the model ("cpu" or "cuda").

    Returns
    -------
    model : torch.nn.Module
        Loaded model.
    optimizer_state_dict : dict
        Optimizer state from checkpoint.
    checkpoint_info : dict
        Extra info (e.g., epoch).
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters and instantiate the model
    hyperparams = checkpoint["model_hyperparams"]
    model = model_class(
        d_model_in=hyperparams["num_properties"],
        d_model_out=hyperparams["fragment_vocab_size"],
        d_ff=hyperparams["d_ff"],
        dropout=hyperparams["dropout"],
        num_layers=hyperparams["num_layers"],
    )

    # Load the model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Extract optimizer state dict
    optimizer_state_dict = checkpoint["optimizer_state_dict"]

    # Extract additional checkpoint information
    checkpoint_info = {
        "epoch": checkpoint["epoch"],
    }

    return model, optimizer_state_dict, checkpoint_info


def linear_lr_scheduler(
    optimizer: Optimizer,
    seen_batches: int,
    total_num_batches: int,
    initial_lr: float,
    final_lr: float,
    warmup_ratio: float = 0.05,
) -> None:
    """
    Linear warmup followed by linear decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to update.
    seen_batches : int
        Number of batches seen so far.
    total_num_batches : int
        Total batches expected.
    initial_lr : float
        Peak learning rate.
    final_lr : float
        Minimum learning rate.
    warmup_ratio : float
        Fraction of training used for warmup.

    Returns
    -------
    None
    """
    # Calculate the number of warm-up epochs
    warmup_epochs = int(total_num_batches * warmup_ratio)
    min_lr = final_lr

    if seen_batches < warmup_epochs:
        # Linearly increase learning rate during warm-up
        lr = min_lr + (initial_lr - min_lr) * (seen_batches / warmup_epochs)
    else:
        # Linearly decay learning rate after warm-up
        lr = initial_lr - (initial_lr - final_lr) * (
            (seen_batches - warmup_epochs) / (total_num_batches - warmup_epochs)
        )

    # Ensure the learning rate does not drop below final_lr
    lr = max(lr, final_lr)

    # Set the new learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value.

    Returns
    -------
    None
    """

    random.seed(seed)  # random module
    np.random.seed(seed)  #  numpy
    torch.manual_seed(seed)  # torch cpu

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # torch gpu


def process_smiles(
    smiles: Tuple[str, ...], processor_template: Any, architect_template: Any
) -> List[dict]:
    """
    Convert a SMILES string into a story dict.

    Parameters
    ----------
    smiles : tuple
        (smiles, property1, ..., propertyN).
    processor_template : object
        Template molecule processor.
    architect_template : object
        Template molecule architect.

    Returns
    -------
    list of dict
        Story data or empty list on failure.
    """
    # unpack the smiles and properties, all given by the iterator
    smiles, *properties = smiles

    # Make thread-local copies of the processor and architect
    processor = copy.deepcopy(processor_template)
    architect = copy.deepcopy(architect_template)
    architect.reset()  # just in case

    # Create the molecule and coarse information
    coarse_info = processor.coarsify_molecule(Chem.MolFromSmiles(smiles))
    coarse_info["properties"] = torch.tensor(properties, dtype=torch.float32)

    try:
        # Create the story
        story = create_story(
            coarse_info=coarse_info,
            architect=architect,
            vocab_attachments_df=vocab_attachments_df,
            vocab_fragments_df=vocab_fragments_df,
            return_snapshots=False,  # remmeber to remove
        )
    except Exception as e:
        print(
            f"Warning: Error while creating a story for smiles: {smiles}, {type(e).__name__}, {e}"
        )
        del processor, architect, coarse_info
        gc.collect()
        return []

    del processor, architect, coarse_info
    gc.collect()

    return story


def worker_process(
    smiles: Tuple[str, ...], processor_template: Any, architect_template: Any
) -> List[Dict[str, Any]]:
    """
    Process SMILES and convert tensors to NumPy.

    Parameters
    ----------
    smiles : tuple
        SMILES and properties.
    processor_template : object
        Molecule processor.
    architect_template : object
        Molecule architect.

    Returns
    -------
    list of dict
        Story dicts with NumPy arrays.
    """
    results = process_smiles(
        smiles=smiles,
        processor_template=processor_template,
        architect_template=architect_template,
    )

    # Convert tensors inside dictionary to NumPy to avoid shared memory issues
    safe_results = []
    for result in results:
        safe_result = {
            key: (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in result.items()
        }
        safe_results.append(safe_result)
    return safe_results


def init_worker(
    vocab_fragments: pd.DataFrame,
    vocab_attachments: pd.DataFrame,
    vocab_anchors: pd.DataFrame,
) -> None:
    """
    Initialize vocab globals for worker pool.

    Parameters
    ----------
    vocab_fragments : pd.DataFrame
        Fragment vocabulary.
    vocab_attachments : pd.DataFrame
        Attachment vocabulary.
    vocab_anchors : pd.DataFrame
        Anchor vocabulary.

    Returns
    -------
    None
    """
    global vocab_fragments_df, vocab_attachments_df, vocab_anchors_df
    vocab_fragments_df = vocab_fragments
    vocab_attachments_df = vocab_attachments
    vocab_anchors_df = vocab_anchors


def parallel_process_smiles(
    dataset_file: str,
    processor_template: Any,
    architect_template: Any,
    vocab_fragments_df: pd.DataFrame,
    vocab_attachments_df: pd.DataFrame,
    vocab_anchors_df: pd.DataFrame,
    max_workers: int,
    batch_size: int,
    scaler: Any,
    cutoff: int,
    shared_data: Optional[Any] = None,
    lock: Optional[Lock] = None,
) -> Optional[DataLoader]:
    """
    Run SMILES processing in parallel, return DataLoader.

    Parameters
    ----------
    dataset_file : str
        Path to SMILES CSV file.
    processor_template : object
        Template molecule processor.
    architect_template : object
        Template molecule architect.
    vocab_fragments_df : pd.DataFrame
        Fragment vocab.
    vocab_attachments_df : pd.DataFrame
        Attachment vocab.
    vocab_anchors_df : pd.DataFrame
        Anchor vocab.
    max_workers : int
        Max processes.
    batch_size : int
        DataLoader batch size.
    scaler : object
        Property scaler.
    cutoff : int
        Max molecules to process.
    shared_data : object, optional
        Shared memory state object.
    lock : Lock, optional
        Lock for writing to shared_data.

    Returns
    -------
    DataLoader or None
        Dataloader if not using shared memory.
    """
    try:
        while True:
            if shared_data != None:
                if shared_data.done_flag == True:
                    print("Terminating...")
                    break
            worker_partial = partial(
                worker_process,
                processor_template=processor_template,
                architect_template=architect_template,
            )

            # Create a pool of processes
            with multiprocessing.Pool(
                processes=min(multiprocessing.cpu_count(), max_workers),
                initializer=init_worker,
                initargs=(vocab_fragments_df, vocab_attachments_df, vocab_anchors_df),
            ) as pool:
                results = pool.imap_unordered(
                    worker_partial,
                    read_smiles_generator(
                        dataset_file,
                        cutoff=cutoff,
                        chunk_size=max_workers,
                        scaler=scaler,
                    ),
                )
                # Flatten and collect into memory after the pool is closed
                combined_stories = list(itertools.chain.from_iterable(results))

            del results
            gc.collect()

            # Create the dataloader from the new story (batch of data)
            dataloader = DataLoader(
                combined_stories,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                pin_memory=False,
                num_workers=min(multiprocessing.cpu_count(), max_workers),
            )
            if shared_data == None:
                return dataloader
            else:
                with lock:
                    shared_data.dataloader = dataloader
                    shared_data.ctr = shared_data.ctr + 1

                del dataloader
                gc.collect()

    except KeyboardInterrupt:
        print("SMILES processing interrupted!")


def combine_pickles_incrementally(directory_path: str) -> List[Any]:
    """
    Load and merge all lists from .pkl files in a directory.

    Parameters
    ----------
    directory_path : str
        Directory containing .pkl files.

    Returns
    -------
    list
        Combined list from all pickle files.

    Raises
    ------
    ValueError
        If any pickle file does not contain a list.
    """
    combined_list = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pkl"):  # Check if the file is a pickle file
            file_path = os.path.join(directory_path, filename)

            # Open and process each file incrementally
            with open(file_path, "rb") as file:
                data = pickle.load(file)

                # Ensure the data is a list before combining
                if isinstance(data, list):
                    combined_list += data  # Extend incrementally
                else:
                    raise ValueError(f"Data in {filename} is not a list.")

    return combined_list
