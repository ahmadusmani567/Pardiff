# Suppress all warnings to keep output clean
import warnings
warnings.filterwarnings('ignore')

# Standard and ML imports
import os
import logging
import pickle
import torch
import numpy as np
from torch_geometric.loader import DataLoader  # For graph data batching

# Utility and model-related imports from PARDiff
from pardiff.utils import find_checkpoint_with_lowest_val_loss
from pardiff.parallel.task import AutoregressiveDiffusion, PredictBlockProperties
from pardiff.dataset import DATA_INFO
from pardiff.analysis.spectre_utils import SpectreSamplingMetrics
from pardiff.analysis.rdkit_functions import BasicMolecularMetrics
from pardiff.utils import from_batch_onehot_to_list, check_block_id_train_vs_generation

# Molecular benchmark utility
from moses.metrics.metrics import get_all_metrics

# Set PyTorch to use a fixed number of CPU threads for data processing
torch.set_num_threads(20)

def evaluate_model(
    device,
    dataset_name,
    diffusion_path,
    block_predictor_path=None,
    mode='best',
    batch_size=128,
    max_hops=3
):
    """
    Evaluates a pretrained diffusion model on a graph generation task.

    Args:
        device (int): CUDA device index to use.
        dataset_name (str): Dataset identifier (e.g., 'qm9').
        diffusion_path (str): Path to the trained diffusion model checkpoints.
        block_predictor_path (str): Optional path to pretrained block predictor.
        mode (str): Evaluation mode - 'best', 'last', 'latest', or 'all'.
        batch_size (int): Batch size for generation and evaluation.
        max_hops (int): Max hop distance used in block generation.
    """
    assert mode in ['best', 'last', 'latest', 'all']

    # Set up logging to a file within the checkpoint directory
    logging.basicConfig(
        filename=os.path.join(diffusion_path, 'eval_metrics.log'),
        encoding='utf-8',
        level=logging.DEBUG
    )

    # Load dataset-specific metadata
    data_cfg = DATA_INFO[dataset_name]
    atom_decoder = data_cfg.get('atom_decoder')
    metric_provider = data_cfg.get('metric_class')

    # Load train/val/test splits for the selected dataset
    data_splits = {
        split: data_cfg['class'](**{**data_cfg['default_args'], 'split': split})
        for split in ['train', 'val', 'test']
    }

    # Wrap each split in a PyG DataLoader for batching
    data_loaders = {
        split: DataLoader(
            data_splits[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=12
        )
        for split in ['train', 'val', 'test']
    }

    # Choose metric provider (Spectre or RDKit-based) based on dataset type
    if atom_decoder is None:
        metrics = metric_provider(data_loaders) if metric_provider else SpectreSamplingMetrics(data_loaders)
    else:
        train_smiles = data_splits['train'].get_smiles(False) if dataset_name in ['qm9', 'zinc250k'] else None
        test_smiles = data_splits['test'].get_smiles(False) if dataset_name in ['qm9', 'zinc250k'] else None
        metrics = BasicMolecularMetrics(atom_decoder, train_smiles, test_smiles)

    # Determine whether to use a joint model (with block predictor inside)
    use_joint_model = True
    block_model = None

    # Load external block predictor model if provided
    if block_predictor_path:
        use_joint_model = False
        block_ckpt = find_checkpoint_with_lowest_val_loss(block_predictor_path)
        block_model = PredictBlockProperties.load_from_checkpoint(block_ckpt, map_location=f'cuda:{device}')

    # Choose diffusion model checkpoints to evaluate
    if mode == 'all':
        ckpts = [os.path.join(diffusion_path, f) for f in os.listdir(diffusion_path) if f.endswith('.ckpt')]
    elif mode == 'best':
        ckpts = [find_checkpoint_with_lowest_val_loss(diffusion_path)]
    elif mode == 'latest':
        ckpts = [find_checkpoint_with_lowest_val_loss(diffusion_path, return_latest=True)]
    else:
        ckpts = [os.path.join(diffusion_path, 'last.ckpt')]

    for model_ckpt in ckpts:
        # Load trained diffusion model
        model = AutoregressiveDiffusion.load_from_checkpoint(model_ckpt, map_location=f'cuda:{device}')
        model.combine_training = use_joint_model
        model.blocksize_model = block_model

        logging.info('=' * 100)
        logging.info(f"Evaluating checkpoint: {os.path.basename(model_ckpt)}")
        print("Sampling in progress...")

        # Number of graphs to generate
        total_samples = data_cfg.get('num_eval_generation', len(data_splits['test']))
        gen_batch_size = min(batch_size, total_samples)

        generated_graphs = []
        block_id_match = []

        # Sample graphs until we reach the target count
        while total_samples > 0:
            try:
                batch_output = model.generate(batch_size=gen_batch_size).cpu()
                graph_data = from_batch_onehot_to_list(batch_output.nodes, batch_output.edges)
                block_check = check_block_id_train_vs_generation(
                    batch_output.nodes,
                    batch_output.edges,
                    batch_output.nodes_blockid,
                    train_max_hops=max_hops
                )
                generated_graphs.extend(graph_data)
                block_id_match.extend(block_check)
                total_samples -= gen_batch_size
            except Exception as err:
                print(f"Sampling error: {err}. Retrying...")
                continue

        # Log block ID consistency (i.e., match with training config)
        match_ratio = 100 * sum(block_id_match) / len(block_id_match)
        logging.info("Block ID match with training:")
        logging.info(f"{match_ratio:.2f}%")

        print("Running evaluation...")

        # Save generated graphs to disk
        with open(os.path.join(diffusion_path, 'generated_graphs.pkl'), 'wb') as f:
            pickle.dump(generated_graphs, f)

        # Evaluate graph quality
        if atom_decoder is None:
            results = metrics(generated_graphs)
        else:
            val_metrics, dist_dict, unique_smiles, all_smiles = metrics(generated_graphs)
            np.save(os.path.join(diffusion_path, 'generated_smiles.npy'), np.array(all_smiles))
            results = [val_metrics, dist_dict]

            # Optional: benchmark using MOSES library
            benchmark = get_all_metrics(gen=all_smiles, k=None, test=test_smiles, train=train_smiles)
            logging.info('-' * 50)
            logging.info(str(benchmark))
            logging.info('-' * 50)

        # Final log
        logging.info(str(results))


# Optional entry point for script-based execution
if __name__ == '__main__':
    evaluate_model(
        device=4,
        dataset_name='qm9',
        diffusion_path='checkpoints/local_denoising/qm9.3hops.ppgnTrans-Parallel.BlockID11.bn.PreNorm=1.H256.E64.L8-lr0.0004.cosine-ires1.blocktime0.uni_noise1.T20.cosine.vlb1.ce0.1.combine=False',
        block_predictor_path='checkpoints/block_prediction/qm9.3hops.ppgnTrans-Parallel.BlockID11.bn.PreNorm=1.H256.E64.L8-lr0.0004.cosine',
        mode='best',
        batch_size=1024,
        max_hops=3
    )
