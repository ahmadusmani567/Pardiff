# Suppress warning messages (for cleaner output during training or inference)
import warnings
warnings.filterwarnings('ignore')

# Standard libraries and dependencies
import os, sys
import torch
import lightning as L
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from functools import partial
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# Project-specific imports
from pardiff.config import cfg, update_cfg
from pardiff.dataset import DATA_INFO
from pardiff.parallel.transform import ToParallelBlocks, ToOneHot
from pardiff.parallel.task import PredictBlockProperties, AutoregressiveDiffusion
from pardiff.parallel import utils as parallel_utils
from pardiff.utils import find_checkpoint_with_lowest_val_loss

# ------------------------------- Configuration -----------------------------------

# Load dataset argument from command-line input
dataset_arg = sys.argv[sys.argv.index('dataset') + 1]

# Load dataset-specific configuration file
cfg.merge_from_file(f'pardiff/configs/{dataset_arg}.yaml')
cfg = update_cfg(cfg)
cfg.dataset = dataset_arg.split('-')[0]

# Set CUDA device and PyTorch threads
if isinstance(cfg.device, int):
    torch.cuda.set_device(cfg.device)
torch.set_num_threads(cfg.num_workers)

# Ensure task type is supported
assert cfg.task in ['block_prediction', 'local_denoising']

# Check whether batched sequential modeling is enabled
batched_seq = cfg.model.batched_sequential

# ------------------------------- Dataset Setup -----------------------------------

print(f'Loading dataset: {cfg.dataset}')
info = DATA_INFO[cfg.dataset]

# Create one-hot encoding transform for node and edge features
one_hot = ToOneHot(
    info['num_node_features'], 
    info['num_edge_features'],
    virtual_node_type=cfg.diffusion.num_node_virtual_types,
    virtual_edge_type=cfg.diffusion.num_edge_virtual_types,
    has_zero_edgetype=(info['start_edge_type'] == 0)
)

# Prepare block transformation logic
block_transform = ToParallelBlocks(
    max_hops=cfg.diffusion.max_hops,
    add_virtual_blocks=(cfg.task == 'local_denoising' and not batched_seq),
    to_batched_sequential=batched_seq
)

# Compose the transformations
from torch_geometric.transforms import Compose
combined_transform = Compose([one_hot, block_transform])

# Load dataset splits with transformations
dataset_splits = {
    split: info['class'](**{
        **info['default_args'],
        'split': split,
        'transform': combined_transform
    })
    for split in ['train', 'val', 'test']
}

# Materialize the datasets for each split (use tqdm for progress bar)
dataset_splits = {
    key: [sample for sample in tqdm(dataset_splits[key], desc=f'Processing {key} split')]
    for key in dataset_splits
}

# Merge training and validation data to compute empirical distributions
merged_data = dataset_splits['train'] + dataset_splits['val']

# Compute node/edge distributions and block statistics for training
node_dist, edge_dist = parallel_utils.get_node_edge_marginal_distribution(merged_data)
init_size_dist, init_degree_dist, block_counts = parallel_utils.get_init_block_size_degree_marginal_distrbution(merged_data)

# Print dataset and block decomposition stats
max_blocks = max(block_counts)
mean_blocks = sum(block_counts) / len(block_counts)
max_block_size = len(init_size_dist)
max_block_degree = len(init_degree_dist)

print(f'Max blocks: {max_blocks}, Max block size: {max_block_size}, Max degree: {max_block_degree}')
print(f'Avg blocks: {mean_blocks:.2f}, Avg steps: {mean_blocks * cfg.diffusion.num_steps:.2f}')

# ------------------------------- Data Loaders -----------------------------------

# Define PyTorch Geometric data loaders
loaders = {
    split: DataLoader(
        dataset_splits[split],
        batch_size=cfg.train.batch_size if split == 'train' else cfg.train.batch_size * 4,
        shuffle=(split == 'train'),
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    for split in ['train', 'val', 'test']
}

# ------------------------------- Model Setup -----------------------------------

print('Initializing model...')

# Select model class based on task
if cfg.task == 'block_prediction':
    model = PredictBlockProperties(
        one_hot.num_node_classes,
        one_hot.num_edge_classes,
        max_blocks + 3,
        max_block_size,
        max_block_degree,
        channels=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        norm=cfg.model.norm,
        add_transpose=cfg.model.add_transpose,
        prenorm=cfg.model.prenorm,
        edge_channels=cfg.model.edge_hidden,
        n_head=cfg.model.num_heads,
        transformer_only=cfg.model.transformer_only,
        lr=cfg.train.lr,
        wd=cfg.train.wd,
        lr_patience=cfg.train.lr_patience,
        lr_warmup=cfg.train.lr_warmup,
        lr_scheduler=cfg.train.lr_scheduler,
        lr_epochs=cfg.train.epochs,
        use_relative_blockid=cfg.model.use_relative_blockid,
        use_absolute_blockid=cfg.model.use_absolute_blockid,
        batched_sequential=batched_seq
    )
elif cfg.task == 'local_denoising':
    model = AutoregressiveDiffusion(
        one_hot.num_node_classes,
        one_hot.num_edge_classes,
        max_blocks + 3,
        max_block_size,
        max_block_degree,
        channels=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        norm=cfg.model.norm,
        add_transpose=cfg.model.add_transpose,
        prenorm=cfg.model.prenorm,
        edge_channels=cfg.model.edge_hidden,
        n_head=cfg.model.num_heads,
        transformer_only=cfg.model.transformer_only,
        use_input=cfg.model.input_residual,
        lr=cfg.train.lr,
        wd=cfg.train.wd,
        lr_patience=cfg.train.lr_patience,
        lr_warmup=cfg.train.lr_warmup,
        lr_scheduler=cfg.train.lr_scheduler,
        lr_epochs=cfg.train.epochs,
        coeff_ce=cfg.diffusion.ce_coeff,
        ce_only=cfg.diffusion.ce_only,
        num_diffusion_steps=cfg.diffusion.num_steps,
        noise_schedule_type=cfg.diffusion.noise_schedule_type,
        noise_schedule_args={},
        uniform_noise=cfg.diffusion.uniform_noise,
        blockwise_timestep=cfg.diffusion.blockwise_time,
        node_marginal_distribution=node_dist,
        edge_marginal_distribution=edge_dist,
        initial_blocksize_distribution=init_size_dist,
        blocksize_model=None,
        combine_training=cfg.diffusion.combine_training,
        use_relative_blockid=cfg.model.use_relative_blockid,
        use_absolute_blockid=cfg.model.use_absolute_blockid,
        batched_sequential=batched_seq
    )
else:
    raise ValueError("Unsupported task type")

# Initialize model weights using Xavier initialization
def init_weights(module, gain=1.0):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
        torch.nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

model.apply(partial(init_weights, gain=1.0 if cfg.model.norm == 'ln' else 0.5))

# ------------------------------- Training Setup -----------------------------------

print('Preparing trainer...')

# Generate a unique identifier for experiment tracking and logging
tag = 'ppgnTrans' if cfg.model.add_transpose else ''
mode_label = '-BatchedSeq' if batched_seq else '-Parallel'
cfg.handtune += tag + mode_label

# Build experiment name from config tags
base_name = f'{cfg.dataset}.{cfg.diffusion.max_hops}hops.{cfg.diffusion.num_node_virtual_types}-{cfg.diffusion.num_edge_virtual_types}typeadded.{cfg.handtune}'
block_id_tag = f'.BlockID{int(cfg.model.use_absolute_blockid)}{int(cfg.model.use_relative_blockid)}'
params_tag = f'.{cfg.model.norm}.PreNorm={int(cfg.model.prenorm)}.H{cfg.model.hidden_size}.E{cfg.model.edge_hidden}.L{cfg.model.num_layers}-lr{cfg.train.lr}.{cfg.train.lr_scheduler}'
diff_tag = f'-ires{int(cfg.model.input_residual)}.blocktime{int(cfg.diffusion.blockwise_time)}.uni_noise{int(cfg.diffusion.uniform_noise)}.T{cfg.diffusion.num_steps}.{cfg.diffusion.noise_schedule_type}'
loss_tag = f'.vlb{int(not cfg.diffusion.ce_only)}.ce{int(cfg.diffusion.ce_only)+cfg.diffusion.ce_coeff}.combine={cfg.diffusion.combine_training}'

full_name = f'{base_name}{block_id_tag}{params_tag}'
if cfg.task == 'local_denoising':
    full_name += diff_tag + loss_tag
if cfg.model.transformer_only:
    full_name = 'TF-' + full_name

# Setup checkpointing and experiment directory
ckpt_dir = f'checkpoints/{cfg.task}/{full_name}'
experiment_dir = f'exps/{cfg.task}/{full_name}'
checkpoint_cb = ModelCheckpoint(
    dirpath=ckpt_dir,
    monitor='val_loss',
    save_top_k=5,
    save_last=True,
    mode='min',
    filename='{epoch}-{val_loss:.3f}'
)

# Resume training from best or last checkpoint if available
resume_ckpt = None
if cfg.train.resume:
    best_ckpt = find_checkpoint_with_lowest_val_loss(ckpt_dir)
    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
    if os.path.exists(best_ckpt) and cfg.train.resume_mode == 'best':
        resume_ckpt = best_ckpt
        print(f'Resuming from best: {best_ckpt}')
    elif os.path.exists(last_ckpt) and cfg.train.resume_mode == 'last':
        resume_ckpt = last_ckpt
        print(f'Resuming from last: {last_ckpt}')
    else:
        print('No valid checkpoint found for resuming.')

# Initialize loggers (TensorBoard + optionally WandB)
tb_logger = TensorBoardLogger('tb', name=f'{cfg.task}.{full_name}')
wb_logger = None if cfg.eval_only else WandbLogger(
    project=f'ParallelDiffusion-{cfg.task}', 
    name=full_name, 
    config=cfg
)

# Configure PyTorch Lightning trainer
trainer = L.Trainer(
    default_root_dir=experiment_dir,
    devices=[cfg.device] if isinstance(cfg.device, int) else cfg.device,
    max_epochs=cfg.train.epochs,
    callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval='epoch')],
    logger=[tb_logger, wb_logger],
    precision=cfg.train.precision
)

# ------------------------------- Training and Evaluation -----------------------------------

if not cfg.eval_only:
    print("Training started...")
    trainer.fit(model, loaders['train'], loaders['val'], ckpt_path=resume_ckpt)
    print("Running final test...")
    trainer.test(model, loaders['test'], ckpt_path='best')
