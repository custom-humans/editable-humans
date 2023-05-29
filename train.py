import os, sys
from datetime import datetime
import logging as log
import numpy as np
import torch
import random
import shutil
import tempfile
import wandb
import pickle

from torch.utils.data import DataLoader
from lib.datasets.customhumans_dataset import CustomHumanDataset
from lib.models.trainer import Trainer
from lib.models.evaluator import Evaluator
from lib.utils.config import *


def create_archive(save_dir, config):

    with tempfile.TemporaryDirectory() as tmpdir:

        shutil.copy(config, os.path.join(tmpdir, 'config.yaml'))
        shutil.copy('train.py', os.path.join(tmpdir, 'train.py'))
        shutil.copy('test.py', os.path.join(tmpdir, 'test.py'))

        shutil.copytree(
            os.path.join('lib'),
            os.path.join(tmpdir, 'lib'),
            ignore=shutil.ignore_patterns('__pycache__'))

        shutil.make_archive(
            os.path.join(save_dir, 'code_copy'),
            'zip',
            tmpdir) 


def main(config):

    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    log_dir = os.path.join(
            config.save_root,
            config.exp_name,
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )

    # Backup code.
    create_archive(log_dir, config.config)
    
    # Initialize dataset and dataloader.

    with open('data/smpl_mesh.pkl', 'rb') as f:
        smpl_mesh = pickle.load(f)

    dataset = CustomHumanDataset(config.num_samples, config.repeat_times)
    dataset.init_from_h5(config.data_root)

    loader = DataLoader(dataset=dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True, 
                        num_workers=config.workers,
                        pin_memory=True)
    

    trainer = Trainer(config, dataset.smpl_V, smpl_mesh['smpl_F'], log_dir)

    evaluator = Evaluator(config, log_dir)


    if config.wandb_id is not None:
        wandb_id = config.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(log_dir, 'wandb_id.txt'), 'w+') as f:
            f.write(wandb_id)

    wandb_mode = "disabled" if (not config.wandb) else "online"
    wandb.init(id=wandb_id,
               project=config.wandb_name,
               config=config,
               name=os.path.basename(log_dir),
               resume="allow",
               settings=wandb.Settings(start_method="fork"),
               mode=wandb_mode,
               dir=log_dir)
    wandb.watch(trainer)

    if config.resume:
        trainer.load_checkpoint(config.resume)


    global_step = trainer.global_step
    start_epoch = trainer.epoch


    for epoch in range(start_epoch, config.epochs):
        for data in loader:
            trainer.step(epoch=epoch, n_iter=global_step, data=data)
            
            if global_step % config.log_every == 0:
                trainer.log(global_step, epoch)

            if config.use_2d_from_epoch >= 0 and \
                epoch >= config.use_2d_from_epoch and \
                global_step % config.log_every == 0:
                trainer.write_images(global_step)

            global_step += 1

        if epoch % config.save_every == 0:
            trainer.save_checkpoint(full=False)

        if epoch % config.valid_every == 0 and epoch > 0:
            evaluator.init_models(trainer)
            evaluator.reconstruction(32, epoch=epoch)

    wandb.finish()

if __name__ == "__main__":

    parser = parse_options()
    args, args_str = argparse_to_str(parser)
    handlers = [log.StreamHandler(sys.stdout)]
    log.basicConfig(level=args.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    log.info(f'Info: \n{args_str}')
    main(args)