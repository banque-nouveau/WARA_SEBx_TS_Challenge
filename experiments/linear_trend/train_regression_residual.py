import argparse
import importlib
import multiprocessing
import warnings
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import logging

from amgm import config as amgm_config
from amgm.utils import common, myplot
from amgm.data.linear_trend import LinearTrendDataModule
from amgm.models.linear_trend.runner import LinearTrendRunner
from trainer_cfg.regression_residual import resolve_balancing_bins

# For parallel run is necessary otherwise, torch will overload each vCPU 
torch.set_num_threads(1)          # limit PyTorch to 1 thread for intra-op parallelism
torch.set_num_interop_threads(1)  # limit inter-op parallelism to 1 thread
    
def main(trainer_cfg, model):
    # Suppress Lightning warning about num_workers=0
    warnings.filterwarnings("ignore", ".*does not have many workers which may be a bottleneck.*")
    multiprocessing.set_start_method("spawn", force=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    wdir = amgm_config.work_dir("linear_trend")

    logger = pl_loggers.TensorBoardLogger(
        name=Path(__file__).stem,
        save_dir=wdir / "logs",
        version=model
    )
    
    print(f"Working directory: {wdir}")
    print(f"Relative log path: {Path(logger.log_dir).relative_to(wdir)}")
    print(f"Full log path: {logger.log_dir}")

    from time import time

    t0 = time()
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        mode="min",
        auto_insert_metric_name=False,
        filename="best-{epoch:02d}-val_acc={val/acc:.4f}",
        save_top_k=1,
        dirpath=Path(logger.log_dir) / "checkpoints"
        )
    
    run_cfg = trainer_cfg["run_cfg"]
    mdl = LinearTrendRunner(**trainer_cfg)
    dmod = LinearTrendDataModule(run_cfg, mdl.dset_cfg, cache_path=wdir / "datasets", rebuild_cache=True)
    trainer = Trainer(max_epochs=run_cfg["max_epochs"], check_val_every_n_epoch=1, 
                        logger=False, accelerator="cpu", callbacks=[checkpoint_callback])
    trainer.fit(mdl, datamodule=dmod)
    train_loss = trainer.callback_metrics.get("train/loss")
    val_loss = trainer.callback_metrics.get("val/loss")
    
    print("Best model score:", checkpoint_callback.best_model_score)
    print(f"Training completed in {time() - t0:.2f} seconds.")

    dmod.rebuild_cache = False  # Disable cache rebuild for validation
    results = trainer.predict(mdl, dataloaders=dmod.val_dataloader())
    merged_results = common.merge_results(results, common.cat_torch, common.cat_numpy)
    y_hat_val = merged_results["y_pred"] 
    y_true_val = merged_results["targets"]
    bins_fn = resolve_balancing_bins(mdl.dset_cfg["balancing_bins"])
    output_range = mdl.dset_cfg["output_range"]
    
    MSE_val, MAE_list, bin_names = common.metric_regression_residual(
        y_true_val,
        y_hat_val,
        window_size_test=mdl.dset_cfg["window_size_test"],
        bins_fn=bins_fn,
    )

    results_train = trainer.predict(mdl, dataloaders=dmod.train_dataloader())
    merged_results = common.merge_results(results_train, common.cat_torch, common.cat_numpy)
    y_hat_train = merged_results["y_pred"] 
    y_true_train = merged_results["targets"]

    MSE_train, MAE_list_train, bin_names_train = common.metric_regression_residual(
        y_true_train,
        y_hat_train,
        window_size_test=mdl.dset_cfg["window_size_test"],
        bins_fn=bins_fn,
    )
    
    return y_true_val, y_hat_val, MSE_val, MAE_list, float(train_loss), float(val_loss), bin_names, output_range, y_true_train, y_hat_train, MAE_list_train

def run_single_simulation(param):
    MC_id, trainer_cfg, model = param
    
    trainer_cfg_copy = deepcopy(trainer_cfg)
    
    second_first_digit = int(f"{datetime.now().second:02d}"[0]) + 1
    seed_everything(int(MC_id) * second_first_digit, workers=True)

    y_true_val, y_hat_val, MSE_val, MAE_list, train_loss, val_loss, \
        bin_names, output_range, y_true_train, y_hat_train, MAE_list_train = main(trainer_cfg_copy, model)

    return y_true_val, y_hat_val, MSE_val, MAE_list, train_loss, val_loss, bin_names, output_range, y_true_train, y_hat_train, MAE_list_train

if __name__ == "__main__":
    # This script saves 10 model checkpoints with best val_acc in Path(logger.log_dir) / "checkpoints"
    
    wdir = amgm_config.work_dir("linear_trend")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer_cfg",
        default="US_Stocks",
        help="Dotted module path (relative to this package) exposing get_trainer_cfg().",
    )
    args = parser.parse_args()

    task = Path(__file__).stem.removeprefix("train_")
    cfg_path = f"trainer_cfg.{task}.{args.trainer_cfg}"
    cfg_module = importlib.import_module(cfg_path, package=__package__ or "experiments.linear_trend")
    trainer_cfg = cfg_module.get_trainer_cfg()

    run_cfg = trainer_cfg["run_cfg"]
    dset_cfg = trainer_cfg["dset_cfg"]
    
    time_bar = dset_cfg["time_bar"]
    w_trn = dset_cfg["window_size_train"]
    w_tst = dset_cfg["window_size_test"]
    trn_dtype = dset_cfg["training_data_type"]

    model = time_bar + "_" + str(w_trn) + "_" + str(w_tst) + "_" + trn_dtype
    
    num_cpus = multiprocessing.cpu_count()
    print(f"Number of CPUs: {num_cpus}")

    MC_count = 1  # Number of Monte Carlo runs
    MC_id_list = list(range(1, MC_count+1)) 
    param_combinations = [(MC_id, trainer_cfg, model) for MC_id in MC_id_list]

    with multiprocessing.Pool(processes=1) as pool:
        results = pool.map(run_single_simulation, param_combinations)  # Ordered results

        y_true_val_list, y_hat_val_list, MSE_val_list, MAE_val_list, \
            train_loss_list, val_loss_list, bin_names_list, output_range_list, \
            y_true_train_list, y_hat_train_list, MAE_list_train_list = zip(*results)
                
        y_true_val_array = np.concatenate(y_true_val_list, axis=0)
        y_hat_val_array = np.concatenate(y_hat_val_list, axis=0)
        
        np.set_printoptions(precision=1, suppress=True)
        print(f"Train Loss List: {train_loss_list}")
        print(f"Validation Loss List: {val_loss_list}")
        
        mean_train_loss = np.mean(train_loss_list)
        std_train_loss = np.std(train_loss_list)
        mean_val_loss = np.mean(val_loss_list)
        std_val_loss = np.std(val_loss_list)
        mean_MAE_list = np.mean(MAE_val_list, axis=0)
        std_MAE_list = np.std(MAE_val_list, axis=0)
        
        print(f"Average Train Loss: {mean_train_loss:.1f} ± {std_train_loss:.1f}")
        print(f"Average Val Loss: {mean_val_loss:.1f} ± {std_val_loss:.1f}")
        print(f"Average MAE List: {mean_MAE_list} ± {std_MAE_list}")

        myplot.regression_residual_performance(y_true_val_list[0], y_hat_val_list[0], 
                                               wdir, bin_names=bin_names_list[0], 
                                               mean_MAE=mean_MAE_list, std_MAE=std_MAE_list, 
                                               dataset_type="validation",
                                               model=model,
                                               output_range=output_range_list[0])  # Plot the first run
        myplot.regression_residual_performance(y_true_train_list[0], y_hat_train_list[0], 
                                               wdir, bin_names=bin_names_list[0], 
                                               mean_MAE=mean_MAE_list, std_MAE=std_MAE_list, 
                                               dataset_type="train", 
                                               model=model,
                                               output_range=output_range_list[0])  # Plot the first run for train data
        