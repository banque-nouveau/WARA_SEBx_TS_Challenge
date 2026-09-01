import random
import gc
import numpy as np
import torch
from lightning import Trainer

import multiprocessing

from amgm import config as amgm_config
from amgm.data.loading import load_sebx_am_data
from amgm.data.linear_trend import LinearTrendDataModule
from amgm.models.linear_trend.runner import LinearTrendRunner
from amgm.utils import common, myplot
from trainer_cfg.regression_residual import resolve_balancing_bins

def run_single_simulation(param_combination):
    MC_id, script_name, model, window_size_train, time_bar, \
        start_date, end_date_train, end_date_test, issue_ids = param_combination  # Extract the parameters from the tuple
    issue_ids = np.asarray(issue_ids, dtype=str)

    wdir = amgm_config.work_dir("linear_trend")
    log_dir = wdir / "logs" / script_name / model
    
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))  # Adjust pattern if needed
    ckpt_path = checkpoint_files[0]  # Use the first checkpoint for initial setup

    mdl = LinearTrendRunner.load_from_checkpoint(ckpt_path, weights_only=False)
    trainer = Trainer(enable_checkpointing=False, logger=False)
    
    ensemble_sum = None
    n_preds = 0
    first_ckpt = True
    
    for ckpt_path in checkpoint_files:
        # Reload the model from checkpoint
        mdl = LinearTrendRunner.load_from_checkpoint(ckpt_path, weights_only=False)
        
        if first_ckpt:
            mdl.dset_cfg["window_size_train"] = window_size_train
            mdl.dset_cfg["start_date"] = start_date
            mdl.dset_cfg["end_date_train"] = end_date_train
            mdl.dset_cfg["end_date_test"] = end_date_test
            mdl.dset_cfg["time_bar"] = time_bar # monthly turns out to be the best 
            mdl.dset_cfg["balancing"] = None
            mdl.dset_cfg["dset_path"] = amgm_config.am_dataset_dir
            
            # In multiprocessing runs, disable cache writes to avoid .tmp rename races.
            dmod = LinearTrendDataModule(
                mdl.run_cfg,
                mdl.dset_cfg,
                issue_ids,
                cache_path=wdir / "datasets",
                rebuild_cache=False,
                save_cache=False,
            )
            dmod.prepare_data()
            val_loader = dmod.val_dataloader()
            first_ckpt = False
        
        # Predict on validation set and gather outputs
        results = trainer.predict(mdl, dataloaders=val_loader)
        # Keep only tensors needed for ensemble metrics to avoid materializing
        # large merged payloads (features/metadata) in memory.
        y_hat_val = torch.cat([batch["y_pred"].reshape(-1).detach().cpu() for batch in results], dim=0)
        y_val = torch.cat([batch["targets"].reshape(-1).detach().cpu() for batch in results], dim=0)
        del results
        
        if ensemble_sum is None:
            ensemble_sum = y_hat_val.to(torch.float32)
        else:
            ensemble_sum += y_hat_val.to(torch.float32)
        n_preds += 1
        
    ensemble_preds = ensemble_sum / n_preds
    print(f"Combined prediction tensor shape: {ensemble_preds.shape}")
    
    y_true = y_val.squeeze()  # Assuming y_val is the ground truth tensor
    bins_fn = resolve_balancing_bins(mdl.dset_cfg["balancing_bins"])
    target_count = mdl.dset_cfg["balancing_target_count"]
    output_range = mdl.dset_cfg["output_range"]
    
    print("Evaluating " + log_dir.name + " model...")
    MSE_val, MAE_list, bin_names = common.metric_regression_residual(
        y_true,
        ensemble_preds,
        window_size_test=mdl.dset_cfg["window_size_test"],
        bins_fn=bins_fn,
    )

    return y_true, ensemble_preds, MSE_val, MAE_list, bin_names, output_range, bins_fn, target_count

if __name__ == "__main__":
    # Make sure the model checkpoints are saved under log_dir for each lookback.
    # Run train_regression_time.py for each lookback configuration to save the checkpoints
    print("This script is taking ensemble over 10 instances of NNs with the same lookback but trained with different epochs...")
    
    wdir = amgm_config.work_dir("linear_trend")
    
    # For parallel run is necessary otherwise, torch will overload each vCPU 
    torch.set_num_threads(1)          # limit PyTorch to 1 thread for intra-op parallelism
    torch.set_num_interop_threads(1)  # limit inter-op parallelism to 1 thread
    
    # Get the number of CPUs
    num_cpus = multiprocessing.cpu_count()
    print(f"Number of CPUs: {num_cpus}")
    
    script_name = "train_regression_residual"    
    model = "daily_252_21_US_Stocks"
    time_bar = str(model.split("_")[0])
    window_size_train = int(model.split("_")[1])
    window_size_test = int(model.split("_")[2])
    training_data_type = "US_Stocks"
    
    start_date = "2018-12-31"
    end_date_train = "2018-12-31"
    end_date_test = "2020-12-31"

    data = load_sebx_am_data(amgm_config.am_dataset_dir)
    all_issue_ids = data["security_data"]["IssueId"].unique().tolist()
    num_iids = 500
    issue_ids_by_run = [
        random.Random(MC_id).sample(all_issue_ids, num_iids)
        for MC_id in range(1, 11)
    ]
    print(f"Sampled {num_iids} issue IDs for each Monte Carlo run.")
    del data
    gc.collect()
    
    MC_count = 10  # Number of Monte Carlo runs
    MC_id_list = list(range(1, MC_count+1))  # 1 to 10 inclusive
    param_combinations = [
        (MC_id, script_name, model, window_size_train, time_bar,
         start_date, end_date_train, end_date_test, issue_ids_by_run[MC_id - 1])
        for MC_id in MC_id_list
    ]
    
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(run_single_simulation, param_combinations)
        
        y_true_val_list, ensemble_preds_list, MSE_list, MAE_list, \
            bin_names_list, output_range_list, bins_fn_list, target_count_list = zip(*results)
        
        MSE_array = np.array(MSE_list)
        MAE_array = np.array(MAE_list)
        
        print(f"shape of MSE array: {MSE_array.shape}")
        print(f"shape of MAE array: {MAE_array.shape}")
        
        np.set_printoptions(precision=1, suppress=True)
        print(f"MSE List: {MSE_array}")
        
        mean_MSE = np.mean(MSE_array)
        std_MSE = np.std(MSE_array)
        mean_MAE = np.mean(MAE_array, axis=0)
        std_MAE = np.std(MAE_array, axis=0)
        avg_MAE_across_label = np.mean(MAE_array, axis=1)
        overall_MAE = np.mean(avg_MAE_across_label)
        overall_MAE_std = np.std(avg_MAE_across_label)
        
        print(f"Average MSE: {mean_MSE:.4f} ± {std_MSE:.4f}")
        print(f"Mean MAE:\n{mean_MAE}")
        print(f"Std MAE:\n{std_MAE}")
        
        print(f"Average MAE across all labels: {overall_MAE:.1f} ± {overall_MAE_std:.1f}")
        
        myplot.regression_residual_performance(y_true_val_list[0], ensemble_preds_list[0], wdir, bin_names=bin_names_list[0], mean_MAE=mean_MAE, std_MAE=std_MAE, model=model, output_range=output_range_list[0])  # Plot the first run

        print(f"min value y_true_val_list[0]: {y_true_val_list[0].min()}, max value y_true_val_list[0]: {y_true_val_list[0].max()}")
        print(f"min value ensemble_preds_list[0]: {ensemble_preds_list[0].min()}, max value ensemble_preds_list[0]: {ensemble_preds_list[0].max()}")
