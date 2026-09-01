import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanAbsoluteError

from amgm import config as amgm_config
from amgm.models.mlp import MLPModel_residual

def get_trainer_cfg():
    output_size = 1

    run_cfg = dict(
        run_idx=1,
        rng_seed=1,
        batch_size=200,
        num_workers=0,
        max_epochs=10,
    )

    dset_cfg = dict(
        split_method="equal",
        num_iids=500,   # I tested upto 2000 and didn't get a significant improvement
        start_date="2014-01-01",        # 2014-01-01 for stocks
        end_date_train="2018-12-31",    # 2018-12-31 for stocks
        end_date_test="2020-12-31",     # 2020-12-31 for stocks
        window_size_train=252,  # 252 for 1Y, 378 for 1.5Y, 504 for 2Y
        window_size_test=21,    # 63 for 3M, 5 for 1W
        task_type="regression",
        label_variable="label_residual",
        time_bar="daily",  # daily turns out to be the best
        trend_value_col="ClAdjLoc",
        feature_names=["ClAdjLoc"],
        normalization=dict(ClAdjLoc="sample_minmax"),
        balancing=dict(train="residual-based", val="residual-based"),
        training_data_type="US_Stocks",
        data_source="local",  # "local" or "yahoo",
        dset_path=amgm_config.dataset_root / "data-20250901",
    )

    model_cfg = dict(
        _target_=MLPModel_residual,
        input_length=dset_cfg["window_size_train"],
        hidden_sizes=[64, 128],  # I test multiple hidden sizes randomly (upto 5000) and found this one to be the best!
        output_size=output_size,
    )
    
    # Use a serializable profile key for balancing bins (avoid checkpoint pickling callables).
    _balancing_cfg = {
        63: {"balancing_profile": "us_stocks_63d",
             "balancing_target_count": 1000,
             "output_range": 50},
        21: {"balancing_profile": "us_stocks_21d",
             "balancing_target_count": 2000,
             "output_range": 20},
        5:  {"balancing_profile": "us_stocks_5d",
             "balancing_target_count": 2000,
             "output_range": 15},
        }

    dset_cfg["balancing_bins"] = _balancing_cfg[dset_cfg["window_size_test"]]["balancing_profile"]
    dset_cfg["balancing_target_count"] = _balancing_cfg[dset_cfg["window_size_test"]]["balancing_target_count"]
    dset_cfg["output_range"] = _balancing_cfg[dset_cfg["window_size_test"]]["output_range"]
    model_cfg["output_range"] = dset_cfg["output_range"]
    
    trainer_cfg = dict(
        run_cfg=run_cfg,
        dset_cfg=dset_cfg,
        model_cfg=model_cfg,
        loss_cfg = dict(_target_=nn.MSELoss),
        acc_cfg = dict(_target_=MeanAbsoluteError),
        optim_cfg=dict(_target_=torch.optim.Adam, lr=1e-3, weight_decay=1e-5),
        sched_cfg=dict(_target_=StepLR, step_size=run_cfg["max_epochs"], gamma=0.2),  # I found no decay works better so I set step_size=max_epochs
    )

    return trainer_cfg


def _residual_bins_63d(residuals):
    """Bins for residual-based balancing when window_size_test == 63 (3-month stocks)."""
    return {
        'lt50':    np.where(residuals <= -50)[0],
        'n40to50': np.where((-50 < residuals) & (residuals < -40))[0],
        'n30to40': np.where((-40 <= residuals) & (residuals < -30))[0],
        'n20to30': np.where((-30 <= residuals) & (residuals < -20))[0],
        'n10to20': np.where((-20 <= residuals) & (residuals < -10))[0],
        'n0to10':  np.where((-10 <= residuals) & (residuals < 0))[0],
        'zero':    np.where(residuals == 0)[0],
        '0to10':   np.where((0 < residuals) & (residuals <= 10))[0],
        '10to20':  np.where((10 < residuals) & (residuals <= 20))[0],
        '20to30':  np.where((20 < residuals) & (residuals <= 30))[0],
        '30to40':  np.where((30 < residuals) & (residuals <= 40))[0],
        '40to50':  np.where((40 < residuals) & (residuals < 50))[0],
        'gt50':    np.where(50 <= residuals)[0],
    }

def _residual_bins_21d(residuals):
    """Bins for residual-based balancing when window_size_test == 21 (1-month stocks)."""
    return {
        'lt20':    np.where(residuals <= -20)[0],
        'n10to20': np.where((-20 < residuals) & (residuals < -10))[0],
        'n5to10': np.where((-10 <= residuals) & (residuals < -5))[0],
        'n2to5': np.where((-5 <= residuals) & (residuals < -2))[0],
        'n1to2': np.where((-2 <= residuals) & (residuals < -1))[0],
        'n0to1':  np.where((-1 <= residuals) & (residuals < 0))[0],
        'zero':    np.where(residuals == 0)[0],
        '0to1':   np.where((0 < residuals) & (residuals <= 1))[0],
        '1to2':  np.where((1 < residuals) & (residuals <= 2))[0],
        '2to5':  np.where((2 < residuals) & (residuals <= 5))[0],
        '5to10':  np.where((5 < residuals) & (residuals <= 10))[0],
        '10to20':  np.where((10 < residuals) & (residuals < 20))[0],
        'gt20':    np.where(20 <= residuals)[0],
    }
    
def _residual_bins_5d(residuals):
    """Bins for residual-based balancing when window_size_test == 5 (1-week stocks)."""
    return {
        'lt15':    np.where(residuals <= -15)[0],
        'n10to15': np.where((-15 < residuals) & (residuals < -10))[0],
        'n5to10':  np.where((-10 <= residuals) & (residuals < -5))[0],
        'n0to5':   np.where((-5 <= residuals) & (residuals < 0))[0],
        'zero':    np.where(residuals == 0)[0],
        '0to5':    np.where((0 < residuals) & (residuals <= 5))[0],
        '5to10':   np.where((5 < residuals) & (residuals <= 10))[0],
        '10to15':  np.where((10 < residuals) & (residuals < 15))[0],
        'gt15':    np.where(15 <= residuals)[0],
    }
