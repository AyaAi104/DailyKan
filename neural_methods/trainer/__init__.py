import neural_methods.trainer.BaseTrainer
import neural_methods.trainer.PhysnetTrainer
import neural_methods.trainer.iBVPNetTrainer
import neural_methods.trainer.TscanTrainer
import neural_methods.trainer.DeepPhysTrainer
import neural_methods.trainer.EfficientPhysTrainer
import neural_methods.trainer.BigSmallTrainer
import neural_methods.trainer.PhysFormerTrainer

# ---- Make PhysMamba optional -----------------------------------------------
try:
    import neural_methods.trainer.PhysMambaTrainer  # requires mamba-ssm + causal-conv1d
except Exception as e:
    import warnings
    warnings.warn(f"PhysMambaTrainer disabled: {e}")
# ---------------------------------------------------------------------------

import neural_methods.trainer.RhythmFormerTrainer
import neural_methods.trainer.FactorizePhysTrainer
import neural_methods.trainer.DailyKanTrainer
import neural_methods.trainer.LiquidPhysTrainer
