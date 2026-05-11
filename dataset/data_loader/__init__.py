import importlib
import warnings

import dataset.data_loader.BaseLoader


def _optional_import(module_name):
    short_name = module_name.rsplit(".", 1)[-1]
    try:
        globals()[short_name] = importlib.import_module(module_name)
    except Exception as e:
        warnings.warn(f"{short_name} disabled: {e}")


for _module_name in [
    "dataset.data_loader.COHFACELoader",
    "dataset.data_loader.UBFCrPPGLoader",
    "dataset.data_loader.PURELoader",
    "dataset.data_loader.iBVPLoader",
    "dataset.data_loader.SCAMPSLoader",
    "dataset.data_loader.MMPDLoader",
    "dataset.data_loader.BP4DPlusLoader",
    "dataset.data_loader.BP4DPlusBigSmallLoader",
    "dataset.data_loader.UBFCPHYSLoader",
    "dataset.data_loader.PhysDriveLoader",
    "dataset.data_loader.LADHLoader",
    "dataset.data_loader.SUMSLoader",
    "dataset.data_loader.AyaLoader2",
    "dataset.data_loader.AyaLoader",
    "dataset.data_loader.DailyLoader",
]:
    _optional_import(_module_name)

del _module_name
