from qiskit.algorithms.optimizers import *

_OPTIMIZER_TO_CLASS_MAP = {
    "ADAM": ADAM,
    "AQGD": AQGD,
    "CG": CG,
    "COBYLA": COBYLA,
    "GSLS": GSLS,
    "GradientDescent": GradientDescent,
    "L_BFGS_B": L_BFGS_B,
    "NELDER_MEAD": NELDER_MEAD,
    "NFT": NFT,
    "P_BFGS": P_BFGS,
    "POWELL": POWELL,
    "SLSQP": SLSQP,
    "SPSA": SPSA,
    "QNSPSA": QNSPSA,
    "TNC": TNC,
    "CRS": CRS,
    "DIRECT_L": DIRECT_L,
    "DIRECT_L_RAND": DIRECT_L_RAND,
    "ESCH": ESCH,
    "ISRES": ISRES,
    "SNOBFIT": SNOBFIT,
    "BOBYQA": BOBYQA,
    "IMFIL": IMFIL
}


def get_optimizer(method: str, **kwargs) -> Optimizer:
    return _OPTIMIZER_TO_CLASS_MAP[method](**kwargs)
