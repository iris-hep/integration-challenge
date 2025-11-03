import awkward as ak
import numpy as np


# functions creating systematic variations
def jet_pt_resolution(pt):
    # normal distribution with 5% variations, shape matches jets
    counts = ak.num(pt)
    pt_flat = ak.flatten(pt)
    resolution_variation = np.random.normal(np.ones_like(pt_flat), 0.05)
    resolution_variation = ak.from_numpy(resolution_variation)
    resolution_variation = ak.to_backend(
        resolution_variation, ak.backend(pt_flat)
    )
    return ak.unflatten(resolution_variation, counts)


def jet_pt_scale():
    return 1.5
