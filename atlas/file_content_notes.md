# Notes on branches and systematics

Run-2 and Run-3 data contain the same branches.

Systematics differ, with mc23 containing some non-closure uncertainties not in the mc20 files:
- `JET_EtaIntercalibration_NonClosure_0p2_PreRec__1[up/down]`
- `JET_FCal_MC23_NonClosure_PreRec__1[up/down]`

Similarly, some branches are present in mc20 but not m23.
In addition to the beamspot weight, there are more FTAG EVs and a few JetETmiss variations
- `weight_beamspot`
- `weight_ftag_effSF_GN2v01_Continuous_FT_EFF_Eigen_B_[24-54]__1[up/down]`
- `JET_EtaIntercalibration_NonClosure_PreRec__1[up/down]`
- `JET_JERUnc_mc20vsmc21_MC20_PreRec__1[up/down]`
- `JET_JERUnc_mc20vsmc21_MC20_PreRec_PseudoData__1[up/down]`
- `JET_JESUnc_mc20vsmc21_MC20_PreRec__1[up/down]`
- `JET_PunchThrough_MC16__1[up/down]`
- `JET_RelativeNonClosure_MC20__1[up/down]`

Fastsim branches:
- `JET_JERUnc_AF3vsFullSim_AF3_PreRec_PseudoData__1[up/down]`
- `JET_JERUnc_AF3vsFullSim_AF3_PreRec__1[up/down]`
- `JET_JERUnc_mc20vsmc21_AF3_PreRec_PseudoData__1[up/down]`
- `JET_JERUnc_mc20vsmc21_AF3_PreRec__1[up/down]`
- `JET_JESUnc_mc20vsmc21_AF3_PreRec__1[up/down]`
- `JET_RelativeNonClosure_AF3__1[up/down]`


to implement:

```python
# systematics to skip (mc20/23 or AF3 specific)
skip = [
    "JET_EtaIntercalibration_NonClosure_0p2_PreRec__1",
    "JET_FCal_MC23_NonClosure_PreRec__1",
    "JET_EtaIntercalibration_NonClosure_PreRec__1",
    "JET_JERUnc_mc20vsmc21_MC20_PreRec__1",
    "JET_JERUnc_mc20vsmc21_MC20_PreRec_PseudoData__1",
    "JET_JESUnc_mc20vsmc21_MC20_PreRec__1",
    "JET_PunchThrough_MC16__1",
    "JET_RelativeNonClosure_MC20__1",
    "JET_JERUnc_AF3vsFullSim_AF3_PreRec_PseudoData__1",
    "JET_JERUnc_AF3vsFullSim_AF3_PreRec__1",
    "JET_JERUnc_mc20vsmc21_AF3_PreRec_PseudoData__1",
    "JET_JERUnc_mc20vsmc21_AF3_PreRec__1",
    "JET_JESUnc_mc20vsmc21_AF3_PreRec__1",
    "JET_RelativeNonClosure_AF3__1"
] + [f"weight_ftag_effSF_GN2v01_Continuous_FT_EFF_Eigen_B_{i}" for i in range(24,55)]
```
