"""
Samples of the H+ -> cb dijet-mass fit, shared by the TRExFitter and cabinetry configs.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Sample:
    """one sample of the fit

    `directory` is the TDirectory in the input file that holds the sample, without the
    region prefix -- TRExFitter uses it as `HistoNameSuff` and cabinetry as `SamplePath`.
    """

    name: str
    directory: str
    title: str = ""
    root_color: int = 0
    color: str = ""

    @property
    def path(self) -> str:
        """`SamplePath` for cabinetry: the directory without the trailing slash"""
        return self.directory.rstrip("/")


DATA_SAMPLE = Sample("data", "data/", "Data")

SIGNAL_SAMPLE = Sample("Hplus_cb", "Hplus_cb/", "H^{+} #rightarrow cb", 632, "#d62728")

BACKGROUND_SAMPLES = [
    Sample("ttbar",    "ttbar_nom/", "t#bar{t}",          800, "#f89c20"),
    Sample("Wt",       "Wt/",        "Wt",                807, "#e76300"),
    Sample("st_tchan", "st_tchan/",  "Single top t-chan", 831, "#2ca089"),
    Sample("st_schan", "st_schan/",  "Single top s-chan", 833, "#1a7a68"),
    Sample("wjets",    "wjets/",     "W+jets",            417, "#5ac25a"),
    Sample("zjets",    "zjets/",     "Z+jets",            861, "#3f90da"),
    Sample("diboson",  "diboson/",   "Diboson",           616, "#c060c0"),
    Sample("ttV",      "ttV/",       "t#bar{t}V",         880, "#964a8b"),
    Sample("rare_top", "rare_top/",  "Rare top",          920, "#9c9ca1"),
]

# all samples a systematic can be applied to
MC_SAMPLES = [SIGNAL_SAMPLE] + BACKGROUND_SAMPLES
