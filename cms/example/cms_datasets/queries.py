"""Cadi entry: https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2624&ancode=B2G-22-006&tp=an&line=B2G-22-006"""

from __future__ import annotations

import re
from operator import itemgetter
from typing import NamedTuple, Any, Callable
from collections.abc import Iterator, MutableMapping
from rich.console import Console
from rich.progress import Progress

from coffea.dataset_tools import rucio_utils


console = Console()


class DuplicateProcessInfoError(KeyError):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class UniqueProcessInfoDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)

    def __getitem__(self, pinfo: ProcessInfo) -> Any:
        return self._dict[pinfo]

    def __setitem__(self, pinfo: ProcessInfo, value: Any) -> None:
        if pinfo in self._dict:
            del value
            raise DuplicateProcessInfoError(f"{pinfo!r} already exists.")
        self._dict[pinfo] = value

    def __delitem__(self, pinfo: ProcessInfo) -> None:
        del self._dict[pinfo]

    def __iter__(self) -> Iterator[ProcessInfo]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def to_dict(self) -> dict[ProcessInfo, Any]:
        return dict(self._dict)


class ProcessInfo(NamedTuple):
    name: str
    xsec: float | None  # in pb


# MC
# Info from AN2019_197_v12.pdf (page 64 ff.)
QMC = "/{0}/RunIISummer20UL{1}NanoAOD*v9-106X*/NANOAODSIM".format
# fmt: off
mc_background_queries = lambda era: {
    # TTbar
    ProcessInfo("TTToSemiLeptonic", 364.31): QMC("TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", era),
    ProcessInfo("TTToHadronic", 380.11): QMC("TTToHadronic_TuneCP5_13TeV-powheg-pythia8", era),
    ProcessInfo("TTTo2L2Nu", 87.33): QMC("TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8", era),
    # WJets (HT-binned)
    ProcessInfo("WJetsToLNu_HT-70To100", 1271.0): QMC("WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("WJetsToLNu_HT-100To200", 1253.0): QMC("WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("WJetsToLNu_HT-200To400", 335.9): QMC("WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("WJetsToLNu_HT-400To600", 45.21): QMC("WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("WJetsToLNu_HT-600To800", 10.99): QMC("WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("WJetsToLNu_HT-800To1200", 4.936 ): QMC("WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("WJetsToLNu_HT-1200To2500", 1.156): QMC("WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("WJetsToLNu_HT-2500ToInf", 0.02623): QMC("WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8", era),
    # DY (HT-binned)
    ProcessInfo("DYJetsToLL_M-50_HT-70to100", 140.1): QMC("DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("DYJetsToLL_M-50_HT-100to200", 140.2): QMC("DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("DYJetsToLL_M-50_HT-200to400", 38.399): QMC("DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("DYJetsToLL_M-50_HT-400to600", 5.21278): QMC("DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("DYJetsToLL_M-50_HT-600to800", 1.26567): QMC("DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("DYJetsToLL_M-50_HT-800to1200", 0.5684304): QMC("DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("DYJetsToLL_M-50_HT-1200to2500", 0.1331514): QMC("DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", era),
    ProcessInfo("DYJetsToLL_M-50_HT-2500toInf", 0.00297803565): QMC("DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", era),
    # Diboson
    ProcessInfo("WW", 118.7): QMC("WW_TuneCP5_13TeV-pythia8", era),
    ProcessInfo("WZ", 46.74): QMC("WZ_TuneCP5_13TeV-pythia8", era),
    ProcessInfo("ZZ", 16.91): QMC("ZZ_TuneCP5_13TeV-pythia8", era),
    # Single top
    ProcessInfo("ST_s-channel_4f", 3.36432): QMC("ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8", era),
    ProcessInfo("ST_t-channel_top_4f", 136.02): QMC("ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8", era),
    ProcessInfo("ST_t-channel_antitop_4f", 80.95): QMC("ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8", era),
    ProcessInfo("ST_tW_top_5f", 19.46655): QMC("ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8", era),
    ProcessInfo("ST_tW_antitop_5f", 19.46655): QMC("ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8", era),
    # QCD
    ProcessInfo("QCD_HT50to100", 185900000.0): QMC("QCD_HT50to100_TuneCP5_PSWeights_13TeV-madgraph-pythia8", era),
    ProcessInfo("QCD_HT100to200", 23610000.0): QMC("QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraph-pythia8", era),
    ProcessInfo("QCD_HT200to300", 1551000.0): QMC("QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8", era),
    ProcessInfo("QCD_HT300to500", 324300.0): QMC("QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8", era),
    ProcessInfo("QCD_HT500to700", 30340.0): QMC("QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8", era),
    ProcessInfo("QCD_HT700to1000", 6440.0): QMC("QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8", era),
    ProcessInfo("QCD_HT1000to1500", 1118.0): QMC("QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8", era),
    ProcessInfo("QCD_HT1500to2000", 108.0): QMC("QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8", era),
    ProcessInfo("QCD_HT2000toInf", 22.0): QMC("QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8", era),
}
 # fmt: on


# fmt: off
mc_signal_queries = lambda era: {
    # Signal samples Z’ with Γ/m = 1%
    # Caution: Currently commented out because they don't exist in DAS -> where did the AN get them from? (private production?)
    # ProcessInfo("ZPrimeToTT_M400_W4", 25.39): QMC("ZPrimeToTT_M400_W4_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M500_W5", 25.59): QMC("ZPrimeToTT_M500_W5_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M600_W6", 17.77): QMC("ZPrimeToTT_M600_W6_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M700_W7", 11.66): QMC("ZPrimeToTT_M700_W7_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M800_W8", 7.728): QMC("ZPrimeToTT_M800_W8_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M900_W9", 5.156): QMC("ZPrimeToTT_M900_W9_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M1000_W10", 3.556): QMC("ZPrimeToTT_M1000_W10_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M1200_W12", 1.738): QMC("ZPrimeToTT_M1200_W12_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M1400_W14", 0.9033): QMC("ZPrimeToTT_M1400_W14_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M1600_W16", 0.4999): QMC("ZPrimeToTT_M1600_W16_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M1800_W18", 0.2826): QMC("ZPrimeToTT_M1800_W18_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M2000_W20", 0.1659): QMC("ZPrimeToTT_M2000_W20_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M2500_W25", 0.04697): QMC("ZPrimeToTT_M2500_W25_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M3000_W30", 0.01491): QMC("ZPrimeToTT_M3000_W30_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M3500_W35", 0.00509): QMC("ZPrimeToTT_M3500_W35_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M4000_W40", 0.0019): QMC("ZPrimeToTT_M4000_W40_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M4500_W45", 0.0007635): QMC("ZPrimeToTT_M4500_W45_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M5000_W50", 3.217e-04): QMC("ZPrimeToTT_M5000_W50_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M6000_W60", 6.06e-05): QMC("ZPrimeToTT_M6000_W60_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M7000_W70", 1.154e-05): QMC("ZPrimeToTT_M7000_W70_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M8000_W80", 1.814e-06): QMC("ZPrimeToTT_M8000_W80_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # ProcessInfo("ZPrimeToTT_M9000_W90", 1.933e-07): QMC("ZPrimeToTT_M9000_W90_TuneCP2_PSweights_13TeV-madgraph-pythia8", era),
    # Signal samples Z’ with Γ/m = 10%
    ProcessInfo("ZPrimeToTT_M400_W40", 2.448): QMC("ZPrimeToTT_M400_W40_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M500_W50", 2.442): QMC("ZPrimeToTT_M500_W50_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M600_W60", 1.736): QMC("ZPrimeToTT_M600_W60_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M700_W70", 1.159): QMC("ZPrimeToTT_M700_W70_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M800_W80", 0.7745): QMC("ZPrimeToTT_M800_W80_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M900_W90", 0.5253): QMC("ZPrimeToTT_M900_W90_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1000_W100", 0.3622): QMC("ZPrimeToTT_M1000_W100_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1200_W120", 0.1821): QMC("ZPrimeToTT_M1200_W120_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1400_W140", 0.09683): QMC("ZPrimeToTT_M1400_W140_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1600_W160", 0.05402): QMC("ZPrimeToTT_M1600_W160_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1800_W180", 0.03153): QMC("ZPrimeToTT_M1800_W180_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M2000_W200", 0.01895): QMC("ZPrimeToTT_M2000_W200_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M2500_W250", 0.005914): QMC("ZPrimeToTT_M2500_W250_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M3000_W300", 0.002112): QMC("ZPrimeToTT_M3000_W300_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M3500_W350", 0.0008526): QMC("ZPrimeToTT_M3500_W350_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M4000_W400", 0.0003889): QMC("ZPrimeToTT_M4000_W400_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M4500_W450", 0.0001967): QMC("ZPrimeToTT_M4500_W450_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M5000_W500", 0.0001082): QMC("ZPrimeToTT_M5000_W500_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M6000_W600", 4.159e-05): QMC("ZPrimeToTT_M6000_W600_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M7000_W700", 1.933e-05): QMC("ZPrimeToTT_M7000_W700_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M8000_W800", 1.051e-05): QMC("ZPrimeToTT_M8000_W800_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M9000_W900", 6.295e-06): QMC("ZPrimeToTT_M9000_W900_TuneCP2_13TeV-madgraph-pythia8", era),
    # Signal samples Z’ with Γ/m = 30%
    ProcessInfo("ZPrimeToTT_M400_W120", 0.7349): QMC("ZPrimeToTT_M400_W120_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M500_W150", 0.6736): QMC("ZPrimeToTT_M500_W150_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M600_W180", 0.4839): QMC("ZPrimeToTT_M600_W180_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M700_W210", 0.329): QMC("ZPrimeToTT_M700_W210_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M800_W240", 0.2239): QMC("ZPrimeToTT_M800_W240_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M900_W270", 0.1542): QMC("ZPrimeToTT_M900_W270_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1000_W300", 0.1081): QMC("ZPrimeToTT_M1000_W300_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1200_W360", 0.05606): QMC("ZPrimeToTT_M1200_W360_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1400_W420", 0.03079): QMC("ZPrimeToTT_M1400_W420_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1600_W480", 0.01783): QMC("ZPrimeToTT_M1600_W480_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M1800_W540", 0.01073): QMC("ZPrimeToTT_M1800_W540_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M2000_W600", 0.006689): QMC("ZPrimeToTT_M2000_W600_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M2500_W750", 0.002338): QMC("ZPrimeToTT_M2500_W750_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M3000_W900", 0.0009523): QMC("ZPrimeToTT_M3000_W900_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M3500_W1050", 0.0004395): QMC("ZPrimeToTT_M3500_W1050_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M4000_W1200", 0.0002273): QMC("ZPrimeToTT_M4000_W1200_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M4500_W1350", 0.0001277): QMC("ZPrimeToTT_M4500_W1350_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M5000_W1500", 7.688e-05): QMC("ZPrimeToTT_M5000_W1500_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M6000_W1800", 3.302e-05): QMC("ZPrimeToTT_M6000_W1800_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M7000_W2100", 1.676e-05): QMC("ZPrimeToTT_M7000_W2100_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M8000_W2400", 9.521e-06): QMC("ZPrimeToTT_M8000_W2400_TuneCP2_13TeV-madgraph-pythia8", era),
    ProcessInfo("ZPrimeToTT_M9000_W2700", 5.782e-06): QMC("ZPrimeToTT_M9000_W2700_TuneCP2_13TeV-madgraph-pythia8", era),
}
# fmt: on


# Data
# Info from AN2019_197_v12.pdf (page 11 ff.)
QData = "/{0}/Run20{1}{2}*UL*MiniAODv2_NanoAODv9-v*/NANOAOD".format
era_run_mapping = {
    # G,H in principle available in 2016 but weren't used in AN2019_197_v12.pdf
    16: ["B", "C", "D", "E", "F"], #, "G", "H"],
    17: ["B", "C", "D", "E", "F"],
    18: ["A", "B", "C", "D"],
}
# fmt: off
data_queries = lambda era: {
    # SingleMuon
    **{
        ProcessInfo(f"SingleMuonRun{run}", None): QData(f"SingleMuon*", era, run)
        for run in era_run_mapping[era]
    },
    # SingleElectron / EGamma
    **(
        # era=16,17
        {
            ProcessInfo(f"SingleElectronRun{run}", None): QData(f"SingleElectron*", era, run)
            for run in era_run_mapping[era]
        }
        if era in (16, 17)
        else 
        # era=18
        {
            ProcessInfo(f"EGammaRun{run}", None): QData(f"EGamma*", era, run)
            for run in era_run_mapping[era]
        }
    ),
}
# fmt: on

_not_resolved = object()


def validate_and_resolve_queries(
    rucio_client,
    queries: dict[ProcessInfo, Any],
    veto_rules: list[Callable[[str], bool]],
    print_only_problematic: bool = True,
    progressbar_description: str = "Resolving queries...",
) -> dict[str, list[ProcessInfo]]:
    out = {}
    with Progress(console=console) as progress:
        task = progress.add_task(
            f"[cyan]{progressbar_description}", total=len(queries), console=console
        )
        for pinfo, query in queries.items():
            outlist = rucio_utils.query_dataset(
                query,
                client=rucio_client,
                tree=False,
                scope="cms",
            )

            # clean with rules
            for rule in veto_rules:
                outlist = list(filter(rule, outlist))

            ext_filter = lambda ds: not re.match(r".+_ext[1-9]-v[1-9]/NANOAODSIM", ds)
            match len(outlist):
                case 0:
                    console.print(
                        f"Warning: no datasets found for query {query}\n",
                        style="bold red",
                    )
                    out[pinfo] = _not_resolved
                case 1:
                    if len(list(filter(ext_filter, outlist))) == 0:
                        console.print(
                            f"Warning: found only an extension, not a nominal dataset for query {query}: {outlist[0]}\n",
                            style="bold red",
                        )
                        out[pinfo] = _not_resolved
                    elif not print_only_problematic:
                        console.print(
                            f"Found dataset for query {query}: {outlist[0]}\n",
                            style="bold green",
                        )
                    out[pinfo] = outlist
                case _:
                    # We allow extensions in addition to 1 unique dataset, i.e. filtering all matches by `_ext...`
                    # should then give us exactly 1 match: the unique dataset. The rest are just extensions of it
                    if len(list(filter(ext_filter, outlist))) == 1:
                        out[pinfo] = outlist
                    # These are truly problematic.
                    else:
                        console.print(
                            f"Warning: multiple datasets found for query {query}: {outlist}\n",
                            style="bold red",
                        )
                        out[pinfo] = _not_resolved
            progress.advance(task)
    return out


if __name__ == "__main__":
    # Before running this make sure you have an active voms-proxy, on coffea-casa use:
    # `voms-proxy-init -voms cms -vomses /etc/vomses -valid 192:00`
    import os
    import json
    from pathlib import Path

    # veto datasets using rules
    ignore_specific_datasets = {
        # there's a newer one than this:
        "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        # there are newer ones than these:
        "/SingleMuon/Run2016B-ver1_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        "/SingleElectron/Run2016B-ver1_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD",
        # preferring the HIPM one like in the AN, instead of:
        "/SingleMuon/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
        "/SingleElectron/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD",
    }
    veto_rules = [
        # no clue what that is -> ignore those for now
        lambda ds: not "preVFP" in ds,
        # ignore specific ones
        lambda ds: ds not in ignore_specific_datasets,
    ]

    rucio_client = rucio_utils.get_rucio_client()
    for era in (16, 17, 18):
        # make sure there are no duplicate ProcessInfo keys across all queries within the era
        era_queries = UniqueProcessInfoDict()
        era_queries.update(mc_background_queries(era))
        era_queries.update(mc_signal_queries(era))
        era_queries.update(data_queries(era))

        resolved_era_queries = validate_and_resolve_queries(
            rucio_client=rucio_client,
            queries=era_queries,
            veto_rules=veto_rules,
            progressbar_description=f"Resolving queries for 20{era}...",
        )

        # check that there isn't a problem, i.e. everything has been resolved, also save name to xsec mapping
        to_disk = {}
        name_to_xsec = {}
        for pinfo, datasets in resolved_era_queries.items():
            if datasets is _not_resolved:
                raise RuntimeError(
                    f"Problematic query found for era {era} and dataset {pinfo}. Resolve first, e.g. using rules."
                )
            name = pinfo.name
            to_disk[name] = datasets
            name_to_xsec[name] = pinfo.xsec

        # if all good, write to disk
        base = Path(__file__).parent / f"20{era}"
        os.makedirs(base, exist_ok=True)
        # resolved queries
        with open(base / f"resolved_nanoaod_queries.json", "w") as f:
            json.dump(to_disk, f, indent=2)
        # dataset name to xsec mapping (after resolution)
        with open(base / f"xsecs.json", "w") as f:
            json.dump(name_to_xsec, f, indent=2)

        # get file names
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Querying files...", total=len(to_disk), console=console
            )
            for name, datasets in to_disk.items():
                progress.console.print(f"[cyan]Querying files for {name}")
                ds_path = base / name
                os.makedirs(ds_path, exist_ok=True)
                for i, dataset in enumerate(datasets):
                    lfns = list(
                        map(
                            itemgetter("name"),
                            rucio_client.list_files(scope="cms", name=dataset),
                        )
                    )
                    # write to disk
                    with open(ds_path / f"{i}.txt", "w") as f:
                        f.write("\n".join(lfns))
                progress.advance(task)
