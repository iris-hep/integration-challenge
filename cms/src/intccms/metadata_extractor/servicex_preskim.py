"""ServiceX skimming before coffea metadata preprocess.

One batched :func:`servicex.deliver` over all datasets; returns a fileset whose ``files``
keys are output URLs.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SERVICEX_DELIVER_KWARGS: Dict[str, Any] = {
    "fail_if_incomplete": True,
    "ignore_local_cache": False,
}

# NanoAOD Events branch names for ServiceX ``UprootRaw`` ``filter_name``.
# Align with ``example_cms/configs/configuration.py`` preprocess branches + mc_only.
_NANOAOD_BRANCHES: Tuple[str, ...] = (
    "nMuon",
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_mass",
    "Muon_miniIsoId",
    "Muon_tightId",
    "Muon_charge",
    "Muon_etaErr",
    "nFatJet",
    "FatJet_particleNet_TvsQCD",
    "FatJet_pt",
    "FatJet_eta",
    "FatJet_phi",
    "FatJet_mass",
    "nJet",
    "Jet_btagDeepB",
    "Jet_jetId",
    "Jet_pt",
    "Jet_eta",
    "Jet_phi",
    "Jet_mass",
    "Jet_hadronFlavour",
    "Jet_area",
    "PuppiMET_pt",
    "PuppiMET_phi",
    "HLT_Mu50",
    "run",
    "luminosityBlock",
    "event",
    "fixedGridRhoFastjetAll",
)
_NANOAOD_MC_ONLY: Tuple[str, ...] = ("Pileup_nTrueInt", "genWeight")

Task = Tuple[str, Dict[str, Any], List[str], str, List[str]]


def _skimming_columns(is_data: bool) -> List[str]:
    """Selection of branches to keep during skimming."""
    if is_data:
        return list(_NANOAOD_BRANCHES)
    return list(_NANOAOD_BRANCHES + _NANOAOD_MC_ONLY)


def _flatten_delivered_urls(value: Any) -> List[str]:
    """Flatten nested ``deliver()`` outputs."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, dict):
        return [u for v in value.values() for u in _flatten_delivered_urls(v)]
    if isinstance(value, Iterable):
        return [u for item in value for u in _flatten_delivered_urls(item)]
    raise TypeError(
        f"Unexpected ServiceX URL entry type {type(value)!r}; "
        "expected str, bytes, dict, or iterable"
    )


def _silence_verbose_servicex_loggers() -> None:
    """Silence INFO loggers to keep notebook clean."""
    for name in (
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.http2",
        "urllib3",
        "urllib3.connectionpool",
        "servicex",
        "servicex.servicex_client",
        "servicex.deliver",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _deliver_tasks(tasks: List[Task], deliver_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build ServiceX query and run one ``servicex.deliver`` call which will submit
    transformer pods in parallel."""
    try:
        from servicex import dataset, query, deliver
    except ImportError as exc:
        raise ImportError(
            "general.servicex.enable_preskim=True requires the 'servicex' package."
        ) from exc

    _silence_verbose_servicex_loggers()
    samples: List[Dict[str, Any]] = []
    for dataset_key, _, input_files, treename, columns in tasks:
        samples.append(
            {
                "Name": f"sx_{dataset_key}",
                "Dataset": dataset.FileList(input_files),
                "Query": query.UprootRaw(
                    [
                        {
                            "treename": treename,
                            "filter_name": columns,
                            "fail_on_missing_trees": True,
                        },
                        {"treename": "LuminosityBlocks"},
                        {"treename": "Runs"},
                    ]
                ),
            }
        )
    spec = {"General": {"Delivery": "URLs"}, "Sample": samples}
    return deliver(spec, **deliver_kwargs)


class ServiceXMetadataSkimmer:
    """Run ServiceX skimming and return an updated fileset."""

    def run(
        self,
        input_fileset: Dict[str, Dict[str, Any]],
        servicex_deliver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the fileset with ``files`` keys replaced by ServiceX output paths/urls.

        Parameters
        ----------
        servicex_deliver_kwargs
            Passed to ``servicex.deliver`` after merging over
            :data:`DEFAULT_SERVICEX_DELIVER_KWARGS`.
        """
        deliver_kwargs = {
            **DEFAULT_SERVICEX_DELIVER_KWARGS,
            **(servicex_deliver_kwargs or {}),
        }
        tasks: List[Task] = []
        for dataset_key, entry in input_fileset.items():
            metadata = entry.get("metadata", {})
            is_data = bool(metadata.get("is_data", False))
            files = entry.get("files", {})
            input_files = list(files.keys())
            if not input_files:
                logger.warning(
                    "Dataset '%s' has no input files, skipping ServiceX",
                    dataset_key,
                )
                continue
            treename = next(iter(files.values()), "Events")
            tasks.append(
                (dataset_key, metadata, input_files, treename, _skimming_columns(is_data))
            )

        if not tasks:
            return dict(input_fileset)

        logger.info(
            "Submitting ServiceX transformers: %s samples in one deliver() call",
            len(tasks),
        )
        delivered = _deliver_tasks(tasks, deliver_kwargs)

        out: Dict[str, Dict[str, Any]] = {}
        for dataset_key, metadata, _, treename, _ in tasks:
            name = f"sx_{dataset_key}"
            urls = _flatten_delivered_urls(delivered[name])
            out[dataset_key] = {
                "files": {u: treename for u in urls},
                "metadata": metadata,
            }
        return out
