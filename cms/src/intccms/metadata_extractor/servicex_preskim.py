"""ServiceX skimming before coffea metadata preprocess.

One batched :func:`servicex.deliver` over all datasets; returns a fileset whose ``files``
keys are output URLs.
"""

import inspect
import logging
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from intccms.utils.tools import dict_to_branches

logger = logging.getLogger(__name__)

Task = Tuple[str, Dict[str, Any], List[str], str, List[str], Optional[str]]


def _skimming_columns(preprocess_config: Any, is_data: bool) -> List[str]:
    """Flat NanoAOD branch names for ServiceX; strip ``mc_branches`` when ``is_data``."""
    flat = dict_to_branches(
        preprocess_config.branches, nanoaod_collection_counts=True
    )
    if is_data:
        mc_only: Set[str] = set(dict_to_branches(preprocess_config.mc_branches))
        return [c for c in flat if c not in mc_only]
    return flat


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


def _selection_to_cut(selection_config: Any, columns: List[str]) -> str:
    """Build a ServiceX ``PythonFunction`` source string from a selection config.

    The returned string defines ``run_query(input_filenames=None)`` that embeds
    the selection function, reads ``columns`` (assumed to already include all
    branches needed for the selection), assembles NanoAOD collections via
    ``ak.zip``, and returns ``{"Events": filtered, "LuminosityBlocks": ..., "Runs": ...}``.
    ``PackedSelection.all(*sel.names)`` is used to extract the combined mask.
    """
    func = selection_config.function
    use_spec = selection_config.use or []

    collections_strings = ""
    call_args: List[str] = []
    for obj, _ in use_spec:
        obj_cols = [col for col in columns if col.startswith(f"{obj}_")]
        zip_dict = ", ".join(f'"{col[len(obj) + 1:]}": events["{col}"]' for col in obj_cols)
        var = obj.lower()
        collections_strings += f"        {var} = ak.zip({{{zip_dict}}})\n"
        call_args.append(var)

    columns_needed = list(columns)
    func_src = textwrap.indent(textwrap.dedent(inspect.getsource(func)).strip(), "    ")

    return f"""\
def run_query(input_filenames=None):
    import uproot
    import awkward as ak
    from coffea.analysis_tools import PackedSelection

{func_src}

    columns_needed = {columns_needed!r}
    with uproot.open(input_filenames) as f:
        columns_avail = set(f['Events'].keys())
        events = f['Events'].arrays([b for b in columns_needed if b in columns_avail], library='ak')
{collections_strings}
        sel = {func.__name__}({', '.join(call_args)})
        mask = sel.all(*sel.names)
        lumi = f['LuminosityBlocks'].arrays(library='ak') if 'LuminosityBlocks' in f else ak.Array([])
        runs = f['Runs'].arrays(library='ak') if 'Runs' in f else ak.Array([])
    return {{'Events': events[mask], 'LuminosityBlocks': lumi, 'Runs': runs}}"""


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
    for dataset_key, _, input_files, treename, columns, cut in tasks:
        if cut is not None:
            servicex_query = query.PythonFunction(cut)
        else:
            servicex_query = query.UprootRaw(
                [
                    {
                        "treename": treename,
                        "filter_name": columns,
                        "fail_on_missing_trees": True,
                    },
                    {"treename": "LuminosityBlocks"},
                    {"treename": "Runs"},
                ]
            )
        samples.append(
            {
                "Name": f"sx_{dataset_key}",
                "Dataset": dataset.FileList(input_files),
                "Query": servicex_query,
            }
        )
    spec = {"General": {"Delivery": "URLs"}, "Sample": samples}
    return deliver(spec, **deliver_kwargs)


class ServiceXMetadataSkimmer:
    """Run ServiceX skimming and return an updated fileset."""

    def run(
        self,
        input_fileset: Dict[str, Dict[str, Any]],
        preprocess_config: Any,
        servicex_deliver_kwargs: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Any] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the fileset with ``files`` keys replaced by ServiceX output paths/urls.

        Parameters
        ----------
        preprocess_config
            Preprocess config with ``branches`` and ``mc_branches`` (e.g.
            ``config.preprocess``). ServiceX ``filter_name`` lists are built via
            :func:`intccms.utils.tools.dict_to_branches`.
        servicex_deliver_kwargs
            Keyword arguments forwarded to ``servicex.deliver`` (e.g.
            ``general.servicex.deliver_kwargs`` from the analysis config).
        selection_config
            Optional baseline selection config (e.g. ``config.baseline_selection``).
            When provided, its ``function`` and ``use`` spec are compiled into a
            ServiceX ``PythonFunction`` query that filters events on the transformer
            workers before delivery.
        """
        deliver_kwargs: Dict[str, Any] = dict(servicex_deliver_kwargs or {})
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
            columns = _skimming_columns(preprocess_config, is_data)
            cut = (
                _selection_to_cut(selection_config, columns)
                if selection_config is not None
                else None
            )
            if cut is not None:
                logger.debug(
                    "Compiled ServiceX PythonFunction for '%s':\n%s",
                    dataset_key,
                    cut,
                )
            tasks.append(
                (
                    dataset_key,
                    metadata,
                    input_files,
                    treename,
                    columns,
                    cut,
                )
            )

        if not tasks:
            return dict(input_fileset)

        logger.info(
            "Submitting ServiceX transformers: %s samples in one deliver() call",
            len(tasks),
        )
        delivered = _deliver_tasks(tasks, deliver_kwargs)

        out: Dict[str, Dict[str, Any]] = {}
        for dataset_key, metadata, _, treename, _, _ in tasks:
            name = f"sx_{dataset_key}"
            urls = _flatten_delivered_urls(delivered[name])
            out[dataset_key] = {
                "files": {u: treename for u in urls},
                "metadata": metadata,
            }
        return out
