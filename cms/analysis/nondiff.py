import glob
import logging
import os
import warnings
from collections import defaultdict
from typing import Any, Literal, Optional, Dict, List, Tuple
import awkward as ak
import cabinetry
import hist
import numpy as np
import vector
from coffea.analysis_tools import PackedSelection

from analysis.base import Analysis
from user.cuts import lumi_mask
from utils.output_files import (
    save_histograms_to_pickle,
    save_histograms_to_root,
)
from utils.output_manager import OutputDirectoryManager
from utils.stats import get_cabinetry_rebinning_router
from utils.tools import get_function_arguments

# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()

# -----------------------------
# Logging Configuration
# -----------------------------
logger = logging.getLogger("NonDiffAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# -----------------------------
# ZprimeAnalysis Class Definition
# -----------------------------
class NonDiffAnalysis(Analysis):

    def __init__(self, config: dict[str, Any], processed_datasets: Dict[str, List[Tuple[Any, Dict[str, Any]]]], output_manager: OutputDirectoryManager) -> None:
        """
        Initialize ZprimeAnalysis with configuration and processed datasets.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'systematics', 'corrections', 'channels',
            and 'general'.
        processed_datasets : Dict[str, List[Tuple[Any, Dict[str, Any]]]]
            Pre-processed datasets from skimming (required)
        output_manager : OutputDirectoryManager
            Centralized output directory manager (required)
        """
        super().__init__(config, processed_datasets, output_manager)
        self.nD_hists_per_region = self._init_histograms()


    def _init_histograms(self) -> dict[str, dict[str, hist.Hist]]:
        """
        Initialize histograms for each analysis channel based on configuration.

        Returns
        -------
        dict
            Dictionary of channel name to hist.Hist object.
        """
        histograms = defaultdict(dict)
        for channel in self.channels:
            channel_name = channel.name
            if (req_channels := self.config.general.channels) is not None:
                if channel_name not in req_channels:
                    continue

            for observable in channel.observables:

                observable_label = observable.label
                observable_binning = observable.binning
                observable_name = observable.name

                if isinstance(observable_binning, str):
                    low, high, nbins = map(
                        float, observable_binning.split(",")
                    )
                    axis = hist.axis.Regular(
                        int(nbins),
                        low,
                        high,
                        name="observable",
                        label=observable_label,
                    )
                else:
                    axis = hist.axis.Variable(
                        observable_binning,
                        name="observable",
                        label=observable_label,
                    )

                histograms[channel_name][observable_name] = hist.Hist(
                    axis,
                    hist.axis.StrCategory([], name="process", growth=True),
                    hist.axis.StrCategory([], name="variation", growth=True),
                    storage=hist.storage.Weight(),
                )
        return histograms

    def histogramming(
        self,
        object_copies: dict[str, ak.Array],
        events: ak.Array,
        process: str,
        variation: str,
        xsec_weight: float,
        analysis: str,
        event_syst: Optional[dict[str, Any]] = None,
        direction: Literal["up", "down", "nominal"] = "nominal",
    ) -> Optional[ak.Array]:
        """
        Apply physics selections and fill histograms.

        Parameters
        ----------
        object_copies : dict
            Corrected event-level objects.
        events : ak.Array
            Original NanoAOD event collection.
        process : str
            Sample name.
        variation : str
            Systematic variation label.
        xsec_weight : float
            Normalization weight.
        analysis : str
            Analysis name string.
        event_syst : dict, optional
            Event-level systematic to apply.
        direction : str, optional
            Systematic direction: 'up', 'down', or 'nominal'.

        Returns
        -------
        dict
            Updated histogram dictionary.
        """
        if process == "data" and variation != "nominal":
            return

        for channel in self.channels:
            channel_name = channel.name
            if (req_channels := self.config.general.channels) is not None:
                if channel_name not in req_channels:
                    continue
            logger.info(f"Applying selection for {channel_name} in {process}")
            mask = 1
            if (selection_funciton := channel.selection.function) is not None:
                selection_args = get_function_arguments(
                    channel.selection.use,
                    object_copies,
                    function_name=channel.selection.function.__name__,
                )
                packed_selection = selection_funciton(*selection_args)
                if not isinstance(packed_selection, PackedSelection):
                    raise ValueError(
                        f"PackedSelection expected, got {type(packed_selection)}"
                    )
                mask = ak.Array(
                    packed_selection.all(packed_selection.names[-1])
                )

            if process == "data":
                mask = mask & lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                )

            if ak.sum(mask) == 0:
                logger.warning(
                    f"{analysis}:: No events left in {channel_name} for {process} with "
                    + "variation {variation}"
                )
                continue

            object_copies_channel = {
                collection: variable[mask]
                for collection, variable in object_copies.items()
            }

            if process != "data":
                weights = (
                    events[mask][self.config.general.weight_branch]
                    * xsec_weight
                    / abs(events[mask][self.config.general.weight_branch])
                )
            else:
                weights = np.ones(ak.sum(mask))

            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, object_copies_channel
                )

            logger.info(
                f"Number of weighted events in {channel_name}: {ak.sum(weights):.2f}"
            )
            logger.info(
                f"Number of raw events in {channel_name}: {ak.sum(mask)}"
            )
            for observable in channel.observables:
                observable_name = observable.name
                logger.info(f"Computing observable {observable_name}")
                observable_args = get_function_arguments(
                    observable.use,
                    object_copies_channel,
                    function_name=observable.function.__name__,
                )
                observable_vals = observable.function(*observable_args)
                self.nD_hists_per_region[channel_name][observable_name].fill(
                    observable=observable_vals,
                    process=process,
                    variation=variation,
                    weight=weights,
                )

    def process(
        self,
        events: ak.Array,
        metadata: dict[str, Any],
    ) -> None:
        """
        Run the full analysis logic on a batch of events.

        Parameters
        ----------
        events : ak.Array
            Input NanoAOD events.
        metadata : dict
            Metadata with keys 'process', 'xsec', 'nevts', and 'dataset'.

        Returns
        -------
        dict
            Histogram dictionary after processing.
        """
        analysis = self.__class__.__name__

        process = metadata["process"]
        variation = metadata.get("variation", "nominal")
        logger.debug(f"Processing {process} with variation {variation}")
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]

        lumi = self.config.general.lumi
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0

        # Nominal processing
        obj_copies = self.get_object_copies(events)
        # Filter objects
        obj_copies = self.apply_object_masks(obj_copies)

        # Apply baseline selection
        baseline_args = get_function_arguments(
            self.config.baseline_selection.use,
            obj_copies,
            function_name=self.config.baseline_selection.function.__name__,
        )

        packed_selection = self.config.baseline_selection.function(
            *baseline_args
        )
        mask = ak.Array(packed_selection.all(packed_selection.names[-1]))
        obj_copies = {
            collection: variable[mask]
            for collection, variable in obj_copies.items()
        }
        # apply ghost observables
        obj_copies = self.compute_ghost_observables(
            obj_copies,
        )

        # apply event-level corrections
        # apply nominal corrections
        obj_copies_corrected = self.apply_object_corrections(
            obj_copies, self.corrections, direction="nominal"
        )

        # apply selection and fill histograms
        self.histogramming(
            obj_copies_corrected,
            events,
            process,
            "nominal",
            xsec_weight,
            analysis,
        )

        if self.config.general.run_systematics:
            # Systematic variations
            for syst in self.systematics + self.corrections:
                if syst.name == "nominal":
                    continue
                for direction in ["up", "down"]:
                    # Filter objects
                    obj_copies = self.apply_object_masks(obj_copies)

                    # apply corrections
                    obj_copies_corrected = self.apply_object_corrections(
                        obj_copies, [syst], direction=direction
                    )
                    varname = f"{syst.name}_{direction}"
                    self.histogramming(
                        obj_copies_corrected,
                        events,
                        process,
                        varname,
                        xsec_weight,
                        analysis,
                        event_syst=syst,
                        direction=direction,
                    )

    def run_fit(
        self, cabinetry_config: dict[str, Any]
    ) -> tuple[Any, Any, Any, Any]:
        """
        Run the fit using cabinetry.

        Parameters
        ----------
        cabinetry_config : dict
            Configuration for cabinetry.
        """

        # what do we do with this
        rebinning_router = get_cabinetry_rebinning_router(
            cabinetry_config, rebinning=slice(110j, None, hist.rebin(2))
        )
        # build the templates
        cabinetry.templates.build(cabinetry_config, router=rebinning_router)
        # optional post-processing (e.g. smoothing, symmetrise)
        cabinetry.templates.postprocess(cabinetry_config)
        # build the workspace
        ws = cabinetry.workspace.build(cabinetry_config)
        # save the workspace
        workspace_path = self.output_manager.get_statistics_dir() / "workspace.json"
        cabinetry.workspace.save(ws, workspace_path)
        # build the model and data
        model, data = cabinetry.model_utils.model_and_data(ws)
        # get pre-fit predictions
        prefit_prediction = cabinetry.model_utils.prediction(model)
        # perform the fit
        results = cabinetry.fit.fit(
            model,
            data,
        )  # perform the fit
        postfit_prediction = cabinetry.model_utils.prediction(
            model, fit_results=results
        )

        return data, results, prefit_prediction, postfit_prediction

    def run_analysis_chain(self):
        """
        Run the complete non-differentiable analysis chain using pre-processed datasets.
        """
        config = self.config

        if not self.processed_datasets:
            raise ValueError("No processed datasets provided to analysis")

        # Loop over processed datasets
        for dataset_name, events_list in self.processed_datasets.items():
            logger.info("========================================")
            logger.info(f"🚀 Processing dataset: {dataset_name}")

            # Process each (events, metadata) tuple
            if config.general.run_histogramming:
                for events, metadata in events_list:
                    logger.info(f"📘 Processing events for {dataset_name}")
                    logger.info("📈 Processing for non-differentiable analysis")
                    self.process(events, metadata)
                    logger.info("📈 Non-differentiable histogram-filling complete.")

            logger.info(f"🏁 Finished dataset: {dataset_name}\n")

        # Report end of processing
        logger.info("✅ All datasets processed.")

        # Save histograms for non-differentiable analysis
        if config.general.run_histogramming:
            save_histograms_to_root(
                self.nD_hists_per_region,
                output_file=self.output_manager.get_histograms_dir() / "histograms.root",
            )
            save_histograms_to_pickle(
                self.nD_hists_per_region,
                output_file=self.output_manager.get_histograms_dir() / "histograms.pkl",
            )
        # Run statistics for non-differentiable analysis
        if config.general.run_statistics:
            cabinetry_config = cabinetry.configuration.load(
                config.statistics.cabinetry_config
            )
            data, fit_results, pre_fit_predictions, postfit_predictions = (
                self.run_fit(cabinetry_config=cabinetry_config)
            )
            cabinetry.visualize.data_mc(
                pre_fit_predictions,
                data,
                close_figure=False,
                config=cabinetry_config,
            )
            cabinetry.visualize.data_mc(
                postfit_predictions,
                data,
                close_figure=False,
                config=cabinetry_config,
            )
            cabinetry.visualize.pulls(fit_results, close_figure=False)
