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

from intccms.analysis.base import Analysis
from intccms.utils.output_files import (
    save_histograms_to_pickle,
    save_histograms_to_root,
)
from intccms.utils.output_manager import OutputDirectoryManager
from intccms.utils.stats import get_cabinetry_rebinning_router
from intccms.utils.tools import get_function_arguments
from intccms.utils.logging import setup_logging

# -----------------------------
# Register backends
# -----------------------------
vector.register_awkward()

# -----------------------------
# Logging Configuration
# -----------------------------
setup_logging()
logger = logging.getLogger("NonDiffAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# -----------------------------
# ZprimeAnalysis Class Definition
# -----------------------------
class NonDiffAnalysis(Analysis):
    """Non-differentiable analysis implementation.

    This class is designed to work with UnifiedProcessor for distributed processing.
    The processor calls process() method per-chunk, and histograms accumulate
    via coffea's merge mechanism.
    """

    def __init__(self, config: dict[str, Any], output_manager: OutputDirectoryManager) -> None:
        """
        Initialize NonDiffAnalysis with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'systematics', 'corrections', 'channels',
            and 'general'.
        output_manager : OutputDirectoryManager
            Centralized output directory manager (required)
        """
        super().__init__(config, output_manager)
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
        is_data: bool = False,
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
        if is_data and variation != "nominal":
            return

        for channel in self.channels:
            channel_name = channel.name
            if (req_channels := self.config.general.channels) is not None:
                if channel_name not in req_channels:
                    continue
            logger.info(f"Applying selection for {channel_name} in {process} "
                         f"with variation {variation}")
            mask = 1
            if (selection_funciton := channel.selection.function) is not None:
                selection_args, selection_static_kwargs = get_function_arguments(
                    channel.selection.use,
                    object_copies,
                    function_name=channel.selection.function.__name__,
                    static_kwargs=channel.selection.get("static_kwargs"),
                )
                packed_selection = selection_funciton(
                    *selection_args, **selection_static_kwargs
                )
                if not isinstance(packed_selection, PackedSelection):
                    raise ValueError(
                        f"PackedSelection expected, got {type(packed_selection)}"
                    )
                mask = ak.Array(
                    packed_selection.all(packed_selection.names[-1])
                )

            if ak.sum(mask) == 0:
                logger.warning(
                    f"{analysis}:: No events left in {channel_name} for {process} with "
                    + f"variation {variation}"
                )
                continue

            object_copies_channel = {
                collection: variable[mask]
                for collection, variable in object_copies.items()
            }

            if not is_data:
                weights = (
                    events[mask][self.config.general.weight_branch]
                    * xsec_weight
                    / abs(events[mask][self.config.general.weight_branch])
                )
            else:
                weights = np.ones(ak.sum(mask))

            if event_syst and not is_data:
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
                observable_args, observable_static_kwargs = get_function_arguments(
                    observable.use,
                    object_copies_channel,
                    function_name=observable.function.__name__,
                    static_kwargs=observable.get("static_kwargs"),
                )
                observable_vals = observable.function(
                    *observable_args, **observable_static_kwargs
                )
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
        is_data = metadata.get("is_data", False)
        logger.debug(f"Processing {process} with variation {variation}")
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]

        lumi = self.config.general.lumi
        xsec_weight = 1.0 if is_data else (xsec * lumi / n_gen)

        # Nominal processing
        obj_copies = self.get_object_copies(events)
        # Filter objects
        obj_copies = self.apply_object_masks(obj_copies)

        # Apply baseline selection
        baseline_args, baseline_static_kwargs = get_function_arguments(
            self.config.baseline_selection.use,
            obj_copies,
            function_name=self.config.baseline_selection.function.__name__,
            static_kwargs=self.config.baseline_selection.get("static_kwargs"),
        )

        packed_selection = self.config.baseline_selection.function(
            *baseline_args, **baseline_static_kwargs
        )
        mask = ak.Array(packed_selection.all(packed_selection.names[-1]))
        obj_copies = {
            collection: variable[mask]
            for collection, variable in obj_copies.items()
        }

        # Apply lumi_mask for data if configured
        if is_data and (lumi_mask_config := metadata.get("lumi_mask_config")):
            lumi_args, lumi_static_kwargs = get_function_arguments(
                lumi_mask_config.use,
                obj_copies,
                function_name=lumi_mask_config.function.__name__,
                static_kwargs=lumi_mask_config.get("static_kwargs"),
            )
            lumi_mask_result = lumi_mask_config.function(
                *lumi_args, **lumi_static_kwargs
            )
            obj_copies = {
                collection: variable[lumi_mask_result]
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
            is_data=is_data,
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
                        is_data=is_data,
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

    def run_statistics(self, cabinetry_config_path: str) -> None:
        """
        Run statistical analysis and create visualizations.

        This method should be called after histograms have been filled
        (either via processor workflow or directly). It runs the cabinetry
        fit and creates visualization plots.

        Parameters
        ----------
        cabinetry_config_path : str
            Path to cabinetry configuration file

        Examples
        --------
        >>> # After processor completes and histograms are filled
        >>> analysis = NonDiffAnalysis(config, [], output_manager)
        >>> analysis.nD_hists_per_region = output["histograms"]  # From processor
        >>> analysis.run_statistics(config.statistics.cabinetry_config)
        """
        logger.info("Running statistical analysis and visualization")

        # Load cabinetry configuration
        cabinetry_config = cabinetry.configuration.load(cabinetry_config_path)

        # Run the fit
        data, fit_results, prefit_prediction, postfit_prediction = (
            self.run_fit(cabinetry_config=cabinetry_config)
        )

        # Create visualizations
        stats_dir = self.output_manager.get_statistics_dir()

        logger.info("Creating pre-fit data/MC plots")
        cabinetry.visualize.data_mc(
            prefit_prediction,
            data,
            close_figure=False,
            config=cabinetry_config,
            figure_folder=stats_dir,
        )

        logger.info("Creating post-fit data/MC plots")
        cabinetry.visualize.data_mc(
            postfit_prediction,
            data,
            close_figure=False,
            config=cabinetry_config,
            figure_folder=stats_dir,
        )

        logger.info("Creating pull plots")
        cabinetry.visualize.pulls(
            fit_results,
            close_figure=False,
            figure_folder=stats_dir,
        )

        logger.info(f"✅ Statistical analysis complete. Plots saved to {stats_dir}")
