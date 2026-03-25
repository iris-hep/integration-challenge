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
from intccms.utils.output import (
    OutputDirectoryManager,
    save_histograms_to_pickle,
    save_histograms_to_root,
)
from intccms.utils.stats import get_cabinetry_rebinning_router
from intccms.utils.functors import ObservableExecutor, SelectionExecutor
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
                observable_binning = observable.binning  # Already array of edges from schema
                observable_name = observable.name

                # Binning is already parsed to edges by schema validation
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
        corrections: Optional[list] = None,
        sys_values: Optional[Dict[str, str]] = None,
        year: Optional[str] = None,
    ) -> None:
        """Apply channel selections and fill histograms with combined event weights.

        All event-level corrections are applied multiplicatively to weights.
        sys_values maps each correction name to its systematic string —
        nominal for unvaried corrections, varied string for the one being varied.

        Parameters
        ----------
        object_copies : dict
            Filtered event objects (after prepare_objects).
        events : ak.Array
            Filtered NanoAOD events (same event count as object_copies).
        process : str
            Sample name.
        variation : str
            Systematic variation label for histogram axis.
        xsec_weight : float
            Cross section x luminosity / n_gen normalization.
        analysis : str
            Analysis identifier.
        is_data : bool
            Whether processing data (skips systematic weights).
        corrections : list, optional
            All corrections and systematics; event-level ones are applied as
            weights.
        sys_values : dict, optional
            Maps correction name to systematic string for the Sys() marker.
            If None or missing entry, falls back to correction's nominal_idx.
        year : str, optional
            Correction year for year-keyed configs.
        """
        if is_data and variation != "nominal":
            return

        for channel in self.channels:
            channel_name = channel.name
            if (req_channels := self.config.general.channels) is not None:
                if channel_name not in req_channels:
                    continue
            logger.debug(f"Applying selection for {channel_name} in {process} "
                         f"with variation {variation}")
            mask = 1
            if channel.selection.function is not None:
                executor = SelectionExecutor(channel.selection)
                mask = executor.execute(object_copies)

            if ak.sum(mask) == 0:
                logger.debug(
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

            # Apply all event-level corrections to weights
            if corrections and not is_data:
                for corr in corrections:
                    if corr.type != "event":
                        continue
                    sys_value = (sys_values[corr.name] if sys_values
                                 else corr.nominal_idx)
                    # Resolve variation function from uncertainty source
                    syst_function = None
                    if (not corr.use_correctionlib
                            and corr.uncertainty_sources
                            and sys_value != corr.nominal_idx):
                        for source in corr.uncertainty_sources:
                            if sys_value in source.up_and_down_idx:
                                idx = source.up_and_down_idx.index(sys_value)
                                syst_function = (source.up_function if idx == 0
                                                 else source.down_function)
                                break
                    weights = self.apply_event_weight_correction(
                        weights, corr, sys_value,
                        object_copies_channel, year,
                        syst_function=syst_function)

            logger.info(
                f"Number of weighted events in {channel_name}: {ak.sum(weights):.2f}"
            )
            logger.info(
                f"Number of raw events in {channel_name}: {ak.sum(mask)}"
            )
            for observable in channel.observables:
                observable_name = observable.name
                logger.debug(f"Computing observable {observable_name}")
                executor = ObservableExecutor(observable)
                observable_vals = executor.execute(object_copies_channel)
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

        Three-block structure:
        1. Nominal: prepare objects, fill histograms with all event weights nominal
        2. Object systematics (JEC): re-prepare objects with varied jet pT,
           fill histograms with combined weight overrides via varies_with
        3. Weight systematics: use nominal objects, vary one event weight at a time

        Parameters
        ----------
        events : ak.Array
            Input NanoAOD events.
        metadata : dict
            Metadata with keys 'process', 'xsec', 'nevts', 'dataset', and optionally 'year'.
        """
        analysis = self.__class__.__name__

        process = metadata["process"]
        variation = metadata.get("variation", "nominal")
        is_data = metadata.get("is_data", False)
        year = metadata.get("year")
        logger.debug(f"Processing {process} with variation {variation}, year={year}")
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]

        lumi = self.config.general.lumi
        xsec_weight = 1.0 if is_data else (xsec * lumi / n_gen)

        # Get year-specific corrections
        corrections = self.get_corrections_for_year(year)

        # SystematicConfig is not supported; use uncertainty_sources on
        # CorrectionConfig instead.
        systematics = self.get_systematics_for_year(year)
        if systematics:
            raise NotImplementedError(
                f"SystematicConfig is not supported. Migrate to "
                f"uncertainty_sources on CorrectionConfig: "
                f"{[s.name for s in systematics]}"
            )

        # Filter corrections by data/MC applicability
        corrections = [
            c for c in corrections
            if c.applies_to in ("both", "data" if is_data else "mc")
        ]

        # Split corrections by type
        object_corrs = [c for c in corrections if c.type == "object"]

        # Nominal sys_values: every event correction at its nominal string
        nominal_sys_values = {
            c.name: c.nominal_idx
            for c in corrections if c.type == "event"
        }

        lumi_mask_config = metadata.get("lumi_mask_config") if is_data else None

        # Block 1: Nominal
        nom_objects, nom_events = self.prepare_objects(
            events, object_corrs, is_data=is_data,
            lumi_mask_config=lumi_mask_config, year=year,
        )

        if self.config.general.run_histogramming:
            self.histogramming(
                nom_objects, nom_events, process, "nominal",
                xsec_weight, analysis,
                is_data=is_data, corrections=corrections,
                sys_values=nominal_sys_values, year=year,
            )

        if self.config.general.run_systematics and not is_data:

            # Block 2: Object systematics
            for obj_corr in object_corrs:
                if not obj_corr.uncertainty_sources:
                    continue
                for source in obj_corr.uncertainty_sources:
                    for direction in ["up", "down"]:
                        objects, filtered_events = self.prepare_objects(
                            events, object_corrs, varied_corr=obj_corr,
                            varied_source=source, direction=direction,
                            year=year,
                        )
                        # Override event corrections linked via varies_with
                        sys_values = nominal_sys_values.copy()
                        for evt_corr in corrections:
                            if evt_corr.type != "event" or not evt_corr.uncertainty_sources:
                                continue
                            for evt_source in evt_corr.uncertainty_sources:
                                if evt_source.varies_with and source.name in evt_source.varies_with:
                                    direction_idx = 0 if direction == "up" else 1
                                    sys_values[evt_corr.name] = evt_source.up_and_down_idx[direction_idx]
                        if self.config.general.run_histogramming:
                            self.histogramming(
                                objects, filtered_events, process,
                                f"{source.name}_{direction}",
                                xsec_weight, analysis,
                                is_data=is_data, corrections=corrections,
                                sys_values=sys_values, year=year,
                            )

            # Block 3: Weight-only systematics
            for corr in corrections:
                if corr.type != "event" or not corr.uncertainty_sources:
                    continue
                for source in corr.uncertainty_sources:
                    if source.varies_with:
                        continue
                    for direction_idx, direction in enumerate(["up", "down"]):
                        sys_values = nominal_sys_values.copy()
                        sys_values[corr.name] = source.up_and_down_idx[direction_idx]
                        if self.config.general.run_histogramming:
                            self.histogramming(
                                nom_objects, nom_events, process,
                                f"{source.name}_{direction}",
                                xsec_weight, analysis,
                                is_data=is_data, corrections=corrections,
                                sys_values=sys_values, year=year,
                            )

            # Block 4: Foreign-year nominal fills
            # Year-decorrelated variations from other years need this year's
            # nominal contribution to produce complete histograms.
            if self._year_keyed_corrections and self.config.general.run_histogramming:
                current_variations = self._collect_produced_variation_names(corrections)
                all_variations = self._collect_all_variation_names()
                foreign_year_variations = all_variations - current_variations

                for var_name in sorted(foreign_year_variations):
                    self.histogramming(
                        nom_objects, nom_events, process, var_name,
                        xsec_weight, analysis,
                        is_data=is_data, corrections=corrections,
                        sys_values=nominal_sys_values, year=year,
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
        workspace_path = self.output_manager.statistics_dir / "workspace.json"
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
        stats_dir = self.output_manager.statistics_dir

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
