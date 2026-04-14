from collections import defaultdict                                                                                                                                                   
from typing import Any, Dict, Optional                                                                                                                                                
                                                                                                                                                                                    
import awkward as ak                                                                                                                                                                  
import hist                                                                                                                                                                           
import logging                                                                                                                                                                        
import numpy as np                                                                                                                                                                  
import cloudpickle
import grpc

from histserv import Client
from intccms.analysis.cms import CMSAnalysis                                                                                                                                          
from intccms.utils.functors import ObservableExecutor, SelectionExecutor                                                                                                              
from intccms.utils.logging import setup_logging 

# -----------------------------
# Logging Configuration
# -----------------------------
setup_logging()
logger = logging.getLogger("HistServAnalysis")

# -----------------------------
# HistServAnalysis Class Definition
# -----------------------------
class HistServAnalysis(CMSAnalysis):
    """A CMS analysis implementation with HistServ for HaaS instead of reduce-aggregate.

    This class is designed to work with coffea processors for distributed processing.
    The processor should call process() method per-chunk.
    """

    def _init_histograms(self) -> dict[str, dict[str, hist.Hist]]:
        """
        Initialize histograms for each analysis channel based on configuration.

        Returns
        -------
        dict
            Dictionary of channel name to hist.Hist object.
        """
        histserv_client = Client(address="oksana-2eshadura-40cern-2ech.dask-worker.cmsaf-dev.flatiron.hollandhpc.org:8788")
        histograms = defaultdict(dict)
        for channel in self.channels:
            channel_name = channel.name
            if (req_channels := self.config.general.channels) is not None:
                if channel_name not in req_channels:
                    continue

            for observable in channel.observables:

                observable_label = observable.label
                observable_binning = np.asarray(observable.binning).tolist()  # Already array of edges from schema
                observable_name = observable.name

                # Binning is already parsed to edges by schema validation
                axis = hist.axis.Variable(
                    observable_binning,
                    name="observable",
                    label=observable_label,
                )

                h = hist.Hist(
                    axis,
                    hist.axis.StrCategory(["p"], name="process", growth=True),
                    hist.axis.StrCategory(["v"], name="variation", growth=True),
                    storage=hist.storage.Weight(),
                )

                client_hist = histserv_client.init(h)
                client_hist.reset()
                histograms[channel_name][observable_name] = client_hist
                
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

            logger.debug(
                f"Number of weighted events in {channel_name}: {ak.sum(weights):.2f}"
            )
            logger.debug(
                f"Number of raw events in {channel_name}: {ak.sum(mask)}"
            )
            for observable in channel.observables:
                observable_name = observable.name
                logger.debug(f"Computing observable {observable_name}")
                executor = ObservableExecutor(observable)
                observable_vals = executor.execute(object_copies_channel)
                self._fill_buffer[(channel_name, observable_name)].append({
                    "observable": np.asarray(observable_vals),
                    "process": process,
                    "variation": variation,
                    "weight": np.asarray(weights),
                })

    def _flush_fills(self):
        """Send all buffered fills to histserv in batch."""
        for (channel_name, obs_name), fill_kwargs in self._fill_buffer.items():
            try:
                self.nD_hists_per_region[channel_name][obs_name].fill_many(fill_kwargs)
            except grpc.RpcError as exc:
                raise exc
        self._fill_buffer.clear()

    def process(self, events, metadata):
        self._fill_buffer = defaultdict(list)
        super().process(events, metadata)
        self._flush_fills()
