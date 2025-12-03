"""
Plotting utilities for physics analysis with CMS-style formatting.

This module provides plotting functions for particle physics analysis,
including histogram plotting, parameter evolution tracking, and MVA score visualization.
Functions support JAX arrays and include error handling and logging.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from intccms.utils.logging import setup_logging

# Configure matplotlib global settings
rcParams.update(
    {
        "axes.formatter.use_mathtext": True,
        "text.usetex": True,
        "font.family": "serif",
        "axes.linewidth": 1.2,
    }
)

# Type alias for array-like objects (supports JAX, NumPy, etc.)
ArrayLike = Union[np.ndarray, Any]

# Set up module-specific logger with setup_logging function
setup_logging("INFO")
logger = logging.getLogger(__name__)


def format_scientific_latex(value: float, significant_digits: int = 2) -> str:
    """
    Convert a floating-point number to LaTeX scientific notation.

    Parameters
    ----------
    value : float
        Numeric value to format
    significant_digits : int, optional
        Number of significant digits to retain, by default 2

    Returns
    -------
    str
        LaTeX-formatted string in scientific notation

    Examples
    --------
    >>> format_scientific_latex(2.487e-5)
    '2.487\\times10^{-5}'

    Raises
    ------
    ValueError
        If significant_digits is not positive
    TypeError
        If value is not a number
    """
    if not isinstance(value, (int, float, np.number, np.ndarray)):
        raise TypeError(f"Value must be a number, got {type(value)}")
    if significant_digits <= 0:
        raise ValueError("significant_digits must be positive")
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(
                "Value must be a scalar, got shape {}".format(value.shape)
            )
        value = value.item()

    try:
        formatted = f"{value:.{significant_digits}e}"
        mantissa, exponent_part = formatted.split("e")
        exponent = int(exponent_part)
        return rf"{mantissa}\times10^{{{exponent}}}"
    except (ValueError, OverflowError) as e:
        logger.error(f"Failed to format {value} in scientific notation: {e}")
        return str(value)


def convert_to_numpy(array_like: ArrayLike) -> np.ndarray:
    """
    Convert JAX arrays or tracers to host-memory NumPy arrays.

    Safe to call outside any JAX-transformed function. Handles device
    transfer and conversion to standard NumPy arrays.

    Parameters
    ----------
    array_like : ArrayLike
        Input array (JAX array, tracer, or NumPy array)

    Returns
    -------
    np.ndarray
        Converted NumPy array on host memory

    Raises
    ------
    ValueError
        If array_like cannot be converted to numpy array
    """
    try:
        device_array = jax.device_get(array_like)
        return np.asarray(device_array)
    except Exception as e:
        logger.error(f"Failed to convert array to numpy: {e}")
        raise ValueError(f"Cannot convert input to numpy array: {e}")


def create_cms_histogram(
    bin_edges: ArrayLike,
    data: ArrayLike,
    templates: Dict[str, ArrayLike],
    fitted_params: Optional[Dict[str, float]] = None,
    plot_settings: Optional[Dict[str, Any]] = None,
    show_signal: bool = True,
    figsize: Tuple[float, float] = (12, 12),
    ratio_ylim: Tuple[float, float] = (0.5, 1.5),
    xlabel: str = "",
    ylabel: str = "Events",
    title: str = "",
) -> plt.Figure:
    """
    Create a CMS-style pre-fit/post-fit histogram plot with ratio panel.

    Parameters
    ----------
    bin_edges : ArrayLike
        Bin edges for histogram
    data : ArrayLike
        Observed data values
    templates : Dict[str, ArrayLike]
        Dictionary of template histograms (process name → bin counts)
    fitted_params : Dict[str, float], optional
        Fitted parameters for scaling templates, by default None
    plot_settings : Dict[str, Any], optional
        Customization settings dictionary containing:
        - process_order: Ordered process names
        - process_colors: Color mapping for processes
        - process_labels: Label mapping for processes
        - jax.fit_param_labels: LaTeX labels for parameters
        by default None
    show_signal : bool, optional
        Whether to display signal process, by default True
    figsize : Tuple[float, float], optional
        Figure dimensions, by default (12, 12)
    ratio_ylim : Tuple[float, float], optional
        Y-axis limits for ratio panel, by default (0.5, 1.5)
    xlabel : str, optional
        X-axis label, by default ""
    ylabel : str, optional
        Y-axis label for main panel, by default "Events"
    title : str, optional
        Figure title, by default ""

    Returns
    -------
    plt.Figure
        Generated matplotlib figure

    Raises
    ------
    ValueError
        If templates dictionary is empty or data/bin_edges have incompatible shapes
    TypeError
        If required parameters are not of expected types

    Notes
    -----
    - Uses CMS plotting style via mplhep
    - Creates two panels: main histogram and data/prediction ratio
    - Includes parameter value annotations when fitted_params are provided
    """
    # Input validation
    if not templates:
        raise ValueError("Templates dictionary cannot be empty")

    try:
        # Apply CMS style and convert inputs
        hep.style.use("CMS")
        data_array = convert_to_numpy(data)
        edges_array = convert_to_numpy(bin_edges)

        # Validate array shapes
        if len(data_array) != len(edges_array) - 1:
            raise ValueError(
                f"Data length ({len(data_array)}) must match "
                f"number of bins ({len(edges_array) - 1})"
            )

        bin_centers = 0.5 * (edges_array[:-1] + edges_array[1:])

        # Determine process drawing order
        config = plot_settings or {}
        process_names = list(templates.keys())
        process_order = config.get("process_order")

        if process_order:
            process_order = [
                p
                for p in process_order
                if p in templates and (show_signal or p.lower() != "signal")
            ]
        else:
            background_processes = sorted(
                p for p in process_names if p.lower() != "signal"
            )
            signal_present = "signal" in templates and show_signal
            process_order = background_processes + (
                ["signal"] if signal_present else []
            )

        logger.debug(f"Process order: {process_order}")

        # Scale templates according to fitted parameters
        signal_scale = fitted_params.get("mu", 1.0) if fitted_params else 1.0
        ttbar_scale = (
            fitted_params.get("norm_ttbar_semilep", 1.0)
            if fitted_params
            else 1.0
        )

        scaled_templates = {}
        for process, values in templates.items():
            scale_factor = 1.0
            if process.lower() == "signal":
                scale_factor = signal_scale
            elif process == "ttbar_semilep":
                scale_factor = ttbar_scale
            scaled_templates[process] = convert_to_numpy(values) * scale_factor

        # Create figure with two panels (main + ratio)
        figure = plt.figure(figsize=figsize)
        grid_spec = GridSpec(
            2, 1, height_ratios=(3, 1), hspace=0.1, figure=figure
        )
        main_axis = figure.add_subplot(grid_spec[0])
        ratio_axis = figure.add_subplot(grid_spec[1], sharex=main_axis)

        # Plot stacked backgrounds and data points
        color_map = config.get("process_colors", {})
        label_map = config.get("process_labels", {})

        hep.histplot(
            [scaled_templates[p] for p in process_order],
            edges_array,
            stack=True,
            ax=main_axis,
            label=[label_map.get(p, p) for p in process_order],
            color=[color_map.get(p) for p in process_order],
            edgecolor="k",
            histtype="fill",
            linewidth=0.5,
        )

        hep.histplot(
            data_array,
            edges_array,
            yerr=np.sqrt(data_array),
            ax=main_axis,
            marker="o",
            color="k",
            label="Data",
            markersize=5,
            capsize=2,
            histtype="errorbar",
        )

        main_axis.set_ylabel(ylabel, fontsize=20)
        main_axis.set_title(title, fontsize=18)
        main_axis.legend(frameon=False, fontsize=16, ncol=2, loc="upper right")
        plt.setp(main_axis.get_xticklabels(), visible=False)

        # Add parameter annotation box if fitted parameters are provided
        if fitted_params:
            param_labels = config.get("jax", {}).get("fit_param_labels", {})
            annotation_lines = []
            for param, value in fitted_params.items():
                latex_label = param_labels.get(param, param)
                if not (
                    latex_label.startswith("$") and latex_label.endswith("$")
                ):
                    latex_label = f"${latex_label}$"
                annotation_lines.append(f"{latex_label} = {value:.3f}")

            main_axis.text(
                0.05,
                0.94,
                "\n".join(annotation_lines),
                transform=main_axis.transAxes,
                va="top",
                ha="left",
                fontsize=18,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "black",
                    "boxstyle": "round,pad=0.5",
                    "alpha": 0.8,
                },
            )

        # Compute and plot ratio panel
        total_prediction = np.sum(
            [scaled_templates[p] for p in process_order], axis=0
        )
        ratio_values = np.divide(
            data_array,
            total_prediction,
            out=np.ones_like(data_array),
            where=total_prediction > 0,
        )
        ratio_errors = np.divide(
            np.sqrt(data_array),
            total_prediction,
            out=np.zeros_like(data_array),
            where=total_prediction > 0,
        )

        ratio_axis.errorbar(
            bin_centers,
            ratio_values,
            yerr=ratio_errors,
            fmt="o",
            color="k",
            capsize=2,
        )
        ratio_axis.axhline(1, color="r", linestyle="--")
        ratio_axis.set_ylim(ratio_ylim)
        ratio_axis.set_xlabel(xlabel, fontsize=14)
        ratio_axis.set_ylabel(
            "Data/Pred.", fontsize=14, ha="center", labelpad=15
        )

        return figure

    except Exception as e:
        logger.error(f"Failed to create CMS histogram: {e}")
        raise


def plot_pvalue_vs_parameters(
    pvalue_history: Sequence[float],
    auxiliary_history: Dict[str, Sequence[float]],
    mle_history: Dict[str, Sequence[float]],
    gradients: Dict[str, float],
    learning_rates: Dict[str, float],
    filename: str = "pvalue_vs_parameters.png",
    plot_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Plot p-values versus parameter values during optimization.

    Creates a grid of subplots showing the relationship between parameter values
    and p-values. Each subplot shows the trajectory of one parameter versus p-value.

    Parameters
    ----------
    pvalue_history : Sequence[float]
        History of p-values during optimization
    auxiliary_history : Dict[str, Sequence[float]]
        History of auxiliary parameters
    mle_history : Dict[str, Sequence[float]]
        History of maximum likelihood estimation parameters
    gradients : Dict[str, float]
        Gradient values for parameters
    learning_rates : Dict[str, float]
        Learning rates used during optimization
    filename : str, optional
        Output filename, by default "pvalue_vs_parameters.png"
    plot_settings : Dict[str, Any], optional
        Dictionary containing:
        - jax.aux_param_labels: LaTeX labels for auxiliary parameters
        - jax.fit_param_labels: LaTeX labels for MLE parameters
        by default None

    Returns
    -------
    None
        Saves plot to specified file

    Raises
    ------
    ValueError
        If no varying parameters are found or input data is invalid
    IOError
        If file cannot be saved
    """
    logger.info(f"Creating p-value vs parameters plot: {filename}")

    try:
        # Combine parameter histories
        parameter_history = {}
        for name, history in auxiliary_history.items():
            if "__NN" not in name:  # Exclude neural network parameters
                parameter_history[f"aux__{name}"] = history
        for name, history in mle_history.items():
            parameter_history[f"mle__{name}"] = history

        # Filter out constant parameters
        non_constant_params = {}
        for name, history in parameter_history.items():
            arr = np.asarray(history)
            if not np.allclose(arr, arr[0]):
                non_constant_params[name] = history

        if not non_constant_params:
            logger.warning("No varying parameters found - skipping plot")
            return

        logger.debug(f"Found {len(non_constant_params)} varying parameters")

        # Create subplot grid
        num_params = len(non_constant_params)
        grid_size = math.ceil(math.sqrt(num_params))
        fig, axes = plt.subplots(
            grid_size,
            grid_size,
            figsize=(4 * grid_size, 3 * grid_size),
            sharey=True,
            squeeze=False,
        )
        axes_flat = axes.flatten()

        # Get labeling information
        config = plot_settings or {}
        jax_config = config.get("jax", {})
        aux_labels = jax_config.get("aux_param_labels", {})
        fit_labels = jax_config.get("fit_param_labels", {})
        param_labels = {**aux_labels, **fit_labels}

        # Plot each parameter's history against p-values
        for idx, (ax, (param_name, history)) in enumerate(
            zip(axes_flat, non_constant_params.items())
        ):
            # Extract base parameter name
            base_name = param_name.split("__", 1)[-1]

            # Create title string with gradient and learning rate
            title_parts = []
            if base_name in gradients.get("aux", {}):
                grad_val = gradients["aux"][base_name]
                grad_val = convert_to_numpy(grad_val)
                title_parts.append(
                    r"$\Delta_{\theta}(p_s) = "
                    f"{format_scientific_latex(grad_val)}$"
                )
            if base_name in learning_rates:
                lr_val = learning_rates[base_name]
                title_parts.append(
                    r"$\eta = " f"{format_scientific_latex(lr_val)}$"
                )
            title_text = ", ".join(title_parts) if title_parts else ""

            # Plot trajectory
            ax.plot(history, pvalue_history, "o-", ms=5)
            ax.set_title(title_text, fontsize=10)

            # Configure axes
            display_name = param_labels.get(base_name, base_name)
            ax.set_xlabel(f"{display_name} Value", fontsize=10)
            if idx % grid_size == 0:
                ax.set_ylabel(r"$p$-value", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Configure axes formatting
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        for ax in axes_flat[:num_params]:
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(axis="both", labelsize=10)
            ax.yaxis.offsetText.set_fontsize(10)

        # Hide unused subplots
        for ax in axes_flat[num_params:]:
            ax.set_visible(False)

        plt.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

        logger.info(
            f"Successfully saved p-value vs parameters plot to {filename}"
        )

    except Exception as e:
        logger.error(f"Failed to create p-value vs parameters plot: {e}")
        raise


def plot_parameters_over_iterations(
    pvalue_history: Sequence[float],
    auxiliary_history: Dict[str, Sequence[float]],
    mle_history: Dict[str, Sequence[float]],
    gradients: Dict[str, float],
    learning_rates: Dict[str, float],
    filename: str = "parameters_vs_iterations.png",
    plot_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Plot parameter values and p-value over optimization iterations.

    Creates a grid of subplots showing the evolution of parameters and p-value
    as a function of iteration number.

    Parameters
    ----------
    pvalue_history : Sequence[float]
        History of p-values during optimization
    auxiliary_history : Dict[str, Sequence[float]]
        History of auxiliary parameters
    mle_history : Dict[str, Sequence[float]]
        History of maximum likelihood estimation parameters
    gradients : Dict[str, float]
        Gradient values for parameters
    learning_rates : Dict[str, float]
        Learning rates used during optimization
    filename : str, optional
        Output filename, by default "parameters_vs_iterations.png"
    plot_settings : Dict[str, Any], optional
        Dictionary containing:
        - jax.aux_param_labels: LaTeX labels for auxiliary parameters
        - jax.fit_param_labels: LaTeX labels for MLE parameters
        by default None

    Returns
    -------
    None
        Saves plot to specified file

    Raises
    ------
    ValueError
        If no varying parameters are found or input data is invalid
    IOError
        If file cannot be saved
    """
    logger.info(f"Creating parameters vs iterations plot: {filename}")

    try:
        # Combine parameter histories
        parameter_history = {}
        for name, history in auxiliary_history.items():
            if "__NN" not in name:  # Exclude neural network parameters
                parameter_history[f"aux__{name}"] = history
        for name, history in mle_history.items():
            parameter_history[f"mle__{name}"] = history

        # Filter out constant parameters
        non_constant_params = {}
        for name, history in parameter_history.items():
            arr = np.asarray(history)
            if not np.allclose(arr, arr[0]):
                non_constant_params[name] = history

        # Add p-value to parameters
        non_constant_params["pvalue"] = np.asarray(pvalue_history)
        if not non_constant_params:
            logger.warning("No varying parameters found - skipping plot")
            return

        logger.debug(
            f"Found {len(non_constant_params)} varying parameters (including p-value)"
        )

        # Create subplot grid
        num_params = len(non_constant_params)
        num_iterations = len(pvalue_history)
        iteration_steps = np.arange(num_iterations)
        grid_size = math.ceil(math.sqrt(num_params))
        fig, axes = plt.subplots(
            grid_size,
            grid_size,
            figsize=(4 * grid_size, 3 * grid_size),
            sharex=True,
            squeeze=False,
        )
        axes_flat = axes.flatten()

        # Get labeling information
        config = plot_settings or {}
        jax_config = config.get("jax", {})
        aux_labels = jax_config.get("aux_param_labels", {})
        fit_labels = jax_config.get("fit_param_labels", {})
        param_labels = {**aux_labels, **fit_labels}

        # Plot each parameter's evolution
        for idx, (ax, (param_name, history)) in enumerate(
            zip(axes_flat, non_constant_params.items())
        ):
            # Extract base parameter name
            base_name = (
                param_name.split("__", 1)[-1]
                if "__" in param_name
                else param_name
            )

            # Create title string with gradient and learning rate
            title_parts = []
            if base_name in gradients.get("aux", {}):
                grad_val = gradients["aux"][base_name]
                grad_val = convert_to_numpy(grad_val)
                title_parts.append(
                    r"$\Delta_{\theta}(p_s) = "
                    f"{format_scientific_latex(grad_val)}$"
                )
            if base_name in learning_rates:
                lr_val = learning_rates[base_name]
                title_parts.append(
                    r"$\eta = " f"{format_scientific_latex(lr_val)}$"
                )
            title_text = ", ".join(title_parts) if title_parts else ""

            # Plot trajectory
            ax.plot(iteration_steps, history, "o-", ms=3)
            ax.set_title(title_text, fontsize=10)

            # Configure axes
            if param_name == "pvalue":
                display_name = r"$p$-value"
            else:
                display_name = param_labels.get(base_name, base_name)
            ax.set_ylabel(display_name, fontsize=10)
            ax.set_xlabel("Iteration", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Configure axes formatting
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        for ax in axes_flat:
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(axis="both", labelsize=10)

        # Hide unused subplots
        for ax in axes_flat[num_params:]:
            ax.set_visible(False)

        plt.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

        logger.info(
            f"Successfully saved parameters vs iterations plot to {filename}"
        )

    except Exception as e:
        logger.error(f"Failed to create parameters vs iterations plot: {e}")
        raise


def _setup_process_ordering(
    data_dict: Dict[str, Any], plot_config: Dict[str, Any]
) -> List[str]:
    """
    Set up consistent process ordering for plotting.

    Parameters
    ----------
    data_dict : Dict[str, Any]
        Dictionary containing process data
    plot_config : Dict[str, Any]
        Plotting configuration with process_order

    Returns
    -------
    List[str]
        Ordered list of process names

    Raises
    ------
    ValueError
        If data_dict is empty
    """
    if not data_dict:
        raise ValueError("Data dictionary cannot be empty")

    process_order = plot_config.get("process_order", list(data_dict.keys()))

    # Build ordered list: first those in process_order then others
    # in original insertion order
    ordered_processes = [p for p in process_order if p in data_dict]
    ordered_processes += [
        p for p in data_dict.keys() if p not in process_order
    ]

    logger.debug(f"Process ordering: {ordered_processes}")
    return ordered_processes


def _setup_plot_style_and_figure(
    figsize: Tuple[float, float] = (8, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Set up consistent plot style and create figure.

    Parameters
    ----------
    figsize : Tuple[float, float], optional
        Figure size, by default (8, 6)

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Created figure and axes objects

    Raises
    ------
    ValueError
        If figsize contains non-positive values
    """
    if figsize[0] <= 0 or figsize[1] <= 0:
        raise ValueError("Figure size must contain positive values")

    try:
        hep.style.use("CMS")
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    except Exception as e:
        logger.error(f"Failed to create figure: {e}")
        raise


def _save_plot_with_logging(
    fig: plt.Figure, plot_filename: Path, plot_description: str, dpi: int = 300
) -> None:
    """
    Save plot with consistent logging and cleanup.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    plot_filename : Path
        Path to save the plot
    plot_description : str
        Description for logging
    dpi : int, optional
        DPI for saving, by default 300

    Raises
    ------
    IOError
        If file cannot be saved
    """
    try:
        # Ensure parent directory exists
        plot_filename.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(plot_filename, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {plot_description} to {plot_filename}")

    except Exception as e:
        logger.error(f"Failed to save plot {plot_filename}: {e}")
        plt.close(fig)  # Ensure figure is closed even on error
        raise IOError(f"Cannot save plot to {plot_filename}: {e}")


def plot_mva_feature_distributions(
    feature_data: Dict[str, Dict[str, ArrayLike]],
    mva_config: Dict[str, Any],
    plot_config: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Plot distributions of MVA input features for different processes.

    Parameters
    ----------
    feature_data : Dict[str, Dict[str, ArrayLike]]
        Nested dictionary mapping process name to feature name to data array.
        Example: {"ttbar": {"n_jet": [1,2,3]}, "wjets": {"n_jet": [4,5,6]}}
    mva_config : Dict[str, Any]
        Configuration for the MVA model, containing feature definitions.
    plot_config : Dict[str, Any]
        Plotting configuration with colors, labels, and process order.
    output_dir : str
        Directory to save the plots.
    """
    logger.info(f"Creating MVA feature distribution plots in {output_dir}")

    output_path = Path(output_dir) / "features"
    output_path.mkdir(parents=True, exist_ok=True)

    process_colors = plot_config.get("process_colors", {})
    process_labels = plot_config.get("process_labels", {})
    ordered_processes = _setup_process_ordering(feature_data, plot_config)

    for feature in mva_config.get("features", []):
        feature_name = feature["name"]
        feature_label = feature.get("label", feature_name)
        feature_binning = feature.get("binning", None)

        # Binning already parsed to edges by schema validation
        prelim_bins = feature_binning  # Already np.ndarray or None

        for scaling_version in ["scaled", "unscaled"]:
            fig, ax = _setup_plot_style_and_figure()

            bins = prelim_bins
            if prelim_bins is None:
                # Determine binning from data if not specified
                all_values = np.concatenate(
                    [
                        data[feature_name][scaling_version]
                        for proc, data in feature_data.items()
                        if feature_name in data
                    ]
                )
                bins = np.linspace(np.min(all_values), np.max(all_values), 50)
            else:
                if scaling_version == "scaled":
                    scaling = getattr(feature, "scale", None)
                    if scaling is not None:
                        # Scale feature data if specified in MVA config
                        bins = scaling(bins)
                    else:
                        bins = prelim_bins
                else:
                    bins = prelim_bins

            for process_name in ordered_processes:
                logger.debug(
                    f"Plotting {scaling_version} feature data "
                    f"'{feature_name}' for process '{process_name}'"
                )
                if (
                    process_name not in feature_data
                    or feature_name not in feature_data[process_name]
                ):
                    continue

                values = convert_to_numpy(
                    feature_data[process_name][feature_name][scaling_version]
                )

                ax.hist(
                    values,
                    bins=bins,
                    color=process_colors.get(process_name, "gray"),
                    label=process_labels.get(process_name, process_name),
                    alpha=0.7,
                    density=True,
                    histtype="stepfilled",
                    linewidth=1.5,
                )

            ax.set_xlabel(feature_label, fontsize=14)
            ax.set_ylabel("a.u.", fontsize=14)
            ax.legend(frameon=False, fontsize=12)
            ax.tick_params(axis="both", labelsize=12)
            fig.tight_layout()

            plot_filename = (
                output_path
                / f"{mva_config['name']}_feats_{feature_name}_{scaling_version}.pdf"
            )
            _save_plot_with_logging(
                fig,
                plot_filename,
                f"MVA feature plot ({feature_name}, {scaling_version})",
            )


def plot_mva_scores(
    scores: Dict[str, ArrayLike],
    plot_config: Dict[str, Any],
    output_dir: str,
    prefix: str = "",
    bins: int = 50,
    score_range: Tuple[float, float] = (0, 1),
) -> None:
    """
    Plot MVA scores for different processes.

    Parameters
    ----------
    scores : Dict[str, ArrayLike]
        Dictionary mapping process name to an array of MVA scores.
    plot_config : Dict[str, Any]
        Plotting configuration with colors, labels, and process order.
    output_dir : str
        Directory to save the plot.
    prefix : str, optional
        Prefix for output filename, by default ""
    bins : int, optional
        Number of bins for the histogram, by default 50.
    score_range : Tuple[float, float], optional
        The range of scores to plot, by default (0, 1).

    Raises
    ------
    ValueError
        If scores dictionary is empty or bins is not positive
    """
    if not scores:
        raise ValueError("Scores dictionary cannot be empty")
    if bins <= 0:
        raise ValueError("Number of bins must be positive")

    logger.info(
        f"Creating MVA scores plot with prefix '{prefix}' in {output_dir}"
    )

    try:
        output_path = Path(output_dir) / "scores"
        output_path.mkdir(parents=True, exist_ok=True)

        process_colors = plot_config.get("process_colors", {})
        process_labels = plot_config.get("process_labels", {})
        ordered_processes = _setup_process_ordering(scores, plot_config)

        fig, ax = _setup_plot_style_and_figure()

        bin_edges = np.linspace(score_range[0], score_range[1], bins + 1)
        max_height = 0

        for process_name in ordered_processes:
            logger.debug(f"Plotting MVA scores for process '{process_name}'")
            if process_name not in scores:
                continue

            process_scores = convert_to_numpy(scores[process_name])
            if len(process_scores) == 0:
                logger.warning(f"No scores found for process '{process_name}'")
                continue

            counts, _ = np.histogram(
                process_scores, bins=bin_edges, density=True
            )
            if len(counts) > 0:
                max_height = max(max_height, counts.max())

            ax.hist(
                process_scores,
                bins=bin_edges,
                color=process_colors.get(process_name, "gray"),
                label=process_labels.get(process_name, process_name),
                alpha=0.7,
                density=True,
                histtype="stepfilled",
                linewidth=1.5,
            )

        ax.set_xlabel("MVA Score", fontsize=14)
        ax.set_ylabel("a.u.", fontsize=14)
        ax.set_ylim(0, max_height * 1.15)
        ax.legend(frameon=False, fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        fig.tight_layout()

        plot_filename = output_path / f"{prefix}mva_score.pdf"
        _save_plot_with_logging(
            fig, plot_filename, f"MVA score plot [{prefix}]"
        )

    except Exception as e:
        logger.error(f"Failed to create MVA scores plot: {e}")
        raise
