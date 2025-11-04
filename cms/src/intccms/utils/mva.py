# =============================================================================
# Imports
# =============================================================================

# Standard library imports
import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad
from itertools import chain
from sklearn.model_selection import train_test_split

# Local application imports
from intccms.utils.functors import FeatureExecutor

# Initialize logger
logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Class
# =============================================================================


class BaseNetwork(ABC):
    """Abstract base class for MVA networks (e.g., JAX, TensorFlow).

    Handles common functionality including:
    - Data preparation and feature extraction
    - Class balancing strategies
    - Train/test splitting
    - Label generation

    Subclasses must implement network initialization, training, and prediction.

    Attributes
    ----------
    mva_cfg : Dict[str, Any]
        Configuration dictionary for the network.
    name : str
        Name of the network.
    parameters : Dict[str, Any]
        Network parameters (used by JAX models).
    model : Any
        Model object (used by TensorFlow models).
    process_to_features_map : Dict[str, Dict[str, np.ndarray]]
        Mapping from process names to scaled and unscaled feature values.

    Methods
    -------
    _split_train_test(features, labels)
        Split data into training and validation sets.
    _extract_features(object_collections, feature_configs)
        Compute and stack features from input data.
    _make_labels(num_events, process_name, class_definitions)
        Generate integer class labels for events.
    _balance_dataset(features, labels)
        Balance dataset using specified strategy.
    prepare_inputs(events_per_process)
        Prepare full training/validation dataset.
    init_network()
        Initialize network architecture (abstract).
    train(training_inputs, training_labels, validation_inputs, validation_labels)
        Train the model (abstract).
    predict(inputs, **kwargs)
        Perform inference (abstract).
    """

    def __init__(self, mva_cfg: Dict[str, Any]) -> None:
        """Initialize the network with configuration.

        Parameters
        ----------
        mva_cfg : Dict[str, Any]
            Configuration dictionary specifying network architecture and
            training options.
        """
        self.mva_cfg = mva_cfg
        self.name = mva_cfg.name
        self.parameters: Dict[str, Any] = {}  # For JAX models
        # process -> feature name -> {'scaled': np.ndarray, 'unscaled': np.ndarray}
        self.process_to_features_map: Dict[
            str, Dict[str, Dict[str, np.ndarray]]
        ] = {}
        # process -> MVA scores array
        self.process_to_scores_map: Dict[str, np.ndarray] = {}
        self.model: Any = None  # For TensorFlow models

    def _split_train_test(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split features and labels into training and validation sets.

        Parameters
        ----------
        features : np.ndarray
            Full feature matrix of shape (n_samples, n_features).
        labels : np.ndarray
            Corresponding class labels of shape (n_samples,).

        Returns
        -------
        X_train : np.ndarray
            Training features.
        X_val : np.ndarray
            Validation features.
        y_train : np.ndarray
            Training labels.
        y_val : np.ndarray
            Validation labels.
        """
        test_size = self.mva_cfg.validation_split
        random_state = self.mva_cfg.random_state

        # Stratify if balancing is active
        stratify = labels if self.mva_cfg.balance_strategy != "none" else None

        return train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )

    def _extract_features(
        self,
        object_collections: Dict[str, Any],
        feature_configs: List[Any],
    ) -> tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
        """Compute and stack all features for one event batch.

        Parameters
        ----------
        object_collections : Dict[str, Any]
            Dictionary of NanoAOD-style objects per event.
        feature_configs : List[Any]
            List of feature configuration objects, each defining a callable
            and its required input objects.

        Returns
        -------
        np.ndarray
            2D array of shape (n_events, n_features) with stacked features.
        Dict[str, Dict[str, np.ndarray]]
            Dictionary mapping feature names to their scaled and unscaled values.
        """
        feature_arrays = []
        features_dict = {}
        for feature_cfg in feature_configs:
            executor = FeatureExecutor(feature_cfg)

            # Compute scaled version (for training/inference)
            feature_values_scaled = executor.execute(object_collections, apply_scaling=True)
            feature_arrays.append(feature_values_scaled)

            # Compute unscaled version (for storage/analysis)
            feature_values_unscaled = executor.execute(object_collections, apply_scaling=False)

            features_dict[feature_cfg.name] = {
                "scaled": feature_values_scaled,
                "unscaled": feature_values_unscaled,
            }

        # Stack all features as columns
        return np.stack(feature_arrays, axis=1).astype(float), features_dict

    def _make_labels(
        self,
        num_events: int,
        process_name: str,
        class_definitions: List[Union[str, Dict[str, Tuple[str]]]],
    ) -> np.ndarray:
        """Generate integer class labels for events belonging to a process.

        Parameters
        ----------
        num_events : int
            Number of events to label.
        process_name : str
            Name of the process (e.g., 'ttbar').
        class_definitions : list
            List of class definitions, each being either:
            - String: Single process name
            - Dict: {class_name: (process1, process2, ...)}

        Returns
        -------
        np.ndarray
            Integer label array of shape (num_events,).

        Raises
        ------
        RuntimeError
            If the process is not found in any class definition.
        """
        for class_index, class_entry in enumerate(class_definitions):
            # Case 1: Simple class (string match)
            if isinstance(class_entry, str) and process_name == class_entry:
                return np.full(num_events, class_index, dtype=int)

            # Case 2: Merged class (dictionary match)
            elif isinstance(class_entry, dict):
                class_name = next(iter(class_entry.keys()))
                if process_name == class_name:
                    return np.full(num_events, class_index, dtype=int)

        raise RuntimeError(
            f"Process '{process_name}' not found in class definitions: "
            f"{class_definitions}"
        )

    def _balance_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, float]]]:
        """Balance dataset using specified strategy.

        Supports:
        - Undersampling
        - Oversampling
        - Class weighting

        Parameters
        ----------
        features : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        labels : np.ndarray
            Integer class labels of shape (n_samples,).

        Returns
        -------
        balanced_features : np.ndarray
            Balanced feature matrix.
        balanced_labels : np.ndarray
            Balanced label array.
        class_weights : Optional[Dict[int, float]]
            Class weights dictionary if using 'class_weight' strategy.

        Raises
        ------
        ValueError
            If unknown balancing strategy is specified.
        """
        strategy = self.mva_cfg.balance_strategy
        random_state = self.mva_cfg.random_state

        # Return original data if no balancing
        if strategy == "none":
            return features, labels, None

        # Initialize RNG and compute class counts
        rng = np.random.RandomState(random_state)
        class_counts = Counter(labels)
        unique_labels = sorted(class_counts)

        if strategy in ("undersample", "oversample"):
            # Determine target samples per class
            target = (
                min(class_counts.values())  # Undersample to minority
                if strategy == "undersample"
                else max(class_counts.values())  # Oversample to majority
            )

            balanced_features = []
            balanced_labels = []

            # Resample each class
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                replace = (
                    strategy == "oversample"
                )  # Allow replacement for oversampling
                resampled_indices = rng.choice(
                    label_indices,
                    size=target,
                    replace=replace,
                )
                balanced_features.append(features[resampled_indices])
                balanced_labels.append(labels[resampled_indices])

            # Combine and shuffle
            features_stacked = np.concatenate(balanced_features, axis=0)
            labels_stacked = np.concatenate(balanced_labels, axis=0)
            permuted_indices = rng.permutation(len(labels_stacked))

            return (
                features_stacked[permuted_indices],
                labels_stacked[permuted_indices],
                None,
            )

        if strategy == "class_weight":
            # Compute inverse frequency weights
            total_samples = float(len(labels))
            class_weights = {
                label: total_samples / (len(unique_labels) * count)
                for label, count in class_counts.items()
            }
            return features, labels, class_weights

        raise ValueError(f"Unknown balance_strategy: {strategy}")

    def prepare_inputs(
        self,
        events_per_process: Dict[str, List[Tuple[Dict, int]]],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[Dict[int, float]],
    ]:
        """Prepare training/validation data from event-level inputs.

        Extract features for all classes (training and plot), but only
        generate labels and populate the training set for classes in
        self.mva_cfg.classes (not plot_classes).
        """
        all_features = []
        all_labels = []
        process_to_features_map = defaultdict(list)

        # combine class definitions, preserving order, avoiding duplicates
        seen = set()
        combined = chain(self.mva_cfg.classes, self.mva_cfg.plot_classes)
        for entry in combined:
            # parse entry to (class_name, proc_names)
            if isinstance(entry, str):
                class_name = entry
                proc_names = [entry]
            elif isinstance(entry, dict):
                class_name, proc_names = next(iter(entry.items()))
            else:
                raise TypeError(f"Invalid class definition: {entry}")

            if class_name in seen:
                continue
            seen.add(class_name)

            # skip if no events
            batches = events_per_process.get(class_name, [])
            if not batches:
                logger.warning(
                    f"No events for class '{class_name}'. Skipping."
                )
                continue

            for obj_dict, n_events in batches:
                # always extract features
                feats, feat_dict = self._extract_features(
                    obj_dict, self.mva_cfg.features
                )
                # collect for process-to-features map
                process_to_features_map[class_name].append(feat_dict)

                # only assign labels & append to training if class is in training set
                if entry in self.mva_cfg.classes:
                    labels = self._make_labels(
                        n_events, class_name, self.mva_cfg.classes
                    )
                    all_features.append(feats)
                    all_labels.append(labels)

        # Finalize process_to_features_map: combine per-feature arrays across batches
        self.process_to_features_map = {}
        for class_name, dict_list in process_to_features_map.items():
            # All dicts share same keys
            combined = {
                feature: {
                    "scaled": np.concatenate(
                        [d[feature]["scaled"] for d in dict_list]
                    ),
                    "unscaled": np.concatenate(
                        [d[feature]["unscaled"] for d in dict_list]
                    ),
                }
                for feature in dict_list[0].keys()
            }
            self.process_to_features_map[class_name] = combined

        if not all_features or not all_labels:
            raise ValueError(
                "No valid training data found for any training class."
            )

        # build and balance dataset
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        X_bal, y_bal, class_weights = self._balance_dataset(X, y)

        # split
        X_train, X_val, y_train, y_val = self._split_train_test(X_bal, y_bal)

        return X_train, y_train, X_val, y_val, class_weights

    @abstractmethod
    def init_network(self) -> None:
        """Initialize network architecture and parameters.

        Must be implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        training_inputs: Union[jnp.ndarray, Any],
        training_labels: Union[jnp.ndarray, Any],
        validation_inputs: Optional[Union[jnp.ndarray, Any]] = None,
        validation_labels: Optional[Union[jnp.ndarray, Any]] = None,
    ) -> Any:
        """Train the model.

        Parameters
        ----------
        training_inputs : Union[jnp.ndarray, Any]
            Training feature array.
        training_labels : Union[jnp.ndarray, Any]
            Training labels.
        validation_inputs : Optional[Union[jnp.ndarray, Any]]
            Validation features (default: None).
        validation_labels : Optional[Union[jnp.ndarray, Any]]
            Validation labels (default: None).

        Returns
        -------
        Any
            Trained model or parameter state.
        """
        raise NotImplementedError

    def generate_scores_for_processes(
        self,
        events_per_process: Dict[str, List[Tuple[Dict, int]]],
        custom_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate MVA scores for plotting processes.

        This method extracts features and generates MVA scores for processes
        that should be included in plots. It uses the same logic as prepare_inputs
        for determining which processes to include based on plot_classes configuration.

        Parameters
        ----------
        events_per_process : Dict[str, List[Tuple[Dict, int]]]
            Dictionary mapping class names to list of (object_dict, event_count) tuples.
        custom_parameters : Optional[Dict[str, Any]]
             Optional custom parameters to use for prediction
             (e.g. optimized parameters).
            If None, uses the current model parameters.

        Returns
        -------
        Dict[str, np.ndarray]
             Dictionary mapping process names to MVA score arrays
             ready for plot_mva_scores.
        """
        # Determine which processes to include based on plot_classes
        plot_processes = getattr(self.mva_cfg, "plot_processes", None)
        if plot_processes is None:
            # Fall back to extracting process names from plot_classes
            plot_processes = []
            for entry in self.mva_cfg.plot_classes:
                if isinstance(entry, str):
                    plot_processes.append(entry)
                elif isinstance(entry, dict):
                    class_name, proc_names = next(iter(entry.items()))
                    plot_processes.extend(proc_names)

        process_scores = {}

        # Generate scores for each process
        for class_name, batches in events_per_process.items():
            # Skip if this class is not in our plot processes
            if class_name not in plot_processes:
                continue

            if not batches:
                logger.warning(
                    f"No events for class '{class_name}'. Skipping score generation."
                )
                continue

            all_scores = []
            for obj_dict, n_events in batches:
                # Extract features
                features, _ = self._extract_features(
                    obj_dict, self.mva_cfg.features
                )

                # Generate predictions using custom parameters if provided
                if custom_parameters is not None:
                    scores = self._predict_with_custom_params(
                        features, custom_parameters
                    )
                else:
                    scores = self.predict(features)

                all_scores.append(np.asarray(scores))

            # Concatenate all scores for this process
            if all_scores:
                process_scores[class_name] = np.concatenate(all_scores)

        # Store in the instance attribute
        self.process_to_scores_map = process_scores
        return process_scores

    def _predict_with_custom_params(
        self,
        inputs: Union[jnp.ndarray, np.ndarray],
        custom_parameters: Dict[str, Any],
    ) -> np.ndarray:
        """Generate predictions using custom parameters.

        This method should be overridden by subclasses to handle custom parameter usage.
        Default implementation falls back to regular predict method.

        Parameters
        ----------
        inputs : Union[jnp.ndarray, np.ndarray]
            Input features for prediction.
        custom_parameters : Dict[str, Any]
            Custom parameters to use for prediction.

        Returns
        -------
        np.ndarray
            Prediction scores.
        """
        logger.warning(
            f"Custom parameters not supported for "
            f"{self.__class__.__name__}. Using default parameters."
        )
        return self.predict(inputs)

    @abstractmethod
    def predict(
        self,
        inputs: Union[jnp.ndarray, Any],
        **kwargs,
    ) -> Any:
        """Perform inference with trained model.

        Parameters
        ----------
        inputs : Union[jnp.ndarray, Any]
            Feature input for prediction.
        **kwargs
            Optional keyword arguments for inference.

        Returns
        -------
        Any
            Prediction results.
        """
        raise NotImplementedError


# =============================================================================
# JAXNetwork Implementation
# =============================================================================


class JAXNetwork(BaseNetwork):
    """JAX-based MVA implementation with manual gradient descent.

    Attributes
    ----------
    Inherits all attributes from BaseNetwork.

    Methods
    -------
    forward_pass(parameters, inputs)
        Forward propagation through network.
    compute_loss(parameters, inputs, targets)
        Calculate loss value.
    compute_accuracy(parameters, inputs, targets)
        Compute classification accuracy.
    _update_step(parameters, inputs, targets)
        Single optimization step.
    """

    def init_network(self) -> None:
        """Initialize weights and biases for each layer.

        Uses configured random seed for reproducibility.
        """
        layer_dimensions = [len(self.mva_cfg.features)] + [
            layer.ndim for layer in self.mva_cfg.layers
        ]
        seed_value = int(self.mva_cfg.random_state)
        rng_key = random.PRNGKey(seed_value)
        layer_keys = random.split(rng_key, len(self.mva_cfg.layers))

        for layer_index, layer_config in enumerate(self.mva_cfg.layers):
            weight_key, bias_key = random.split(layer_keys[layer_index], 2)
            input_dim = layer_dimensions[layer_index]
            output_dim = layer_dimensions[layer_index + 1]

            weight_name = f"__NN_{self.name}_{layer_config.weights}"
            bias_name = f"__NN_{self.name}_{layer_config.bias}"

            # Initialization scaled by 0.1
            self.parameters[weight_name] = (
                random.normal(weight_key, (input_dim, output_dim)) * 0.1
            )
            self.parameters[bias_name] = jnp.zeros(output_dim)

    def forward_pass(
        self, parameters: Dict[str, jnp.ndarray], inputs: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward propagation through network layers.

        Parameters
        ----------
        parameters : Dict[str, jnp.ndarray]
            Dictionary of weights and biases.
        inputs : jnp.ndarray
            Input data of shape (n_samples, n_features)

        Returns
        -------
        jnp.ndarray
            Output predictions of shape (n_samples,).
        """
        activations = inputs
        for layer_config in self.mva_cfg.layers:
            weights = parameters[f"__NN_{self.name}_{layer_config.weights}"]
            biases = parameters[f"__NN_{self.name}_{layer_config.bias}"]
            activations = layer_config.activation(activations, weights, biases)
        return activations.squeeze()

    def compute_loss(
        self,
        parameters: Dict[str, jnp.ndarray],
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute scalar loss value.

        Parameters
        ----------
        parameters : Dict[str, jnp.ndarray]
            Network parameters.
        inputs : jnp.ndarray
            Input features.
        targets : jnp.ndarray
            True labels.

        Returns
        -------
        jnp.ndarray
            Scalar loss value.
        """
        predictions = self.forward_pass(parameters, inputs)
        return self.mva_cfg.loss(predictions, targets)

    def compute_accuracy(
        self,
        parameters: Dict[str, jnp.ndarray],
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute binary classification accuracy.

        Parameters
        ----------
        parameters : Dict[str, jnp.ndarray]
            Model parameters.
        inputs : jnp.ndarray
            Input feature matrix.
        targets : jnp.ndarray
            Ground truth labels.

        Returns
        -------
        jnp.ndarray
            Mean classification accuracy.
        """
        logits = self.forward_pass(parameters, inputs)
        predictions = (logits > 0).astype(jnp.float32)
        return jnp.mean(predictions == targets)

    def _update_step(
        self,
        parameters: Dict[str, jnp.ndarray],
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        """Perform one gradient descent update step.

        Parameters
        ----------
        parameters : Dict[str, jnp.ndarray]
            Current parameter dictionary.
        inputs : jnp.ndarray
            Feature batch.
        targets : jnp.ndarray
            Label batch.

        Returns
        -------
        updated_parameters : Dict[str, jnp.ndarray]
            Updated parameters after gradient step.
        loss_value : jnp.ndarray
            Current loss value.
        """
        learning_rate = self.mva_cfg.learning_rate
        loss_value, gradients = value_and_grad(self.compute_loss)(
            parameters, inputs, targets
        )
        updated_parameters = jax.tree.map(
            lambda p, g: p - learning_rate * g, parameters, gradients
        )
        return updated_parameters, loss_value

    def train(
        self,
        training_inputs: jnp.ndarray,
        training_labels: jnp.ndarray,
        validation_inputs: Optional[jnp.ndarray] = None,
        validation_labels: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Train model with optional mini-batching and validation.

        Parameters
        ----------
        training_inputs : jnp.ndarray
            Training feature matrix.
        training_labels : jnp.ndarray
            Training label vector.
        validation_inputs : Optional[jnp.ndarray]
            Validation features (default: None).
        validation_labels : Optional[jnp.ndarray]
            Validation labels (default: None).

        Returns
        -------
        Dict[str, jnp.ndarray]
            Final trained parameters.
        """
        total_epochs = getattr(self.mva_cfg, "epochs", 1000)
        batch_size = getattr(self.mva_cfg, "batch_size", None)
        validation_fraction = getattr(self.mva_cfg, "validation_split", 0.0)
        log_frequency = getattr(self.mva_cfg, "log_interval", 100)
        total_samples = training_inputs.shape[0]

        # JIT compile the update step
        self._update_step = jit(self._update_step)

        # Create validation split if needed
        if validation_fraction > 0:
            split_index = int(total_samples * (1 - validation_fraction))
            train_x, train_y = (
                training_inputs[:split_index],
                training_labels[:split_index],
            )
            valid_x, valid_y = (
                training_inputs[split_index:],
                training_labels[split_index:],
            )
        else:
            train_x, train_y = training_inputs, training_labels
            valid_x = valid_y = None

        parameters = self.parameters
        for epoch in range(1, total_epochs + 1):
            rng_key = random.PRNGKey(epoch)
            shuffled_indices = random.permutation(rng_key, train_x.shape[0])
            shuffled_inputs = train_x[shuffled_indices]
            shuffled_labels = train_y[shuffled_indices]

            # Full batch training
            if batch_size is None:
                parameters, current_loss = self._update_step(
                    parameters, shuffled_inputs, shuffled_labels
                )

            # Mini-batch training
            else:
                current_loss = 0.0
                for batch_start in range(
                    0, shuffled_inputs.shape[0], batch_size
                ):
                    batch_end = min(
                        batch_start + batch_size, shuffled_inputs.shape[0]
                    )
                    batch_inputs = shuffled_inputs[batch_start:batch_end]
                    batch_labels = shuffled_labels[batch_start:batch_end]
                    parameters, current_loss = self._update_step(
                        parameters, batch_inputs, batch_labels
                    )
                    # current_loss += batch_loss * (
                    #     batch_end - batch_start
                    # ) / train_x.shape[0]

            # Log progress
            if epoch % log_frequency == 0 or epoch == total_epochs:
                train_acc = self.compute_accuracy(parameters, train_x, train_y)
                msg = (
                    f"{self.mva_cfg.name} | Epoch {epoch}: "
                    f"loss={current_loss:.4f}, acc={train_acc:.4f}"
                )
                if valid_x is not None:
                    val_acc = self.compute_accuracy(
                        parameters, valid_x, valid_y
                    )
                    msg += f", val_acc={val_acc:.4f}"
                logger.info(msg)

        self.parameters = parameters
        return self.parameters

    def predict(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Generate predictions from trained network.

        Parameters
        ----------
        inputs : jnp.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        jnp.ndarray
            Model output logits.
        """
        return self.forward_pass(self.parameters, inputs)

    def _predict_with_custom_params(
        self,
        inputs: Union[jnp.ndarray, np.ndarray],
        custom_parameters: Dict[str, Any],
    ) -> np.ndarray:
        """Generate predictions using custom parameters for JAX networks.

        Parameters
        ----------
        inputs : Union[jnp.ndarray, np.ndarray]
            Input features for prediction.
        custom_parameters : Dict[str, Any]
            Custom parameters to use for prediction. Should contain NN parameters
            with keys like "__NN_{model_name}_{param_name}".

        Returns
        -------
        np.ndarray
            Prediction scores.
        """
        # Extract NN parameters for this model from the flattened parameter dict
        model_params = {}
        for param_name, param_value in custom_parameters.items():
            if param_name.startswith(f"__NN_{self.name}_"):
                model_params[param_name] = param_value

        # If no custom parameters found for this model, use default parameters
        if not model_params:
            logger.warning(
                f"No custom parameters found for model '{self.name}'. "
                f"Using default parameters."
            )
            return np.asarray(self.predict(jnp.asarray(inputs)))

        # Use custom parameters for prediction
        inputs_jax = jnp.asarray(inputs)
        predictions = self.forward_pass(model_params, inputs_jax)
        return np.asarray(predictions)


# =============================================================================
# TFNetwork Implementation
# =============================================================================


class TFNetwork(BaseNetwork):
    """TensorFlow/Keras-based MVA implementation using built-in .fit().

    Attributes
    ----------
    Inherits all attributes from BaseNetwork.
    """

    def init_network(self) -> None:
        import tensorflow as tf

        """Construct and compile Keras Sequential model."""
        input_dimension = len(self.mva_cfg.features)
        keras_layers = []

        for layer_index, layer_config in enumerate(self.mva_cfg.layers):
            # Layer configuration
            dense_layer_args: Dict[str, Any] = {
                "units": layer_config.ndim,
                "activation": layer_config.activation,
            }

            # Add input shape for first layer
            if layer_index == 0:
                dense_layer_args["input_shape"] = (input_dimension,)

            keras_layers.append(tf.keras.layers.Dense(**dense_layer_args))

        self.model = tf.keras.Sequential(keras_layers)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.mva_cfg.learning_rate
            ),
            loss=self.mva_cfg.loss,
            metrics=["accuracy"],
        )

    def train(
        self,
        training_inputs: Any,
        training_labels: Any,
        validation_inputs: Optional[Any] = None,
        validation_labels: Optional[Any] = None,
    ) -> "tf.keras.Model":
        """Train model using Keras .fit() method.

        Parameters
        ----------
        training_inputs : Any
            Training feature array.
        training_labels : Any
            Training labels.
        validation_inputs : Optional[Any]
            Validation features (default: None).
        validation_labels : Optional[Any]
            Validation labels (default: None).

        Returns
        -------
        tf.keras.Model
            Trained Keras model.
        """
        total_epochs = getattr(self.mva_cfg, "epochs", 50)
        batch_size = getattr(self.mva_cfg, "batch_size", 32)
        validation_split_fraction = getattr(
            self.mva_cfg, "validation_split", 0.0
        )

        training_arguments = {
            "x": training_inputs,
            "y": training_labels,
            "epochs": total_epochs,
            "batch_size": batch_size,
            "validation_split": validation_split_fraction,
            "verbose": 1,
        }

        # Use explicit validation data if provided
        if validation_inputs is not None and validation_labels is not None:
            training_arguments.pop("validation_split")
            training_arguments["validation_data"] = (
                validation_inputs,
                validation_labels,
            )

        self.model.fit(**training_arguments)
        return self.model

    def predict(
        self, inputs: Any, batch_size: Optional[int] = None, **kwargs
    ) -> Any:
        """Generate predictions using trained model.

        Parameters
        ----------
        inputs : Any
            Input features for prediction.
        batch_size : Optional[int]
            Batch size for inference (default: None).
        **kwargs
            Additional arguments for Keras predict().

        Returns
        -------
        Any
            Model predictions.
        """
        return self.model.predict(inputs, batch_size=batch_size, **kwargs)
