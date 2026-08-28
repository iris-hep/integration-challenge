import json
import os

import onnxruntime as ort
import yaml
import numpy as np
import awkward as ak

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype, InferenceServerException
    from google.protobuf import json_format
except ImportError as e:  # tritonclient is optional, only needed with use_triton=True
    grpcclient = None
    np_to_triton_dtype = None
    InferenceServerException = None
    json_format = None
    triton_import_error = e
else:
    triton_import_error = None


def pad_and_stack(fields, max_len):
    """Pad ragged per-object fields to `max_len` and stack them densely."""

    padded = [ak.pad_none(field, max_len, clip=True) for field in fields]

    # True where the slot is padding rather than a real object
    mask = ak.to_numpy(ak.is_none(padded[0], axis=1))

    features = np.stack(
        [ak.to_numpy(ak.fill_none(field, 0.0)) for field in padded], axis=-1
    ).astype(np.float32, copy=False)

    return features, mask


class TritonInference:

    def load_triton_client(self, num_instances=None, kind=None):
        """Connect to the server and make sure the model is loaded and ready."""
        if triton_import_error is not None:
            raise ImportError(
                "use_triton=True requires tritonclient and protobuf, which failed "
                f"to import: {triton_import_error}"
            ) from triton_import_error

        self.triton_client = grpcclient.InferenceServerClient(url=self.triton_url)

        if not (self.triton_client.is_server_live() and self.triton_client.is_server_ready()):
            raise RuntimeError(f"Server at {self.triton_url} not live")

        if not self.triton_client.is_model_ready(self.triton_model):
            self.load_model()

        config_override = self.instance_group_override(num_instances, kind)
        if config_override is not None:
            self.load_model(config_override)

        # take the output names from the server so they stay in sync with config.pbtxt
        metadata = self.triton_client.get_model_metadata(self.triton_model)
        self.triton_outputs = [out.name for out in metadata.outputs]

    def load_model(self, config=None):
        try:
            self.triton_client.load_model(self.triton_model, config=config)
        except InferenceServerException as e:
            raise RuntimeError(
                f"Failed to load model '{self.triton_model}': {e}"
            ) from e

    def instance_group_override(self, num_instances=None, kind=None):
        """Build the model config that requests the given instance_group."""

        if num_instances is None and kind is None:
            return None

        config = self.triton_client.get_model_config(self.triton_model).config
        config = json_format.MessageToDict(config, preserving_proto_field_name=True)

        group = {}
        if num_instances is not None:
            group["count"] = num_instances
        if kind is not None:
            group["kind"] = kind

        running = config["instance_group"]
        if len(running) == 1:
            if all(running[0].get(key) == value for key, value in group.items()):
                return None  # already loaded with the requested instance_group

        config["instance_group"] = [group]
        return json.dumps(config)

    def run_triton(self, feed):
        if self.triton_client is None:
            raise ValueError("Triton client not loaded")

        inputs = []
        for name, array in feed.items():
            array = np.ascontiguousarray(array)
            inp = grpcclient.InferInput(
                name, array.shape, np_to_triton_dtype(array.dtype)
            )
            inp.set_data_from_numpy(array)
            inputs.append(inp)

        outputs = [grpcclient.InferRequestedOutput(name) for name in self.triton_outputs]

        response = self.triton_client.infer(
            model_name=self.triton_model, inputs=inputs, outputs=outputs
        )
        return [response.as_numpy(name) for name in self.triton_outputs]

    def cleanup_triton(self):
        """Unload the model from the server and close the client connection."""
        if self.triton_client is None:
            self.triton_client = grpcclient.InferenceServerClient(url=self.triton_url)

        try:
            if self.triton_client.is_model_ready(self.triton_model):
                self.triton_client.unload_model(self.triton_model)
        except InferenceServerException as e:
            print(f"warning: failed to unload {self.triton_model}: {e}")

        self.triton_client.close()
        self.triton_client = None


class MLModelEvent(TritonInference):
    def __init__(self, use_triton=False, num_instances=None, kind=None):
        self.nn_path = "/data/acordeir/integ-challenge/network.onnx"
        self.norm_path = "/data/acordeir/integ-challenge/IC-input-norms-final.yaml"

        self.use_triton = use_triton
        self.triton_model = "jet_network"
        self.triton_url = "triton-traefik.triton.svc.cluster.local:8001"
        self.num_instances = num_instances
        self.kind = kind

        self.ort_model = None
        self.triton_client = None
        self.triton_outputs = None
        self.norm_dict = None

    def load(self):
        self.load_norms()
        if self.use_triton:
            self.load_triton_client(self.num_instances, self.kind)
        else:
            self.load_onnx()

    def load_onnx(self):
        self.ort_model = ort.InferenceSession(self.nn_path)

    def load_norms(self):
        with open(self.norm_path) as f:
            self.norm_dict = yaml.safe_load(f)

    def norm_features(self, event):
        norm = self.norm_dict
        if norm is None:
            raise ValueError("Normalization not loaded")

        jets = event.jet
        els  = event.el

        # --- convert to numpy ---
        jet_pt  = ak.to_numpy(jets.pt)
        jet_eta = ak.to_numpy(jets.eta)
        jet_phi = ak.to_numpy(jets.phi)
        jet_b   = ak.to_numpy(jets.GN2v01_FixedCutBEff_77_select)

        el_pt  = ak.to_numpy(els.pt)
        el_eta = ak.to_numpy(els.eta)
        el_phi = ak.to_numpy(els.phi)

        # --- normalize ---
        jet_arr = np.stack([
            (jet_pt  - norm["jet"]["pt"]["mean"])  / norm["jet"]["pt"]["std"],
            (jet_eta - norm["jet"]["eta"]["mean"]) / norm["jet"]["eta"]["std"],
            (jet_phi - norm["jet"]["phi"]["mean"]) / norm["jet"]["phi"]["std"],
            (jet_b   - norm["jet"]["btag"]["mean"]) / norm["jet"]["btag"]["std"],
        ], axis=1).astype(np.float32)

        el_arr = np.stack([
            (el_pt  - norm["el"]["pt"]["mean"])  / norm["el"]["pt"]["std"],
            (el_eta - norm["el"]["eta"]["mean"]) / norm["el"]["eta"]["std"],
            (el_phi - norm["el"]["phi"]["mean"]) / norm["el"]["phi"]["std"],
        ], axis=1).astype(np.float32)

        return jet_arr, el_arr

    def run_event(self, event):
        if self.use_triton:
            if self.triton_client is None:
                raise ValueError("Triton client not loaded")
        elif self.ort_model is None:
            raise ValueError("ONNX model not loaded")

        # skip empty events
        if len(event.jet.pt) == 0 or len(event.el.pt) == 0:
            return None

        jet_arr, el_arr = self.norm_features(event)

        feed = {
            "jet_features": jet_arr,
            "el_features": el_arr,
        }

        if self.use_triton:
            out = self.run_triton(feed)
        else:
            out = self.ort_model.run(None, feed)

        prob = 1 / (1 + np.exp(-out[0].item()))
        return prob


class MLModel(TritonInference):
    def __init__(self, use_triton=False, num_instances=None, kind=None):
        self.nn_path = "/data/acordeir/integ-challenge/network_batch.onnx"
        self.norm_path = "/data/acordeir/integ-challenge/IC-input-norms-final.yaml"

        self.use_triton = use_triton
        self.triton_model = "jet_network_batch"
        self.triton_url = "triton-traefik.triton.svc.cluster.local:8001"
        self.num_instances = num_instances
        self.kind = kind

        self.ort_model = None
        self.triton_client = None
        self.triton_outputs = None
        self.norm_dict = None

    def load(self):
        self.load_norms()
        if self.use_triton:
            self.load_triton_client(self.num_instances, self.kind)
        else:
            self.load_onnx()

    def load_onnx(self):
        self.ort_model = ort.InferenceSession(self.nn_path)

    def load_norms(self):
        with open(self.norm_path) as f:
            self.norm_dict = yaml.safe_load(f)

    def prep_features(self, events):
        norm = self.norm_dict

        max_jets = int(ak.max(ak.num(events.jet.pt)))
        max_els  = int(ak.max(ak.num(events.el.pt)))

        jets = events.jet
        els = events.el

        jet_fields = [
            (jets.pt  - norm["jet"]["pt"]["mean"])  / norm["jet"]["pt"]["std"],
            (jets.eta - norm["jet"]["eta"]["mean"]) / norm["jet"]["eta"]["std"],
            (jets.phi - norm["jet"]["phi"]["mean"]) / norm["jet"]["phi"]["std"],
            (jets.GN2v01_FixedCutBEff_77_select - norm["jet"]["btag"]["mean"])
                / norm["jet"]["btag"]["std"],
        ]

        el_fields = [
            (els.pt  - norm["el"]["pt"]["mean"])  / norm["el"]["pt"]["std"],
            (els.eta - norm["el"]["eta"]["mean"]) / norm["el"]["eta"]["std"],
            (els.phi - norm["el"]["phi"]["mean"]) / norm["el"]["phi"]["std"],
        ]

        jet_features, jet_mask = pad_and_stack(jet_fields, max_jets)
        el_features, el_mask = pad_and_stack(el_fields, max_els)

        return jet_features, jet_mask, el_features, el_mask

    def run_inference(self, events):
        if self.use_triton:
            if self.triton_client is None:
                raise ValueError("Triton client not loaded")
        elif self.ort_model is None:
            raise ValueError("ONNX model not loaded")

        jet_features, jet_mask, el_features, el_mask = self.prep_features(events)

        feed = {
            "jet_features": jet_features,
            "jet_features_mask": jet_mask,
            "el_features": el_features,
            "el_features_mask": el_mask,
        }

        if self.use_triton:
            out = self.run_triton(feed)
        else:
            out = self.ort_model.run(None, feed)

        prob = 1 / (1 + np.exp(-out[0]))
        return prob
