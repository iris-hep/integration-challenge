import onnxruntime as ort
import yaml
import numpy as np
import awkward as ak


class MLModel:
    def __init__(self):
        self.nn_path = "/data/acordeir/integ-challenge/network.onnx" 
        self.norm_path = "/data/acordeir/integ-challenge/IC-input-norms-final.yaml"

        self.ort_model = None
        self.norm_dict = None

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
        if self.ort_model is None:
            raise ValueError("ONNX model not loaded")

        # skip empty events
        if len(event.jet.pt) == 0 or len(event.el.pt) == 0:
            return None

        jet_arr, el_arr = self.norm_features(event)

        out = self.ort_model.run(None, {
            "jet_features": jet_arr,
            "el_features": el_arr,
        })

        prob = 1 / (1 + np.exp(-out[0].item()))
        return prob