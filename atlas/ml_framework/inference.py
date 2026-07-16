import onnxruntime as ort
import yaml
import numpy as np
import awkward as ak


class MLModelEvent:
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


class MLModel:
    """
    Inferencing utilities for batched dimension ONNX models
    """
    def __init__(self):
        self.nn_path = "/data/acordeir/integ-challenge/network_batch.onnx"
        self.norm_path = "/data/acordeir/integ-challenge/IC-input-norms-final.yaml"

        self.ort_model = None
        self.norm_dict = None

    def load_onnx(self):
        self.ort_model = ort.InferenceSession(self.nn_path)
        
    def load_norms(self):
        with open(self.norm_path) as f:
            self.norm_dict = yaml.safe_load(f)
            
    def prep_features(self, events):
        norm = self.norm_dict
    
        max_jets = int(ak.max(ak.num(events.jet.pt)))
        max_els  = int(ak.max(ak.num(events.el.pt)))
    
        jet_arr = ak.zip([
            (events.jet.pt  - norm["jet"]["pt"]["mean"])  / norm["jet"]["pt"]["std"],
            (events.jet.eta - norm["jet"]["eta"]["mean"]) / norm["jet"]["eta"]["std"],
            (events.jet.phi - norm["jet"]["phi"]["mean"]) / norm["jet"]["phi"]["std"],
            (events.jet.GN2v01_FixedCutBEff_77_select - norm["jet"]["btag"]["mean"])
                / norm["jet"]["btag"]["std"],
        ])
    
        el_arr = ak.zip([
            (events.el.pt  - norm["el"]["pt"]["mean"])  / norm["el"]["pt"]["std"],
            (events.el.eta - norm["el"]["eta"]["mean"]) / norm["el"]["eta"]["std"],
            (events.el.phi - norm["el"]["phi"]["mean"]) / norm["el"]["phi"]["std"],
        ])


        jet_arr = ak.pad_none(jet_arr, max_jets)
        el_arr = ak.pad_none(el_arr, max_els)
    
        jet_mask = ak.is_none(jet_arr, axis = 1)
        el_mask = ak.is_none(el_arr, axis = 1)
        
        jet_arr = ak.fill_none(
            jet_arr,
            (0.0, 0.0, 0.0, 0.0),
        )
    
        el_arr = ak.fill_none(
            el_arr,
            (0.0, 0.0, 0.0),
        )

        jet_features = np.array(ak.to_list(jet_arr), dtype=np.float32)
        el_features = np.array(ak.to_list(el_arr), dtype=np.float32)
        
        jet_mask = np.array(ak.to_list(jet_mask), dtype=bool)
        el_mask = np.array(ak.to_list(el_mask), dtype=bool)
    
        return jet_features, jet_mask, el_features, el_mask
    
    def run_inference(self, events):
        if self.ort_model is None:
            raise ValueError("ONNX model not loaded")

        jet_features, jet_mask, el_features, el_mask = self.prep_features(events)

        out = self.ort_model.run(  
            None,
            {
                "jet_features": jet_features,
                "jet_features_mask": jet_mask,
                "el_features": el_features,
                "el_features_mask": el_mask,
            }
        )

        prob = 1 / (1 + np.exp(-out[0]))
        return prob