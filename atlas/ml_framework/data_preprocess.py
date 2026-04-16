import awkward as ak
import numpy as np
import h5py
import yaml



import awkward as ak
import numpy as np
import h5py


def to_hdf5(samples_dict, output_file, max_events_per_sample=-1):

    arrays = []

    for sample, files in samples_dict.items():

        label = 1 if sample.startswith("Hplus") else 0

        arr = ak.from_parquet(list(files)) #files is ServiceX Guardlist wrapper

        if max_events_per_sample != -1:
            arr = arr[:max_events_per_sample]

        # add fields
        arr = ak.with_field(arr, label, "label")
   #     arr = ak.with_field(arr, sample, "sample") 

        print(f"\Loading {sample} ({len(files)} files, label={label})")


        arrays.append(arr)

        print(f"Loaded {len(arr)} events")

    # -----------------
    # merge all samples
    # -----------------

    arr = ak.concatenate(arrays)
    num_events = len(arr)

    print(f"\nTotal events: {num_events}")

    max_jets = int(ak.max(ak.num(arr.jet_pt_NOSYS)))
    max_els  = int(ak.max(ak.num(arr.el_pt_NOSYS)))

    # -----------------
    # dtypes
    # -----------------

    event_dtype = np.dtype([
        ("met", "f4"),
        ("met_phi", "f4"),
        ("met_sig", "f4"),
        ("met_sumet", "f4"),
        ("label", "i4"),
#        ("sample", "S64"), 
    ])

    jet_dtype = np.dtype([
        ("valid", "?"),
        ("pt", "f4"),
        ("eta", "f4"),
        ("phi", "f4"),
        ("btag", "?"),
    ])

    el_dtype = np.dtype([
        ("valid", "?"),
        ("pt", "f4"),
        ("eta", "f4"),
        ("phi", "f4"),
    ])

    # -----------------
    # allocate
    # -----------------

    event = np.zeros(num_events, dtype=event_dtype)
    jet   = np.zeros((num_events, max_jets), dtype=jet_dtype)
    el    = np.zeros((num_events, max_els), dtype=el_dtype)

    # event-level (vectorized)
    event["met"]       = ak.to_numpy(arr.met_met_NOSYS)
    event["met_phi"]   = ak.to_numpy(arr.met_phi_NOSYS)
    event["met_sig"]   = ak.to_numpy(arr.met_significance_NOSYS)
    event["met_sumet"] = ak.to_numpy(arr.met_sumet_NOSYS)
    event["label"]     = ak.to_numpy(arr.label)
#    event["sample"]    = ak.to_numpy(arr.sample).astype("S64")

    # -----------------
    # loop objects
    # -----------------

    #log_step = max(1, num_events // 20)

    # -----------------
    # vectorized objects
    # -----------------
    
    # ---- Jets ----
    max_jets = int(ak.max(ak.num(arr.jet_pt_NOSYS)))
    #pad_none default axis is 1
    jet_pt   = ak.pad_none(arr.jet_pt_NOSYS, max_jets, clip=True)
    jet_eta  = ak.pad_none(arr.jet_eta, max_jets, clip=True)
    jet_phi  = ak.pad_none(arr.jet_phi, max_jets, clip=True)
    jet_btag = ak.pad_none(arr.jet_GN2v01_FixedCutBEff_77_select, max_jets, clip=True)
    
    jet_valid = ~ak.is_none(jet_pt, axis=-1)
    
    jet_pt   = ak.fill_none(jet_pt, 0)
    jet_eta  = ak.fill_none(jet_eta, 0)
    jet_phi  = ak.fill_none(jet_phi, 0)
    jet_btag = ak.fill_none(jet_btag, False)
    
    jet["valid"] = ak.to_numpy(jet_valid)
    jet["pt"]    = ak.to_numpy(jet_pt)
    jet["eta"]   = ak.to_numpy(jet_eta)
    jet["phi"]   = ak.to_numpy(jet_phi)
    jet["btag"]  = ak.to_numpy(jet_btag)
    
    
    # ---- Electrons ----
    max_els = int(ak.max(ak.num(arr.el_pt_NOSYS)))
    
    el_pt  = ak.pad_none(arr.el_pt_NOSYS, max_els, clip=True)
    el_eta = ak.pad_none(arr.el_eta, max_els, clip=True)
    el_phi = ak.pad_none(arr.el_phi, max_els, clip=True)
    
    el_valid = ~ak.is_none(el_pt, axis=-1)
    
    el_pt  = ak.fill_none(el_pt, 0)
    el_eta = ak.fill_none(el_eta, 0)
    el_phi = ak.fill_none(el_phi, 0)
    
    el["valid"] = ak.to_numpy(el_valid)
    el["pt"]    = ak.to_numpy(el_pt)
    el["eta"]   = ak.to_numpy(el_eta)
    el["phi"]   = ak.to_numpy(el_phi)

    # -----------------
    # write
    # -----------------

    with h5py.File(output_file, "w") as f:
        f.create_dataset("event", data=event)
        f.create_dataset("jet", data=jet)
        f.create_dataset("el", data=el)

    print(f"\nWritten {output_file}")


def split_h5(input_file, train_frac=0.7, val_frac=0.15):

    with h5py.File(input_file, "r") as f:
        event = f["event"][:]
        jet   = f["jet"][:]
        el    = f["el"][:]

    n = len(event)

    # shuffle indices
    idx = np.random.permutation(n)

    n_train = int(train_frac * n)
    n_val   = int(val_frac * n)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    # write files
    for name, indices in splits.items():
        
        out_file = input_file[:-3]+"_"+name+".h5"
        with h5py.File(out_file, "w") as f:
            f.create_dataset("event", data=event[indices])
            f.create_dataset("jet", data=jet[indices])
            f.create_dataset("el", data=el[indices])

        print(f"Wrote {out_file} ({len(indices)} events)")




def make_norm_dict(h5file, output):

    norm = {}

    with h5py.File(h5file, "r") as f:

        for obj in ["event", "jet", "el"]:

            norm[obj] = {}

            data = f[obj][:]

            for name in data.dtype.names:

                if name == "valid" or name == "label" or name == "sample":
                    continue
                    
                values = data[name]
                
                if "valid" in data.dtype.names:
                    mask = data["valid"]
                    values = values[mask]

                values = data[name].reshape(-1)

                norm[obj][name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values) + 1e-6), #!Div by 0 
                }

    with open(output, "w") as out:
        yaml.dump(norm, out)