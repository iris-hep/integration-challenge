import awkward as ak
import numpy as np
import h5py
import yaml



def to_hdf5(
    files,   # {label: file or list_of_files}
    labels,
    output_file,
    max_events=-1,
):

    # -----------------
    # load each label group and assign labels
    # -----------------
    arrays = []

    for label, f in zip(list(labels), list(files)):

        arr = ak.from_parquet(f)

        if max_events != -1:
            arr = arr[:max_events]

        # add label field
        arr = ak.with_field(arr, int(label), "label")

        arrays.append(arr)

        print(f"Loaded label {label}: {len(arr)} events")

    # merge all
    arr = ak.concatenate(arrays)

    num_events = len(arr)

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

    # vectorized event copy
    event["met"]       = ak.to_numpy(arr.met_met_NOSYS)
    event["met_phi"]   = ak.to_numpy(arr.met_phi_NOSYS)
    event["met_sig"]   = ak.to_numpy(arr.met_significance_NOSYS)
    event["met_sumet"] = ak.to_numpy(arr.met_sumet_NOSYS)
    event["label"]     = ak.to_numpy(arr.label)

    # -----------------
    # loop jagged objects
    # -----------------

    log_step = max(1, num_events // 20)

    for i in range(num_events):

        if i % log_step == 0:
            print(f"Processed {i}/{num_events} ({100*i/num_events:.1f}%)")

        # jets
        pts   = arr.jet_pt_NOSYS[i]
        etas  = arr.jet_eta[i]
        phis  = arr.jet_phi[i]
        btags = arr.jet_GN2v01_FixedCutBEff_77_select[i]

        n = len(pts)

        jet[i, :n]["valid"] = True
        jet[i, :n]["pt"]    = ak.to_numpy(pts)
        jet[i, :n]["eta"]   = ak.to_numpy(etas)
        jet[i, :n]["phi"]   = ak.to_numpy(phis)
        jet[i, :n]["btag"]  = ak.to_numpy(btags)

        # electrons
        pts  = arr.el_pt_NOSYS[i]
        etas = arr.el_eta[i]
        phis = arr.el_phi[i]

        n = len(pts)

        el[i, :n]["valid"] = True
        el[i, :n]["pt"]    = ak.to_numpy(pts)
        el[i, :n]["eta"]   = ak.to_numpy(etas)
        el[i, :n]["phi"]   = ak.to_numpy(phis)

    # -----------------
    # write
    # -----------------

    with h5py.File(output_file, "w") as f:
        f.create_dataset("event", data=event)
        f.create_dataset("jet", data=jet)
        f.create_dataset("el", data=el)

    print(f"Written {output_file}")
    print(f"Total events: {num_events}")



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

        with h5py.File(input_file[:-3]+name+".h5", "w") as f:
            f.create_dataset("event", data=event[indices])
            f.create_dataset("jet", data=jet[indices])
            f.create_dataset("el", data=el[indices])

        print(f"Wrote {name}.h5 ({len(indices)} events)")




def make_norm_dict(h5file, output):

    norm = {}

    with h5py.File(h5file, "r") as f:

        for obj in ["event", "jet", "el"]:

            norm[obj] = {}

            data = f[obj][:]

            for name in data.dtype.names:

                if name == "valid" or name == "label":
                    continue

                values = data[name].reshape(-1)

                if name == "valid":
                    continue

                norm[obj][name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values) + 1e-6),
                }

    with open(output, "w") as out:
        yaml.dump(norm, out)