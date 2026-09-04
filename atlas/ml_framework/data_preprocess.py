import awkward as ak
import numpy as np
import h5py
import yaml



import awkward as ak
import numpy as np
import h5py


def to_hdf5(files_dict, labels_dict, config, output_file, max_events=-1):
    """
    Configurable translation layer from ServiceX to Salt
    Writes out an .h5 file containing object descriptions for specified files
    labels and files dict are built similarly

    example config: {
        "event": {
            "branches": {
                "met": "met_met_NOSYS",
                "met_phi": "met_phi_NOSYS",
            },
            "attach_label": True,}, #Attach training labels to global variable for Salt
        "jet": {
            "branches": {
                "pt": "jet_pt_NOSYS",
                "eta": "jet_eta",
            }}
            }
    
    """
        
    # ----------------------------------
    # Load arrays and label per sample
    # ----------------------------------
    
    arrays = []
    print("LOADING")
    print("=" * 30)
    for sample, files in files_dict.items():
        arr = ak.from_parquet(list(files))
        if max_events > 0:
            arr = arr[:max_events]
        arr = ak.with_field(arr, labels_dict[sample], "label")
        arrays.append(arr)
        print(f"{sample} - {len(arr)} events")

    all_samples = ak.concatenate(arrays)
    num_events = len(all_samples)
    print("\n--- Combined dataset ---")
    print(f"Total events : {num_events}")

    # -----------------
    # Validate config 
    # -----------------
    for obj, obj_cfg in config.items():
        for out_name, branch in obj_cfg.get("branches", {}).items():
            if branch not in ak.fields(all_samples):
                raise ValueError(
                    f"Branch '{branch}' (mapped to '{obj}/{out_name}') not found. "
                    "Verify config or file content."
                )
        if obj_cfg.get("attach_label", False) and "label" not in ak.fields(all_samples):
            raise ValueError(
                f"config['{obj}']['attach_label'] is True but no 'label' field "
                "was found on the loaded samples."
            )

    # ----------------------------------
    # Build h5 objects from config 
    # ----------------------------------

    output_arrays = {}

    for obj, obj_cfg in config.items():
        branches = obj_cfg.get("branches", {})
        attach_label = obj_cfg.get("attach_label", False)

        if branches:
            first_branch = list(branches.values())[0]
            # check nesting
            is_jagged = all_samples[first_branch].ndim > 1
            max_length = int(ak.max(ak.num(all_samples[first_branch]))) if is_jagged else None
        else: #when no branches (only global to attach label)
            is_jagged = False
            max_length = None

        print(f"{obj}: {'jagged (max=' + str(max_length) + ')' if is_jagged else 'flat'}")

        converted = {} #to hold on numpy arrays of the current object
        valid_arr = None #to hold valid masks 

        for out_name, branch in branches.items():
            branch_arr = all_samples[branch]

            if is_jagged:
                padded = ak.pad_none(branch_arr, max_length, clip=True)

                if valid_arr is None:
                    valid_arr = ak.to_numpy(~ak.is_none(padded, axis=-1))
                    
                np_arr = ak.to_numpy(ak.fill_none(padded, 0))

                converted[out_name] = np_arr
            else:
                converted[out_name] = ak.to_numpy(branch_arr) #no padding

        if attach_label: #attached only to specified object
            converted["label"] = ak.to_numpy(all_samples["label"])

        dtype_fields = [(out_name, arr.dtype) for out_name, arr in converted.items()]
        if is_jagged:
            dtype_fields.append(("valid", "?"))  #required by salt to read var len
        obj_dtype = np.dtype(dtype_fields)
        print(f"{obj} dtype (derived): {obj_dtype}")

        shape = (num_events, max_length) if is_jagged else (num_events,)
        out = np.zeros(shape, dtype=obj_dtype) #np obj to allocate all features

        for out_name, np_arr in converted.items():
            out[out_name] = np_arr
        if is_jagged:
            out["valid"] = valid_arr

        output_arrays[obj] = out

    # -----------------
    # write
    # -----------------
    with h5py.File(output_file, "w") as f:
        for obj, arr in output_arrays.items():
            f.create_dataset(obj, data=arr)
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