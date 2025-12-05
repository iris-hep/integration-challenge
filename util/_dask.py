"""
Example usage:

    `Map` and `Reduce` functions for Dask distributed client.

    >>> from util._dask import dask_map, dask_reduce

    # map and keep track of future <-> workitem mapping
    >>> futures, futurekey2item = dask_map(
    >>>     Processor().process,
    >>>     workitems,
    >>>     client=client,
    >>>     NanoEventsFactory_kwargs={
    >>>         "preload": lambda b: b.name in {'Jet_pt', 'Jet_eta'},
    >>>         "schemaclass": NtupleSchema,
    >>>     }
    >>> )

    # perform reduction and track failures
    >>> final_future, failed_items = dask_reduce(
    >>>     futures,
    >>>     futurekey2item=futurekey2item,
    >>>     client=client,
    >>>     treereduction=16,
    >>> )

See also: `if __name__ == "__main__":` block for a complete example.
"""

from __future__ import annotations

import dataclasses
import time
import typing as tp
from collections import Counter, defaultdict
from functools import partial

import awkward as ak
import uproot
from coffea.nanoevents import NanoEventsFactory
from coffea.processor import Accumulatable, accumulate
from coffea.processor.executor import WorkItem
from coffea.util import rich_bar
from dask.distributed import Client
from dask.tokenize import tokenize
from rich.console import Group
from rich.live import Live
from rich.progress import Progress

from util._futures import DynamicAsCompleted, FutureLike

try:
    from coffea.util import coffea_console
except ImportError:
    from rich.console import Console

    coffea_console = Console()


_processing_sentinel = object()
_final_merge_sentinel = object()


# group of progress bars for dask/future executor
def pbar_group(datasets: list[str]) -> tuple[Live, dict[tp.Any, Progress]]:
    pbars = {_processing_sentinel: rich_bar()}
    pbars.update({ds: rich_bar() for ds in datasets})
    pbars[_final_merge_sentinel] = rich_bar()
    return Live(Group(*pbars.values()), console=coffea_console), pbars


Result: tp.TypeAlias = Accumulatable | BaseException


@dataclasses.dataclass(frozen=True, slots=True)
class Failure:
    item: WorkItem
    reason: BaseException


class ReduceSchedulingError(RuntimeError): ...


acc = partial(accumulate, accum=None)


def failed_future(future: FutureLike) -> bool:
    # if we return an exception as a value, consider it failed (see wrapped_process)
    # we catch any exception, but no RuntimeError. Maybe the user wants to raise that?
    return issubclass(future.type, BaseException) or future.status == "error"


def wrapped_process(
    process_func: tp.Callable[[ak.Array], Result],
    workitem: WorkItem,
    /,
    *,
    NanoEventsFactory_kwargs: dict[str, tp.Any] | None = None,
) -> Result:
    t0 = time.time() # move t0 here to include potential preloading
    f = uproot.open(workitem.filename)
    if NanoEventsFactory_kwargs is None:
        NanoEventsFactory_kwargs = {}
    events = NanoEventsFactory.from_root(
        f,
        treepath=workitem.treename,
        mode="virtual",
        access_log=(access_log := []),
        entry_start=workitem.entrystart,
        entry_stop=workitem.entrystop,
        **NanoEventsFactory_kwargs,
    ).events()
    events.metadata.update(workitem.usermeta)
    try:
        out = process_func(events)
        t1 = time.time()
    except Exception as err:
        # return err as value, no metrics
        return err
    bytesread = f.file.source.num_requested_bytes
    report = {
        "bytesread": bytesread,
        "entries": workitem.entrystop - workitem.entrystart,
        "processtime": t1 - t0,
        "chunks": 1,
        "columns": access_log,
        "chunk_info": {
            (workitem.filename, workitem.entrystart, workitem.entrystop): (
                t0,
                t1,
                bytesread,
            )
        },
    }
    return {"out": out, "report": report}


def dask_map(
    process_func: tp.Callable[[ak.Array], Result],
    workitems: tp.Iterable[WorkItem],
    /,
    *,
    client: Client,
    NanoEventsFactory_kwargs: dict[str, tp.Any] | None = None,
) -> tuple[list[FutureLike], dict[str, WorkItem]]:
    futures = client.map(
        partial(
            wrapped_process,
            process_func,
            NanoEventsFactory_kwargs=NanoEventsFactory_kwargs,
        ),
        workitems,
        pure=True,
        key="process",
        priority=0,
    )
    return futures, {f.key: wi for f, wi in zip(futures, workitems)}


def dask_reduce(
    futures: tp.Iterable[FutureLike],
    *,
    futurekey2item: tp.Mapping[str, WorkItem],
    client: Client,
    treereduction: int = 1 << 4,
) -> tuple[FutureLike, defaultdict[list[Failure]]]:
    items = list(futurekey2item.values())
    datasets = [it.dataset for it in items]
    unique_datasets = sorted(set(datasets))

    live, pbars = pbar_group(unique_datasets)

    with live:
        # prepare some metadata for merging
        # dataset -> number of items to do
        ds2todo = Counter(datasets)
        # create a buffer for each dataset (what we merge)
        ds2buf = defaultdict(list)
        # future.key -> dataset item
        key2ds = {fk: wi.dataset for fk, wi in futurekey2item.items()}

        # initialize progress bars
        processing_task = pbars[_processing_sentinel].add_task(
            "Processing", total=len(futures), unit="chunk"
        )
        dataset_merge_tasks = {}
        for ds in unique_datasets:
            total = ds2todo[ds]
            dataset_merge_tasks[ds] = pbars[ds].add_task(
                f"[cyan]Merging {total} [italic]{ds}[/italic] datasets into 1",
                total=total,
                unit="merge",
            )

        failed_items: defaultdict[list[Failure]] = defaultdict(list)
        dynac = DynamicAsCompleted(futures)

        # in-dataset merging loop, we merge first within datasets to avoid large accumulators in memory
        # some reasonable value for the batch_size:
        # yield in batches of treereduction, and we want at least 1 item per batch
        batch_size = min(
            treereduction, max(int(len(futures) / 100), 1)
        )  # this is heuristic, can be tuned
        for batch in dynac.iter_batches(batch_size=batch_size):
            for future in batch:
                ds = key2ds[future.key]

                # subtract from todo
                if not future.key.startswith("accumulate-"):
                    ds2todo[ds] -= 1

                # get buffer
                buf = ds2buf[ds]

                if failed_future(future):
                    # let merge failures raise right away
                    if future.key.startswith("accumulate-"):
                        raise future.exception() from None

                    # just collect bad futures coming from the processing step, do not merge them
                    reason = future.result()
                    item = futurekey2item[future.key]
                    failure = Failure(item=item, reason=reason)

                    # append to failed items
                    failed_items[ds].append(failure)

                    # all failed for this dataset
                    if len(buf) == 0 and ds2todo[ds] == 0:
                        del ds2todo[ds]
                        del ds2buf[ds]

                    # nothing to merge, skip
                    continue

                # update progress bars only for successful items
                if future.key.startswith("accumulate-"):
                    # merging task
                    pbars[ds].update(dataset_merge_tasks[ds], advance=1)
                else:
                    pbars[_processing_sentinel].update(processing_task, advance=1)

                # add future to buffer for merging
                if future in buf:
                    raise ReduceSchedulingError("Future already in buffer!")
                buf.append(future)

                # if this is the last item for this dataset, skip merging
                # as we schedule it for final cross-dataset merging
                if len(buf) == 1 and ds2todo[ds] == 0:
                    continue

                # submit treereduction merge if we have enough items
                if len(buf) >= min(ds2todo[ds], treereduction) and len(buf) > 1:
                    work = client.submit(
                        acc,
                        buf,
                        key=f"accumulate-{tokenize(buf)}",
                        priority=1,
                    )

                    # release explicit retention
                    for f in buf:
                        f.release()

                    # add merged item to key2item, just use the first one of the
                    # buffer in order to access the dataset later again
                    key2ds[work.key] = ds

                    # reset buffer
                    buf.clear()

                    # add back to the ac, recursively merge
                    dynac.add(work)

        del dynac

        # make sure there's only 1 future per dataset in the buffer for the final merge
        final_merge_futures = {}
        for ds, todo in ds2todo.items():
            buf = ds2buf[ds]
            if todo != 0 or len(buf) != 1:
                msg = f"dataset {ds} has {len(buf)} items in merge-buffer (should only be 1); chunks left to merge: {todo}"
                raise ReduceSchedulingError(msg)
            pbars[ds].update(dataset_merge_tasks[ds], advance=1)
            final_merge_futures[ds] = buf[0]

        final_total = 0
        final_merge_task = pbars[_final_merge_sentinel].add_task(
            f"[cyan]Merging {len(final_merge_futures)} merged datasets [italic](final)",
            total=total,
            unit="merge",
        )

        # not needed anymore
        del ds2buf, ds2todo

        # final merge across datasets
        buf = []

        dynac = DynamicAsCompleted(final_merge_futures.values())
        for future in dynac:
            if failed_future(future):
                raise future.exception()

            if future not in final_merge_futures.values():
                # final merge progress
                pbars[_final_merge_sentinel].update(
                    final_merge_task,
                    advance=1,
                    total=final_total,
                )

            buf.append(future)
            if len(buf) >= min(len(buf), treereduction) and len(buf) > 1:
                future = client.submit(
                    acc,
                    buf,
                    key=f"accumulate-{tokenize(buf)}",
                    priority=2,
                )

                # release explicit retention
                for f in buf:
                    f.release()

                # add merged item to key2item, just use the first one of the
                # buffer in order to access the dataset later again
                key2ds[future.key] = ds

                # reset buffer
                buf.clear()

                # add back to the ac, recursively merge
                dynac.add(future)

                # add one to the pbar
                final_total += 1

        del dynac

        # final result
        assert len(buf) == 1
        future = buf[0]
        return future, failed_items


if __name__ == "__main__":
    # Run with: `python -m util._dask`
    from coffea.nanoevents import NanoAODSchema
    from dask.distributed import Client, LocalCluster

    workitems = [
        WorkItem(
            filename="../coffea/tests/samples/nano_dy.root",
            treename="Events",
            entrystart=i * 10,
            entrystop=(i + 1) * 10,
            dataset="dy",
            usermeta={"dataset": "dy"},
            fileuuid="1234abcd",
        )
        for i in range(4)
    ] + [
        WorkItem(
            filename="../coffea/tests/samples/nano_dimuon.root",
            treename="Events",
            entrystart=i * 10,
            entrystop=(i + 1) * 10,
            dataset="data",
            usermeta={"dataset": "data"},
            fileuuid="5678efgh",
        )
        for i in range(4)
    ]

    def process(events: ak.Array) -> ak.Array:
        import random

        if random.random() < 0.4:
            raise ValueError("Random failure during processing!")
        return ak.mean(events.Jet.pt)

    with (
        LocalCluster(n_workers=4, threads_per_worker=1) as cluster,
        Client(cluster) as client,
    ):
        # map and keep track of future <-> workitem mapping
        futures, futurekey2item = dask_map(
            process,
            workitems,
            client=client,
            NanoEventsFactory_kwargs={
                "preload": lambda b: b.name in {"Jet_pt", "Jet_eta"},
                "schemaclass": NanoAODSchema,
            },
        )

        # perform reduction and track failures
        final_future, failed_items = dask_reduce(
            futures,
            futurekey2item=futurekey2item,
            client=client,
            treereduction=3,
        )

        coffea_console.print("Failed items:", failed_items)
        result = final_future.result()
        coffea_console.print("Output:", result["out"])
        coffea_console.print("Metrics:", result["report"])
