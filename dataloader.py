from typing import Any, Callable, List, Sequence, Union, Tuple, Dict, Optional
import numpy as np

class DatasetProtocol:
    """Any dataset with __len__ and __getitem__ is acceptable for the dataloader."""

    def __len__(self) -> int: ... # noqa: D105, Convention in typing is to uses ellipsis (...) 
    def __getitem__(self, idx: int) -> Any: ... # noqa: ANN401, D105

def _default_collate(batch: List[Any]) -> Any:
    """Collate: to take multiple individual samples from the dataset and
    assemble them in to a single batch in the format the model expects.
    Stacks numpy arrays along a new first dimension when shapes match.
    Recursively handles tuples, lists and dicts.

    Args:
        batch (List[Any]): List of samples from the dataset.

    Returns:
        Any: Collated batch, which can be a numpy array, tuple, list

    """ # noqa: D205
    elem = batch[0]

    # numpy arrays
    if isinstance(elem, np.ndarray):
        # Verify shapes match
        shapes = [x.shape for x in batch]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(f"Array shapes differ in batch: {shapes}")
        return np.stack(batch, axis=0)

    # numbers
    if isinstance(elem, (int, float, np.integer, np.floating)):
        return np.array(batch)

    # tuples
    if isinstance(elem, tuple): 
        transposed = list(zip(*batch))
        return tuple(_default_collate(list(x)) for x in transposed)

    # lists
    if isinstance(elem, list):
        transposed = list(zip(*batch))
        return [_default_collate(list(x)) for x in transposed]

    # dicts
    if isinstance(elem, dict):
        keys = elem.keys()
        if not all(d.keys() == keys for d in batch):
            raise ValueError("All dict samples must have the same keys.") #noqa:TRY003, EM101
        return {k: _default_collate([d[k] for d in batch]) for k in keys}

    return batch  # If no known type, return as is


class Dataloader: 
    """Training data management class."""

    def __init__(self,
                 dataset: DatasetProtocol,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 collate_fn: Optional[Callable[[List[Any]], Any]] = None,
                 seed: Optional[int] = 42,
                 indices: Optional[Sequence[int]] = None,
                 ):
        if batch_size <= 0:
             raise ValueError("batch size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last 
        self.collate_fn = collate_fn or _default_collate
        self.base_seed = seed
        self.epoch = 0

        n = len(dataset)
        if indices is None:
            self.indices = np.arange(n, dtype=int)
        else:
            self.indices = np.array(indices, dtype=np.int64)
            if np.any(self.indices < 0) or np.any(self.indices >= n):
                raise ValueError("indices out of bouds")

        self._compute_num_batches()


    def _compute_num_batches(self):
        N = len(self.indices)
        if self.drop_last:
            self.num_batches = N // self.batch_size
        else: 
            self.num_batches = (N + self.base_seed - 1) // self.batch_size


    def set_epoch(self, epoch:int):
        """Set curretnt epoch (affects shuffle RNG state deterministically.)"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        order = self.indices.copy() 
        if self.shuffle:
            rng = np.random.default_rng(None if self.base_seed else self.base_seed + self.epoch)
            rng.shuffle(order)

        N = len(order)
        bs = self.batch_size
        limit = (N // bs) * bs if self.drop_last else N

        for start in range(0, limit, bs):
            end = min(start + bs, N)
            batch_indices = order[start:end].tolist() 
            batch_samples = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(batch_samples)