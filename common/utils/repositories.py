import pyarrow as pa
import pyarrow.compute as pc
from typing import Iterable, Any


def list_contains_mask(list_col: pa.Array | pa.ChunkedArray, value: Any) -> pa.BooleanArray:
    """
    Return a boolean mask of length == len(list_col) indicating whether each list row contains `value`.
    - Null list rows are treated as False.
    - Works for ChunkedArray (combines chunks).
    """
    # Normalize to a single Array
    if isinstance(list_col, pa.ChunkedArray):
        arr = list_col.combine_chunks()
    else:
        arr = list_col

    # Fast path: empty table
    if len(arr) == 0:
        return pa.array([], type=pa.bool_())

    # Flatten the list elements
    flat = pc.list_flatten(arr)                        # elements array
    parent_idx = pc.list_parent_indices(arr)           # same length as `flat`

    # Cast value to the elements' dtype
    val_scalar = pa.scalar(value, type=flat.type) if not isinstance(value, pa.Scalar) else value

    # Which flattened elements match `value`?
    matches_flat = pc.equal(flat, val_scalar)          # bool per element

    # Map matching elements back to their parent row indices
    nz = pc.indices_nonzero(matches_flat)              # int64 indices into `flat`
    matching_parents = parent_idx.take(nz)
    unique_parents = pc.unique(matching_parents)

    # Build row index array [0..n-1] and test membership
    row_indices = pa.array(range(len(arr)), type=unique_parents.type)
    mask = pc.is_in(row_indices, unique_parents)

    # Treat null list rows as False
    if pc.any(pc.is_null(arr)).as_py():
        mask = pc.and_(mask, pc.invert(pc.is_null(arr)))

    return mask


def list_contains_any_mask(list_col: pa.Array | pa.ChunkedArray, values: Iterable[Any]) -> pa.BooleanArray:
    """
    OR of `list_contains_mask` for multiple values.
    If `values` is empty, returns an all-True mask (i.e., keep all rows).
    """
    values = list(values)
    if not values:
        return pa.array([True] * len(list_col), type=pa.bool_())
    mask = None
    for v in values:
        m = list_contains_mask(list_col, v)
        mask = m if mask is None else pc.or_(mask, m)
    return mask


def list_contains_all_mask(list_col: pa.Array | pa.ChunkedArray, values: Iterable[Any]) -> pa.BooleanArray:
    """
    AND of `list_contains_mask` for multiple values.
    If `values` is empty, returns an all-True mask (i.e., keep all rows).
    """
    values = list(values)
    if not values:
        return pa.array([True] * len(list_col), type=pa.bool_())
    mask = None
    for v in values:
        m = list_contains_mask(list_col, v)
        mask = m if mask is None else pc.and_(mask, m)
    return mask
