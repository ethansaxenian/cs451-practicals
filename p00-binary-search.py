from typing import List, Optional

# Turn this on for loud print-statements.
DEBUG = False


def TODO(for_what: str) -> None:
    """Because crashing should be legible."""
    raise ValueError(f"TODO: {for_what}")


def __binary_search_rec(data: List[int], query: int, left: int,
                        right: int) -> Optional[int]:
    """
    (private) Return the position of the element 'query' in 'data', or None if not present.
    Starts at 'left' (inclusive) and goes to 'right' (exclusive).
    """
    # Compute the midpoint between the [left, right) indices.
    mid = (right - left) // 2 + left

    if DEBUG:
        print(f"debugging: query={query}, left={left}, mid={mid}, right={right}")

    if left >= right:
        # base-case: no more to check
        return None
    elif query == data[mid]:
        # base-case: found it!
        return mid
    elif query < data[mid]:
        return __binary_search_rec(data, query, left, mid)
    else:
        return __binary_search_rec(data, query, mid+1, right)


def binary_search(data: List[int], query: int) -> Optional[int]:
    """(public) return the position of the element 'query' in 'data', or None if not present."""
    return __binary_search_rec(data, query, 0, len(data))


if __name__ == '__main__':
    example = [11, 22, 33, 44, 55, 66, 77, 88, 99]
    assert (3 == binary_search(example, 44))
    assert (None == binary_search(example, 100))
    assert (None == binary_search(example, 5))
    assert (None == binary_search(example, 35))
