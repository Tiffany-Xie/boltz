import pytest
import torch

from boltz.model.modules.utils import split_sample_ids


@pytest.mark.parametrize(
    ("num_samples", "max_parallel_samples", "expected_chunks"),
    [
        (500, 5, 100),
        (503, 5, 101),
        (3, 5, 1),
    ],
)
def test_split_sample_ids(
    num_samples: int,
    max_parallel_samples: int,
    expected_chunks: int,
) -> None:
    sample_ids = torch.arange(num_samples)

    chunks = split_sample_ids(sample_ids, max_parallel_samples)

    assert len(chunks) == expected_chunks
    assert max(map(len, chunks)) <= max_parallel_samples
    assert torch.equal(torch.cat(chunks), sample_ids)


def test_split_sample_ids_rejects_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        split_sample_ids(torch.arange(5), 0)
