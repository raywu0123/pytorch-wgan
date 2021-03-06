import torch

from ..utils import pairwise_euclidean_square_distance


def test_pairwise_euclidean_distance():
    function = pairwise_euclidean_square_distance
    a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    b = torch.Tensor([[7, 8], [9, 10], [11, 12]])
    pairwise_distance = function(a, b)
    expected_result = torch.Tensor([
        [72, 128, 200],
        [32, 72, 128],
        [8, 32, 72],
    ])
    assert torch.all(pairwise_distance == expected_result)
