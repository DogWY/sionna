#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the AWGN channel block"""

import pytest
import torch

from sionna.phy import dtypes
from sionna.phy.channel import AWGN


class TestAWGN:
    """Tests for the AWGN class"""

    def test_output_shape(self, device, precision):
        """Test that output shape matches input shape"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        input_shape = [64, 32, 16]
        x = torch.randn(input_shape, dtype=cdtype, device=device)
        no = 0.1

        awgn = AWGN(precision=precision, device=device)
        y = awgn(x, no)

        assert list(y.shape) == input_shape

    def test_output_dtype(self, device, precision):
        """Test that output dtype matches input dtype"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(64, 16, dtype=cdtype, device=device)
        no = 0.1

        awgn = AWGN(precision=precision, device=device)
        y = awgn(x, no)

        assert y.dtype == cdtype
        assert y.is_complex()

    def test_noise_variance_scalar(self, device):
        """Test that noise variance is correct for scalar no"""
        # Use double precision for more accurate variance estimation
        cdtype = dtypes["double"]["torch"]["cdtype"]
        batch_size = 10000
        num_symbols = 100
        x = torch.zeros(batch_size, num_symbols, dtype=cdtype, device=device)
        no = 0.5

        awgn = AWGN(precision="double", device=device)
        y = awgn(x, no)

        # Noise should be the only component in y since x is zeros
        # Variance of the noise should be approximately no
        actual_var = torch.var(y).item()
        # Allow 10% tolerance due to statistical sampling
        assert abs(actual_var - no) / no < 0.1

    def test_noise_variance_tensor(self, device):
        """Test that noise variance is correct for tensor no with batch broadcasting"""
        cdtype = dtypes["double"]["torch"]["cdtype"]
        num_batches = 4
        samples_per_batch = 10000
        x = torch.zeros(num_batches, samples_per_batch, dtype=cdtype, device=device)
        # Different noise variance for each batch (broadcasted along last axis)
        no = torch.tensor([0.1, 0.5, 1.0, 2.0], device=device)

        awgn = AWGN(precision="double", device=device)
        y = awgn(x, no)

        # Check variance for each batch
        for i, expected_var in enumerate(no.tolist()):
            actual_var = torch.var(y[i, :]).item()
            # Allow 10% tolerance due to statistical sampling
            assert abs(actual_var - expected_var) / expected_var < 0.1

    def test_broadcasting_rank(self, device, precision):
        """Test that no is correctly broadcast to match x rank"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        # 4D input tensor
        x = torch.randn(8, 4, 16, 32, dtype=cdtype, device=device)
        # 1D noise variance (should be broadcast)
        no = torch.tensor([0.1], device=device)

        awgn = AWGN(precision=precision, device=device)
        y = awgn(x, no)

        assert y.shape == x.shape

    def test_zero_noise(self, device, precision):
        """Test that zero noise returns input unchanged (statistically)"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(64, 16, dtype=cdtype, device=device)
        no = 0.0

        awgn = AWGN(precision=precision, device=device)
        y = awgn(x, no)

        # With zero noise, output should equal input
        assert torch.allclose(y, x)

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly"""
        awgn_channel = AWGN(device=device)
        x = torch.randn(64, 16, dtype=torch.complex64, device=device)
        no = 0.1
        y = awgn_channel(x, no)
        assert y.shape == torch.Size([64, 16])
        assert y.is_complex()

    def test_per_sample_noise(self, device):
        """Test applying different noise to each sample in a batch"""
        cdtype = dtypes["double"]["torch"]["cdtype"]
        batch_size = 10000
        x = torch.zeros(batch_size, dtype=cdtype, device=device)
        # Different noise for even and odd samples
        no = torch.zeros(batch_size, device=device)
        no[::2] = 0.1  # Even samples
        no[1::2] = 1.0  # Odd samples

        awgn = AWGN(precision="double", device=device)
        y = awgn(x, no)

        # Check variance for even and odd samples separately
        var_even = torch.var(y[::2]).item()
        var_odd = torch.var(y[1::2]).item()

        # Allow 15% tolerance due to statistical sampling with fewer samples
        assert abs(var_even - 0.1) / 0.1 < 0.15
        assert abs(var_odd - 1.0) / 1.0 < 0.15

    def test_device_placement(self, device, precision):
        """Test that output is on the correct device"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(32, 16, dtype=cdtype, device=device)
        no = 0.1

        awgn = AWGN(precision=precision, device=device)
        y = awgn(x, no)

        assert y.device == x.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda_updates_logical_device_and_output_device(self):
        """Test that AWGN follows .to(cuda) even without other tensor state."""
        awgn = AWGN(device="cpu")

        awgn.to("cuda:0")

        x = torch.randn(8, 4, dtype=torch.complex64, device="cuda:0")
        y = awgn(x, 0.1)
        assert awgn.device == "cuda:0"
        assert y.device == torch.device("cuda:0")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_parent_module_to_cuda_updates_child_awgn(self):
        """Test that a parent nn.Module moves child AWGN's device buffer."""

        class Parent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.awgn = AWGN(device="cpu")

            def forward(self, x, no):
                return self.awgn(x, no)

        parent = Parent()
        parent.to("cuda:0")

        x = torch.randn(8, 4, dtype=torch.complex64, device="cuda:0")
        y = parent(x, 0.1)
        assert parent.awgn.device == "cuda:0"
        assert y.device == torch.device("cuda:0")
