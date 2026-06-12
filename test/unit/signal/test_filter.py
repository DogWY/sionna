#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for filter classes: Filter, CustomFilter, RaisedCosineFilter, RootRaisedCosineFilter, SincFilter"""

import numpy as np
import pytest
import torch

from sionna.phy import dtypes
from sionna.phy.signal import (
    CustomFilter,
    CustomWindow,
    RaisedCosineFilter,
    RootRaisedCosineFilter,
    SincFilter,
)


class TestCustomFilter:
    """Tests for the CustomFilter class"""

    @pytest.mark.parametrize("inp_complex", [False, True])
    @pytest.mark.parametrize("filt_complex", [False, True])
    @pytest.mark.parametrize("padding", ["valid"])
    def test_dtype(self, precision, device, inp_complex, filt_complex, padding):
        """Test the output dtype for all possible combinations of input and filter dtypes"""
        batch_size = 64
        inp_length = 1000
        span_in_symbols = 8
        samples_per_symbol = 4
        filter_length = samples_per_symbol * span_in_symbols + 1

        rdtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]

        # Generate inputs
        if inp_complex:
            inp = torch.randn(batch_size, inp_length, dtype=cdtype, device=device)
        else:
            inp = torch.randn(batch_size, inp_length, dtype=rdtype, device=device)

        # Generate filter coefficients
        if filt_complex:
            coefficients = torch.randn(filter_length, dtype=cdtype, device=device)
        else:
            coefficients = torch.randn(filter_length, dtype=rdtype, device=device)

        if inp_complex or filt_complex:
            out_dtype = cdtype
        else:
            out_dtype = rdtype

        # No windowing
        filt = CustomFilter(
            samples_per_symbol, coefficients, precision=precision, device=device
        )
        out = filt(inp, padding=padding)
        assert out.dtype == out_dtype

        # With windowing
        win_coeff = torch.randn(filter_length, dtype=rdtype, device=device)
        window = CustomWindow(
            coefficients=win_coeff, precision=precision, device=device
        )
        filt = CustomFilter(
            samples_per_symbol,
            coefficients,
            window=window,
            precision=precision,
            device=device,
        )
        out = filt(inp, padding=padding)
        assert out.dtype == out_dtype

    def test_shape(self, device):
        """Test the output shape"""
        input_shape = [16, 8, 24, 1000]
        inp = torch.randn(input_shape, device=device)

        for span_in_symbols in (7, 8):
            for samples_per_symbol in (1, 3, 4):
                filter_length = span_in_symbols * samples_per_symbol
                if (filter_length % 2) == 0:
                    filter_length = filter_length + 1

                win_coeff = torch.randn(filter_length, device=device)
                window = CustomWindow(coefficients=win_coeff, device=device)

                for win in (None, window):
                    coefficients = torch.randn(filter_length, device=device)
                    filt = CustomFilter(
                        samples_per_symbol,
                        coefficients=coefficients,
                        window=win,
                        device=device,
                    )

                    # 'valid' padding
                    out = filt(inp, "valid")
                    out_shape = input_shape[:-1] + [input_shape[-1] - filter_length + 1]
                    assert list(out.shape) == out_shape

                    # 'same' padding
                    out = filt(inp, "same")
                    out_shape = input_shape
                    assert list(out.shape) == out_shape

                    # 'full' padding
                    out = filt(inp, "full")
                    out_shape = input_shape[:-1] + [input_shape[-1] + filter_length - 1]
                    assert list(out.shape) == out_shape

    @pytest.mark.parametrize("inp_complex", [False, True])
    @pytest.mark.parametrize("fil_complex", [False, True])
    @pytest.mark.parametrize("padding", ["valid", "same", "full"])
    @pytest.mark.parametrize("span_in_symbols", [7, 8])
    @pytest.mark.parametrize("samples_per_symbol", [1, 3, 4])
    def test_computation(
        self,
        device,
        inp_complex,
        fil_complex,
        padding,
        span_in_symbols,
        samples_per_symbol,
    ):
        """Test the calculation"""
        input_length = 1000
        filter_length = span_in_symbols * samples_per_symbol
        if (filter_length % 2) == 0:
            filter_length = filter_length + 1

        rdtype = dtypes["double"]["torch"]["dtype"]
        cdtype = dtypes["double"]["torch"]["cdtype"]

        win_coeff = torch.randn(filter_length, dtype=rdtype, device=device)
        window = CustomWindow(coefficients=win_coeff, precision="double", device=device)

        for win in (None, window):
            if inp_complex:
                inp = torch.randn(1, input_length, dtype=cdtype, device=device)
            else:
                inp = torch.randn(1, input_length, dtype=rdtype, device=device)

            if fil_complex:
                fil_coeff = torch.randn(filter_length, dtype=cdtype, device=device)
            else:
                fil_coeff = torch.randn(filter_length, dtype=rdtype, device=device)

            filt = CustomFilter(
                samples_per_symbol,
                coefficients=fil_coeff,
                window=win,
                normalize=False,
                precision="double",
                device=device,
            )

            # No conjugate
            out = filt(inp, padding, conjugate=False)
            if win:
                ker_ref = fil_coeff.cpu().numpy() * win_coeff.cpu().numpy()
            else:
                ker_ref = fil_coeff.cpu().numpy()
            out_ref = np.convolve(inp.cpu().numpy()[0], ker_ref, mode=padding)
            max_err = np.max(np.abs(out.cpu().numpy()[0] - out_ref))
            assert max_err <= 1e-10

            # Conjugate
            out = filt(inp, padding, conjugate=True)
            ker_ref = np.conj(ker_ref)
            out_ref = np.convolve(inp.cpu().numpy()[0], ker_ref, mode=padding)
            max_err = np.max(np.abs(out.cpu().numpy()[0] - out_ref))
            assert max_err <= 1e-10

    @pytest.mark.parametrize("fil_complex", [False, True])
    def test_normalization(self, device, fil_complex):
        """Test the normalization"""
        span_in_symbols = 8
        samples_per_symbol = 4
        filter_length = samples_per_symbol * span_in_symbols + 1

        rdtype = dtypes["double"]["torch"]["dtype"]
        cdtype = dtypes["double"]["torch"]["cdtype"]

        if fil_complex:
            fil_coeff = torch.randn(filter_length, dtype=cdtype, device=device)
        else:
            fil_coeff = torch.randn(filter_length, dtype=rdtype, device=device)

        filt = CustomFilter(
            samples_per_symbol,
            coefficients=fil_coeff,
            normalize=True,
            precision="double",
            device=device,
        )

        x = torch.tensor([1.0], dtype=rdtype, device=device)
        out = filt(x)

        # Normalized filter should have unit energy
        energy = torch.sum(torch.abs(out) ** 2).item()
        assert abs(energy - 1.0) < 1e-5

    @pytest.mark.parametrize("fil_complex", [False, True])
    def test_conjugate_filter(self, device, fil_complex):
        """Test the conjugate filter application"""
        span_in_symbols = 8
        samples_per_symbol = 4
        filter_length = samples_per_symbol * span_in_symbols + 1

        rdtype = dtypes["double"]["torch"]["dtype"]
        cdtype = dtypes["double"]["torch"]["cdtype"]

        if fil_complex:
            fil_coeff = torch.randn(filter_length, dtype=cdtype, device=device)
        else:
            fil_coeff = torch.randn(filter_length, dtype=rdtype, device=device)

        filt = CustomFilter(
            samples_per_symbol,
            coefficients=fil_coeff,
            normalize=False,
            precision="double",
            device=device,
        )

        x = torch.tensor([1.0], dtype=rdtype, device=device)

        f = filt(x, conjugate=True)
        # Output should be the conjugate of coefficients
        expected = torch.conj(filt.coefficients) if fil_complex else filt.coefficients
        max_err = torch.max(torch.abs(f - expected)).item()
        assert max_err < 1e-10

        f = filt(x, conjugate=False)
        max_err = torch.max(torch.abs(f - filt.coefficients)).item()
        assert max_err < 1e-10

    @pytest.mark.parametrize("fil_complex", [False, True])
    def test_trainable_coefficients(self, device, fil_complex):
        """Test gradient computation if coefficients are variables"""
        batch_size = 64
        span_in_symbols = 8
        samples_per_symbol = 4
        filter_length = samples_per_symbol * span_in_symbols + 1
        input_length = 1024

        rdtype = dtypes["double"]["torch"]["dtype"]
        cdtype = dtypes["double"]["torch"]["cdtype"]

        inp = torch.randn(batch_size, input_length, dtype=cdtype, device=device)

        if fil_complex:
            fil_coeff = torch.randn(
                filter_length, dtype=cdtype, device=device, requires_grad=True
            )
        else:
            fil_coeff = torch.randn(
                filter_length, dtype=rdtype, device=device, requires_grad=True
            )

        filt = CustomFilter(
            samples_per_symbol,
            coefficients=fil_coeff,
            precision="double",
            device=device,
        )

        out = filt(inp)
        loss = torch.mean(torch.abs(out) ** 2)
        loss.backward()

        assert fil_coeff.grad is not None
        assert fil_coeff.grad.shape == torch.Size([filter_length])

        if fil_complex:
            assert fil_coeff.grad.real.sum().item() != 0
            assert fil_coeff.grad.imag.sum().item() != 0
        else:
            assert fil_coeff.grad.sum().item() != 0

    def test_parameter_coefficients_are_trainable(self, device):
        """Test that Parameter coefficients remain module parameters"""
        samples_per_symbol = 4
        filter_length = samples_per_symbol * 8 + 1
        rdtype = dtypes["double"]["torch"]["dtype"]
        coeff = torch.nn.Parameter(
            torch.randn(filter_length, dtype=rdtype, device=device)
        )

        filt = CustomFilter(
            samples_per_symbol,
            coefficients=coeff,
            precision="double",
            device=device,
        )

        params = list(filt.parameters())
        assert len(params) == 1
        assert params[0] is coeff
        assert "_coefficients" in filt.state_dict()


class TestRaisedCosineFilter:
    """Tests for the RaisedCosineFilter class"""

    def test_creation(self, device, precision):
        """Test that filter can be created with various parameters"""
        for beta in [0.0, 0.35, 1.0]:
            rc = RaisedCosineFilter(
                span_in_symbols=8,
                samples_per_symbol=4,
                beta=beta,
                precision=precision,
                device=device,
            )
            assert rc.beta == beta
            assert rc.span_in_symbols == 8
            assert rc.samples_per_symbol == 4

    def test_filter_length(self, device, precision):
        """Test that filter length is computed correctly (odd)"""
        rc = RaisedCosineFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            beta=0.35,
            precision=precision,
            device=device,
        )
        # 8 * 4 = 32, which is even, so length should be 33
        assert rc.length == 33

    def test_application(self, device, precision):
        """Test filter application"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        rc = RaisedCosineFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            beta=0.35,
            precision=precision,
            device=device,
        )
        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = rc(x, padding="same")
        assert y.shape == x.shape


class TestRootRaisedCosineFilter:
    """Tests for the RootRaisedCosineFilter class"""

    def test_creation(self, device, precision):
        """Test that filter can be created with various parameters"""
        for beta in [0.0, 0.35, 1.0]:
            rrc = RootRaisedCosineFilter(
                span_in_symbols=8,
                samples_per_symbol=4,
                beta=beta,
                precision=precision,
                device=device,
            )
            assert rrc.beta == beta
            assert rrc.span_in_symbols == 8
            assert rrc.samples_per_symbol == 4

    def test_filter_length(self, device, precision):
        """Test that filter length is computed correctly (odd)"""
        rrc = RootRaisedCosineFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            beta=0.35,
            precision=precision,
            device=device,
        )
        # 8 * 4 = 32, which is even, so length should be 33
        assert rrc.length == 33

    def test_application(self, device, precision):
        """Test filter application"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        rrc = RootRaisedCosineFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            beta=0.35,
            precision=precision,
            device=device,
        )
        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = rrc(x, padding="same")
        assert y.shape == x.shape

    def test_matched_filter(self, device, precision):
        """Test that RRC with itself forms a raised cosine (matched filter property)"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        rrc = RootRaisedCosineFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            beta=0.35,
            normalize=False,
            precision=precision,
            device=device,
        )
        # Impulse response (not normalized for easier testing)
        x = torch.zeros(100, dtype=rdtype, device=device)
        x[50] = 1.0
        y1 = rrc(x, padding="same")
        y2 = rrc(y1, padding="same")

        # The cascade of two RRC filters should approximate a raised cosine
        # Check that the output has the expected peak at the center
        peak_idx = torch.argmax(torch.abs(y2)).item()
        assert abs(peak_idx - 50) <= 1  # Peak should be near the center


class TestSincFilter:
    """Tests for the SincFilter class"""

    def test_creation(self, device, precision):
        """Test that filter can be created"""
        sinc = SincFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            precision=precision,
            device=device,
        )
        assert sinc.span_in_symbols == 8
        assert sinc.samples_per_symbol == 4

    def test_filter_length(self, device, precision):
        """Test that filter length is computed correctly (odd)"""
        sinc = SincFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            precision=precision,
            device=device,
        )
        # 8 * 4 = 32, which is even, so length should be 33
        assert sinc.length == 33

    def test_application(self, device, precision):
        """Test filter application"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        sinc = SincFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            precision=precision,
            device=device,
        )
        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = sinc(x, padding="same")
        assert y.shape == x.shape


class TestFilterACLR:
    """Tests for filter ACLR computation"""

    def test_aclr_positive(self, device, precision):
        """Test that ACLR is positive"""
        rrc = RootRaisedCosineFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            beta=0.35,
            precision=precision,
            device=device,
        )
        aclr = rrc.aclr
        assert aclr.item() > 0

    def test_aclr_differentiable(self, device):
        """Test if ACLR computation is differentiable"""
        span_in_symbols = 8
        samples_per_symbol = 4
        filter_length = samples_per_symbol * span_in_symbols + 1

        rdtype = dtypes["double"]["torch"]["dtype"]
        fil_coeff = torch.randn(
            filter_length, dtype=rdtype, device=device, requires_grad=True
        )

        filt = CustomFilter(
            samples_per_symbol,
            coefficients=fil_coeff,
            precision="double",
            device=device,
        )

        aclr = filt.aclr
        loss = torch.abs(aclr) ** 2
        loss.backward()

        assert fil_coeff.grad is not None
        assert fil_coeff.grad.shape == torch.Size([filter_length])
        assert fil_coeff.grad.sum().item() != 0
