#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""5G NR transport block decoding for Sionna PHY."""

from typing import Optional, Tuple, Union
import numpy as np
import torch

from sionna.phy import Block
from sionna.phy.fec.crc import CRCDecoder
from sionna.phy.fec.scrambling import Descrambler
from sionna.phy.fec.ldpc import LDPC5GDecoder
from sionna.phy.nr.tb_encoder import TBEncoder


__all__ = ["TBDecoder"]


class TBDecoder(Block):
    # pylint: disable=line-too-long
    r"""5G NR transport block (TB) decoder as defined in TS 38.214
    :cite:p:`3GPPTS38214`.

    The transport block decoder takes as input a sequence of noisy channel
    observations and reconstructs the corresponding `transport block` of
    information bits. The detailed procedure is described in TS 38.214
    :cite:p:`3GPPTS38214` and TS 38.211 :cite:p:`3GPPTS38211`.

    :param encoder: Associated transport block encoder used for encoding
        of the signal.
    :param num_bp_iter: Number of BP decoder iterations. Defaults to 20.
    :param cn_update: Check node update rule for BP decoding.
        One of `"boxplus-phi"`, `"boxplus"`, `"minsum"`,
        `"offset-minsum"`, `"identity"`, or a callable.
        If a callable is provided, it will be used instead as CN update.
        Defaults to `"boxplus-phi"`.
    :param vn_update: Variable node update rule for BP decoding.
        One of `"sum"`, `"identity"`, or a callable.
        If a callable is provided, it will be used instead as VN update.
        Defaults to `"sum"`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.

    :input inputs: [..., num_coded_bits], `torch.float`.
        2+D tensor containing channel logits/LLR values of the (noisy)
        channel observations.

    :output b_hat: [..., target_tb_size], `torch.float`.
        2+D tensor containing hard decided bit estimates of all information
        bits of the transport block.

    :output tb_crc_status: [...], `torch.bool`.
        Transport block CRC status indicating if a transport block was
        (most likely) correctly recovered. Note that false positives are
        possible.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.nr import TBEncoder, TBDecoder

        # Create encoder and decoder
        encoder = TBEncoder(
            target_tb_size=1000,
            num_coded_bits=2000,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            n_rnti=1,
            n_id=1
        )
        decoder = TBDecoder(encoder, num_bp_iter=20)

        # Encode and decode
        bits = torch.randint(0, 2, (10, 1000), dtype=torch.float32)
        coded = encoder(bits)
        llr = 10.0 * (2.0 * coded - 1.0)  # High SNR LLRs
        bits_hat, crc_ok = decoder(llr)
        print(bits_hat.shape, crc_ok.shape)
        # torch.Size([10, 1000]) torch.Size([10])
    """

    def __init__(
        self,
        encoder: TBEncoder,
        num_bp_iter: int = 20,
        cn_update: Union[str, callable] = "boxplus-phi",
        vn_update: Union[str, callable] = "sum",
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(encoder, TBEncoder):
            raise TypeError("encoder must be TBEncoder.")
        self._tb_encoder = encoder

        self._num_cbs = encoder.num_cbs

        # Initialize BP decoder
        self._decoder = LDPC5GDecoder(
            encoder=encoder.ldpc_encoder,
            num_iter=num_bp_iter,
            cn_update=cn_update,
            vn_update=vn_update,
            hard_out=True,
            return_infobits=True,
            precision=precision,
            device=device,
        )

        # Initialize descrambler
        if encoder.scrambler is not None:
            self._descrambler = Descrambler(
                encoder.scrambler,
                binary=False,
                precision=precision,
                device=device,
            )
        else:
            self._descrambler = None

        # Initialize CRC decoders
        self._tb_crc_decoder = CRCDecoder(
            encoder.tb_crc_encoder,
            precision=precision,
            device=device,
        )

        if encoder.cb_crc_encoder is not None:
            self._cb_crc_decoder = CRCDecoder(
                encoder.cb_crc_encoder,
                precision=precision,
                device=device,
            )
        else:
            self._cb_crc_decoder = None

        # Cache output_perm_inv on decoder's device to avoid .to() on every call
        self.register_buffer(
            "_output_perm_inv",
            encoder.output_perm_inv.to(self.device),
            persistent=False,
        )

    #########################################
    # Public methods and properties
    #########################################

    @property
    def tb_size(self) -> int:
        """Number of information bits per TB."""
        return self._tb_encoder.tb_size

    @property
    def k(self) -> int:
        """Number of input information bits. Equals TB size."""
        return self._tb_encoder.tb_size

    @property
    def n(self) -> int:
        """Total number of output codeword bits."""
        return self._tb_encoder.n

    def build(self, input_shape: tuple) -> None:
        """Test input shapes for consistency."""
        if input_shape[-1] != self.n:
            raise ValueError(f"Invalid input shape. Expected input length is {self.n}.")

    def call(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transport block decoding."""
        input_shape = list(inputs.shape)
        llr_ch = inputs.float()

        llr_ch = llr_ch.reshape(-1, self._tb_encoder.num_tx, self._tb_encoder.n)

        # Undo scrambling
        if self._descrambler is not None:
            llr_scr = self._descrambler(llr_ch)
        else:
            llr_scr = llr_ch

        # Undo CB interleaving and puncturing
        num_fillers = (self._tb_encoder.ldpc_encoder.n * self._tb_encoder.num_cbs
                       - self._tb_encoder.cw_lengths_sum)
        filler_shape = list(llr_scr.shape)
        filler_shape[-1] = num_fillers
        fillers = torch.zeros(filler_shape, dtype=llr_scr.dtype, device=llr_scr.device)
        llr_int = torch.cat([llr_scr, fillers], dim=-1)
        llr_int = torch.index_select(llr_int, -1, self._output_perm_inv)

        # Undo CB concatenation
        llr_cb = llr_int.reshape(
            -1, self._tb_encoder.num_tx, self._num_cbs,
            self._tb_encoder.ldpc_encoder.n)

        # LDPC decoding
        u_hat_cb = self._decoder(llr_cb)

        # CB CRC removal
        if self._cb_crc_decoder is not None:
            u_hat_cb_crc, _ = self._cb_crc_decoder(u_hat_cb)
        else:
            u_hat_cb_crc = u_hat_cb

        # Undo CB segmentation
        u_hat_tb = u_hat_cb_crc.reshape(
            -1, self._tb_encoder.num_tx,
            self.tb_size + self._tb_encoder.tb_crc_encoder.crc_length)

        # TB CRC removal
        u_hat, tb_crc_status = self._tb_crc_decoder(u_hat_tb)

        # Restore input shape
        output_shape = input_shape.copy()
        output_shape[-1] = self.tb_size
        u_hat = u_hat.reshape(output_shape)

        # Also reshape CRC status
        crc_shape = input_shape.copy()
        crc_shape[-1] = 1
        tb_crc_status = tb_crc_status.reshape(crc_shape)

        # Handle tb_size vs target_tb_size mismatch due to quantization
        if self._tb_encoder.k_padding > 0:
            # tb_size > target_tb_size: remove zero-padding
            u_hat = u_hat[..., :-self._tb_encoder.k_padding]
        elif self._tb_encoder.k_padding < 0:
            # tb_size < target_tb_size: pad with zeros to match target_tb_size
            padding_shape = list(u_hat.shape)
            padding_shape[-1] = -self._tb_encoder.k_padding
            padding = torch.zeros(padding_shape, dtype=u_hat.dtype, device=u_hat.device)
            u_hat = torch.cat([u_hat, padding], dim=-1)

        # Cast to output dtype
        u_hat = u_hat.to(self.dtype)
        tb_crc_status = tb_crc_status.squeeze(-1).bool()

        return u_hat, tb_crc_status
