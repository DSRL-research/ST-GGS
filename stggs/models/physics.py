"""Differentiable branch flows and Kirchhoff residual from per-branch angle differences.

For a branch f -> t with series admittance Ys = 1/(r + jx), charging b and tap
ratio T = tap * exp(j * shift):

    S_from = vm_f^2 conj(Yff) + vm_f vm_t conj(Yft) exp( j dtheta)
    S_to   = vm_t^2 conj(Ytt) + vm_t vm_f conj(Ytf) exp(-j dtheta)

with dtheta = theta_f - theta_t. Kirchhoff at a bus reads p_inj = sum of branch P
and q_inj + q_shunt = sum of branch Q.
"""
import numpy as np
import torch
import torch.nn as nn


class BranchPhysics(nn.Module):
    """Per-branch admittance terms of one topology, held as non-persistent buffers."""

    def __init__(self, topology):
        super().__init__()
        r = np.asarray(topology.line_r, dtype=np.float64)
        x = np.asarray(topology.line_x, dtype=np.float64)
        b = np.asarray(topology.line_b, dtype=np.float64)
        tap = np.asarray(topology.line_tap, dtype=np.float64)
        tap = np.where(tap == 0.0, 1.0, tap)
        shift = np.deg2rad(np.asarray(topology.line_shift, dtype=np.float64))

        impedance = r + 1j * x
        impedance = np.where(np.abs(impedance) < 1e-12, 1e-12 + 0j, impedance)
        series = 1.0 / impedance
        ratio = tap * np.exp(1j * shift)

        admittances = {
            "yff": (series + 1j * b / 2.0) / (ratio * np.conj(ratio)),
            "yft": -series / np.conj(ratio),
            "ytf": -series / ratio,
            "ytt": series + 1j * b / 2.0,
        }
        for name, values in admittances.items():
            self.register_buffer(f"{name}_r", torch.tensor(values.real, dtype=torch.float32), persistent=False)
            self.register_buffer(f"{name}_i", torch.tensor(values.imag, dtype=torch.float32), persistent=False)
        self.register_buffer("f", torch.tensor(np.asarray(topology.line_or_to_subid), dtype=torch.long),
                             persistent=False)
        self.register_buffer("t", torch.tensor(np.asarray(topology.line_ex_to_subid), dtype=torch.long),
                             persistent=False)
        self.n_bus = int(topology.n_sub)
        self.n_branch = len(r)

    def flows(self, vm: torch.Tensor, dtheta: torch.Tensor):
        """(p_from, q_from, p_to, q_to) in MW/MVAr, shaped (B, n_branch)."""
        vm = vm.view(-1, self.n_bus)
        dtheta = dtheta.view(-1, self.n_branch)
        vf, vt = vm[..., self.f], vm[..., self.t]
        cos_d, sin_d = torch.cos(dtheta), torch.sin(dtheta)

        a_r = self.yft_r * cos_d + self.yft_i * sin_d
        a_i = self.yft_i * cos_d - self.yft_r * sin_d
        p_from = vf * vf * self.yff_r + vf * vt * a_r
        q_from = -(vf * vf * self.yff_i) - vf * vt * a_i

        b_r = self.ytf_r * cos_d - self.ytf_i * sin_d
        b_i = self.ytf_i * cos_d + self.ytf_r * sin_d
        p_to = vt * vt * self.ytt_r + vt * vf * b_r
        q_to = -(vt * vt * self.ytt_i) - vt * vf * b_i
        return p_from * 100.0, q_from * 100.0, p_to * 100.0, q_to * 100.0

    def kcl_residual(self, vm, dtheta, p_inj, q_inj, q_shunt):
        """Per-bus (P, Q) Kirchhoff residual in MW/MVAr, flattened over the batch."""
        p_from, q_from, p_to, q_to = self.flows(vm, dtheta)
        vm = vm.view(-1, self.n_bus)
        p_bus = torch.zeros_like(vm).index_add(-1, self.f, p_from).index_add(-1, self.t, p_to)
        q_bus = torch.zeros_like(vm).index_add(-1, self.f, q_from).index_add(-1, self.t, q_to)
        residual_p = p_inj.view(-1, self.n_bus) - p_bus
        residual_q = q_inj.view(-1, self.n_bus) + q_shunt.view(-1, self.n_bus) - q_bus
        return residual_p.reshape(-1), residual_q.reshape(-1)
