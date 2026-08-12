"""
Differentiable Ecological ODE Layer (DEcoDE)
=============================================
Embeds a modified logistic growth equation as a differentiable constraint
via torchdiffeq, enforcing temporal biomass consistency.

ODE:  dB/dt = r(c_t) * B * (1 - B/K(c_t)) - d(c_t) * B

Where:
  r(c_t) = Softplus(MLP_r(c_t))    -- intrinsic growth rate
  K(c_t) = Softplus(MLP_K(c_t))+e  -- carrying capacity  
  d(c_t) = Sigmoid(MLP_d(c_t))*d_max -- mortality rate

Parameters are learned MLPs conditioned on ERA5 climate vectors.

Expected ecological ranges (mangrove reference):
  r: 0.01-0.15 yr^-1   (Komiyama et al., 2008)
  K: 150-400 Mg/ha      (Simard et al., 2019)
  d: increases under high salinity / low precipitation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Climate-Conditioned Parameter Networks
# ---------------------------------------------------------------------------

class GrowthRateNet(nn.Module):
    """MLP_r: climate vector -> intrinsic growth rate r > 0."""
    def __init__(self, climate_dim=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(climate_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Softplus()  # ensures r > 0
        )
        # Initialize bias so initial r ~ 0.08 yr^-1 (mid-literature range)
        # Softplus(x) ~ x for x > 1, so set final bias ~ 0.08
        with torch.no_grad():
            self.net[-2].bias.fill_(0.08)
    
    def forward(self, climate):
        """climate: (B, climate_dim) -> r: (B, 1)"""
        return self.net(climate)


class CarryingCapacityNet(nn.Module):
    """MLP_K: climate vector -> carrying capacity K > epsilon."""
    def __init__(self, climate_dim=3, hidden=32, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(
            nn.Linear(climate_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Softplus()  # ensures K > 0
        )
        # Initialize so initial K ~ 250 Mg/ha (mid-literature: 150-400)
        with torch.no_grad():
            self.net[-2].bias.fill_(249.0)  # Softplus(249) ~ 249, + epsilon=1 -> 250
    
    def forward(self, climate):
        """climate: (B, climate_dim) -> K: (B, 1)"""
        return self.net(climate) + self.epsilon


class MortalityRateNet(nn.Module):
    """MLP_d: climate vector -> mortality rate d in [0, d_max]."""
    def __init__(self, climate_dim=3, hidden=32, d_max=0.3):
        super().__init__()
        self.d_max = d_max
        self.net = nn.Sequential(
            nn.Linear(climate_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # output in [0, 1]
        )
        # Initialize so initial d ~ 0.04 yr^-1 (mid-literature: 0.01-0.10)
        # Sigmoid(x) = 0.04/0.3 ~ 0.133, inv_sigmoid(0.133) ~ -1.87
        with torch.no_grad():
            self.net[-2].bias.fill_(-1.87)
    
    def forward(self, climate):
        """climate: (B, climate_dim) -> d: (B, 1)"""
        return self.net(climate) * self.d_max


# ---------------------------------------------------------------------------
# Ecological Growth ODE Function
# ---------------------------------------------------------------------------

class EcologicalODEFunc(nn.Module):
    """
    ODE right-hand-side:
      dB/dt = r * B * (1 - B/K) - d * B
    
    This is a modified logistic growth equation with mortality.
    Parameters (r, K, d) are precomputed from climate vectors and
    held fixed during ODE integration.
    """
    def __init__(self):
        super().__init__()
        # Parameters are set externally before integration
        self.r = None  # (B, 1)
        self.K = None  # (B, 1)
        self.d = None  # (B, 1)
    
    def set_params(self, r, K, d):
        self.r = r
        self.K = K
        self.d = d
    
    def forward(self, t, B):
        """
        Args:
            t: scalar time
            B: (B_batch, 1) biomass state
        Returns:
            dBdt: (B_batch, 1)
        """
        # Clamp B to avoid numerical issues (negative biomass)
        B_clamped = torch.clamp(B, min=1e-6)
        
        growth = self.r * B_clamped * (1.0 - B_clamped / self.K)
        mortality = self.d * B_clamped
        dBdt = growth - mortality
        
        return dBdt


# ---------------------------------------------------------------------------
# DEcoDE Layer (Full Module)
# ---------------------------------------------------------------------------

class DEcoDE(nn.Module):
    """
    Differentiable Ecological ODE Layer.
    
    Given:
      - Initial biomass prediction B_hat_t0 from spatial encoder
      - Climate vector c_t
      - Time span [t0, t1]
    
    Produces:
      - B_hat_t1_ode: ODE-integrated biomass prediction
      - Learned ecological parameters (r, K, d)
    
    Uses torchdiffeq for differentiable ODE solving with RK4.
    Falls back to manual Euler integration if torchdiffeq unavailable.
    """
    
    def __init__(self, climate_dim=3, hidden=32, d_max=0.3):
        super().__init__()
        self.r_net = GrowthRateNet(climate_dim, hidden)
        self.K_net = CarryingCapacityNet(climate_dim, hidden)
        self.d_net = MortalityRateNet(climate_dim, hidden, d_max)
        self.ode_func = EcologicalODEFunc()
        
        # Check torchdiffeq availability
        self._has_torchdiffeq = False
        try:
            from torchdiffeq import odeint
            self._has_torchdiffeq = True
        except ImportError:
            pass
    
    def forward(self, B_init, climate_vec, dt=1.0):
        """
        Integrate the ecological ODE forward by dt time units.
        
        Args:
            B_init:      (B, 1) initial biomass prediction
            climate_vec: (B, climate_dim) ERA5 climate context
            dt:          float, time step (default 1.0 = 1 year for prototype)
        
        Returns:
            B_ode:  (B, 1) ODE-predicted biomass at t0 + dt
            params: dict with 'r', 'K', 'd' each (B, 1)
        """
        # Compute ecological parameters from climate
        r = self.r_net(climate_vec)  # (B, 1)
        K = self.K_net(climate_vec)  # (B, 1)
        d = self.d_net(climate_vec)  # (B, 1)
        
        # Set parameters in ODE function
        self.ode_func.set_params(r, K, d)
        
        # Integrate ODE
        t_span = torch.tensor([0.0, dt], dtype=B_init.dtype, device=B_init.device)
        
        if self._has_torchdiffeq:
            from torchdiffeq import odeint
            # odeint returns (T, B, 1) where T = len(t_span)
            trajectory = odeint(
                self.ode_func, B_init, t_span,
                method='rk4',
                options={'step_size': dt / 4}  # 4 RK4 steps
            )
            B_ode = trajectory[-1]  # (B, 1) at time t0+dt
        else:
            # Manual RK4 fallback
            B_ode = self._rk4_step(B_init, dt, n_steps=4)
        
        params = {'r': r, 'K': K, 'd': d}
        return B_ode, params
    
    def _rk4_step(self, B, dt, n_steps=4):
        """Manual RK4 integration as fallback."""
        h = dt / n_steps
        for _ in range(n_steps):
            k1 = self.ode_func(0, B)
            k2 = self.ode_func(0, B + 0.5 * h * k1)
            k3 = self.ode_func(0, B + 0.5 * h * k2)
            k4 = self.ode_func(0, B + h * k3)
            B = B + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            B = torch.clamp(B, min=1e-6)  # physical constraint: B >= 0
        return B


# ---------------------------------------------------------------------------
# Temporal Consistency Loss
# ---------------------------------------------------------------------------

def ode_consistency_loss(B_spatial, B_ode):
    """
    Penalizes discrepancy between spatial encoder prediction and
    ODE-integrated prediction.
    
    L_ODE = (1/N) * sum || B_spatial - B_ode ||^2
    
    This acts as a soft regularizer, NOT a replacement for task loss.
    
    Args:
        B_spatial: (B, 1) spatial encoder's AGB prediction
        B_ode:     (B, 1) ODE-integrated AGB prediction
    
    Returns:
        loss: scalar
    """
    return F.mse_loss(B_spatial, B_ode)
