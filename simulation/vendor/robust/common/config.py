from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Config:
    # Mission model (Section III)
    n_tasks: int = 20
    n_mec: int = 3
    t_max: float = 1750.0
    e_max_kj: float = 500.0
    uav_altitude_m: float = 120.0
    speed_mps: float = 17.0
    flight_power_w: float = 255.0
    hover_power_w: float = 185.0
    tx_power_w: float = 0.12
    rx_power_w: float = 0.10
    mec_tx_power_w: float = 0.7
    ul_bw_hz: float = 18e6
    dl_bw_hz: float = 12e6
    fc_hz: float = 2e9
    noise_psd_w_hz: float = 10 ** ((-174.0 - 30.0) / 10.0)
    los_a: float = 9.61
    los_b: float = 0.16
    eta_los: float = 10 ** (1.0 / 10.0)
    eta_nlos: float = 10 ** (20.0 / 10.0)
    ul_min_mbps: float = 14.0
    dl_min_mbps: float = 9.333333333333334
    mec_cpu_ghz: tuple = (25.0, 25.0, 25.0)
    local_f_min_ghz: float = 1.2
    local_f_max_ghz: float = 5.0
    kappa_kj_per_gcycle_ghz2: float = 1.6e-4
    output_ratio: float = 0.08

    # Algorithm 2 parameters from the latest Section IV.
    gh_order: int = 7
    beam_width: int = 2
    saa_samples: int = 5

    # Pressure-aware MTE-Sub.  q=0 and beta=0 recover the legacy cost.
    sub_pressure_q: float = 1.0
    sub_bottleneck_beta: float = 0.5

    # Scenario controls used only by sensitivity experiments.
    uncertainty_scale: float = 1.0
    workload_scale: float = 4.5

    # Development controls for the spatial-priority uncertainty scenario.
    spatial_priority: bool = True
    sigma_min: float = 0.32
    sigma_span: float = 0.45
    priority_distance_weight: float = 0.8
    priority_offshore_weight: float = 0.2
    deadline_scale: float = 1.0
    mean_spatial_span: float = 0.0
    priority_scheme: str = "balanced"
    deadline_mode: str = "legacy"
    deadline_slack_s: float = 140.0
    deadline_factor_high: float = 1.25
    deadline_factor_low: float = 2.8

BASE_LONLAT = (113.5460, 22.2100)
MEC_LONLAT = (
    (113.5416, 22.2024),
    (113.5607, 22.1558),
    (113.5695, 22.1175),
)
REGION_CENTERS = (
    (113.5525, 22.1990),
    (113.5630, 22.1570),
    (113.5715, 22.1420),
    (113.5635, 22.1190),
)
REGION_SCALES_DEG = (
    (0.0065, 0.0070),
    (0.0075, 0.0060),
    (0.0080, 0.0050),
    (0.0090, 0.0075),
)

def with_overrides(cfg: Config, **kwargs) -> Config:
    return replace(cfg, **kwargs)
