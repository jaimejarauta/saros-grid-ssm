"""Synthetic Spanish REE 50-Node Transmission Grid Data Generator and Loader.

Models realistic 15-minute dispatch interval (96 timesteps/day) generation profiles,
transmission line thermal capacities, regional renewable potential, and curtailment events.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

SPANISH_50_NODES: List[str] = [
    # Madrid & Central Load Center (Nodes 0-3)
    "Madrid-Norte", "Madrid-Sur", "Madrid-Este", "Madrid-Oeste",
    # Andalusia - High Solar (Nodes 4-11)
    "Sevilla", "Cordoba", "Granada", "Cadiz", "Huelva", "Jaen", "Malaga", "Almeria",
    # Extremadura - High Solar (Nodes 12-13)
    "Caceres", "Badajoz",
    # Castile-La Mancha - Solar & Wind (Nodes 14-18)
    "Toledo", "Ciudad-Real", "Albacete", "Cuenca", "Guadalajara",
    # Aragon - High Wind (Nodes 19-22)
    "Zaragoza", "Huesca", "Teruel", "Ebro-Central",
    # Castile and Leon - Wind & Hydro (Nodes 23-31)
    "Valladolid", "Burgos", "Leon", "Palencia", "Salamanca", "Segovia", "Soria", "Zamora", "Avila",
    # Galicia - High Wind & Hydro (Nodes 32-35)
    "Coruna", "Lugo", "Ourense", "Pontevedra-Vigo",
    # Northern Spain / Basque / Cantabria / Asturias (Nodes 36-41)
    "Bilbao", "San-Sebastian", "Vitoria", "Pamplona", "Logrono", "Santander",
    # Catalonia - Heavy Load & Nuclear (Nodes 42-45)
    "Barcelona-Norte", "Barcelona-Sur", "Tarragona", "Girona",
    # Levante & Murcia (Nodes 46-49)
    "Valencia", "Alicante", "Castellon", "Murcia"
]

def build_spanish_grid_adjacency(num_nodes: int = 50) -> np.ndarray:
    """Builds an unweighted symmetric adjacency matrix reflecting 400kV/220kV transmission trunk lines."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # Transmission corridors: (node_i, node_j)
    corridors = [
        # Madrid internal ring and central links
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 18), (1, 14), (2, 18), (3, 31), (3, 28),
        # Andalusia internal network
        (4, 5), (4, 7), (4, 8), (5, 9), (6, 9), (6, 10), (6, 11), (7, 10), (10, 11),
        # Andalusia to Extremadura & Castile-La Mancha
        (4, 13), (5, 15), (9, 15), (11, 49),
        # Extremadura
        (12, 13), (12, 14), (13, 15), (12, 27),
        # Castile-La Mancha internal & connections
        (14, 15), (14, 17), (15, 16), (16, 17), (17, 18),
        (16, 46), (16, 49), (17, 21), (18, 29),
        # Aragon (Ebro Valley wind hub)
        (19, 20), (19, 21), (19, 22), (20, 22), (21, 22),
        (19, 39), (19, 44), (20, 40), (22, 44),
        # Castile and Leon internal network
        (23, 24), (23, 25), (23, 26), (23, 27), (23, 28), (23, 30), (23, 31),
        (24, 26), (24, 29), (24, 38), (25, 26), (25, 30), (25, 33),
        (27, 30), (28, 31), (29, 40),
        # Galicia
        (32, 33), (32, 35), (33, 34), (34, 35),
        (32, 41), (33, 25), (34, 30),
        # Northern Spain
        (36, 37), (36, 38), (36, 41), (37, 38), (37, 39), (38, 39), (39, 40),
        (41, 25),
        # Catalonia
        (42, 43), (42, 44), (42, 45), (43, 44),
        (44, 48), (44, 19),
        # Levante & Murcia
        (46, 47), (46, 48), (47, 49), (48, 44), (49, 11)
    ]
    
    for u, v in corridors:
        if u < num_nodes and v < num_nodes:
            adj[u, v] = 1.0
            adj[v, u] = 1.0

    return adj

def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Computes symmetric normalized adjacency: D^{-1/2} (A + I) D^{-1/2}."""
    adj_loop = adj + np.eye(adj.shape[0], dtype=np.float32)
    degrees = np.sum(adj_loop, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ adj_loop @ d_mat

def generate_synthetic_ree_grid(
    num_days: int = 14,
    num_nodes: int = 50,
    seed: int = 2026
) -> Dict[str, np.ndarray]:
    """Generates synthetic 15-minute dispatch profiles for 50 Spanish REE transmission nodes."""
    rng = np.random.RandomState(seed)
    timesteps_per_day = 96
    total_timesteps = num_days * timesteps_per_day

    # 1. Base Capacities per Node (MW)
    # Solar capacity: high in South & Extremadura, low in North
    solar_cap = np.zeros(num_nodes, dtype=np.float32)
    # Wind capacity: high in Ebro (Aragon), Galicia, Castile & Leon
    wind_cap = np.zeros(num_nodes, dtype=np.float32)
    # Base peak demand (MW)
    base_demand = np.zeros(num_nodes, dtype=np.float32)
    # Transmission export line thermal limit (MW)
    export_limit = np.zeros(num_nodes, dtype=np.float32)

    for i in range(num_nodes):
        if i < 4:  # Madrid center: massive demand, low renewables
            solar_cap[i] = rng.uniform(50.0, 150.0)
            wind_cap[i] = rng.uniform(20.0, 80.0)
            base_demand[i] = rng.uniform(2500.0, 4200.0)
            export_limit[i] = rng.uniform(3000.0, 4500.0)
        elif 4 <= i <= 13:  # Andalusia & Extremadura: massive solar
            solar_cap[i] = rng.uniform(1800.0, 3500.0)
            wind_cap[i] = rng.uniform(300.0, 900.0)
            base_demand[i] = rng.uniform(600.0, 1400.0)
            export_limit[i] = rng.uniform(1200.0, 2000.0)  # Export bottleneck prone
        elif 14 <= i <= 18:  # Castile-La Mancha: solar + wind
            solar_cap[i] = rng.uniform(1200.0, 2400.0)
            wind_cap[i] = rng.uniform(800.0, 1800.0)
            base_demand[i] = rng.uniform(400.0, 900.0)
            export_limit[i] = rng.uniform(1400.0, 2200.0)
        elif 19 <= i <= 22:  # Aragon: high wind
            solar_cap[i] = rng.uniform(400.0, 900.0)
            wind_cap[i] = rng.uniform(2200.0, 4000.0)
            base_demand[i] = rng.uniform(500.0, 1200.0)
            export_limit[i] = rng.uniform(1500.0, 2400.0)  # Wind curtailment prone
        elif 23 <= i <= 31:  # Castile and Leon: wind & hydro
            solar_cap[i] = rng.uniform(500.0, 1100.0)
            wind_cap[i] = rng.uniform(1500.0, 3200.0)
            base_demand[i] = rng.uniform(400.0, 1000.0)
            export_limit[i] = rng.uniform(1600.0, 2500.0)
        elif 32 <= i <= 35:  # Galicia: high wind
            solar_cap[i] = rng.uniform(100.0, 400.0)
            wind_cap[i] = rng.uniform(1600.0, 2900.0)
            base_demand[i] = rng.uniform(600.0, 1300.0)
            export_limit[i] = rng.uniform(1300.0, 2100.0)
        elif 42 <= i <= 45:  # Catalonia: heavy load
            solar_cap[i] = rng.uniform(300.0, 700.0)
            wind_cap[i] = rng.uniform(400.0, 1000.0)
            base_demand[i] = rng.uniform(2000.0, 3800.0)
            export_limit[i] = rng.uniform(2500.0, 3800.0)
        else:  # Basque, Cantabria, Levante, Murcia
            solar_cap[i] = rng.uniform(400.0, 1200.0)
            wind_cap[i] = rng.uniform(400.0, 1200.0)
            base_demand[i] = rng.uniform(800.0, 1800.0)
            export_limit[i] = rng.uniform(1400.0, 2200.0)

    # 2. Time Array & Normalized Diurnal Profiles
    # 96 steps/day: step 0 = 00:00, step 28 = 07:00, step 56 = 14:00, step 84 = 21:00
    t_steps = np.arange(total_timesteps)
    day_step = t_steps % timesteps_per_day
    hour_of_day = (day_step * 15.0) / 60.0

    # Solar Diurnal Profile (Zero outside 07:00 - 21:00, peak at 14:00)
    solar_diurnal = np.maximum(0.0, np.sin(np.pi * (hour_of_day - 7.0) / 14.0))
    solar_diurnal[hour_of_day < 7.0] = 0.0
    solar_diurnal[hour_of_day > 21.0] = 0.0
    # Add non-linear parabolic zenith sharpness
    solar_diurnal = np.power(solar_diurnal, 1.3)

    # Wind Multi-Harmonic Profile (Nocturnal breeze + afternoon convection)
    wind_diurnal = (
        0.55
        + 0.25 * np.cos(2 * np.pi * (hour_of_day - 3.0) / 24.0)
        + 0.15 * np.sin(4 * np.pi * (hour_of_day - 15.0) / 24.0)
    )

    # Demand Profile (Bimodal: Morning peak 11:00, Evening peak 21:00, Night valley 05:00)
    demand_diurnal = (
        0.60
        + 0.25 * np.exp(-0.5 * ((hour_of_day - 11.5) / 2.5) ** 2)
        + 0.30 * np.exp(-0.5 * ((hour_of_day - 21.0) / 2.2) ** 2)
    )

    # Allocate feature matrices
    # in_features: [solar_potential, wind_potential, demand, export_limit]
    inputs = np.zeros((total_timesteps, num_nodes, 4), dtype=np.float32)
    # injection target: [actual_solar_mw, actual_wind_mw]
    injections = np.zeros((total_timesteps, num_nodes, 2), dtype=np.float32)
    # curtailment risk: [bottleneck_indicator]
    curtailments = np.zeros((total_timesteps, num_nodes, 1), dtype=np.float32)

    # Weather front progression across days
    weather_phase = rng.uniform(0, 2 * np.pi, size=(num_nodes,))

    for t in range(total_timesteps):
        day_idx = t // timesteps_per_day
        # Synoptic daily weather variations
        weather_cloud = 0.85 + 0.15 * np.sin(2 * np.pi * (day_idx / 7.0) + weather_phase)
        weather_wind_gust = 0.90 + 0.25 * np.cos(2 * np.pi * (day_idx / 5.0) + weather_phase)

        if solar_diurnal[t] > 0.0:
            sol_pot = solar_cap * solar_diurnal[t] * weather_cloud
            sol_pot += rng.normal(0.0, 5.0, size=(num_nodes,)) * solar_diurnal[t]
            sol_pot = np.maximum(0.0, sol_pot)
        else:
            sol_pot = np.zeros(num_nodes, dtype=np.float32)

        wnd_pot = wind_cap * wind_diurnal[t] * weather_wind_gust
        wnd_pot += rng.normal(0.0, 10.0, size=(num_nodes,))
        wnd_pot = np.maximum(0.0, wnd_pot)

        dem = base_demand * demand_diurnal[t] * (1.0 + rng.normal(0.0, 0.02, size=(num_nodes,)))

        inputs[t, :, 0] = sol_pot
        inputs[t, :, 1] = wnd_pot
        inputs[t, :, 2] = dem
        inputs[t, :, 3] = export_limit

        # Physical Grid Bottleneck & Curtailment calculation
        # Net nodal generation excess
        total_ren = sol_pot + wnd_pot
        net_excess = total_ren - dem - export_limit

        is_curtailed = net_excess > 0.0
        curtailments[t, :, 0] = is_curtailed.astype(np.float32)

        # Curtailed dispatch: excess generation cannot be injected
        curtailed_solar = np.where(is_curtailed, sol_pot - 0.6 * net_excess, sol_pot)
        curtailed_wind = np.where(is_curtailed, wnd_pot - 0.4 * net_excess, wnd_pot)

        injections[t, :, 0] = np.maximum(0.0, curtailed_solar)
        injections[t, :, 1] = np.maximum(0.0, curtailed_wind)

    adj = build_spanish_grid_adjacency(num_nodes)
    adj_norm = normalize_adjacency(adj)

    return {
        "inputs": inputs,
        "injections": injections,
        "curtailments": curtailments,
        "adjacency": adj,
        "adjacency_norm": adj_norm,
        "node_names": SPANISH_50_NODES[:num_nodes],
    }

class REESpanishGridDataset(Dataset):
    """PyTorch Dataset yielding 24-hour (96-timestep) day-ahead grid dispatch windows."""
    def __init__(
        self,
        inputs: np.ndarray,
        injections: np.ndarray,
        curtailments: np.ndarray,
        window_size: int = 96,
        stride: int = 96
    ):
        self.inputs = torch.from_numpy(inputs).float()
        self.injections = torch.from_numpy(injections).float()
        self.curtailments = torch.from_numpy(curtailments).float()
        self.window_size = window_size
        self.stride = stride

        # Number of non-overlapping or strided daily windows
        self.total_timesteps = inputs.shape[0]
        self.indices = list(range(0, self.total_timesteps - window_size + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.window_size
        x_window = self.inputs[start:end]              # (96, num_nodes, 4)
        y_inj_window = self.injections[start:end]      # (96, num_nodes, 2)
        y_curt_window = self.curtailments[start:end]   # (96, num_nodes, 1)
        return x_window, y_inj_window, y_curt_window
