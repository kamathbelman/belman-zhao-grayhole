"""Fast Python/NumPy simulator of gray-hole attacks in a mobile ad hoc network.

Nodes move under a mobility model, discover routes with an AODV-like protocol,
and forward CBR data packets hop by hop. Gray-hole nodes route normally but drop
a fraction (drop_ratio) of the data packets they are asked to forward. Benign
nodes also lose packets to congestion and broken links, so the forwarding ratio
is a noisy signal. Telemetry uses the same schema as the NS-3 pipeline in ../ns3.

Run: python3 manet_sim.py (smoke test). Use generate_dataset.py for the grid.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    n_nodes: int = 40
    area: float = 1000.0             # meters (square field side)
    comm_range: float = 250.0        # meters
    duration: float = 300.0          # seconds of simulated time
    dt: float = 1.0                  # seconds per simulation step
    window: float = 20.0             # seconds per telemetry window
    mobility: str = "random_waypoint"  # random_waypoint | gauss_markov | manhattan
    speed_min: float = 1.0           # m/s
    speed_max: float = 15.0          # m/s
    n_flows: int = 12                # concurrent CBR flows
    cbr_interval: float = 0.5        # seconds between data packets in a flow
    gray_fraction: float = 0.20      # fraction of nodes that are gray-holes
    drop_ratio: float = 0.4          # fraction of transit data a gray-hole drops
    benign_loss_base: float = 0.03   # baseline random loss per hop
    collision_loss: float = 0.05     # extra loss scaling with local congestion
    route_ttl: float = 15.0          # seconds a cached route stays valid
    # "static" drops at drop_ratio throughout. "onoff" is the pulsed attacker:
    # it alternates between twice drop_ratio and zero with period attack_period,
    # which keeps the same mean drop rate.
    attack_mode: str = "static"
    attack_period: float = 40.0
    seed: int = 0

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class Mobility:
    """Generates node positions over time for the chosen mobility model."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.pos = rng.uniform(0, cfg.area, size=(cfg.n_nodes, 2))
        self.vel = np.zeros((cfg.n_nodes, 2))
        # random_waypoint targets
        self.target = rng.uniform(0, cfg.area, size=(cfg.n_nodes, 2))
        self.speed = rng.uniform(cfg.speed_min, cfg.speed_max, size=cfg.n_nodes)
        # gauss-markov state
        self.gm_speed = rng.uniform(cfg.speed_min, cfg.speed_max, size=cfg.n_nodes)
        self.gm_dir = rng.uniform(0, 2 * np.pi, size=cfg.n_nodes)

    def step(self):
        cfg = self.cfg
        if cfg.mobility == "random_waypoint":
            self._random_waypoint()
        elif cfg.mobility == "gauss_markov":
            self._gauss_markov()
        elif cfg.mobility == "manhattan":
            self._manhattan()
        else:
            raise ValueError(f"unknown mobility {cfg.mobility}")
        np.clip(self.pos, 0, cfg.area, out=self.pos)
        return self.pos

    def _random_waypoint(self):
        cfg = self.cfg
        d = self.target - self.pos
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        stepv = (d / dist) * (self.speed[:, None] * cfg.dt)
        arrived = dist[:, 0] < (self.speed * cfg.dt)
        self.pos = np.where(arrived[:, None], self.target, self.pos + stepv)
        if arrived.any():
            k = int(arrived.sum())
            self.target[arrived] = self.rng.uniform(0, cfg.area, size=(k, 2))
            self.speed[arrived] = self.rng.uniform(cfg.speed_min, cfg.speed_max, size=k)

    def _gauss_markov(self):
        cfg = self.cfg
        alpha = 0.75
        mean_speed = (cfg.speed_min + cfg.speed_max) / 2
        self.gm_speed = (alpha * self.gm_speed
                         + (1 - alpha) * mean_speed
                         + np.sqrt(1 - alpha ** 2) * self.rng.normal(0, 2, cfg.n_nodes))
        self.gm_dir = (alpha * self.gm_dir
                       + (1 - alpha) * (np.pi / 2)
                       + np.sqrt(1 - alpha ** 2) * self.rng.normal(0, 0.4, cfg.n_nodes))
        self.gm_speed = np.clip(self.gm_speed, cfg.speed_min, cfg.speed_max)
        self.pos[:, 0] += self.gm_speed * np.cos(self.gm_dir) * cfg.dt
        self.pos[:, 1] += self.gm_speed * np.sin(self.gm_dir) * cfg.dt
        # bounce off walls by flipping direction component
        for ax in (0, 1):
            lo = self.pos[:, ax] < 0
            hi = self.pos[:, ax] > cfg.area
            self.gm_dir[lo | hi] = np.pi - self.gm_dir[lo | hi]

    def _manhattan(self):
        # grid streets: nodes move along axis-aligned lines, occasional turns
        cfg = self.cfg
        if not hasattr(self, "_axis"):
            self._axis = self.rng.integers(0, 2, cfg.n_nodes)  # 0=x-move,1=y-move
            self._sign = self.rng.choice([-1, 1], cfg.n_nodes)
        turn = self.rng.random(cfg.n_nodes) < 0.05
        self._axis[turn] = 1 - self._axis[turn]
        self._sign[turn] = self.rng.choice([-1, 1], int(turn.sum()))
        move = self.speed * cfg.dt * self._sign
        self.pos[np.arange(cfg.n_nodes), self._axis] += move
        # reflect at boundaries
        for ax in (0, 1):
            m = (self.pos[:, ax] < 0) | (self.pos[:, ax] > cfg.area)
            self._sign[m] *= -1


@dataclass
class NodeStats:
    # per-node counters for the current window, reset at each flush
    data_rx_forward: int = 0   # transit data packets received for forwarding
    data_forwarded: int = 0
    data_dropped: int = 0
    rreq_originated: int = 0
    rreq_forwarded: int = 0
    rrep_sent: int = 0
    data_originated: int = 0   # packets this node sourced
    data_delivered_as_dest: int = 0
    neighbor_sum: int = 0      # for the average neighbor count
    neighbor_obs: int = 0

    def reset_window(self):
        for f in self.__dataclass_fields__:
            setattr(self, f, 0)


class ManetSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.mob = Mobility(cfg, self.rng)
        n = cfg.n_nodes
        n_gray = max(1, int(round(cfg.gray_fraction * n)))
        self.is_gray = np.zeros(n, dtype=bool)
        self.is_gray[self.rng.choice(n, size=n_gray, replace=False)] = True
        self.stats = [NodeStats() for _ in range(n)]
        self.seq = [[] for _ in range(n)]          # list of per-window feature dicts
        self.route_cache = {}                       # (src,dst) -> (path, expiry_time)
        # neighbor contact counts over time, the adjacency source for HRM tier 2
        self.adj_counts = np.zeros((n, n), dtype=np.int32)
        # active flows: list of (src, dst, next_emit_time)
        self.flows = []
        self._init_flows()

    def _neighbors(self, positions):
        n = self.cfg.n_nodes
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        adj = (dist <= self.cfg.comm_range) & (dist > 0)
        return adj, dist

    def _init_flows(self):
        n = self.cfg.n_nodes
        for _ in range(self.cfg.n_flows):
            s, d = self.rng.choice(n, size=2, replace=False)
            self.flows.append([int(s), int(d), 0.0])

    def _find_route(self, adj, src, dst, t):
        key = (src, dst)
        cached = self.route_cache.get(key)
        # As in AODV, an unexpired cached route is used without checking it
        # against the current topology. A broken link is found in transit
        # (_send_data), where the packet is lost and the route is invalidated.
        if cached and cached[1] > t:
            return cached[0]
        # route discovery: BFS shortest path on the current adjacency
        n = self.cfg.n_nodes
        prev = {src: -1}
        frontier = [src]
        self.stats[src].rreq_originated += 1
        found = False
        while frontier and not found:
            nxt = []
            for u in frontier:
                nbrs = np.where(adj[u])[0]
                for v in nbrs:
                    if v not in prev:
                        prev[v] = u
                        # only intermediate nodes count a relayed RREQ
                        if v != dst:
                            self.stats[v].rreq_forwarded += 1
                        if v == dst:
                            found = True
                            break
                        nxt.append(v)
                if found:
                    break
            frontier = nxt
        if dst not in prev:
            return None
        path = []
        u = dst
        while u != -1:
            path.append(u)
            u = prev[u]
        path.reverse()
        # rreq_forwarded holds relayed RREQs only. rreq_originated is tracked
        # separately and is not a model feature (same split as the NS-3 data).
        # Each node on the reverse path sends an RREP. Benign and gray nodes are
        # counted alike: no eager route advertisement is modeled.
        for node in path[1:]:
            self.stats[node].rrep_sent += 1
        self.route_cache[key] = (path, t + self.cfg.route_ttl)
        return path

    def _send_data(self, path, adj, dist, degree):
        cfg = self.cfg
        src, dst = path[0], path[-1]
        self.stats[src].data_originated += 1
        # walk the path hop by hop; a drop at any hop loses the packet
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            forwarder = u if i > 0 else None  # src just originates; intermediates forward
            if i > 0:
                st = self.stats[u]
                st.data_rx_forward += 1
                if self.is_gray[u]:
                    # gray-hole: drop at drop_ratio plus a small base loss
                    rate = cfg.drop_ratio
                    if cfg.attack_mode == "onoff":
                        # pulsed attacker: twice drop_ratio, then zero
                        on = (self._t % cfg.attack_period) < (cfg.attack_period / 2)
                        rate = 2.0 * cfg.drop_ratio if on else 0.0
                    p_drop = rate + cfg.benign_loss_base * 0.5
                else:
                    # benign loss grows with local congestion (node degree)
                    congestion = cfg.collision_loss * (degree[u] / max(1, cfg.n_nodes * 0.15))
                    p_drop = cfg.benign_loss_base + min(0.15, congestion)
                if self.rng.random() < p_drop:
                    st.data_dropped += 1
                    return False
                else:
                    st.data_forwarded += 1
            # Stale route: the next hop moved out of range after route discovery.
            # The loss is charged to the upstream node, as a watchdog-style
            # observer would see it, and the route is invalidated.
            if not adj[u, v]:
                if i > 0:
                    self.stats[u].data_dropped += 1
                    self.stats[u].data_forwarded -= 1
                self.route_cache.pop((src, dst), None)
                return False
        self.stats[dst].data_delivered_as_dest += 1
        return True

    def _flush_window(self):
        for i, st in enumerate(self.stats):
            rx = st.data_rx_forward
            # Ratios are defined over resolved outcomes (forwarded + dropped), so
            # they lie in [0,1]. Every packet resolves within its step here, so
            # rx == forwarded + dropped. The NS-3 data uses the same definition.
            den = st.data_forwarded + st.data_dropped
            fwd_ratio = (st.data_forwarded / den) if den > 0 else 1.0
            drop_ratio = (st.data_dropped / den) if den > 0 else 0.0
            assert rx == den, "bootstrap invariant: rx == forwarded + dropped"
            avg_nbr = (st.neighbor_sum / st.neighbor_obs) if st.neighbor_obs else 0.0
            ctrl = st.rreq_forwarded + st.rrep_sent
            feat = {
                "fwd_ratio": fwd_ratio,
                "drop_ratio": drop_ratio,
                "data_rx_forward": rx,
                "data_forwarded": st.data_forwarded,
                "data_dropped": st.data_dropped,
                "rreq_forwarded": st.rreq_forwarded,
                "rrep_sent": st.rrep_sent,
                "data_originated": st.data_originated,
                "avg_neighbors": avg_nbr,
                "ctrl_to_data": ctrl / (st.data_forwarded + 1),
            }
            self.seq[i].append(feat)
            st.reset_window()

    def run(self):
        cfg = self.cfg
        t = 0.0
        next_window = cfg.window
        self._t = 0.0
        while t < cfg.duration:
            self._t = t
            positions = self.mob.step()
            adj, dist = self._neighbors(positions)
            degree = adj.sum(1)
            self.adj_counts += adj.astype(np.int32)
            for i in range(cfg.n_nodes):
                self.stats[i].neighbor_sum += int(degree[i])
                self.stats[i].neighbor_obs += 1
            for flow in self.flows:
                s, d, nxt_emit = flow
                if t >= nxt_emit:
                    path = self._find_route(adj, s, d, t)
                    if path and len(path) > 1:
                        self._send_data(path, adj, dist, degree)
                    flow[2] = t + cfg.cbr_interval
            if t >= next_window:
                self._flush_window()
                next_window += cfg.window
            t += cfg.dt
        self._flush_window()
        return self._collect()

    def _collect(self):
        n = self.cfg.n_nodes
        feat_keys = list(self.seq[0][0].keys())
        # sequences: (n_nodes, n_windows, n_feat)
        n_windows = min(len(s) for s in self.seq)
        seq_arr = np.zeros((n, n_windows, len(feat_keys)), dtype=np.float32)
        for i in range(n):
            for w in range(n_windows):
                seq_arr[i, w] = [self.seq[i][w][k] for k in feat_keys]
        # aggregate features: mean, std and min over windows, per feature
        agg_mean = seq_arr.mean(1)
        agg_std = seq_arr.std(1)
        agg_min = seq_arr.min(1)
        agg = np.concatenate([agg_mean, agg_std, agg_min], axis=1)
        agg_keys = ([f"{k}_mean" for k in feat_keys]
                    + [f"{k}_std" for k in feat_keys]
                    + [f"{k}_min" for k in feat_keys])
        # adjacency: connected for more than 20% of the time
        obs = self.cfg.duration / self.cfg.dt
        adj_bool = (self.adj_counts / obs) > 0.2
        return {
            "labels": self.is_gray.astype(np.int64),
            "seq": seq_arr,
            "seq_feat_keys": feat_keys,
            "agg": agg,
            "agg_keys": agg_keys,
            "adj": adj_bool,
            "config": self.cfg.as_dict(),
        }


if __name__ == "__main__":
    # quick smoke test
    cfg = SimConfig(n_nodes=40, duration=200, seed=1)
    out = ManetSim(cfg).run()
    print("labels (gray count):", out["labels"].sum(), "/", len(out["labels"]))
    print("seq shape:", out["seq"].shape)
    print("agg shape:", out["agg"].shape)
    g = out["labels"] == 1
    fr_idx = out["seq_feat_keys"].index("fwd_ratio")
    print("mean fwd_ratio benign:", out["seq"][~g, :, fr_idx].mean().round(3),
          "gray:", out["seq"][g, :, fr_idx].mean().round(3))
