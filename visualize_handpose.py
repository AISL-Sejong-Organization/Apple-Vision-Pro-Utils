import argparse
import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List

# ────────────────────────────── 상수 & 유틸 ────────────────────────────── #
JOINT_LABELS = [
    "wrist",
    "thumbKnuckle", "thumbIntermediateBase", "thumbIntermediateTip", "thumbTip",
    "indexFingerMetacarpal", "indexFingerKnuckle", "indexFingerIntermediateBase",
    "indexFingerIntermediateTip", "indexFingerTip",
    "middleFingerMetacarpal", "middleFingerKnuckle", "middleFingerIntermediateBase",
    "middleFingerIntermediateTip", "middleFingerTip",
    "ringFingerMetacarpal", "ringFingerKnuckle", "ringFingerIntermediateBase",
    "ringFingerIntermediateTip", "ringFingerTip",
    "littleFingerMetacarpal", "littleFingerKnuckle", "littleFingerIntermediateBase",
    "littleFingerIntermediateTip", "littleFingerTip",
]
FINGER_JOINT_COUNTS = [4, 5, 5, 5, 5]  # thumb~little

def frames_from(mat: np.ndarray) -> List[np.ndarray]:
    """flat 16원소 → (N,4,4) / 혹은 이미 (N,4,4)인 배열 반환"""
    arr = np.asarray(mat).squeeze()
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return [m for m in arr]                       # (N,4,4)
    if arr.ndim == 0:
        return []
    flat = arr.ravel()
    if flat.size % 16 != 0:
        return []
    return [m.reshape(4, 4) for m in flat.reshape(-1, 16)]

def np2tensor(data: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor | List[torch.Tensor]] = {}
    for k, v in data.items():
        if k in ("left_wrist", "right_wrist", "head"):
            mats = frames_from(v)
            out[k] = (torch.tensor(np.stack(mats), dtype=torch.float32, device=device)
                      if mats else torch.zeros((1, 4, 4), dtype=torch.float32, device=device))
        elif k in ("left_fingers", "right_fingers"):
            mats = frames_from(v)
            out[k] = [torch.tensor(m, dtype=torch.float32, device=device) for m in mats]
        else:
            out[k] = torch.tensor(v, dtype=torch.float32, device=device)
    return out

def draw_frame(ax, M, scale=0.1, label=None):
    o, R = M[:3, 3], M[:3, :3]
    ax.quiver(*o, *R[:, 0], length=scale, normalize=True, color="r")
    ax.quiver(*o, *R[:, 1], length=scale, normalize=True, color="g")
    ax.quiver(*o, *R[:, 2], length=scale, normalize=True, color="b")
    if label:
        ax.text(*o, label, fontsize=8)

# ────────────────────────────── HDF5 Writer ────────────────────────────── #
class HandHDF5Writer:
    """
    한 파일 안에 /left, /right 그룹
        wrist           (N,4,4)
        fingers         (N,25,4,4)  ← 25관절
        pinch_distance  (N,)
        wrist_roll      (N,)
    """
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = h5py.File(path, "a")        # append 모드

    @staticmethod
    def _req(g, name, init_shape, max_shape):
        return g[name] if name in g else g.create_dataset(
            name, shape=init_shape, maxshape=max_shape,
            dtype="f4", compression="gzip", chunks=True
        )

    def append(self, hand: str, wrist, fingers, pinch, roll):
        g = self.f.require_group(hand)
        n = g["wrist"].shape[0] if "wrist" in g else 0
        self._req(g, "wrist", (0, 4, 4), (None, 4, 4))
        self._req(g, "fingers", (0, 25, 4, 4), (None, 25, 4, 4))
        self._req(g, "pinch_distance", (0,), (None,))
        self._req(g, "wrist_roll", (0,), (None,))

        g["wrist"].resize(n + 1, 0); g["wrist"][n] = wrist
        g["fingers"].resize(n + 1, 0); g["fingers"][n] = fingers
        g["pinch_distance"].resize(n + 1, 0); g["pinch_distance"][n] = pinch
        g["wrist_roll"].resize(n + 1, 0); g["wrist_roll"][n] = roll
        self.f.flush()
        return n

    def close(self):
        self.f.flush()
        self.f.close()

# ────────────────────────────── 시각화 + 저장 환경 ────────────────────────────── #
class MatplotlibVisualizerEnv:
    def __init__(self, args):
        self.args = args
        self.device = "cpu"

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.ion()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # 저장 관련
        self.active_hand: str | None = None           # "left" | "right" | None
        self.writer: HandHDF5Writer | None = None
        self.save_dir = args.save_dir
        if not args.load:
            os.makedirs(self.save_dir, exist_ok=True)

    def _next_fname(self, hand: str) -> str:
        files = [f for f in os.listdir(self.save_dir)
                 if f.startswith(f"{hand}_hand_data_") and f.endswith(".hdf5")]
        idx = (max(int(f.split('_')[-1].split('.')[0]) for f in files) + 1) if files else 0
        return os.path.join(self.save_dir, f"{hand}_hand_data_{idx:03d}.hdf5")

    # ── 키 이벤트 ──
    def on_key_press(self, e):
        key = e.key.lower()

        if key in ("l", "r"):
            self.active_hand = "left" if key == "l" else "right"
            fname = self._next_fname(self.active_hand)
            self.writer = HandHDF5Writer(fname)
            print(f"[{self.active_hand.upper()}] recording {os.path.basename(fname)}")

        if key == "n" and self.writer:
            self.active_hand = "left" if key == "l" else "right"
            fname = self._next_fname(self.active_hand)
            self.writer.close()
            self.writer = None
            print(f"finished")
            self.active_hand = None

        if self.writer:
            return


    # ── 프레임 저장 ──
    def _save_current(self, tr: Dict[str, torch.Tensor], hand: str):
        flist = tr[f"{hand}_fingers"]
        if len(flist) == 0:
            return
        # 손목 + 24관절 = 25개 쌓기
        wrist_pose = tr[f"{hand}_wrist"][0].cpu().numpy()
        fingers = [m.cpu().numpy() for m in flist]  # (25,4,4)
        fingers = np.stack(fingers)
        pinch = float(tr.get(f"{hand}_pinch_distance", torch.zeros(1)))
        roll = float(tr.get(f"{hand}_wrist_roll", torch.zeros(1)))
        idx = self.writer.append(hand, wrist_pose, fingers, pinch, roll)
        print(f"[{hand.upper()}] frame #{idx}")

    # ── 메인 스텝 ──
    def step(self, tr: Dict[str, torch.Tensor]):
        if self.writer and self.active_hand:
            self._save_current(tr, self.active_hand)
        self._render(tr)

    # ── 3-D 렌더링 ──
    def _render(self, tr: Dict[str, torch.Tensor]):
        ax = self.ax; ax.cla()
        ax.set(xlim=(-.7, .7), ylim=(-.7, .7), zlim=(0, 2),
               title="3D Hand Pose", xlabel="X", ylabel="Y", zlabel="Z")

        # head
        if (h := tr.get("head")) is not None:
            hp = h[0]
            draw_frame(ax, hp.cpu().numpy(), .08, "Head")
            pos = hp[:3, 3].cpu().numpy()
            ax.scatter(*pos, c="r", s=50)
            ax.text(*pos, "H")

        # hands
        for side, col in [("left", "g"), ("right", "b")]:
            wk, fk = f"{side}_wrist", f"{side}_fingers"
            w, flist = tr.get(wk), tr.get(fk, [])
            if w is None or not flist:
                continue
            w_pose = w[0]; w_pos = w_pose[:3, 3].cpu().numpy()
            draw_frame(ax, w_pose.cpu().numpy(), .1, f"{side}_wrist")
            pts = [w_pos]
            for rel in flist:
                p = (w_pose @ rel)[:3, 3].cpu().numpy()
                pts.append(p); ax.scatter(*p, c=col, s=20)
            skip = {(5, 6), (10, 11), (15, 16), (20, 21)}
            for i in range(len(pts) - 1):
                if (i, i+1) in skip:
                    continue
                xa, ya, za = zip(pts[i], pts[i+1])
                ax.plot(xa, ya, za, color=col, linewidth=2)

        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.writer:
            self.writer.close()

# ────────────────────────────── 스트림 / 재생 래퍼 ────────────────────────────── #
class LiveVisualizer:
    def __init__(self, args):
        from avp_stream import VisionProStreamer
        self.streamer = VisionProStreamer(args.ip, args.record)
        self.env = MatplotlibVisualizerEnv(args)

    def run(self):
        while plt.fignum_exists(self.env.fig.number):
            raw = self.streamer.latest
            if not raw:
                plt.pause(0.01)
                continue
            self.env.step(np2tensor(raw, self.env.device))
        self.env.close()

class H5Player:
    """hands_dataset.h5 재생용"""
    def __init__(self, path, args):
        self.f = h5py.File(path, "r")
        self.env = MatplotlibVisualizerEnv(args)
        self.N = max(self.f.get("left/wrist", np.zeros((0,))).shape[0],
                     self.f.get("right/wrist", np.zeros((0,))).shape[0])

    def run(self, fps=10):
        for i in range(self.N):
            sample = {}
            for hand in ("left", "right"):
                if hand not in self.f:
                    continue
                g = self.f[hand]
                if g["wrist"].shape[0] <= i:
                    continue
                sample[f"{hand}_wrist"] = g["wrist"][i]
                sample[f"{hand}_fingers"] = g["fingers"][i]
                sample[f"{hand}_pinch_distance"] = g["pinch_distance"][i]
                sample[f"{hand}_wrist_roll"] = g["wrist_roll"][i]
            self.env.step(np2tensor(sample, "cpu"))
            plt.pause(1 / fps)
        self.f.close()
        self.env.close()

# ────────────────────────────── main ────────────────────────────── #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ip", type=str, default="192.168.0.70")
    p.add_argument("--record", action="store_true")
    p.add_argument("--save_dir", default="data")
    p.add_argument("--load", help="hands_dataset.h5 재생")
    args = p.parse_args()

    if args.load:
        H5Player(args.load, args).run()
    else:
        print("Figure 창에 포커스 후  l / r  눌러 왼손 / 오른손 녹화, n 눌러 정지")
        LiveVisualizer(args).run()
