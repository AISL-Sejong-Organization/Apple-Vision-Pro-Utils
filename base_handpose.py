import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict

# ARKit 관절 순서에 맞춘 라벨
JOINT_LABELS = [
    "wrist",
    "thumbKnuckle",
    "thumbIntermediateBase",
    "thumbIntermediateTip",
    "thumbTip",
    "indexFingerMetacarpal",
    "indexFingerKnuckle",
    "indexFingerIntermediateBase",
    "indexFingerIntermediateTip",
    "indexFingerTip",
    "middleFingerMetacarpal",
    "middleFingerKnuckle",
    "middleFingerIntermediateBase",
    "middleFingerIntermediateTip",
    "middleFingerTip",
    "ringFingerMetacarpal",
    "ringFingerKnuckle",
    "ringFingerIntermediateBase",
    "ringFingerIntermediateTip",
    "ringFingerTip",
    "littleFingerMetacarpal",
    "littleFingerKnuckle",
    "littleFingerIntermediateBase",
    "littleFingerIntermediateTip",
    "littleFingerTip",
]

FINGER_JOINT_COUNTS = [4, 5, 5, 5, 5]


def frames_from(mat):
    """리스트 형태의 flat 16원소 배열 → 4×4 행렬 리스트로 변환"""
    arr = np.asarray(mat).squeeze()
    if arr.ndim == 0:
        return []
    flat = arr.ravel()
    if flat.size % 16 != 0:
        return []
    mats = flat.reshape(-1, 16)
    return [m.reshape(4, 4) for m in mats]


def np2tensor(data: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    for key in data.keys():
        if key in ["left_wrist", "right_wrist", "head"]:
            mats = frames_from(data[key])
            if mats:
                data[key] = torch.tensor(
                    np.stack(mats), dtype=torch.float32, device=device
                )
            else:
                data[key] = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
        elif key in ["left_fingers", "right_fingers"]:
            mats = frames_from(data[key])
            if mats:
                data[key] = [
                    torch.tensor(m, dtype=torch.float32, device=device) for m in mats
                ]
            else:
                data[key] = []
        else:
            data[key] = torch.tensor(data[key], dtype=torch.float32, device=device)
    return data


def draw_frame(ax, M, scale=0.1, label=None):
    """한 프레임(4×4)에서 XYZ 축과 라벨을 그림"""
    origin = M[:3, 3]
    R = M[:3, :3]
    ax.quiver(*origin, *R[:, 0], length=scale, normalize=True, color="r")
    ax.quiver(*origin, *R[:, 1], length=scale, normalize=True, color="g")
    ax.quiver(*origin, *R[:, 2], length=scale, normalize=True, color="b")
    if label:
        ax.text(*origin, label, color="k", fontsize=8)


def apply_finger_chain(wrist_pose, relative_fingers):
    """손목 포즈와 상대 행렬 리스트를 곱해서 절대 finger 포즈 리스트 반환"""
    current_pose = wrist_pose.clone()
    absolute_poses = []
    for rel in relative_fingers:
        current_pose = current_pose @ rel
        absolute_poses.append(current_pose.clone())
    return absolute_poses


class MatplotlibVisualizerEnv:
    def __init__(self, args):
        self.args = args
        self.device = "cpu"
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.ion()

    def step(self, transformation: Dict[str, torch.Tensor]):
        self.render(transformation)

    def render(self, transformation: Dict[str, torch.Tensor]):
        self.ax.cla()
        # 축 범위 조정
        self.ax.set_xlim(-0.7, 0.7)
        self.ax.set_ylim(-0.7, 0.7)
        self.ax.set_zlim(0.0, 2.0)
        self.ax.set_title("3D Hand Pose")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # 1) 머리 위치 계산 & 프레임 그리기
        head = transformation.get("head")
        head_pos = None
        if head is not None:
            h_pose = head[0]
            head_pos = h_pose[:3, 3].cpu().numpy()
            draw_frame(self.ax, h_pose.cpu().numpy(), scale=0.08, label="Head")
            # 머리 점도 찍고 번호를 매기려면
            self.ax.scatter(*head_pos, c="red", s=500)
            self.ax.text(*head_pos, "H", color="red", fontsize=10)

        for side, color in [("left", "green"), ("right", "blue")]:
            wrist_key = f"{side}_wrist"
            finger_key = f"{side}_fingers"

            mats_w = transformation.get(wrist_key)
            rel_fingers = transformation.get(finger_key, [])
            if mats_w is None or not rel_fingers:
                continue

            w_pose = mats_w[0]
            wrist_pos = w_pose[:3, 3].cpu().numpy()
            draw_frame(self.ax, w_pose.cpu().numpy(), scale=0.1, label=f"{side}_wrist")

            # positions 리스트 구성
            positions = [wrist_pos]
            for rel in rel_fingers:
                abs_pose = w_pose @ rel
                pos = abs_pose[:3, 3].cpu().numpy()
                positions.append(pos)
                self.ax.scatter(pos[0], pos[1], pos[2], c=color, s=20)

            # 스킵할 세그먼트 인덱스 쌍
            skip = {(5, 6), (10, 11), (15, 16), (20, 21)}

            # 각 세그먼트마다 그리기
            for i in range(len(positions) - 1):
                if (i, i + 1) in skip:
                    continue
                x = [positions[i][0], positions[i + 1][0]]
                y = [positions[i][1], positions[i + 1][1]]
                z = [positions[i][2], positions[i + 1][2]]
                self.ax.plot(x, y, z, color=color, linewidth=2)

            # # 번호 텍스트 (원하면 유지)
            # for idx, (x, y, z) in enumerate(positions):
            #     self.ax.text(x, y, z, str(idx), fontsize=8, color=color)

        plt.draw()
        plt.pause(0.01)


class MatplotlibVisualizer:
    def __init__(self, args):
        from avp_stream import VisionProStreamer

        self.streamer = VisionProStreamer(args.ip, args.record)
        self.env = MatplotlibVisualizerEnv(args)

    def run(self):
        while True:
            latest = self.streamer.latest  # Dict[str, np.ndarray]
            tensor_data = np2tensor(latest, self.env.device)
            self.env.step(tensor_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=False, default="192.168.0.70")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()

    vis = MatplotlibVisualizer(args)
    vis.run()
