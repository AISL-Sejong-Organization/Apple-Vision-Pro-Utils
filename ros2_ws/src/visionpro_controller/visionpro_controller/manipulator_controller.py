#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# VisionProStreamer 모듈 경로 (로컬 경로에 맞게 수정)
sys.path.append("/home/hyeonsu/Documents/AISL-apple-visionpro-modules")
from avp_stream import VisionProStreamer
import torch


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


def np2tensor(data: dict, device) -> dict:
    for key in list(data.keys()):
        if key in ["left_wrist", "right_wrist", "head"]:
            mats = frames_from(data[key])
            data[key] = torch.tensor(np.stack(mats), dtype=torch.float32, device=device) if mats else torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
        elif key in ["left_fingers", "right_fingers"]:
            mats = frames_from(data[key])
            data[key] = [torch.tensor(m, dtype=torch.float32, device=device) for m in mats] if mats else []
        else:
            data[key] = torch.tensor(data[key], dtype=torch.float32, device=device)
    return data


def draw_frame(ax, M, scale=0.1, label=None):
    origin = M[:3, 3]
    Rmat = M[:3, :3]
    ax.quiver(*origin, *Rmat[:, 0], length=scale, normalize=True)
    ax.quiver(*origin, *Rmat[:, 1], length=scale, normalize=True)
    ax.quiver(*origin, *Rmat[:, 2], length=scale, normalize=True)
    if label:
        ax.text(*origin, label, fontsize=8)


class MatplotlibVisualizerEnv:
    def __init__(self, args):
        self.device = 'cpu'
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()

    def step(self, transformation: dict):
        self.render(transformation)

    def render(self, transformation: dict):
        self.ax.cla()
        self.ax.set_xlim(0.0, 0.3)
        self.ax.set_ylim(-0.2, 0.2)
        self.ax.set_zlim(0.5, 1.1)
        self.ax.set_title('3D Hand Pose')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # 머리 프레임 그리기
        head = transformation.get('head')
        if head is not None:
            h_pose = head[0].cpu().numpy()
            draw_frame(self.ax, h_pose, scale=0.08, label='Head')
            head_pos = h_pose[:3, 3]
            self.ax.scatter(*head_pos, c='red', s=100)

        # 양쪽 손목 및 손가락 렌더링
        for side, color in [('left', 'green'), ('right', 'blue')]:
            wrist_key = f'{side}_wrist'
            finger_key = f'{side}_fingers'
            mats_w = transformation.get(wrist_key)
            rel_fingers = transformation.get(finger_key, [])
            if mats_w is None or not rel_fingers:
                continue

            w_pose = mats_w[0].cpu().numpy()
            draw_frame(self.ax, w_pose, scale=0.1, label=f'{side}_wrist')
            wrist_pos = w_pose[:3, 3]
            positions = [wrist_pos]
            for rel in rel_fingers:
                abs_pose = w_pose @ rel.cpu().numpy()
                pos = abs_pose[:3, 3]
                positions.append(pos)
                self.ax.scatter(*pos, c=color, s=20)

            skip = {(5,6), (10,11), (15,16), (20,21)}
            for i in range(len(positions)-1):
                if (i, i+1) in skip:
                    continue
                xs = [positions[i][0], positions[i+1][0]]
                ys = [positions[i][1], positions[i+1][1]]
                zs = [positions[i][2], positions[i+1][2]]
                self.ax.plot(xs, ys, zs, linewidth=2, color=color)

        plt.draw()
        plt.pause(0.001)


class VisionProControllerNode(Node):
    def __init__(self, args):
        super().__init__('visionpro_controller')
        self.streamer = VisionProStreamer(args.ip, args.record)
        self.env = MatplotlibVisualizerEnv(args)
        # /follow_target 토픽으로 6-DOF follow data 퍼블리시
        self.pub = self.create_publisher(String, 'follow_target', 10)
        # 약 30Hz 주기로 콜백
        self.create_timer(1 / 30.0, self.timer_callback)

    def timer_callback(self):
        latest = self.streamer.latest
        tensor_data = np2tensor(latest, self.env.device)
        self.env.step(tensor_data)

        # 항상 퍼블리시
        wrist = tensor_data.get('right_wrist')
        if wrist is not None and len(wrist) > 0:
            M = wrist[0].cpu().numpy()
            x, y, z = M[:3, 3]
            roll, pitch, yaw = R.from_matrix(M[:3, :3]).as_euler('xyz', degrees=True)
            data = np.array([x, y, z, roll, pitch, yaw])
            msg = String()
            msg.data = ' '.join(f'{v:.6f}' for v in data)
            self.get_logger().debug(f'Publishing follow_target: {msg.data}')
            self.pub.publish(msg)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='192.168.0.17')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()

    rclpy.init()
    node = VisionProControllerNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
