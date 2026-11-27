import numpy as np
from frankapy import FrankaArm

arm = FrankaArm()
pose = arm.get_pose()

outward = pose.translation[:2] / np.linalg.norm(pose.translation[:2])
offset_base = np.array([0.08 * outward[0], 0.08 * outward[1], -0.04])
offset_tool = pose.rotation.T @ offset_base

print("opening_offset_m:", offset_tool.tolist())
arm.goto_gripper(0.035, force=20, grasp=True)
