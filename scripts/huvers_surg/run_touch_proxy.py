# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Touch a rigid organ proxy with a gripper (Newton backend)."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Huvers surg: touch rigid proxy.")
parser.add_argument("--num_steps", type=int, default=600, help="Total simulation steps.")
parser.add_argument("--approach_steps", type=int, default=300, help="Steps to interpolate to target pose.")
parser.add_argument(
    "--target_joints",
    type=str,
    default="",
    help="Comma-separated 7 joint targets for the arm (overrides heuristic).",
)
parser.add_argument("--headless", action="store_true", help="Run without GUI.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab.utils import close_simulation, is_simulation_running

from huvers_surg.envs.touch_proxy_env import TouchProxyEnv, TouchProxyEnvCfg
from huvers_surg.utils import wp_to_torch


def parse_target_joints(arg: str):
    if not arg:
        return None
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) != 7:
        raise ValueError("--target_joints must have 7 comma-separated values.")
    return torch.tensor([float(p) for p in parts])


def main():
    cfg = TouchProxyEnvCfg()
    if args.headless:
        cfg.sim.render_interval = cfg.decimation

    env = TouchProxyEnv(cfg, render_mode=None if args.headless else "human")
    env.reset()

    # compute target arm joints
    default_joint_pos = wp_to_torch(env.robot.data.default_joint_pos)[0]
    arm_start = default_joint_pos[env._arm_joint_ids].clone()

    target_override = parse_target_joints(args.target_joints)
    if target_override is not None:
        arm_target = target_override.to(env.device)
    else:
        arm_target = arm_start.clone()
        # heuristic nudge towards the proxy
        if arm_target.numel() >= 4:
            arm_target[1] -= 0.3
            arm_target[3] += 0.3

    for step in range(args.num_steps):
        if not is_simulation_running(simulation_app, env.unwrapped.sim):
            break

        alpha = min(1.0, step / max(1, args.approach_steps))
        arm_cmd = (1.0 - alpha) * arm_start + alpha * arm_target
        env.command_arm_joints(arm_cmd.unsqueeze(0))
        env.command_gripper(env.cfg.gripper_open)

        env.step(env._action_buf)
        env.tissue.step_visual_update()

        left = env.left_contact_features.update()
        right = env.right_contact_features.update()

        if step % 30 == 0:
            print(
                f"[step {step:04d}] left_force={left.force_mag[0].item():.3f} right_force={right.force_mag[0].item():.3f} "
                f"in_contact=({bool(left.in_contact[0])},{bool(right.in_contact[0])})"
            )

    env.close()


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
