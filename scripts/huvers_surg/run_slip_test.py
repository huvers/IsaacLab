# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slip test harness with solver parameter sweeps."""

import argparse
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Huvers surg: slip test sweep.")
parser.add_argument("--headless", action="store_true", help="Run without GUI.")
parser.add_argument("--impratios", type=str, default="1,3,5,10", help="Comma-separated impratio values.")
parser.add_argument("--cones", type=str, default="pyramidal,elliptic", help="Comma-separated cone modes.")
parser.add_argument("--output_dir", type=str, default="outputs/huvers_surg", help="Output directory.")
parser.add_argument("--approach_steps", type=int, default=200)
parser.add_argument("--close_steps", type=int, default=150)
parser.add_argument("--lift_steps", type=int, default=300)
parser.add_argument("--total_steps", type=int, default=800)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab.utils import close_simulation, is_simulation_running

from huvers_surg.envs.touch_proxy_env import TouchProxyEnv, TouchProxyEnvCfg
from huvers_surg.utils import relative_pose, wp_to_torch


def parse_list(arg: str, cast=float):
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [cast(p) for p in parts]


def run_trial(impratio: float, cone: str):
    cfg = TouchProxyEnvCfg()
    cfg.solver_cfg.impratio = impratio
    cfg.solver_cfg.cone = cone
    cfg.newton_cfg.solver_cfg = cfg.solver_cfg

    env = TouchProxyEnv(cfg, render_mode=None if args.headless else "human")
    env.reset()

    default_joint_pos = wp_to_torch(env.robot.data.default_joint_pos)[0]
    arm_start = default_joint_pos[env._arm_joint_ids].clone()
    arm_approach = arm_start.clone()
    if arm_approach.numel() >= 4:
        arm_approach[1] -= 0.3
        arm_approach[3] += 0.3

    arm_lift = arm_approach.clone()
    if arm_lift.numel() >= 3:
        arm_lift[1] += 0.25

    anchor = None
    max_drift = 0.0

    for step in range(args.total_steps):
        if not is_simulation_running(simulation_app, env.unwrapped.sim):
            break

        if step < args.approach_steps:
            alpha = step / max(1, args.approach_steps)
            arm_cmd = (1.0 - alpha) * arm_start + alpha * arm_approach
            grip_cmd = env.cfg.gripper_open
        elif step < args.approach_steps + args.close_steps:
            arm_cmd = arm_approach
            progress = (step - args.approach_steps) / max(1, args.close_steps)
            grip_cmd = env.cfg.gripper_open * (1.0 - progress)
        else:
            alpha = (step - args.approach_steps - args.close_steps) / max(1, args.lift_steps)
            alpha = min(1.0, alpha)
            arm_cmd = (1.0 - alpha) * arm_approach + alpha * arm_lift
            grip_cmd = env.cfg.gripper_closed

        env.command_arm_joints(arm_cmd.unsqueeze(0))
        env.command_gripper(grip_cmd)
        env.step(env._action_buf)
        env.tissue.step_visual_update()

        left = env.left_contact_features.update()
        right = env.right_contact_features.update()
        jaw_gap = env.get_jaw_gap()[0].item()

        gripper_state = env.get_gripper_state()
        obj_pos, obj_quat = env.get_proxy_pose()

        if anchor is None and bool(left.in_contact[0]) and bool(right.in_contact[0]) and jaw_gap < 0.01:
            anchor = env.tissue.compute_grasp_anchor(gripper_state)
            env.tissue.apply_grasp_constraint(anchor, gripper_state)

        if anchor is not None:
            env.tissue.apply_grasp_constraint(anchor, gripper_state)
            rel_pos, _ = relative_pose(gripper_state.pos_w, gripper_state.quat_w, obj_pos, obj_quat)
            drift = torch.linalg.norm(rel_pos - anchor.rel_pos, dim=-1)[0].item()
            max_drift = max(max_drift, drift)

    env.close()

    return {
        "impratio": impratio,
        "cone": cone,
        "solver_cfg": cfg.solver_cfg.to_dict(),
        "max_drift": max_drift,
        "drop": max_drift > 0.02,
    }


def main():
    impratios = parse_list(args.impratios, cast=float)
    cones = parse_list(args.cones, cast=str)

    results = []
    for cone in cones:
        for impratio in impratios:
            print(f"[sweep] cone={cone} impratio={impratio}")
            results.append(run_trial(impratio, cone))

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"slip_test_{timestamp}.json")
    payload = {
        "timestamp": timestamp,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved slip sweep results to {out_path}")


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
