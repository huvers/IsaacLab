# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted grasp + lift + release + regrasp demo."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Huvers surg: grasp demo.")
parser.add_argument("--headless", action="store_true", help="Run without GUI.")
parser.add_argument("--steps", type=int, default=1200, help="Total simulation steps.")
parser.add_argument("--approach_steps", type=int, default=300, help="Steps to approach proxy.")
parser.add_argument("--close_steps", type=int, default=200, help="Steps to close gripper.")
parser.add_argument("--lift_steps", type=int, default=300, help="Steps to lift." )
parser.add_argument("--release_steps", type=int, default=150, help="Steps to release/open.")
parser.add_argument("--regrasp_steps", type=int, default=250, help="Steps to regrasp.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab.utils import close_simulation, is_simulation_running

from huvers_surg.controllers import GraspFsm, GraspFsmCfg, GraspState
from huvers_surg.envs.touch_proxy_env import TouchProxyEnv, TouchProxyEnvCfg
from huvers_surg.utils import relative_pose, wp_to_torch


def main():
    cfg = TouchProxyEnvCfg()
    env = TouchProxyEnv(cfg, render_mode=None if args.headless else "human")
    env.reset()

    fsm = GraspFsm(GraspFsmCfg())

    default_joint_pos = wp_to_torch(env.robot.data.default_joint_pos)[0]
    arm_start = default_joint_pos[env._arm_joint_ids].clone()
    arm_approach = arm_start.clone()
    if arm_approach.numel() >= 4:
        arm_approach[1] -= 0.3
        arm_approach[3] += 0.3

    arm_lift = arm_approach.clone()
    if arm_lift.numel() >= 3:
        arm_lift[1] += 0.2

    anchor = None

    for step in range(args.steps):
        if not is_simulation_running(simulation_app, env.unwrapped.sim):
            break

        close_cmd = False
        open_cmd = False

        if step < args.approach_steps:
            alpha = step / max(1, args.approach_steps)
            arm_cmd = (1.0 - alpha) * arm_start + alpha * arm_approach
            grip_cmd = env.cfg.gripper_open
        elif step < args.approach_steps + args.close_steps:
            arm_cmd = arm_approach
            close_cmd = True
            progress = (step - args.approach_steps) / max(1, args.close_steps)
            grip_cmd = env.cfg.gripper_open * (1.0 - progress)
        elif step < args.approach_steps + args.close_steps + args.lift_steps:
            alpha = (step - args.approach_steps - args.close_steps) / max(1, args.lift_steps)
            arm_cmd = (1.0 - alpha) * arm_approach + alpha * arm_lift
            grip_cmd = env.cfg.gripper_closed
        elif step < args.approach_steps + args.close_steps + args.lift_steps + args.release_steps:
            arm_cmd = arm_lift
            open_cmd = True
            grip_cmd = env.cfg.gripper_open
        else:
            # regrasp: move back down and close
            remaining = step - (args.approach_steps + args.close_steps + args.lift_steps + args.release_steps)
            alpha = min(1.0, remaining / max(1, args.regrasp_steps))
            arm_cmd = (1.0 - alpha) * arm_lift + alpha * arm_approach
            close_cmd = True
            grip_cmd = env.cfg.gripper_open * (1.0 - alpha)

        env.command_arm_joints(arm_cmd.unsqueeze(0))
        env.command_gripper(grip_cmd)
        env.step(env._action_buf)

        env.tissue.step_visual_update()

        left = env.left_contact_features.update()
        right = env.right_contact_features.update()
        jaw_gap = env.get_jaw_gap()[0]

        gripper_state = env.get_gripper_state()
        obj_pos, obj_quat = env.get_proxy_pose()

        drift = 0.0
        if anchor is not None:
            rel_pos, _ = relative_pose(gripper_state.pos_w, gripper_state.quat_w, obj_pos, obj_quat)
            drift = torch.linalg.norm(rel_pos - anchor.rel_pos, dim=-1)[0].item()

        fsm.update(
            close_cmd=close_cmd,
            open_cmd=open_cmd,
            left_in_contact=bool(left.in_contact[0]),
            right_in_contact=bool(right.in_contact[0]),
            jaw_gap=jaw_gap.item(),
            constraint_engaged=env.grasp_constraint.engaged,
            object_drift=drift,
            force_left=left.force_mag[0].item(),
            force_right=right.force_mag[0].item(),
        )

        if fsm.state == GraspState.CANDIDATE_GRASP and anchor is None:
            anchor = env.tissue.compute_grasp_anchor(gripper_state)
            env.tissue.apply_grasp_constraint(anchor, gripper_state)

        if fsm.state == GraspState.HELD and anchor is not None:
            env.tissue.apply_grasp_constraint(anchor, gripper_state)

        if fsm.state == GraspState.OPENING and env.grasp_constraint.engaged:
            env.tissue.release_grasp()
            anchor = None

        if step % 60 == 0:
            print(
                f"[step {step:04d}] state={fsm.state} gap={jaw_gap.item():.3f} "
                f"forces=({left.force_mag[0].item():.2f},{right.force_mag[0].item():.2f}) drift={drift:.3f}"
            )

    env.close()


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
