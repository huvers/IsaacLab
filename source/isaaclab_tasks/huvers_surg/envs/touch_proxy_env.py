"""Touch-proxy environment for surgical interaction (Newton)."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_CFG

from huvers_surg.assets import OrganAssetCfg, spawn_organ_asset
from huvers_surg.controllers import ContactFeatures, GraspConstraint, GraspConstraintCfg
from huvers_surg.interfaces import RigidTissueProxy
from huvers_surg.utils import wp_to_torch


@configclass
class TouchProxySceneCfg(InteractiveSceneCfg):
    """Scene config with no predefined assets (spawned in env)."""

    pass


@configclass
class TouchProxyEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    action_scale = 1.0
    action_space = 8  # 7 arm joints + 1 gripper command
    observation_space = 1
    state_space = 0

    solver_cfg = MJWarpSolverCfg(
        njmax=64,
        nconmax=64,
        ls_iterations=20,
        cone="pyramidal",
        impratio=1,
        ls_parallel=True,
        integrator="implicit",
    )

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, newton_cfg=newton_cfg)

    # assets
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FRANKA_PANDA_CFG.spawn.replace(activate_contact_sensors=True),
    )

    organ_asset_cfg: OrganAssetCfg = OrganAssetCfg(
        prim_path="/World/envs/env_0/Organs/Gallbladder",
        proxy_init_pos=(0.6, 0.0, 0.02),
    )

    # default jaw names (override if your robot differs)
    left_jaw_body_name: str = "panda_leftfinger"
    right_jaw_body_name: str = "panda_rightfinger"
    hand_body_name: str = "panda_hand"

    # gripper limits
    gripper_open: float = 0.04
    gripper_closed: float = 0.0

    # scene
    scene: InteractiveSceneCfg = TouchProxySceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )


class TouchProxyEnv(DirectRLEnv):
    cfg: TouchProxyEnvCfg

    def __init__(self, cfg: TouchProxyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._action_buf = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # tissue interface + grasp constraint
        self.grasp_constraint = GraspConstraint(GraspConstraintCfg())
        self.tissue = RigidTissueProxy(
            rigid_object=self.organ_proxy,
            visual_prim_path=self._organ_visual_prim_path,
            physics_prim_path=self._organ_physics_prim_path,
            grasp_constraint=self.grasp_constraint,
        )

        # contact feature helpers (initialized after sensors are created)
        self.left_contact_features = ContactFeatures(self.scene.sensors["left_jaw_contact"])  # type: ignore[index]
        self.right_contact_features = ContactFeatures(self.scene.sensors["right_jaw_contact"])  # type: ignore[index]

    def _resolve_joint_and_body_ids(self) -> None:
        # joints
        _, _, self._arm_joint_ids = self.robot.find_joints("panda_joint.*")
        _, _, self._finger_joint_ids = self.robot.find_joints("panda_finger_joint.*")
        # body indices (gripper + jaws)
        self._hand_body_name = self._resolve_body_name(self.cfg.hand_body_name, fallback_patterns=["panda_hand", "hand"])  # type: ignore[attr-defined]
        self._left_jaw_body_name = self._resolve_body_name(
            self.cfg.left_jaw_body_name,
            fallback_patterns=["panda_leftfinger", "leftfinger", "left_finger", "left_jaw"],
        )
        self._right_jaw_body_name = self._resolve_body_name(
            self.cfg.right_jaw_body_name,
            fallback_patterns=["panda_rightfinger", "rightfinger", "right_finger", "right_jaw"],
        )

        _, _, self._hand_body_ids = self.robot.find_bodies(self._hand_body_name)
        _, _, self._left_jaw_body_ids = self.robot.find_bodies(self._left_jaw_body_name)
        _, _, self._right_jaw_body_ids = self.robot.find_bodies(self._right_jaw_body_name)

    def _resolve_body_name(self, preferred: str, fallback_patterns: Sequence[str]) -> str:
        if preferred in self.robot.body_names:
            return preferred
        for pattern in fallback_patterns:
            _, _, names = self.robot.find_bodies(pattern)
            if len(names) > 0:
                return names[0]
        # fallback to last body
        return self.robot.body_names[-1]

    def _setup_scene(self):
        # robot articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # organ asset (visual + physics proxy)
        env0_root = self.scene.env_fmt.format(0)

        def _to_env0_path(path: str) -> str:
            if "{ENV_REGEX_NS}" in path:
                return path.format(ENV_REGEX_NS=env0_root)
            if path.startswith(env0_root):
                return path
            if path.startswith("/World/envs/"):
                return path
            if path.startswith("/"):
                return f"{env0_root}{path}"
            return f"{env0_root}/{path}"

        organ_cfg = self.cfg.organ_asset_cfg.replace(prim_path=_to_env0_path(self.cfg.organ_asset_cfg.prim_path))
        organ_info = spawn_organ_asset(organ_cfg)
        proxy_regex_path = organ_info.physics_prim_path.replace(env0_root, self.scene.env_regex_ns)
        self.organ_proxy = RigidObject(organ_info.rigid_cfg.replace(prim_path=proxy_regex_path))
        self._organ_visual_prim_path = organ_info.visual_prim_path
        self._organ_physics_prim_path = proxy_regex_path

        # ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # register assets
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["organ_proxy"] = self.organ_proxy

        # resolve joint/body ids after assets are created
        self._resolve_joint_and_body_ids()

        # contact sensors
        proxy_filter = proxy_regex_path
        left_jaw_path = f"{self.scene.env_regex_ns}/Robot/{self._left_jaw_body_name}"
        right_jaw_path = f"{self.scene.env_regex_ns}/Robot/{self._right_jaw_body_name}"

        left_cfg = ContactSensorCfg(
            prim_path=left_jaw_path,
            filter_prim_paths_expr=[proxy_filter],
            history_length=3,
            track_air_time=False,
        )
        right_cfg = ContactSensorCfg(
            prim_path=right_jaw_path,
            filter_prim_paths_expr=[proxy_filter],
            history_length=3,
            track_air_time=False,
        )

        self.scene.sensors["left_jaw_contact"] = ContactSensor(left_cfg)
        self.scene.sensors["right_jaw_contact"] = ContactSensor(right_cfg)

        # add light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._action_buf = actions.clone()

    def _apply_action(self) -> None:
        # arm targets
        arm_targets = self._action_buf[:, : len(self._arm_joint_ids)]
        # gripper target (single scalar replicated for both fingers)
        grip_target = self._action_buf[:, -1].unsqueeze(-1)
        grip_target = torch.clamp(grip_target, self.cfg.gripper_closed, self.cfg.gripper_open)
        grip_targets = grip_target.repeat(1, len(self._finger_joint_ids))

        self.robot.set_joint_position_target(arm_targets, joint_ids=self._arm_joint_ids)
        self.robot.set_joint_position_target(grip_targets, joint_ids=self._finger_joint_ids)

    def _get_observations(self) -> dict:
        # minimal observation: zeros
        obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot
        joint_pos = wp_to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = wp_to_torch(self.robot.data.default_joint_vel)[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset organ proxy
        default_pose = wp_to_torch(self.organ_proxy.data.default_root_pose)[env_ids].clone()
        self.organ_proxy.write_root_pose_to_sim(default_pose, env_ids=env_ids)

    # Helper APIs for scripts
    def get_gripper_state(self):
        body_pos = wp_to_torch(self.robot.data.body_pos_w)
        body_quat = wp_to_torch(self.robot.data.body_quat_w)
        hand_pos = body_pos[:, self._hand_body_ids[0]]
        hand_quat = body_quat[:, self._hand_body_ids[0]]
        jaw_pos = torch.stack(
            [body_pos[:, self._left_jaw_body_ids[0]], body_pos[:, self._right_jaw_body_ids[0]]], dim=1
        )
        jaw_gap = self.get_jaw_gap()
        from huvers_surg.controllers.grasp_anchor import GripperState

        return GripperState(pos_w=hand_pos, quat_w=hand_quat, jaw_pos_w=jaw_pos, jaw_gap=jaw_gap)

    def get_jaw_gap(self) -> torch.Tensor:
        joint_pos = wp_to_torch(self.robot.data.joint_pos)
        finger_pos = joint_pos[:, self._finger_joint_ids]
        return finger_pos.sum(dim=1)

    def command_arm_joints(self, arm_targets: torch.Tensor) -> None:
        self._action_buf[:, : len(self._arm_joint_ids)] = arm_targets

    def command_gripper(self, width: float | torch.Tensor) -> None:
        if isinstance(width, torch.Tensor):
            width_t = width
        else:
            width_t = torch.tensor(width, device=self.device)
        width_t = torch.clamp(width_t, self.cfg.gripper_closed, self.cfg.gripper_open)
        self._action_buf[:, -1] = width_t

    def get_proxy_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = wp_to_torch(self.organ_proxy.data.root_pos_w)
        quat = wp_to_torch(self.organ_proxy.data.root_quat_w)
        return pos, quat
