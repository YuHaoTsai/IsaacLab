# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations
#
# import gymnasium as gym
# import torch
#
# import omni.isaac.lab.sim as sim_utils
# from omni.isaac.lab.assets import Articulation, ArticulationCfg
# from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
# from omni.isaac.lab.envs.ui import BaseEnvWindow
# from omni.isaac.lab.markers import VisualizationMarkers
# from omni.isaac.lab.scene import InteractiveSceneCfg
# from omni.isaac.lab.sim import SimulationCfg
# from omni.isaac.lab.terrains import TerrainImporterCfg
# from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.utils.math import subtract_frame_transforms

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

import numpy as np
from math import pi
from isaaclab.utils.math import matrix_from_quat

##
# Pre-defined configs
##
# from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
# from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

class QuadcopterEnvWindow4(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv4, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg4(DirectMARLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    #action_space = 4
    ### ------------------------------------------
    possible_agents = ["Translation", "Yaw"]
    action_spaces = {"Translation": 4, "Yaw": 1}
    ### ------------------------------------------
    #observation_space = 12
    observation_spaces = {"Translation": 12, "Yaw": 12}
    state_space = -1
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow4

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv4(DirectMARLEnv):
    cfg: QuadcopterEnvCfg4

    def __init__(self, cfg: QuadcopterEnvCfg4, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        #self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        #print("action:")
        #print()
        #print("thrust:")
        #print(self._thrust.shape)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        ### fdcl
        self.e1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.e2 = torch.zeros(self.num_envs, 3, device=self.device)
        self.e3 = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        act1 = actions["Translation"].clone().clamp(-1.0, 1.0)
        act2 = actions["Yaw"].clone().clamp(-1.0, 1.0)

        self.f = act1[:, 0]  # [N]
        self.tau = act1[:, 1:]
        self.M3 = act2  # [Nm]

        self.e1[:, :] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.e2[:, :] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.e3[:, :] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.J = np.diag([0.000023951, 0.000023951, 0.000032347])  # inertia matrix of quad, [kg m2]
        # Limits of states:
        self.x_lim = 1.0  # [m]
        self.v_lim = 2.0  # [m/s]
        self.W_lim = pi  # [rad/s]

        # state = torch.zeros(self.num_envs, 18, device=self.device)
        # state[:, :3] = self._robot.data.root_link_state_w[:, :3]
        # state[:, 3:6] = self._robot.data.root_link_state_w[:, 7:10]
        # state[:, 15:18] = self._robot.data.root_link_state_w[:, 10:13]
        # state[:, 6:15] = (matrix_from_quat(self._robot.data.root_link_state_w[:, 3:7])).view(self.num_envs, 9)

        R = matrix_from_quat(self._robot.data.root_link_state_w[:, 3:7])
        W = self._robot.data.root_link_state_w[:, 10:13] # / self.W_lim
        # print(R[3])
        b1, b2 = R @ self.e1.view(self.num_envs, 3, 1), R @ self.e2.view(self.num_envs, 3, 1)
        # print(torch.transpose(b1, 1, 2).shape) # (32, 1, 3)
        # print(self.tau.view(self.num_envs, 3, 1).shape) # (32, 3, 1)
        # print(torch.matmul(torch.transpose(b1, 1, 2), self.tau.view(self.num_envs, 3, 1)).shape)
        # print((0.035 * W[:, 2] * W[:, 1]).shape)
        self.M1 = torch.squeeze(torch.matmul(torch.transpose(b1, 1, 2), self.tau.view(self.num_envs, 3, 1))) + 0.032347 * W[:, 2] * W[:, 1]  # M1
        self.M2 = torch.squeeze(torch.matmul(torch.transpose(b2, 1, 2), self.tau.view(self.num_envs, 3, 1))) - 0.032347 * W[:, 2] * W[:, 0]  # M2
        # print(self.M1.shape)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.f + 1.0) / 2.0
        self._moment[:, 0, 0] = self.cfg.moment_scale * torch.squeeze(self.tau[:, 1])
        self._moment[:, 0, 1] = self.cfg.moment_scale * torch.squeeze(self.tau[:, 2])
        self._moment[:, 0, 2] = self.cfg.moment_scale * torch.squeeze(self.M3)
        # print("thrust")
        # print(self._thrust)
        # print("moment")
        # print(self._moment)
    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_com_lin_vel_b,
                self._robot.data.root_com_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"Translation": obs,
                        "Yaw": obs,}
        #print("OBS : ")
        #print(observations)
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        lin_vel = torch.sum(torch.square(self._robot.data.root_com_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
            # print("haha")
            # print(reward.shape)

        return {"Translation": reward, "Yaw": reward}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_link_pos_w[:, 2] < 0.1, self._robot.data.root_link_pos_w[:, 2] > 2.0
        )

        ###
        terminated = {agent: died for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}

        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_link_pos_w[env_ids], dim=1
        ).mean()
        '''
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)
        '''

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        #self._actions[env_ids] = 0.0

        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
