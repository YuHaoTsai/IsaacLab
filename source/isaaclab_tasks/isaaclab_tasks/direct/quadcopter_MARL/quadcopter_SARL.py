# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import get_checkpoint_path
from matplotlib import pyplot as plt
##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

import numpy as np
import pandas as pd
from math import pi
from scipy.spatial.transform import Rotation as R
from isaaclab.utils.math import matrix_from_quat


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnvSARL, window_name: str = "IsaacLab"):
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
class QuadcopterEnvCfgSARL(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 2
    action_space = 4
    # observation_space = 12
    observation_space = 15 #21
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
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
    distance_to_goal_reward_scale = 20.0
    encouraging_scale = 0.8


class QuadcopterEnvSARL(DirectRLEnv):
    cfg: QuadcopterEnvCfgSARL

    def __init__(self, cfg: QuadcopterEnvCfgSARL, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "encouraging",
                "exceeding_thrust_limit1",
                "exceeding_thrust_limit2",
                "exceeding_thrust_limit3",
                "exceeding_thrust_limit4",
                "thrust_smoothing",
                "moment_smoothing",
                "goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        ### fdcl
        self.e1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.e2 = torch.zeros(self.num_envs, 3, device=self.device)
        self.e3 = torch.zeros(self.num_envs, 3, device=self.device)

        self.o_b1 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.o_b2 = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # wind drag
        self.wind_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.w_coefficient = 0.01
        self.external_forces = torch.zeros(self.num_envs, 3, device=self.device)

        # last step position
        self.last_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # sum of action
        # self.total_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # last action
        self.last_moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.last_thrust = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_forces = torch.zeros(self.num_envs, 4, device=self.device)
        # T1 ~ T4
        self.Forces = torch.zeros(self.num_envs, 4, device=self.device)
        self.Forces += 0.07
        self.f_check = torch.zeros(self.num_envs, device=self.device)
        self.f_check += 10.0

        self.d = 0.05
        self.c_tf = 0.006
        self.forces_to_fM = [[1.0, 1.0, 1.0, 1.0],
                             [0.0, -self.d, 0.0, self.d],
                             [self.d, 0.0, -self.d, 0.0],
                             [-self.c_tf, self.c_tf, -self.c_tf, self.c_tf]]
        self.M = torch.tensor(self.forces_to_fM, device=self.device)
        self.fM_to_forces = torch.linalg.inv(self.M)

        # distance reward
        self.dFactor = 0.0

        # goal bonus
        self.check = torch.zeros(self.num_envs, 1, device=self.device)
        self.check_count = torch.zeros(self.num_envs, device=self.device)
        self.bonus = torch.zeros(self.num_envs, 1, device=self.device)

        # for plotting
        self.step_count = 0
        #
        self.thrust = []

        self.moment_x = []
        self.moment_y = []
        self.moment_z = []
        #
        self.pos_x = []
        self.pos_y = []
        self.pos_z = []
        #
        self.l_vel_x = []
        self.l_vel_y = []
        self.l_vel_z = []

        self.a_vel_x = []
        self.a_vel_y = []
        self.a_vel_z = []
        #
        self.pic_count_thrust = 0
        self.pic_count_moment = 0
        self.pic_count_pos = 0
        self.pic_count_l_vel = 0
        self.pic_count_a_vel = 0
        self.pic_count_T = 0
        self.pic_count_drag = 0
        self.change_count = 0
        #
        self.T1 = []
        self.T2 = []
        self.T3 = []
        self.T4 = []
        #
        self.dragX = []
        self.dragY = []
        self.dragZ = []
        self.dragX_n = []
        self.dragY_n = []
        self.dragZ_n = []
        #
        self.w_vel_X = []
        self.w_vel_Y = []

        # self.th_change = []
        # self.mo_change = []
        self.T1_change = []
        self.T2_change = []
        self.T3_change = []
        self.T4_change = []

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.last_pos = self._robot.data.root_link_pos_w

        self.e1[:, :] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.e2[:, :] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.e3[:, :] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.J = np.diag([0.000023951, 0.000023951, 0.000032347])  # inertia matrix of quad, [kg m2]

        R = matrix_from_quat(self._robot.data.root_link_state_w[:, 3:7])
        W = self._robot.data.root_link_state_w[:, 10:13]  # / self.W_lim ##need to change
        # print(R[3])
        b1, b2 = R @ self.e1.view(self.num_envs, 3, 1), R @ self.e2.view(self.num_envs, 3, 1)

        self.o_b1 = torch.transpose(b1, 1, 2)
        self.o_b2 = torch.transpose(b2, 1, 2)

        self._actions = actions.clone().clamp(-1.0, 1.0)
        # self.total_actions[:, 0] += self._actions[:, 0] * 0.5
        # self.total_actions[:, 1] += self._actions[:, 1] * 0.1
        # self.total_actions[:, 2] += self._actions[:, 2] * 0.1
        # self.total_actions[:, 3] += self._actions[:, 3] * 0.05

        self.last_thrust = 0.0
        self.last_moment = 0.0
        self.last_forces = 0.0
        self.last_thrust += self._thrust[:, 0, 2]
        self.last_moment += self._moment[:, 0, :]
        self.last_forces += self.Forces
        # print(self.last_moment[0])

        self._thrust[:, 0, :2] = 0.0
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.total_actions[:, 0] + 1.0) / 2.0
        # self._thrust[:, 0, 2] = self._thrust[:, 0, 2].clone().clamp(0.02, 0.6)

        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        # self._moment[:, 0, :] = self.total_actions[:, 1:]
        # self._moment[:, 0, :2] = self._moment[:, 0, :2].clone().clamp(-0.0075, 0.0075)
        # self._moment[:, 0, 2] = self._moment[:, 0, 2].clone().clamp(-0.0018, 0.0018)
        # print("moment")
        # print(self._moment[0])

        # plot section                                                                                                      ***
        fM = torch.zeros(self.num_envs, 4, device=self.device)
        fM[:, 0] = self._thrust[:, 0, 2]
        fM[:, 1:4] = self._moment[:, 0, :]
        # Forces = torch.zeros(self.num_envs, 4, device=self.device)
        for env in range(self.num_envs):
            self.Forces[env, :] = torch.matmul(self.fM_to_forces, fM[env, :])

        # thrust_change = self._thrust[:, 0, 2] - self.last_thrust
        # moment_change = self._moment.squeeze() - self.last_moment
        forces_change = self.Forces - self.last_forces
        #
        # for plotting
        # self.thrust.append(self._thrust[0, 0, 2].item())
        #
        # self.moment_x.append(self._moment[0, 0, 0].item())
        # self.moment_y.append(self._moment[0, 0, 1].item())
        # self.moment_z.append(self._moment[0, 0, 2].item())
        # #
        # self.pos_x.append(self._robot.data.root_link_pos_w[0, 0].item())
        # self.pos_y.append(self._robot.data.root_link_pos_w[0, 1].item())
        # self.pos_z.append(self._robot.data.root_link_pos_w[0, 2].item())
        # #
        # self.l_vel_x.append(self._robot.data.root_link_state_w[0, 7].item())
        # self.l_vel_y.append(self._robot.data.root_link_state_w[0, 8].item())
        # self.l_vel_z.append(self._robot.data.root_link_state_w[0, 9].item())
        #
        # self.a_vel_x.append(self._robot.data.root_link_state_w[0, 10].item())
        # self.a_vel_y.append(self._robot.data.root_link_state_w[0, 11].item())
        # self.a_vel_z.append(self._robot.data.root_link_state_w[0, 12].item())
        # #
        # self.T1.append(self.Forces[0, 0].item())
        # self.T2.append(self.Forces[0, 1].item())
        # self.T3.append(self.Forces[0, 2].item())
        # self.T4.append(self.Forces[0, 3].item())
        #
        # self.th_change.append(thrust_change[0].item())
        # self.mo_change.append(moment_change[0, 2].item())
        # self.T1_change.append(forces_change[0, 0].item())
        # self.T2_change.append(forces_change[0, 1].item())
        # self.T3_change.append(forces_change[0, 2].item())
        # self.T4_change.append(forces_change[0, 3].item())
        # print("now")
        # print(self._thrust[0, 0, 2])
        # print(self.last_thrust[0])

        # wind vec generating                                                                                               ***
        # gauss wind
        self.wind_vel[:, 0] += torch.zeros_like(self.wind_vel[:, 0]).normal_(0.0, 0.01)
        self.wind_vel[:, 1] += torch.zeros_like(self.wind_vel[:, 1]).normal_(0.0, 0.01)
        self.wind_vel[:, 2] = 0.0

        # wind external forces
        wind_vec_b = quat_apply_inverse(self._robot.data.root_link_state_w[:, 3:7], self.wind_vel)
        self.external_forces = -1.0 * self.w_coefficient * (self._robot.data.root_com_lin_vel_b - wind_vec_b)

        # PLOT
        # self.dragX.append(self.external_forces[0, 0].item())
        # self.dragY.append(self.external_forces[0, 1].item())
        # self.dragZ.append(self.external_forces[0, 2].item())
        #
        # self.w_vel_X.append(self.wind_vel[0, 0].item())
        # self.w_vel_Y.append(self.wind_vel[0, 1].item())
        #
        self.step_count += 1  # ***
        ###

    def _apply_action(self):
        # print(self.external_forces)
        thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        thrust[:, 0, 0] = self._thrust[:, 0, 0] + self.external_forces[:, 0]
        thrust[:, 0, 1] = self._thrust[:, 0, 1] + self.external_forces[:, 1]
        thrust[:, 0, 2] = self._thrust[:, 0, 2] + self.external_forces[:, 2]
        self._robot.set_external_force_and_torque(thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )

        noise = torch.zeros_like(self.external_forces).normal_(0.0, 0.003)

        obs = torch.cat(
            [
                self._robot.data.root_com_lin_vel_b,
                self._robot.data.root_com_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                # self.o_b1.squeeze(dim=1),
                # self.o_b2.squeeze(dim=1),
                # wind vel
                self.external_forces #+ noise,
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        # print("last")
        # print(self.last_pos[0, :])
        # print("current")
        # print(self.external_forces[0])

        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_com_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(self.dFactor * distance_to_goal - 2.0)
        thrust_change = torch.square(self._thrust[:, 0, 2] - self.last_thrust)
        moment_change = torch.sum(torch.square(self._moment.squeeze(dim=1) - self.last_moment), dim=1)
        # print(moment_change.shape)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "encouraging": self.cfg.encouraging_scale * (torch.linalg.norm(self._desired_pos_w - self.last_pos, dim=1) - distance_to_goal) * self.step_dt,
            # "exceeding_thrust_limit1": (torch.tanh(4 * (10 * self.Forces[:, 0] + 0.5)) - 0.99) * self.step_dt * 15,
            # "exceeding_thrust_limit2": (torch.tanh(4 * (10 * self.Forces[:, 1] + 0.5)) - 0.99) * self.step_dt * 15,
            # "exceeding_thrust_limit3": (torch.tanh(4 * (10 * self.Forces[:, 2] + 0.5)) - 0.99) * self.step_dt * 15,
            # "exceeding_thrust_limit4": (torch.tanh(4 * (10 * self.Forces[:, 3] + 0.5)) - 0.99) * self.step_dt * 15,
            "thrust_smoothing": 1.5 * thrust_change * self.step_dt,
            "moment_smoothing": 1.5 * moment_change * self.step_dt,
            "goal": self.bonus,
        }
        # print(self.step_dt)
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        died = torch.logical_or(self._robot.data.root_link_pos_w[:, 2] < 0.05, self._robot.data.root_link_pos_w[:, 2] > 3.0)

        died = torch.logical_or(died, self.Forces[:, 0] < 0.0)
        died = torch.logical_or(died, self.Forces[:, 1] < 0.0)
        died = torch.logical_or(died, self.Forces[:, 2] < 0.0)
        died = torch.logical_or(died, self.Forces[:, 3] < 0.0)

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        self.check = torch.where(distance_to_goal < 0.5, 1.0, 0.0)  # hovering factor
        self.check_count += self.check
        self.bonus = torch.where(self.check_count > 300.0, 100, 0.0)
        died = torch.logical_or(died, self.check_count > 301.0)
        ###

        forces_change = self.Forces - self.last_forces
        forces_change[:, 0] += self.f_check
        forces_change[:, 1] += self.f_check
        forces_change[:, 2] += self.f_check
        forces_change[:, 3] += self.f_check

        self.f_check[:] = 0.0
        condition = torch.logical_and(torch.sqrt(torch.square(forces_change)) < 5.0, torch.sqrt(torch.square(forces_change)) > 0.05)
        motor = torch.where(condition, 1.0, 0.0)
        motor_rate = torch.sum(motor, dim=1)
        died = torch.logical_or(died, motor_rate > 1.0)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_link_pos_w[env_ids], dim=1
        ).mean()
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

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # sum of action
        # self.total_actions[env_ids] = 0.0
        # self.total_actions[env_ids, 0] = 0.2

        # Sample new commands
        self.dFactor = 0.75  # 1.25, 0.85, 0.5, ... <=> (2, 3, 5, ...)
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-5.0, 5.0)
        #
        # xp = torch.sqrt(torch.square(self._desired_pos_w[env_ids, :2]))
        # self._desired_pos_w[env_ids, :2] *= (2 - torch.tanh(1.5 * (xp - 2.0)))
        #
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # for testing scenario                                                                                              ***
        # scenario 1.
        # goal point
        # self._desired_pos_w[env_ids, 0] = 0.0
        # self._desired_pos_w[env_ids, 1] = 4.0
        # self._desired_pos_w[env_ids, 2] = 1.5
        # # starting point
        # default_root_state[:, 0] = -1.0
        # default_root_state[:, 1] = 0.0
        # default_root_state[:, 2] = 1.0
        ###
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset wind velocity
        # 1. constant wind., same in all envs. 2. different in each env.
        self.wind_vel[env_ids, 0] = torch.zeros_like(self.wind_vel[env_ids, 0]).normal_(0.0, 1.0)  # -1.0  # m/s
        self.wind_vel[env_ids, 1] = torch.zeros_like(self.wind_vel[env_ids, 1]).normal_(0.0, 1.0)
        self.wind_vel[env_ids, 2] = 0.0
        self.external_forces[env_ids, :] = 0.0

        self.last_pos[env_ids, :] = 0.0
        self.Forces[env_ids, :] = 0.0  # 8/05

        # reset goal bonus
        self.check[env_ids] = 0.0
        self.check_count[env_ids] = 0.0
        self.bonus[env_ids] = 0.0
        self.f_check[env_ids] = 10.0

        # for plotting                                                                                                      ***
        # if torch.any(env_ids == 0):
            # plt.plot(range(self.step_count), self.thrust)
            # pic_name = "thrust:{:05d}.jpg".format(self.pic_count_thrust)
            # plt.savefig(pic_name)
            # plt.close()
            # data_thrust = pd.DataFrame(
            #     {'total thrust f': self.thrust, 't': range(self.step_count)})
            # data_thrust.to_csv("Total thrust:{:05d}.csv".format(self.pic_count_thrust), index=False)
            #
            # self.pic_count_thrust += 1
            # self.thrust = []
            #
            # plt.plot(range(self.step_count), self.moment_x)
            # pic_name = "moment_x:{:05d}.jpg".format(self.pic_count_moment)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.moment_y)
            # pic_name = "moment_y:{:05d}.jpg".format(self.pic_count_moment)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.moment_z)
            # pic_name = "moment_z:{:05d}.jpg".format(self.pic_count_moment)
            # plt.savefig(pic_name)
            # plt.close()
            # data_moment = pd.DataFrame(
            #     {'moment X': self.moment_x, 'moment Y': self.moment_y, 'moment Z': self.moment_z, 't': range(self.step_count)})
            # data_moment.to_csv("Moment:{:05d}.csv".format(self.pic_count_moment), index=False)
            #
            # self.pic_count_moment += 1
            # self.moment_x = []
            # self.moment_y = []
            # self.moment_z = []

            # plt.plot(range(self.step_count), self.pos_x)
            # pic_name = "pos_x:{:05d}.jpg".format(self.pic_count_pos)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.pos_y)
            # pic_name = "pos_y:{:05d}.jpg".format(self.pic_count_pos)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.pos_z)
            # pic_name = "pos_z:{:05d}.jpg".format(self.pic_count_pos)
            # plt.savefig(pic_name)
            # plt.close()
            # data_pos = pd.DataFrame(
            #     {'pos X': self.pos_x, 'pos Y': self.pos_y, 'pos Z': self.pos_z, 't': range(self.step_count)})
            # data_pos.to_csv("Position:{:05d}.csv".format(self.pic_count_pos), index=False)
            #
            # self.pic_count_pos += 1
            # self.pos_x = []
            # self.pos_y = []
            # self.pos_z = []
            #
            # plt.plot(range(self.step_count), self.l_vel_x)
            # pic_name = "linear vel x:{:05d}.jpg".format(self.pic_count_l_vel)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.l_vel_y)
            # pic_name = "linear vel y:{:05d}.jpg".format(self.pic_count_l_vel)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.l_vel_z)
            # pic_name = "linear vel z:{:05d}.jpg".format(self.pic_count_l_vel)
            # plt.savefig(pic_name)
            # plt.close()
            # data_l_vel = pd.DataFrame(
            #     {'l_vel X': self.l_vel_x, 'l_vel Y': self.l_vel_y, 'l_vel Z': self.l_vel_z, 't': range(self.step_count)})
            # data_l_vel.to_csv("Linear velocity:{:05d}.csv".format(self.pic_count_l_vel), index=False)

            # self.pic_count_l_vel += 1
            # self.l_vel_x = []
            # self.l_vel_y = []
            # self.l_vel_z = []
            #
            # plt.plot(range(self.step_count), self.a_vel_x)
            # pic_name = "angular vel x:{:05d}.jpg".format(self.pic_count_a_vel)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.a_vel_y)
            # pic_name = "angular vel y:{:05d}.jpg".format(self.pic_count_a_vel)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.a_vel_z)
            # pic_name = "angular vel z:{:05d}.jpg".format(self.pic_count_a_vel)
            # plt.savefig(pic_name)
            # plt.close()
            # data_a_vel = pd.DataFrame(
            #     {'a_vel X': self.a_vel_x, 'a_vel Y': self.a_vel_y, 'a_vel Z': self.a_vel_z, 't': range(self.step_count)})
            # data_a_vel.to_csv("Angular velocity:{:05d}.csv".format(self.pic_count_a_vel), index=False)
            #
            # self.pic_count_a_vel += 1
            # self.a_vel_x = []
            # self.a_vel_y = []
            # self.a_vel_z = []

            # plt.plot(range(self.step_count), self.T1)
            # pic_name = "T1:{:05d}.jpg".format(self.pic_count_T)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.T2)
            # pic_name = "T2:{:05d}.jpg".format(self.pic_count_T)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.T3)
            # pic_name = "T3:{:05d}.jpg".format(self.pic_count_T)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.T4)
            # pic_name = "T4:{:05d}.jpg".format(self.pic_count_T)
            # plt.savefig(pic_name)
            # plt.close()
            # data_T = pd.DataFrame(
            #     {'T1': self.T1, 'T2': self.T2, 'T3': self.T3, 'T4': self.T4, 't': range(self.step_count)})
            # data_T.to_csv("Each_thrust:{:05d}.csv".format(self.pic_count_T), index=False)

            # self.pic_count_T += 1
            # self.T1 = []
            # self.T2 = []
            # self.T3 = []
            # self.T4 = []
            # #
            # plt.plot(range(self.step_count), self.dragX)
            # pic_name = "drag x:{:05d}.jpg".format(self.pic_count_drag)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.dragY)
            # pic_name = "drag y:{:05d}.jpg".format(self.pic_count_drag)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.dragZ)
            # pic_name = "drag z:{:05d}.jpg".format(self.pic_count_drag)
            # plt.savefig(pic_name)
            # plt.close()

            # plt.plot(range(self.step_count), self.w_vel_X)
            # pic_name = "w_vel x:{:05d}.jpg".format(self.pic_count_drag)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.w_vel_Y)
            # pic_name = "w_vel y:{:05d}.jpg".format(self.pic_count_drag)
            # plt.savefig(pic_name)
            # plt.close()

            # data_w_vel = pd.DataFrame(
            #     {'wind vel X': self.w_vel_X, 'wind vel Y': self.w_vel_Y, 't': range(self.step_count)})
            # data_w_vel.to_csv("Wind_vel:{:05d}.csv".format(self.pic_count_drag), index=False)

            # self.w_vel_X = []
            # self.w_vel_Y = []

            # data_drag = pd.DataFrame(
            #     {'drag X': self.dragX, 'drag Y': self.dragY, 'drag Z': self.dragZ, 'X_n': self.dragX_n, 'Y_n': self.dragY_n, 'Z_n': self.dragZ_n, 't': range(self.step_count)})
            # data_drag.to_csv("Wind_drag:{:05d}.csv".format(self.pic_count_drag), index=False)
            #
            # self.pic_count_drag += 1
            # self.dragX = []
            # self.dragY = []
            # self.dragZ = []
            # self.dragX_n = []
            # self.dragY_n = []
            # self.dragZ_n = []
            #
            # plt.plot(range(self.step_count), self.th_change)
            # pic_name = "th change:{:05d}.jpg".format(self.change_count)
            # plt.savefig(pic_name)
            # plt.close()
            # plt.plot(range(self.step_count), self.mo_change)
            # pic_name = "mo change:{:05d}.jpg".format(self.change_count)
            # plt.savefig(pic_name)
            # # plt.close()
            # plt.plot(range(self.step_count), self.T_change, marker='o')
            # pic_name = "T change:{:05d}.jpg".format(self.change_count)
            # plt.savefig(pic_name)
            # plt.close()
            # data_change = pd.DataFrame(
            #     {'T1_change': self.T1_change, 'T2_change': self.T2_change, 'T3_change': self.T3_change, 'T4_change': self.T4_change, 't': range(self.step_count)})
            # data_change.to_csv("Thrust Change:{:05d}.csv".format(self.change_count), index=False)
            #
            # self.change_count += 1
            # self.th_change = []
            # self.mo_change = []
            # self.T1_change = []
            # self.T2_change = []
            # self.T3_change = []
            # self.T4_change = []
            ###                                                                                                               ***
            # self.step_count = 0
        ###

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
        # self.goal_pos_visualizer.visualize(torch.tensor([[0, 0, 1], [0, 5, 1]], device=self.device))