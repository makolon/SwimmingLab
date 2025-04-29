from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import swimlab_tasks.obstacle_navigation.mdp as mdp


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pose_command = mdp.TargetPoseCommandCfg(
        asset_name="robot",
        body_name="base_link",
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        ranges=mdp.TargetPoseCommandCfg.Ranges(
            pos_x=(-0.5, 0.5),
            pos_y=(24.0, 24.0),
            pos_z=(2.0, 2.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    drone_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_command = ObsTerm(
            func=mdp.relative_to_target,
            params={
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        lidar_scan = ObsTerm(
            func=mdp.lidar_scan,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "lidar_range": 4.0,
                "lidar_resolution": (36, 4),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_body_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-10.0, 10.0),
                "y": (-24.0, -24.0),
                "z": (0.0, 1.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        params={
            "std": 10.0,
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
        weight=0.1,
    )
    close_target_enough = RewTerm(
        func=mdp.distance_to_target,
        params={
            "threshold": 5.0,
            "command_name": "pose_command",
        },
        weight=10.0,
    )
    velocity_alignment = RewTerm(
        func=mdp.velocity_alignment_reward,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
        weight=0.01,
    )
    lidar_safety = RewTerm(
        func=mdp.lidar_safety_reward,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "lidar_range": 4.0,
            "lidar_resolution": (36, 4),
        },
        weight=0.005,
    )
    height_penalty = RewTerm(
        func=mdp.height_penalty,
        params={
            "threshold": 3.0,
            "asset_cfg": SceneEntityCfg("robot")
        },
        weight=0.05,
    )
    uprightness = RewTerm(
        func=mdp.upright_reward,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=0.05,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    drone_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("robot")}
    )
    upper_limit = DoneTerm(
        func=mdp.root_height_above_maximum, params={"maximum_height": 4.0, "asset_cfg": SceneEntityCfg("robot")}
    )


##
# Environment configuration
##


@configclass
class ObstacleNavigationBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the drone obstacle navigation environment."""

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 800.0
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_ccd = True
        self.sim.physx.enable_stabilization = True
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.1
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 * 64
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 64
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.solver_type = 0
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.static_friction = 0.5
        self.sim.physics_material.dynamic_friction = 0.5
        self.sim.physics_material.restitution = 0.0
        self.sim.render.enable_translucency = True
        self.sim.render.enable_reflections = True
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

