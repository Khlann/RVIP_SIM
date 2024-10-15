import argparse
import gymnasium as gym
from transforms3d.euler import euler2quat
from mani_skill2_real2sim.envs.sapien_env import BaseEnv
from mani_skill2_real2sim.utils.sapien_utils import look_at
import numpy as np
from sapien.core import Pose
import time

MS1_ENV_IDS = [
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]



def parse_env_kwargs(opts):
    # 你需要实现这个函数来解析额外的参数
    env_kwargs = {}
    for opt in opts:
        if '=' in opt:
            key, value = opt.split('=', 1)
            env_kwargs[key] = value
        else:
            env_kwargs[opt] = True
    return env_kwargs

def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Robot Control Visualization Script")
    parser.add_argument(
        "-e", "--env-id", type=str, default="PushMultipleDifferentObjectsInScene-v0", help="Environment ID"
    )
    parser.add_argument("-o", "--obs-mode", type=str, default="rgbd", help="Observation mode")
    parser.add_argument("--reward-mode", type=str, help="Reward mode")
    parser.add_argument(
        "-c",
        "--control-mode",
        type=str,
        default="arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
        help="Control mode",
    )
    parser.add_argument(
        "--render-mode", type=str, default="cameras", help="Render mode"
    )
    parser.add_argument(
        "--add-segmentation",
        action="store_true",
        help="Add segmentation to observation",
    )
    parser.add_argument(
        "--enable-sapien-viewer", action="store_true", help="Enable SAPIEN viewer"
    )

    if args_list is None:
        args_list = []

    args, opts = parser.parse_known_args(args_list)
    args.env_kwargs = parse_env_kwargs(opts)
    return args

def process_user_input(
    args,
    key,
    has_base,
    num_arms,
    has_gripper,
    is_google_robot,
    ee_action_scale,
    ee_rot_action_scale,
):
    base_action = np.zeros(4) if has_base else np.zeros(0)
    ee_action_dim = (
        6
        if "pd_ee_delta_pose" in args.control_mode
        or "pd_ee_target_delta_pose" in args.control_mode
        else 3
    )
    ee_action = np.zeros(ee_action_dim)
    gripper_action = 0

    if has_base:
        base_action = process_base_input(key, base_action)

    if num_arms > 0:
        ee_action = process_ee_input(
            key, ee_action, ee_action_scale, ee_rot_action_scale
        )

    if has_gripper:
        gripper_action = process_gripper_input(key, is_google_robot)

    return base_action, ee_action, gripper_action

def process_base_input(key, base_action):
    if key == "w":
        base_action[0] = 1
    elif key == "s":
        base_action[0] = -1
    elif key == "a":
        base_action[1] = 1
    elif key == "d":
        base_action[1] = -1
    elif key == "q" and len(base_action) > 2:
        base_action[2] = 1
    elif key == "e" and len(base_action) > 2:
        base_action[2] = -1
    elif key == "z" and len(base_action) > 2:
        base_action[3] = 1
    elif key == "x" and len(base_action) > 2:
        base_action[3] = -1
    return base_action


def process_ee_input(key, ee_action, ee_action_scale, ee_rot_action_scale):
    if key == "i":
        ee_action[0] = ee_action_scale
    elif key == "k":
        ee_action[0] = -ee_action_scale
    elif key == "j":
        ee_action[1] = ee_action_scale
    elif key == "l":
        ee_action[1] = -ee_action_scale
    elif key == "u":
        ee_action[2] = ee_action_scale
    elif key == "o":
        ee_action[2] = -ee_action_scale
    elif key == "1":
        ee_action[3:6] = [ee_rot_action_scale, 0, 0]
    elif key == "2":
        ee_action[3:6] = [-ee_rot_action_scale, 0, 0]
    elif key == "3":
        ee_action[3:6] = [0, ee_rot_action_scale, 0]
    elif key == "4":
        ee_action[3:6] = [0, -ee_rot_action_scale, 0]
    elif key == "5":
        ee_action[3:6] = [0, 0, ee_rot_action_scale]
    elif key == "6":
        ee_action[3:6] = [0, 0, -ee_rot_action_scale]
    return ee_action


def process_gripper_input(key, is_google_robot):
    if not is_google_robot:
        return 1 if key == "f" else -1 if key == "g" else 0
    else:
        return -1 if key == "f" else 1 if key == "g" else 0

def parse_env_kwargs(opts):
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    return dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))


def get_env_reset_options(env, args):
    if hasattr(env, "prepackaged_config") and env.prepackaged_config:
        return {}

    reset_options = {}
    if "GraspSingle" in args.env_id:
        reset_options = setup_grasp_single_options()
    elif "MoveNear" in args.env_id:
        reset_options = setup_move_near_options()
    elif "Drawer" in args.env_id:
        reset_options = setup_drawer_options(args)
    elif "GraspMultipleSameObjectsInScene" in args.env_id:
        reset_options = setup_grasp_multi_options()
    if args.env_id == "GraspMultipleDifferentObjectsInScene-v0":
        reset_options = setup_grasp_multi_different_options()
    elif "PushMultipleDifferentObjectsInScene" in args.env_id:
        reset_options = setup_push_multi_different_options()
    elif any(
        task in args.env_id
        for task in [
            "PutSpoonOnTableCloth",
            "PutCarrotOnPlate",
            "StackGreenCubeOnYellowCube",
            "PutEggplantInBasket",
        ]
    ):
        reset_options = setup_task_options(env)

    return reset_options


def setup_grasp_multi_options():
    custom_options = {
        "obj_init_options": {
            "init_xy_0": [-0.2, 0.2],  # Custom position for the first can
            "orientation_0": "upright",  # Custom orientation for the first can
            "init_xy_1": [-0.35, 0.2],  # Custom position for the second can
            "orientation_1": "upright",  # Custom orientation for the first can
            "init_xy_2": [-0.2, 0.35],  # Custom position for the third can
            "orientation_2": "upright",  # Custom orientation for the third can
            "init_xy_3": [-0.35, 0.35],  # Custom position for the fourth can
            "orientation_3": "upright",  # Custom orientation for the fourth can
            "init_xy_4": [-0.2, 0.09],  # Custom position for the fifth can
            "orientation_4": "upright",  # Custom orientation for the fifth can
            "init_xy_5": [-0.35, 0.05],  # Custom position for the sixth can
            "orientation_5": "laid_vertically",  # Custom orientation for the sixth can
            "init_xy_6": [-0.47, 0.15],  # Custom position for the seventh can
            "orientation_6": "upright",  # Custom orientation for the seventh can
            "init_xy_7": [-0.5, 0.3],  # Custom position for the eighth can
            "init_rot_quat_7": euler2quat(
                np.pi / 4, np.pi / 4, 0
            ),  # Custom orientation for the second can (45 degrees around y-axis)
            "init_xy_8": [-0.2, -0.15],  # Custom position for the ninth can
            "orientation_8": "upright",  # Custom orientation for the ninth can
            "init_xy_9": [-0.35, 0.65],  # Custom position for the tenth can
            "orientation_9": "laid_vertically",  # Custom orientation for the tenth can
        }
        # "obj_init_options": {
        #     "init_xy_0": [-0.2, 0.2],  # Custom position for the first can
        #     "orientation_0": "upright",  # Custom orientation for the first can
        #     "init_xy_1": [-0.35, 0.2],  # Custom position for the second can
        #     "orientation_1": "upright",  # Custom orientation for the first can
        #     "init_xy_2": [-0.3,0.23],  # Custom position for the third can
        #     "orientation_2": "upright",  # Custom orientation for the third can
        #     "init_xy_3": [0,0],  # Custom position for the fourth can
        #     "orientation_3": "upright",  # Custom orientation for the fourth can
        #     "init_xy_4": [0,0],  # Custom position for the fifth can
        #     "orientation_4": "upright",  # Custom orientation for the fifth can
        #     "init_xy_5": [0,0],  # Custom position for the sixth can
        #     "orientation_5": "laid_vertically",  # Custom orientation for the sixth can
        #     "init_xy_6": [0,0],  # Custom position for the seventh can
        #     "orientation_6": "upright",  # Custom orientation for the seventh can
        #     "init_xy_7": [0,0],  # Custom position for the eighth can
        #     "init_rot_quat_7": euler2quat(
        #         np.pi / 4, np.pi / 4, 0
        #     ),  # Custom orientation for the second can (45 degrees around y-axis)
        #     "init_xy_8": [0,0],  # Custom position for the ninth can
        #     "orientation_8": "upright",  # Custom orientation for the ninth can
        #     "init_xy_9": [0,0],  # Custom position for the tenth can
        #     "orientation_9": "laid_vertically",  # Custom orientation for the tenth can
        # }
    }
    return custom_options


def setup_grasp_multi_different_options():
    custom_options = {
        "model_ids": [
            "bridge_spoon_generated_modified",
            "blue_plastic_bottle",
            "apple",
            "eggplant",
        ],
        "obj_init_options": {
            "init_xy_0": [-0.2, 0.2],
            "orientation_0": "upright",
            "init_xy_1": [-0.35, 0.02],
            "orientation_1": "laid_vertically",
            "init_xy_2": [-0.2, 0.35],
            # "init_rot_quat_2": euler2quat(np.pi / 2, 0, 0),
            "init_xy_3": [-0.35, 0.23],
            "init_rot_quat_3": euler2quat(np.pi / 4, 0, 0),
            "init_xy_4": [-0.2, 0.09],
            "orientation_4": "upright",
            "init_xy_5": [-0.35, 0.05],
            "orientation_5": "laid_vertically",
            "init_xy_6": [-0.47, 0.15],
            "orientation_6": "upright",
            "init_xy_7": [-0.5, 0.3],
            "init_rot_quat_7": euler2quat(np.pi / 4, np.pi / 4, 0),
            "init_xy_8": [-0.2, -0.15],
            "orientation_8": "upright",
            "init_xy_9": [-0.35, 0.65],
            "orientation_9": "laid_vertically",
        },
    }
    return custom_options


def setup_push_multi_different_options():
    custom_options = {
        "model_ids": [
            "bridge_spoon_generated_modified",
            "blue_plastic_bottle",
            "apple",
            "eggplant",
        ],
        "obj_init_options": {
            "init_xy_0": [-0.096, 0.011],
            "orientation_0": "upright",
            "init_xy_1": [-0.141, 0.193],
            "orientation_1": "laid_vertically",
            "init_xy_2": [-0.208, 0.084],
            # "init_rot_quat_2": euler2quat(np.pi / 2, 0, 0),
            "init_xy_3": [-0.2, 0.23],
            "init_xy_4": [-0.2, 0.09],
            "orientation_4": "upright",
            "init_xy_5": [-0.35, 0.05],
            "orientation_5": "laid_vertically",
            "init_xy_6": [-0.47, 0.15],
            "orientation_6": "upright",
            "init_xy_7": [-0.5, 0.3],
            "init_rot_quat_7": euler2quat(np.pi / 4, np.pi / 4, 0),
            "init_xy_8": [-0.2, -0.15],
            "orientation_8": "upright",
            "init_xy_9": [-0.35, 0.65],
            "orientation_9": "laid_vertically",
        },
    }
    return custom_options


def setup_grasp_single_options():
    init_rot_quat = Pose(q=[0, 0, 0, 1]).q
    return {
        "obj_init_options": {"init_xy": [-0.12, 0.2]},
        "robot_init_options": {
            "init_xy": [0.35, 0.20],
            "init_rot_quat": init_rot_quat,
        },
    }


def setup_move_near_options():
    init_rot_quat = (Pose(q=euler2quat(0, 0, -0.09)) * Pose(q=[0, 0, 0, 1])).q
    return {
        "obj_init_options": {"episode_id": 0},
        "robot_init_options": {
            "init_xy": [0.35, 0.21],
            "init_rot_quat": init_rot_quat,
        },
    }


def setup_drawer_options(args):
    init_rot_quat = [0, 0, 0, 1]
    init_xy = [0.652, 0.009] if "PlaceInClosedDrawer" in args.env_id else [0.851, 0.035]
    return {
        "obj_init_options": {"init_xy": [0.0, 0.0]},
        "robot_init_options": {
            "init_xy": init_xy,
            "init_rot_quat": init_rot_quat,
        },
    }


def setup_environment(args):
    if (
        args.env_id in MS1_ENV_IDS
        and args.control_mode is not None
        and not args.control_mode.startswith("base")
    ):
        args.control_mode = "base_pd_joint_vel_arm_" + args.control_mode

    if "robot" in args.env_kwargs:
        setup_camera_pose(args)

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        camera_cfgs={"add_segmentation": args.add_segmentation},
        **args.env_kwargs,
    )

    return env


def setup_task_options(env):
    init_rot_quat = Pose(q=[0, 0, 0, 1]).q
    if env.robot_uid == "widowx":
        init_xy = [0.147, 0.028]
    elif env.robot_uid == "widowx_camera_setup2":
        init_xy = [0.147, 0.070]
    elif env.robot_uid == "widowx_sink_camera_setup":
        init_xy = [0.127, 0.060]
    else:
        init_xy = [0.147, 0.028]

    return {
        "obj_init_options": {"episode_id": 0},
        "robot_init_options": {
            "init_xy": init_xy,
            "init_rot_quat": init_rot_quat,
        },
    }


def setup_camera_pose(args):
    if (
        "google_robot" in args.env_kwargs["robot"]
        or "widowx" in args.env_kwargs["robot"]
    ):
        pose = look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.7])
        args.env_kwargs["render_camera_cfgs"] = {
            "render_camera": dict(p=pose.p, q=pose.q)
        }

def get_image(obs,view,name):
    if name == "Segmentation":
        image = obs["image"][view][name]
        image = image.astype(np.uint8)
        return image
    else:
        image = obs["image"][view][name]
        return image
# def get_robot_info(env):
#     has_base = "base" in env.agent.controller.configs
#     has_gripper = any("gripper" in x for x in env.agent.controller.configs)
#     is_google_robot = "google_robot" in env.agent.robot.name
#     is_widowx = "wx250s" in env.agent.robot.name
#     return has_base, has_gripper, is_google_robot, is_widowx

def get_action_scale(is_google_robot, is_widowx):
    ee_action = 0.02 if (is_google_robot or is_widowx) else 0.1
    ee_rot_action = 0.1 if (is_google_robot or is_widowx) else 1.0
    return ee_action, ee_rot_action

def get_robot_info(env):
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    is_google_robot = "google_robot" in env.agent.robot.name
    is_widowx = "wx250s" in env.agent.robot.name
    is_gripper_delta_target_control = (
        env.agent.controller.controllers["gripper"].config.use_target
        and env.agent.controller.controllers["gripper"].config.use_delta
    )
    return (
        has_base,
        num_arms,
        has_gripper,
        is_google_robot,
        is_widowx,
        is_gripper_delta_target_control,
    )


def create_action_dict(base_action, ee_action, gripper_action, has_gripper):
    action_dict = {"base": base_action, "arm": ee_action}
    if has_gripper:
        action_dict["gripper"] = gripper_action
    return action_dict


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()
    env = setup_environment(args)
    reset_options = get_env_reset_options(env, args)

    obs, info = env.reset(options=reset_options)
    print("Reset info:", info)
    print("Instruction:", env.unwrapped.get_language_instruction())
    print("Robot pose:", env.agent.robot.pose)
    print("Initial qpos:", env.agent.robot.get_qpos())

    while True:
        render_frame = env.render()
        env.render_human()

        if "GraspSingle" in args.env_id:
            print("Object pose:", env.obj.get_pose())
        elif "GraspMultipleSameObjectsInScene" in args.env_id:
            for obj in env.objects[:1]:
                print(f"Object {obj.name} pose: {obj.get_pose()}")

        print("TCP pose wrt world:", env.tcp.pose)

        # Determine object position based on environment type
        if "GraspSingle" in args.env_id:
            object_position = env.obj.get_pose().p
        elif "GraspMultipleSameObjectsInScene" in args.env_id:
            object_position = env.objects[0].get_pose().p
        elif "GraspMultipleDifferentObjectsInScene" in args.env_id:
            object_position = env.objects[0].get_pose().p
        elif "PushMultipleDifferentObjectsInScene" in args.env_id:
            object_position = env.objects[0].get_pose().p
        else:
            object_position = env.episode_objs[0].get_pose().p

        print("Object position:", object_position)

        # Simulate the environment without actions
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        time.sleep(1.0 / 20)  # Limit to 20 FPS

        if terminated or truncated:
            obs, info = env.reset(options=reset_options)
            print("Environment reset")


if __name__ == "__main__":
    main()
