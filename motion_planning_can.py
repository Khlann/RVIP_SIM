import cv2
# from pygame.examples.go_over_there import target_position

from hy_env.example_env.mani_skill2_real2sim.examples.test_oritentation import target_rot
from mani_skill2_real2sim.envs.sapien_env import BaseEnv
from mani_skill2_real2sim.utils.sapien_utils import look_at
from enum import Enum

from component.reference.hy_env_add_manipulation import *

MS1_ENV_IDS = [
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]

class GraspState(Enum):
    APPROACH = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4

class GraspMotionPlanner:
    def __init__(
        self,
        ee_action_scale=0.02,
        ee_rot_action_scale=0.1,
        dr=0.005,
        dH=0.1,
        dh=-0.008,
        dSH=1.05,
        is_google_robot=False,
    ):
        self.ee_action_scale = ee_action_scale
        self.ee_rot_action_scale = ee_rot_action_scale
        self.dr = dr  # tolerance in x and y
        self.dH = dH  # height above object for approach
        self.dSH = dSH  # height above object for lifting
        self.dh = dh  # distance from object for grasping
        self.current_state = GraspState.APPROACH
        self.is_google_robot = is_google_robot

    def moveto(self, object_position, robot_qpos, tcp_pose):
        threshold = 0.1
        x0, y0, z0 = object_position
        p, q = tcp_pose.p, tcp_pose.q
        q = np.array([q[1], q[2], q[3], q[0]])  # turn into [x, y, z, w]
        x1, y1, z1 = p
        self.gripper_pose = robot_qpos[-3]
        # Target orientation (pointing downwards)
        target_q = np.array(
            [0, -0.707, -0.707, 0]
        )  # [w, x, y, z]
        target_q = np.array([target_q[1], target_q[2], target_q[3], target_q[0]])
        # Initialize ee_action
        ee_action = np.zeros(6)
        # Calculate position error
        position_error = np.array([x0 - x1, y0 - y1, z0 - z1])
        pe=position_error
        # print("Position error:", position_error)
        # print(np.linalg.norm(position_error) )
        # Check if the position error is within the threshold
        # if np.linalg.norm(position_error) < threshold or np.linalg.norm(position_error)== threshold:
        #     print("Target position reached.")
        #     return ee_action, 0, True  # Return with success flag

        orientation_error = np.zeros(3)
        # Determine target z based on current state
        target_z = z0 + self.dH
        for i in range(3):
            if i == 2:  # z-axis
                error = z1 - target_z
            else:
                if self.current_state != GraspState.LIFT:
                    error = position_error[i]
                else:
                    error = 0

            ee_action[i] = np.clip(
                -error, -self.ee_action_scale, self.ee_action_scale
            )

        # Determine gripper action
        gripper_action = 0  # Keep gripper open
        # Determine next state
        self._update_state(position_error, orientation_error, z1, z0)

        # print("Current state:", self.current_state)

        return ee_action, gripper_action, pe

    def plan_motion(self, object_position, robot_qpos, tcp_pose):
        x0, y0, z0 = object_position
        p, q = tcp_pose.p, tcp_pose.q  # (x, y, z), (w, x, y, z)
        q = np.array([q[1], q[2], q[3], q[0]])  # turn into [x, y, z, w]
        x1, y1, z1 = p
        self.gripper_pose = robot_qpos[-3]

        # Target orientation (pointing downwards)
        target_q = np.array(
            [0, -0.707, -0.707, 0]
        )  # [w, x, y, z]
        target_q = np.array([target_q[1], target_q[2], target_q[3], target_q[0]])

        # Initialize ee_action
        ee_action = np.zeros(6)

        # Calculate position error
        position_error = np.array([x0 - x1, y0 - y1, z0 - z1])
        print("Position error:", position_error)

        orientation_error = np.zeros(3)

        # Determine target z based on current state
        if self.current_state == GraspState.APPROACH:
            target_z = z0 + self.dH
        elif self.current_state == GraspState.DESCEND:
            target_z = z0 + self.dh
        elif self.current_state == GraspState.LIFT:
            target_z = self.dSH
        else:  # GRASP state
            target_z = z1  # Maintain current z position

        # Calculate ee_action for position
        if self.current_state in [
            GraspState.APPROACH,
            GraspState.DESCEND,
            GraspState.LIFT,
        ]:
            for i in range(3):
                if i == 2:  # z-axis
                    error = z1 - target_z
                else:
                    if self.current_state != GraspState.LIFT:
                        error = position_error[i]
                    else:
                        error = 0

                ee_action[i] = np.clip(
                    -error, -self.ee_action_scale, self.ee_action_scale
                )

        # Determine gripper action
        if self.current_state in [GraspState.GRASP, GraspState.LIFT]:
            gripper_action = 1 if self.is_google_robot else -1
        else:
            gripper_action = 0  # Keep gripper open

        # Determine next state
        self._update_state(position_error, orientation_error, z1, z0)

        print("Current state:", self.current_state)
        return ee_action, gripper_action

    def _update_state(self, position_error, orientation_error, z1, z0):
        GRASP_THRESHOLD = 0.027 if not self.is_google_robot else 0.58
        if self.current_state == GraspState.APPROACH:
            if (
                np.linalg.norm(position_error[:2]) < self.dr
                and abs(z1 - (z0 + self.dH)) < 0.01
            ):
                self.current_state = GraspState.DESCEND
        elif self.current_state == GraspState.DESCEND:
            if (
                np.linalg.norm(position_error[:2]) < self.dr
                and abs(z1 - (z0 + self.dh)) < 0.01
            ):
                self.current_state = GraspState.GRASP
        elif self.current_state == GraspState.GRASP:
            if self.gripper_pose >= GRASP_THRESHOLD:
                self.current_state = GraspState.LIFT
        elif self.current_state == GraspState.LIFT:
            if z1 > z0 + self.dH - 0.01:
                pass  # Maintain lift state

    def get_current_state(self):
        return self.current_state

    def reset(self):
        self.current_state = GraspState.APPROACH

#这个类是为了方便调用环境
class environment:
    def __init__(self):
        args_list = [
            "-e",
            "GraspMultipleSameObjectsInScene-v0",
            "-c",
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
            "-o",
            "rgbd",
            "robot",
            "google_robot_static",
            "rgb_overlay_mode", "background",
            "rgb_overlay_path",
            "/home/khl/khl/Acmu/SENV/component/data/real_inpainting/google_coke_can_real_eval_1.png",
            "rgb_overlay_cameras", "overhead_camera"
        ]
        self.args = args_list


    def parse_args(self):
        parser = argparse.ArgumentParser(description="Robot Control Visualization Script")
        parser.add_argument(
            "-e", "--env-id", type=str, required=True, help="Environment ID"
        )
        parser.add_argument("-o", "--obs-mode", type=str, help="Observation mode")
        parser.add_argument("--reward-mode", type=str, help="Reward mode")
        parser.add_argument(
            "-c",
            "--control-mode",
            type=str,
            default="pd_ee_delta_pose",
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

        self.args, opts = parser.parse_known_args(self.args)
        eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
        self.args.env_kwargs=dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))

        return self.args


class EnvironmentManager:
    def __init__(self, args):
        self.args = args
        self.env = self.setup_environment()

    def setup_environment(self):
        args = self.args
        if (
            args.env_id in MS1_ENV_IDS
            and args.control_mode is not None
            and not args.control_mode.startswith("base")
        ):
            args.control_mode = "base_pd_joint_vel_arm_" + args.control_mode

        if "robot" in args.env_kwargs:
            self.setup_camera_pose()

        env: BaseEnv = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            camera_cfgs={"add_segmentation": args.add_segmentation},
            **args.env_kwargs,
        )

        self.print_env_info(env)
        return env

    def setup_camera_pose(self):
        args = self.args
        if (
            "google_robot" in args.env_kwargs["robot"]
            or "widowx" in args.env_kwargs["robot"]
        ):
            pose = look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.7])
            args.env_kwargs["render_camera_cfgs"] = {
                "render_camera": dict(p=pose.p, q=pose.q)
            }

    def print_env_info(self, env):
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        print("Control mode:", env.control_mode)
        print("Reward mode:", env.reward_mode)

    def get_env_reset_options(self):
        args = self.args
        env = self.env
        if hasattr(env, "prepackaged_config") and env.prepackaged_config:
            return {}

        reset_options = {}
        if "GraspSingle" in args.env_id:
            reset_options = self.setup_grasp_single_options()
        elif "MoveNear" in args.env_id:
            reset_options = self.setup_move_near_options()
        elif "Drawer" in args.env_id:
            reset_options = self.setup_drawer_options()
        elif "GraspMultipleSameObjectsInScene" in args.env_id:
            reset_options = self.setup_grasp_multi_options()
        elif any(
            task in args.env_id
            for task in [
                "PutSpoonOnTableCloth",
                "PutCarrotOnPlate",
                "StackGreenCubeOnYellowCube",
                "PutEggplantInBasket",
            ]
        ):
            reset_options = self.setup_task_options(env)

        return reset_options

    def setup_grasp_single_options(self):
        init_rot_quat = Pose(q=[0, 0, 0, 1]).q
        return {
            "obj_init_options": {"init_xy": [-0.12, 0.2]},
            "robot_init_options": {
                "init_xy": [0.35, 0.20],
                "init_rot_quat": init_rot_quat,
            },
        }

    def setup_move_near_options(self):
        init_rot_quat = (Pose(q=euler2quat(0, 0, -0.09)) * Pose(q=[0, 0, 0, 1])).q
        return {
            "obj_init_options": {"episode_id": 0},
            "robot_init_options": {
                "init_xy": [0.35, 0.21],
                "init_rot_quat": init_rot_quat,
            },
        }

    def setup_drawer_options(self):
        args = self.args
        init_rot_quat = [0, 0, 0, 1]
        init_xy = [0.652, 0.009] if "PlaceInClosedDrawer" in args.env_id else [0.851, 0.035]
        return {
            "obj_init_options": {"init_xy": [0.0, 0.0]},
            "robot_init_options": {
                "init_xy": init_xy,
                "init_rot_quat": init_rot_quat,
            },
        }

    def setup_grasp_multi_options(self):
        custom_options = {
            "obj_init_options": {
                "init_xy_0": [-0.2, 0.2],  # Custom position for the first can
                "orientation_0": "upright",  # Custom orientation for the first can
                "init_xy_1": [-0.35, 0.2],  # Custom position for the second can
                "orientation_1": "upright",
                "init_xy_2": [-0.2, 0.35],  # Custom position for the third can
                "orientation_2": "upright",
                # Add additional configurations as needed
            }
        }
        return custom_options

    def setup_task_options(self, env):
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

class Show:
    def __init__(self):
        self.filename="mpc.mp4"
        self.images=[]#这个用于保存视频帧
        self.points = []
        pass

    def frames_visualization(self):
        frames=self.images
        # 将视频帧保存为视频文件
        height, width, _ = frames[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(self.filename, fourcc, 10, (width, height))  # 帧率为10
        for frame in frames:
            video.write(frame)
        video.release()

    def get_image(self,obs, view, name):
        if name == "Segmentation":
            image = obs["image"][view][name]
            image = image.astype(np.uint8)
            return image
        else:
            image = obs["image"][view][name]
            return image

    def show_robot_info(self,env,info):
        print("Reset info:", info)
        print("Instruction:", env.unwrapped.get_language_instruction())
        print("Robot pose:", env.agent.robot.pose)
        print("Initial qpos:", env.agent.robot.get_qpos())

    def image_mode(self,env,obs,mode):
        #这里选择在线显示还是保存视频
        if mode=="online":
            render_frame = env.render()
            render_frame = cv2.cvtColor(render_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("render", render_frame)
            cv2.waitKey(1)
        elif mode=="offline":
            image = get_image(obs, 'overhead_camera', "rgb")
            self.images.append(image)

    def save_to_video(self):
        for i in range(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2RGB)
        # 将images保存为本地视频
        self.frames_visualization()

    def get_tcp_pose(self,env):
        print("TCP pose wrt world:", env.tcp.pose)

    def get_points(self,frame, delay=1):

        # 定义鼠标回调函数
        def mouse_callback(event, x, y, flags, param):
            # 如果检测到鼠标左键点击事件
            if event == cv2.EVENT_LBUTTONDOWN:
                # 将点击的点加入到列表中
                self.points.append((x, y))
                # 在图像上标记点击的位置
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                # 显示更新后的图像
                cv2.imshow("Image", frame)
                # 打印点击的点
                print(f"Point {len(self.points)}: ({x}, {y})")


        # cv2.imshow("OpenCVViewer", np.zeros((100, 100, 3)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 创建窗口
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", mouse_callback)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(delay)
        if key == -1:  # timeout
            return None
        elif key == 27:  # escape
            exit(0)
        else:
            return chr(key)

    def cal_p2w(self,obs,num):
        # 提取相机内参和外参
        intrinsic_matrix = obs["camera_param"]["base_camera"]["intrinsic_cv"]
        extrinsic_matrix = obs["camera_param"]["base_camera"]["extrinsic_cv"]
        K = intrinsic_matrix
        u, v = self.points[num]
        Z = obs["image"]["base_camera"]["depth"][v, u]  # 这里可能有问题
        R = extrinsic_matrix[:3, :3]
        t = extrinsic_matrix[:3, 3]
        P_uv = np.array([u, v, 1])

        # 计算 K^-1 * ZP_uv
        ZP_uv = Z * P_uv
        K_inv = np.linalg.inv(K)
        camera_coords = K_inv.dot(ZP_uv)

        # 计算世界坐标 P_w
        P_inv = np.linalg.inv(R)
        P_w = P_inv.dot(camera_coords - t)
        print("世界坐标 P_w:", P_w)
        return P_w


class RobotController:
    def __init__(self, env):
        self.env = env
        (
            self.has_base,
            self.num_arms,
            self.has_gripper,
            self.is_google_robot,
            self.is_widowx,
            self.is_gripper_delta_target_control,
        ) = self.get_robot_info()
        (
            self.ee_action_scale,
            self.ee_rot_action_scale,
        ) = self.get_action_scale()

    def get_robot_info(self):
        env = self.env
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

    def get_action_scale(self):
        is_google_robot = self.is_google_robot
        is_widowx = self.is_widowx
        ee_action = 0.02 if (is_google_robot or is_widowx) else 0.1
        ee_rot_action = 0.1 if (is_google_robot or is_widowx) else 1.0
        return ee_action, ee_rot_action



def main():
    np.set_printoptions(suppress=True, precision=3)
    start_up=environment()
    args = start_up.parse_args()
    env = setup_environment(args)
    reset_options = get_env_reset_options(env, args)

    robot_controller = RobotController(env)
    planner = GraspMotionPlanner(
        ee_action_scale=robot_controller.ee_action_scale,
        ee_rot_action_scale=robot_controller.ee_rot_action_scale,
        dh=-0.008 if robot_controller.is_google_robot else 0.005,
        is_google_robot=robot_controller.is_google_robot,
    )
    show=Show()

    # timestep = 0
    obs, info = env.reset(options=reset_options)


    #在这里记录两个点的像素坐标
    while len(show.points) < 2:
        frame = obs["image"]["base_camera"]["rgb"]
        show.get_points(frame)

    p1=show.cal_p2w(obs,num=0)
    p2=show.cal_p2w(obs,num=1)

    p1_down = p1.copy()
    p1_down[2] = p1[2]-0.2
    p2_down = p2.copy()
    p2_down[2] = p2[2]-0.2

    p1_up=p1.copy()
    p1_up[2] = p1[2]+0
    p2_up=p2.copy()
    p2_up[2] = p2[2]+0
    point = [p1,p1,p1,p1_down,p1_down,p1_down,p1_down,p1,p1,p1,p1,p2,p2_down]
    state = 0

    current_error = None
    while True:
        show.image_mode(env,obs,"online")
        object_position = env.objects[0].get_pose().p
        qpos = env.agent.robot.get_qpos()
        # 获取当前末端执行器的位置
        tcp_pose = env.tcp.pose


        previous_error = current_error
        ee_action, gripper_action,error= planner.moveto(point[state], qpos,tcp_pose)

        current_error=error

        if previous_error is not None:
            # error_difference = np.linalg.norm(np.array(current_error) - np.array(previous_error))
            error_difference = (current_error[0]-previous_error[0])**2+(current_error[1]-previous_error[1])**2+(current_error[2]-previous_error[2])**2
            #开根号
            error_difference=error_difference**0.5
            print(error_difference)
            if error_difference < 0.0001:
                print("Target position reached.")
                state+=1
                # break

        base_action = (
            np.zeros(4) if robot_controller.has_base else np.zeros(0)
        )
        action_dict = {
            "base": base_action,
            "arm": ee_action,
        }

        if robot_controller.has_gripper:
            action_dict["gripper"] = gripper_action
        action = env.agent.controller.from_action_dict(action_dict)

        obs, reward, terminated, truncated, info = env.step(action)
    # current_error = None
    # while True:
    #     show.image_mode(env,obs,"online")
    #     object_position = env.objects[0].get_pose().p
    #     qpos = env.agent.robot.get_qpos()
    #     # 获取当前末端执行器的位置
    #     tcp_pose = env.tcp.pose
    #
    #
    #     previous_error = current_error
    #     ee_action, gripper_action,error= planner.moveto(point[state], qpos,tcp_pose)
    #
    #     current_error=error
    #
    #     if previous_error is not None:
    #         error_difference = np.linalg.norm(np.array(current_error) - np.array(previous_error))
    #         print(error_difference)
    #         if error_difference < 0.00002:
    #             print("Target position reached.")
    #             state+=1
    #             break
    #
    #     base_action = (
    #         np.zeros(4) if robot_controller.has_base else np.zeros(0)
    #     )
    #     action_dict = {
    #         "base": base_action,
    #         "arm": ee_action,
    #     }
    #
    #     if robot_controller.has_gripper:
    #         action_dict["gripper"] = gripper_action
    #     action = env.agent.controller.from_action_dict(action_dict)
    #
    #     obs, reward, terminated, truncated, info = env.step(action)

    show.save_to_video()
    env.close()


if __name__ == "__main__":
    main()
