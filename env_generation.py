#这里提供了一个环境生成的类，用于生成环境
from reference.hy_env_add_manipulation import *
from simpler_env.policies.octo.octo_model import OctoInference
from simpler_env.policies.rt1.rt1_model import RT1Inference
arg_x = [  # windowx
    "-e", "PushMultipleDifferentObjectsInScene-v0",
    "-c", "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
    "-o", "rgbd",
    "robot",
    "widowx",
    "sim_freq", "@500",
    "control_freq", "@5",
    "scene_name", "bridge_table_1_v1",
    "rgb_overlay_mode", "background",
    "rgb_overlay_path", "data/real_inpainting/bridge_real_eval_1.png",
    "rgb_overlay_cameras", "3rd_view_camera"
]


arg_google_diff = [#google
    "-e", "GraspMultipleDifferentObjectsInScene-v0",
    "-c", "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
    "-o", "rgbd",
    "robot",
    "google_robot_static",
    "sim_freq", "@500",
    "control_freq", "@5",
    "rgb_overlay_mode", "background",
    "rgb_overlay_path", "data/real_inpainting/google_coke_can_real_eval_1.png",
    "rgb_overlay_cameras", "overhead_camera"
]

arg_google_same = [  # google
    "-e", "GraspMultipleSameObjectsInScene-v0",
    "-c", "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
    "-o", "rgbd",
    "robot",
    "google_robot_static",
    "sim_freq", "@500",
    "control_freq", "@5",
    "rgb_overlay_mode", "background",
    "rgb_overlay_path", "data/real_inpainting/google_coke_can_real_eval_1.png",
    "rgb_overlay_cameras", "overhead_camera"
]

class Method:
    Octo_text = "Octo_text"
    Octo_image = "Octo_image"
    Rtx = "Rtx"

class Rvip_Enivronment:
    def __init__(self,robot_type,model_name,policy_setup):
        self.robot_type = robot_type
        self.env = None
        self.reset_options = None
        self.has_base, self.num_arms, self.has_gripper, self.is_google_robot, self.is_widowx, self.is_gripper_delta_target_control= None,None,None,None,None,None
        self.ee_action_scale, self.ee_rot_action_scale = None,None
        self.action = None
        self.model_name = model_name
        self.policy_setup = policy_setup#widowx_bridge,google_robot
        self.model=None
        self.generate_env()

    def load_model(self):
        if self.model_name == "octo_base":
            self.model = OctoInference(model_type=self.model_name, policy_setup=self.policy_setup, init_rng=0)
        elif self.model_name == "rtx":
            # 请确保ckpt_path路径正确
            ckpt_path = "/home/khl/khl/Acmu/SENV/component/checkpoints /rt_1_x_tf_trained_for_002272480_step"
            self.model = RT1Inference(saved_model_path=ckpt_path, policy_setup=self.policy_setup)

    def parse_env_kwargs(self,opts):
        eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
        return dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))

    def parse_args(self):
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
        args, opts = parser.parse_known_args(self.robot_type)

        args.env_kwargs = self.parse_env_kwargs(opts)
        self.robot_type = args
        return args


    def generate_env(self):
        args = self.parse_args()
        self.env = setup_environment(args)
        self.reset_options = get_env_reset_options(self.env, args)
        (
            self.has_base,self.num_arms,self.has_gripper,self.is_google_robot,self.is_widowx,self.is_gripper_delta_target_control,
        ) = get_robot_info(self.env)
        self.ee_action_scale,self.ee_rot_action_scale = get_action_scale(self.is_google_robot,self.is_widowx)

    def get_action(self,key):
        base_action, ee_action, gripper_action = process_user_input(
        self.robot_type,
        key,
        self.has_base,
        self.num_arms,
        self.has_gripper,
        self.is_google_robot,
        self.ee_action_scale,
        self.ee_rot_action_scale,
        )
        action_dict = create_action_dict(
            base_action, ee_action, gripper_action, self.has_gripper
        )
        self.action = self.env.agent.controller.from_action_dict(action_dict)




    def get_camera_info(self):
        pass