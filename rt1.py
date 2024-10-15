# import os
#
# import cv2
# import numpy as np
# # import simpler_env
# from SimplerEnv.simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
# import mediapy
# from component.reference.hy_env_add_manipulation import *
#
# RT_1_CHECKPOINTS = {
#     "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
#     "rt_1_400k": "rt_1_tf_trained_for_000400120",
#     "rt_1_58k": "rt_1_tf_trained_for_000058240",
#     "rt_1_1k": "rt_1_tf_trained_for_000001120",
# }
#
#
#
#
# def frames_visulization( frames, video_path):
#     # 将视频帧保存为视频文件
#     height, width, _ = frames[0].shape
#
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))  # 帧率为10
#     for frame in frames:
#         video.write(frame)
#     video.release()
#
# import simpler_env
# def run():
#     np.set_printoptions(suppress=True, precision=3)
#     args_list = [
#         "-e", "GraspSingleOpenedCokeCanInScene-v0",
#         "-c", "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
#         "-o", "rgbd",
#         "robot",
#         "google_robot_static",
#         "sim_freq", "@500",
#         "control_freq", "@5",
#         # "scene_name", "bridge_table_1_v1",
#         "rgb_overlay_mode", "background",
#         "rgb_overlay_path", "/home/khl/khl/Acmu/SENV/component/data/real_inpainting/google_coke_can_real_eval_1.png",
#         "rgb_overlay_cameras", "overhead_camera"
#     ]
#
#     # args_list = [
#     #     "-e", "MoveNearGoogleBakedTexInScene-v1",
#     #     "-c", "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
#     #     "--enable-sapien-viewer",
#     #     "-o", "rgbd",
#     #     "robot",
#     #     "google_robot_static",
#     #     "sim_freq", "@501",
#     #     "control_freq", "@3",
#     #     "scene_name", "google_pick_coke_can_1_v4",
#     #     "rgb_overlay_mode", "debug",
#     #     "rgb_overlay_path", "/home/khl/khl/Acmu/SENV/hy_env/example_env/bridge_real_eval_1.png",
#     #     "rgb_overlay_cameras", "overhead_camera"
#     # ]
#
#     args = parse_args(args_list)
#     # args = parse_args()
#     # task_name = "google_robot_pick_coke_can"
#     # env = simpler_env.make(task_name)
#     env = setup_environment(args)
#     reset_options = get_env_reset_options(env, args)
#
#     obs, reset_info = env.reset(options=reset_options)
#     instruction = env.get_language_instruction()
#
#     # instruction = 'move to the coke can'
#     print("Reset info", reset_info)
#     print("Instruction", instruction)
#     policy_setup = "google_robot"
#     model_name = "rt_1_x"
#     from SimplerEnv.simpler_env.policies.rt1.rt1_model import RT1Inference
#     # ckpt_path = get_rt_1_checkpoint(model_name)
#     ckpt_path = "/home/khl/khl/Acmu/SENV/component/checkpoints /rt_1_x_tf_trained_for_002272480_step"
#     model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
#
#     obs, reset_info = env.reset()
#     instruction = env.get_language_instruction()
#     # instruction = 'stop'
#     model.reset(instruction)
#     print(instruction)
#
#     # image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
#     image = get_image(obs,'overhead_camera',"rgb")
#     # image = get_image(obs, '3rd_view_camera', "rgb")
#     images = [image]
#     predicted_terminated, success, truncated = False, False, False
#     timestep = 0
#     # while not (predicted_terminated or truncated):
#     iteration = 0
#     # while iteration < 100:
#     while not (predicted_terminated or truncated):
#         iteration+=1
#         # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
#         raw_action, action = model.step(image)
#         predicted_terminated = bool(action["terminate_episode"][0] > 0)
#         # temp=np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
#         obs, reward, success, truncated, info = env.step(
#             np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
#         )
#         print(timestep, info)
#         # update image observation
#         image = get_image_from_maniskill2_obs_dict(env, obs)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         images.append(image)
#         timestep += 1
#
#     episode_stats = info.get("episode_stats", {})
#     print(f"Episode success: {success}")
#     #将images保存为本地视频
#     frames_visulization(images, "rt1.mp4")
#
# if __name__ == "__main__":
#     run()


import os

import cv2
import numpy as np
# import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy

# from component.octo import policy_setup
from reference.hy_env_add_manipulation import *

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}

def frames_visualization(frames, video_path):
    # 将视频帧保存为视频文件
    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))  # 帧率为10
    for frame in frames:
        video.write(frame)
    video.release()

import simpler_env

def run(env, model, run_id):
    obs, reset_info= env.reset(options=reset_options)

    # print("Reset info", reset_info)
    # # obs, reset_info = env.reset()
    hi=obs["image"]["overhead_camera"]["rgb"]
    # hi=obs["image"]["3rd_view_camera"]["rgb"]
    # # #根据id保存图片
    hi = cv2.cvtColor(hi, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"rt{run_id}_0.png", hi)

    instruction = env.get_language_instruction()
    # instruction = 'Pick the first green cube'
    # instruction = 'Move the apple towards the spoon'#1
    # instruction = "Pick up the Coke can on the far right of the middle row"#2.1
    # instruction = "Pick up the Coke in the bottom right corner"#2.2
    instruction = "Place the blocks of the same color together"#3.1
    # instruction = "Place the green blocks together and the yellow blocks together"#3.2
    # instruction = "Arrange the blocks in a straight line, with the yellow side on one end and the green side on the other."#3.3
    model.reset(instruction)
    print(f"第{run_id}次测试，指令：{instruction}")

    # image = get_image(obs, '3rd_view_camera', "rgb")
    image = get_image(obs, 'overhead_camera', "rgb")
    images = [image]
    predicted_terminated, success, truncated = False, False, False
    timestep = 0

    # while not (predicted_terminated or truncated):
    while timestep < 300:
        # 模型步进
        raw_action, action = model.step(image)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        obs, reward, success, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )
        print(timestep, info)
        # 更新图像观察
        image = get_image_from_maniskill2_obs_dict(env, obs)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})
    print(f"第{run_id}次测试结果，成功：{success}")
    # 根据成功与否设置文件名
    success_tag = "_success" if success else ""
    video_filename = f"_rt{run_id}{success_tag}.mp4"
    #写一个循环，将images的每一帧转为RGB格式
    for i in range(len(images)):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    # 将images保存为本地视频
    frames_visualization(images, video_filename)
    # # 将images保存为本地视频
    # frames_visualization(images, f"rt{run_id}.mp4")

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)
    args_list = [#google
        # "-e", "GraspMultipleSameObjectsInScene-v0",
        "-e", "GraspMultipleDifferentObjectsInScene-v0",
        "-c", "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
        "-o", "rgbd",
        "robot",
        "google_robot_static",
        "sim_freq", "@500",
        "control_freq", "@5",
        # "scene_name", "bridge_table_1_v1",
        "rgb_overlay_mode", "background",
        "rgb_overlay_path", "/home/khl/khl/Acmu/SENV/component/data/real_inpainting/google_coke_can_real_eval_1.png",
        "rgb_overlay_cameras", "overhead_camera"
    ]

    args_list_w = [#windowx
        "-e", "PushMultipleDifferentObjectsInScene-v0",
        "-c", "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        "-o", "rgbd",
        "robot",
        "widowx",
        "sim_freq", "@500",
        "control_freq", "@5",
        "scene_name", "bridge_table_1_v1",
        "rgb_overlay_mode", "background",
        "rgb_overlay_path", "/home/khl/khl/Acmu/SENV/component/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras", "3rd_view_camera"
    ]

    args = parse_args(args_list)
    env = setup_environment(args)
    reset_options = get_env_reset_options(env, args)
    policy_setup = "google_robot"
    # policy_setup = "widowx_bridge"
    model_name = "rt_1_x"
    from simpler_env.policies.rt1.rt1_model import RT1Inference
    # 请确保ckpt_path路径正确
    ckpt_path = "/home/khl/khl/Acmu/SENV/component/checkpoints /rt_1_x_tf_trained_for_002272480_step"
    model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)

    for i in range(1, 30):
        run(env, model, i)
