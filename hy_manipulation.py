from motion_planning import *
from user_interaction import *
from env_generation import *

def run():
    np.set_printoptions(suppress=True, precision=4)
    REnv = Rvip_Enivronment(arg_google_same,"rtx","google_robot")
    obs, info = REnv.env.reset(options=REnv.reset_options)
    # Mp = Motion_planning(obs,"overhead_camera","rgb")
    uplayer = UserInteraction(obs,"overhead_camera","rgb")# #初始化一个UserInteraction类


    # uplayer.ground_truth_instruction()
    # pt = uplayer.points
    #
    # pw=Mp.pixel_to_3d_point(pt)
    # print("pw:",pw)

    while True:

        # REnv.env.render_human()#这个函数是用来显示sapien的图像的

        frame = uplayer.get_image()

        uplayer.obs = obs
        key=uplayer.grt_keyboard_instruction(frame)

        REnv.get_action(key)

        obs, reward, terminated, truncated, info = REnv.env.step(REnv.action)


if __name__ == '__main__':
    run()