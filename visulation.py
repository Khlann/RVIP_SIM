from component.reference.hy_robot import *


def run():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()
    env = setup_environment(args)
    reset_options = get_env_reset_options(env, args)

    obs, info = env.reset(options=reset_options)
    while True:
        render_frame = env.render()
        env.render_human()
        action = np.array([1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0])
        env.step(action)
        # env.step(env.action_space.sample())
        time.sleep(1.0 / 20)  # Limit to 20 FPS
if __name__ == '__main__':
    run()