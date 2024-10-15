#所有人机交互的都在这边
import cv2
import os
#创建一个枚举类
class ImageType:
    RGB = "rgb"
    DEPTH = "depth"
    SEGMENTATION = "segmentation"

class CameraName:
    OVERHEAD_CAMERA = "overhead_camera"
    BASE_CAMERA = "base_camera"
    THIRD_VIEW_CAMERA = "3rd_view_camera"



class UserInteraction:
    def __init__(self,obs,camera_name,image_type):
        self.image_save_path = 'debug_image.png'
        self.points = []#用于存储鼠标点击的点
        self.obs = obs
        self.camera_name = camera_name
        self.image_type = image_type

    # def get_image_with_angle(self):

    def get_image(self):
        displayed_image = self.obs['image'][self.camera_name][self.image_type]
        displayed_image = cv2.cvtColor(displayed_image, cv2.COLOR_BGR2RGB)
        return displayed_image

    def human_control(self):
        img = self.get_image()
        # cv2.imwrite(self.image_save_path, img)

    def ground_truth_instruction(self):
        frame = self.get_image()
        while True:
            # 调用 get_key 方法并传入图像
            key = self.get_key(frame)
            if key:
                print(f"Key pressed: {key}")
                break  # 按任意键退出循环

    def mouse_callback(self, event, x, y, flags, param):
        # 如果检测到鼠标左键点击事件
        if event == cv2.EVENT_LBUTTONDOWN:
            # 将点击的点加入到列表中
            self.points.append((x, y))
            # 在图像上标记点击的位置
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            # 显示更新后的图像
            cv2.imshow("Image", param)
            # 打印点击的点
            print(f"Point {len(self.points)}: ({x}, {y})")

    def grt_keyboard_instruction(self,frame,delay=0):

        cv2.namedWindow("Image")
        cv2.imshow("Image", frame)
        key=cv2.waitKey(delay)
        if key == -1:  # timeout
            return None
        elif key == 27:  # escape
            exit(0)
        else:
            return chr(key)


    def get_key(self, frame, delay=1):
        # 转换图像格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 创建窗口
        cv2.namedWindow("Image")
        # 设置鼠标回调函数，传递图像参数
        cv2.setMouseCallback("Image", self.mouse_callback, frame)
        # 显示图像
        cv2.imshow("Image", frame)
        # 等待按键输入
        key = cv2.waitKey(delay)
        if key == -1:  # timeout
            return None
        elif key == 27:  # escape
            exit(0)
        else:
            return chr(key)

    def get_robot_info(self):
        #获取机器人信息
        base_pose = self.obs["agent"]["base_pose"]
        qpos = self.obs["agent"]["qpos"]
        tcp_pose = self.obs["agent"]["tcp_pose"]
        print("base_pose:",base_pose)
        print("qpos:",qpos)
        print("tcp_pose:",tcp_pose)


    def get_goal_image(self,video_path,output_dir):
        #Function: To select and save frames from a video file
        print("Press 's' to save the current frame.")
        print("Press left/right arrow keys to move to the previous/next frame.")
        # Check if the video file exists
        if not os.path.isfile(video_path):
            print(f"Error: The video file '{video_path}' does not exist.")
            return

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file '{video_path}'.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video information: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps} FPS")

        current_frame = 0
        saved_frames = 0

        # Create a window for displaying the video frames
        window_name = "Frame Selector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            # Set the video capture to the current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()

            if not ret:
                print("Reached the end of the video or failed to read the frame.")
                break

            # Display the current frame number on the frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Frame: {current_frame}/{total_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Show the frame in the window
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print("Exiting script.")
                break
            elif key == ord('s'):
                # Save the current frame
                frame_filename = os.path.join(output_dir, f"frame_{current_frame:06d}.png")
                cv2.imwrite(frame_filename, frame)
                saved_frames += 1
                print(f"Saved frame {current_frame} to '{frame_filename}'")
            elif key == 81 or key == ord('a'):  # Left arrow or 'a' key
                # Move to the previous frame
                if current_frame > 0:
                    current_frame -= 1
                    print(f"Moved to previous frame: {current_frame}")
                else:
                    print("This is the first frame.")
            elif key == 83 or key == ord('d'):  # Right arrow or 'd' key
                # Move to the next frame
                if current_frame < total_frames - 1:
                    current_frame += 1
                    print(f"Moved to next frame: {current_frame}")
                else:
                    print("This is the last frame.")
            else:
                print("Invalid key. Use left/right arrows to browse, 's' to save, 'q' to quit.")

        cap.release()
        cv2.destroyAllWindows()
        print(f"Total saved frames: {saved_frames}")


