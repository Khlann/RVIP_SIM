import numpy as np
class Motion_planning:
    def __init__(self,obs,camera_name,image_type):
        self.obs = obs
        self.camera_name = camera_name
        self.image_type = image_type

    def pixel_to_3d_point(self,points):
        intrinsic_matrix = self.obs["camera_param"][self.camera_name]["intrinsic_cv"]
        extrinsic_matrix = self.obs["camera_param"][self.camera_name]["extrinsic_cv"]
        K = intrinsic_matrix
        u, v = points[0]
        Z = self.obs["image"][self.camera_name][self.image_type][v,u]#这里可能有问题
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
        # print("TCP pose wrt world:", env.tcp.pose)



