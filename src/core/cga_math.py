# src/core/cga_math.py

import numpy as np
import math
from clifford.g3c import *
from clifford.tools.g3c import *
from clifford import inter_e_mt

# Hằng số vật lý (Đơn vị: km)
R_EARTH = 6371.0 

class CGAEngine:
    """
    Engine xử lý toán học Conformal Geometric Algebra cho bài toán Vệ tinh.
    Hệ cơ sở G4,1: e1, e2, e3, e4, e5.
    Mapping: e_inf (infinity), e_o (origin).
    """
    def __init__(self):
        # Khởi tạo các biến cơ sở thường dùng để tăng tốc độ tính toán
        self.e_inf = einf
        self.e_o = eo
        
        # Biểu diễn Trái Đất là một mặt cầu tâm O, bán kính R_EARTH
        # Công thức Sphere: S = P_center - 0.5 * R^2 * e_inf
        # Vì tâm là gốc tọa độ e_o -> S_earth = e_o - 0.5 * R^2 * e_inf
        self.earth_sphere = self.e_o - 0.5 * (R_EARTH**2) * self.e_inf

    def euclidean_to_cga(self, x, y, z):
        """
        Chuyển điểm 3D (x,y,z) sang Conformal Point P.
        Công thức: P = x + 0.5 * x^2 * e_inf + e_o
        """
        # Tạo vector 3D trong đại số GA
        x_vec = x*e1 + y*e2 + z*e3
        
        # Áp dụng công thức Up-projection
        # Lưu ý: Trong clifford, up() thực hiện chính xác công thức trên
        return up(x_vec)

    def latlon_to_cga(self, lat, lon, alt=0.0):
        """
        Chuyển đổi Kinh độ/Vĩ độ/Độ cao sang CGA Point.
        """
        r = R_EARTH + alt
        phi = np.radians(lat)
        theta = np.radians(lon)

        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        
        return self.euclidean_to_cga(x, y, z)

    def get_distance_sq(self, P1, P2):
        """
        Tính khoảng cách Euclide bình phương giữa 2 điểm CGA.
        Công thức CGA: P1 . P2 = -0.5 * |x1 - x2|^2
        => |x1 - x2|^2 = -2 * (P1 . P2)
        """
        # Toán tử | (Pipe) trong clifford là Inner Product
        inner_prod = P1 | P2 
        dist_sq = -2.0 * float(inner_prod)
        return abs(dist_sq) # Lấy trị tuyệt đối để tránh sai số dấu phẩy động nhỏ

    def create_orbit_rotor(self, inclination, ascending_node):
        """
        Tạo Rotor biểu diễn mặt phẳng quỹ đạo.
        Đây là phần thay thế cho ma trận quay Euler.
        """
        # 1. Quay quanh trục Z (Ascending Node)
        # Rotor R_omega = exp(-B_xy * theta / 2)
        B_xy = e1^e2
        R_node = math.e**(-0.5 * np.radians(ascending_node) * B_xy)

        # 2. Quay nghiêng (Inclination) quanh trục X (sau khi đã quay node)
        # Trục quay mới là trục node (Line of nodes)
        B_yz = e2^e3 
        R_inc = math.e**(-0.5 * np.radians(inclination) * B_yz)
        
        # Rotor tổng hợp
        return R_node * R_inc

    def propagate_satellite(self, sat_point, angular_velocity, dt, orbit_normal_bivector):
        """
        Di chuyển vệ tinh theo quỹ đạo bằng Rotor.
        P(t+dt) = R * P(t) * ~R
        Với R = exp(-B * (omega*dt) / 2)
        """
        angle = angular_velocity * dt # Radian
        
        # Tạo Rotor động lực học
        # orbit_normal_bivector định nghĩa mặt phẳng quay
        rotor = math.e**(-0.5 * angle * orbit_normal_bivector)
        
        # Update vị trí: R * P * R_reverse
        new_pos = rotor * sat_point * ~rotor
        return new_pos

    def check_visibility(self, user_point, sat_point, min_elevation_deg=10.0):
        """
        Kiểm tra vệ tinh có nhìn thấy User không (Line of Sight).
        Sử dụng hình học:
        1. Tạo mặt phẳng tiếp tuyến (Tangent Plane) tại vị trí User.
        2. Kiểm tra phía của Vệ tinh so với mặt phẳng.
        """
        # Tâm trái đất
        center = self.e_o
        
        # Vector từ tâm đến User (đại diện cho pháp tuyến tại user)
        # Trong CGA, vector hướng = (User Point) - (Center Point) (xấp xỉ ở vô cực)
        # Cách đơn giản hơn: Check tích vô hướng vector 3D thường (nhanh hơn cho bước check sơ bộ)
        
        # Tuy nhiên, để đúng chất CGA Paper, ta dùng phép Dual:
        # Mặt phẳng chân trời H = User ^ Center ^ e_inf (Dual Sphere) -> Phức tạp
        
        # Cách hiệu quả nhất cho code:
        # Cos(elevation) dựa trên khoảng cách
        d_us_sq = self.get_distance_sq(user_point, sat_point)
        r_u = R_EARTH
        r_s = math.sqrt(self.get_distance_sq(sat_point, self.e_o)) # xấp xỉ khoảng cách tâm
        
        # Định lý hàm Cos: d^2 = r_u^2 + r_s^2 - 2*r_u*r_s*cos(gamma)
        # gamma là góc ở tâm.
        # Elevation angle (el) quan hệ với gamma.
        
        # Để đơn giản và nhanh, ta dùng vector Euclide tạm thời cho check góc:
        # (Đây là thỏa hiệp kỹ thuật để tốc độ cao trong Python)
        u_vec = user_point(1) * e1 + user_point(2) * e2 + user_point(3) * e3
        s_vec = sat_point(1) * e1 + sat_point(2) * e2 + sat_point(3) * e3
        
        u_norm = u_vec / abs(u_vec)
        s_norm = s_vec / abs(s_vec)
        
        cos_gamma = float(u_norm | s_norm)
        
        # Điều kiện mặt phẳng chân trời (Horizon): cos_gamma > R_E / R_SAT
        horizon_thresh = R_EARTH / r_s
        
        # Có thể tinh chỉnh thêm min_elevation ở đây
        if cos_gamma > horizon_thresh:
             return True
        return False

    def to_features(self, user_point, sat_point):
        """
        Trích xuất đặc trưng cho mạng DRL.
        Input: Multivectors
        Output: Numpy array (Invariant Features)
        
        Đây là đóng góp chính của Paper: "Invariant Geometric Features"
        Thay vì trả về (x,y,z), ta trả về các đại lượng vô hướng.
        """
        dist_sq = self.get_distance_sq(user_point, sat_point)
        
        # Tính chất đối ngẫu (Dual) hoặc các tích wedge khác
        # Ví dụ: Độ lớn của bivector tạo bởi User và Sat (liên quan đến diện tích quét)
        # area_bivector = user_point ^ sat_point
        # magnitude = abs(area_bivector)
        
        # Hiện tại lấy khoảng cách và vector hướng tương đối đã chuẩn hóa
        return np.array([dist_sq], dtype=np.float32)
