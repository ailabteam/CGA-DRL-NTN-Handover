import numpy as np
from src.core.cga_math import CGAEngine
from clifford.g3c import e1, e2, e3

class WalkerConstellation:
    """
    Trình tạo chùm vệ tinh theo cấu trúc Walker Delta (i: T/P/F).
    - T: Tổng số vệ tinh.
    - P: Số mặt phẳng quỹ đạo (Orbital Planes).
    - F: Phase factor (độ lệch pha giữa các mặt phẳng).
    - Inclination: Góc nghiêng so với xích đạo.
    """
    def __init__(self, engine: CGAEngine, total_sats=66, n_planes=6, inclination=86.4, altitude=780.0):
        self.engine = engine
        self.total_sats = total_sats
        self.n_planes = n_planes
        self.sats_per_plane = total_sats // n_planes
        self.inclination = inclination
        self.altitude = altitude
        
        # Tính toán vận tốc góc (Angular Velocity) dựa trên định luật Kepler
        # v = sqrt(GM / r) -> omega = v / r = sqrt(GM / r^3)
        # GM trái đất ~ 3.986e5 km^3/s^2
        GM = 3.986004418e5
        r = 6371.0 + altitude
        self.omega = np.sqrt(GM / (r**3)) # rad/s
        
        # Danh sách vệ tinh: Mỗi phần tử là dict {'id': int, 'pos': multivector, 'plane_normal': bivector, 'phase': float}
        self.satellites = []
        self._generate_walker()

    def _generate_walker(self):
        """
        Khởi tạo vị trí ban đầu của tất cả vệ tinh bằng CGA Rotors.
        """
        self.satellites = []
        
        # 1. Tạo điểm gốc tại xích đạo (Lat=0, Lon=0)
        start_point = self.engine.latlon_to_cga(0.0, 0.0, self.altitude)
        
        # 2. Loop qua từng mặt phẳng
        for p in range(self.n_planes):
            # Góc RAAN (Right Ascension of Ascending Node) của mặt phẳng p
            raan = (360.0 / self.n_planes) * p
            
            # Tạo Rotor định nghĩa mặt phẳng quỹ đạo (nghiêng i, xoay raan)
            # Hàm create_orbit_rotor bạn cần bổ sung vào cga_math hoặc tính trực tiếp ở đây
            # Tính trực tiếp để code độc lập:
            
            # Rotor quay quanh Z (RAAN)
            B_xy = e1^e2
            R_node = np.e**(-0.5 * np.radians(raan) * B_xy)
            
            # Rotor nghiêng quanh trục X' (Inclination)
            # Trục quay sau khi rotate node là trục node (nằm trên mp xích đạo)
            # Nhưng để đơn giản, ta quay nghiêng quanh trục Y (e1^e3) hoặc trục vuông góc tương ứng
            # Chuẩn Walker: Nghiêng so với Z.
            # Ta dùng rotor nghiêng quanh trục X (e2^e3) sau khi đã xoay RAAN
            B_yz = e2^e3
            R_inc = np.e**(-0.5 * np.radians(self.inclination) * B_yz)
            
            # Rotor tổng hợp cho mặt phẳng
            R_plane = R_node * R_inc
            
            # Bivector pháp tuyến của quỹ đạo (Dùng để propagate sau này)
            # Mặc định quay quanh trục Z (e1^e2). Sau khi R_plane tác động, trục quay mới là:
            orbit_normal = R_plane * (e1^e2) * ~R_plane
            
            # Phase shift giữa các plane (Walker parameter F)
            # phase_offset = (360 * F / T) * p? 
            # Đơn giản hóa: phase offset = 0 cho PoC, hoặc random
            
            # 3. Loop qua từng vệ tinh trong mặt phẳng
            for s in range(self.sats_per_plane):
                # Góc Mean Anomaly (vị trí trên quỹ đạo)
                mean_anomaly = (360.0 / self.sats_per_plane) * s
                
                # Rotor quay vệ tinh đến vị trí Mean Anomaly (quanh trục Z ban đầu)
                R_sat = np.e**(-0.5 * np.radians(mean_anomaly) * (e1^e2))
                
                # Vị trí ban đầu (chưa nghiêng, chưa xoay plane)
                pos_local = R_sat * start_point * ~R_sat
                
                # Vị trí thực tế (sau khi áp dụng Rotor mặt phẳng)
                pos_final = R_plane * pos_local * ~R_plane
                
                sat_id = p * self.sats_per_plane + s
                
                self.satellites.append({
                    'id': sat_id,
                    'pos': pos_final,
                    'plane_normal': orbit_normal,
                    'plane_id': p
                })

    def propagate(self, dt):
        """
        Di chuyển toàn bộ chùm vệ tinh thêm thời gian dt (giây).
        """
        for sat in self.satellites:
            # Dùng hàm propagate của engine
            sat['pos'] = self.engine.propagate_satellite(
                sat['pos'], 
                self.omega, 
                dt, 
                sat['plane_normal']
            )
