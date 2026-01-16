import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.core.cga_math import CGAEngine
from src.core.constellation import WalkerConstellation

class SatelliteHandoverEnv(gym.Env):
    """
    Môi trường Gym cho bài toán Handover Vệ tinh LEO.
    Observation: Đặc trưng CGA của K vệ tinh gần nhất.
    Action: Chọn vệ tinh để kết nối.
    """
    def __init__(self, k_nearest=5, max_steps=1000):
        super(SatelliteHandoverEnv, self).__init__()
        
        self.k_nearest = k_nearest
        self.max_steps = max_steps
        self.step_count = 0
        
        # Khởi tạo Engine và Chùm vệ tinh (Iridium-like: 66 sats)
        self.cga = CGAEngine()
        self.constellation = WalkerConstellation(self.cga, total_sats=66, n_planes=6, inclination=86.4, altitude=780.0)
        
        # Vị trí User (Cố định tại Hà Nội cho đơn giản hóa phase 1)
        self.user_pos = self.cga.latlon_to_cga(21.028, 105.854, 0.0)
        
        # Action Space: Chọn 1 trong K vệ tinh gần nhất
        self.action_space = spaces.Discrete(k_nearest)
        
        # Observation Space: K x 2 features (Distance_norm, Cosine_similarity) + 1 (Current Connection Flag)
        # Flatten thành vector 1 chiều: Size = K * 3
        self.obs_dim = k_nearest * 3
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32)
        
        # Trạng thái kết nối hiện tại (Satellite ID)
        self.current_sat_id = -1 
        self.last_sat_id = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset lại chùm vệ tinh về t=0 (Hoặc random phase nếu muốn stochastic)
        self.constellation = WalkerConstellation(self.cga) # Re-init
        self.step_count = 0
        self.current_sat_id = -1
        self.last_sat_id = -1
        
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        dt = 10.0 # Mỗi step là 10 giây thực tế
        
        # 1. Di chuyển vệ tinh
        self.constellation.propagate(dt)
        
        # 2. Lấy danh sách vệ tinh quan sát được (Visible Candidate List)
        candidates = self._get_candidates()
        
        # 3. Thực hiện Action (Handover decision)
        # Action là index trong danh sách candidates (0..K-1)
        # Nếu action vượt quá số lượng candidate thực tế -> Penalty
        reward = 0
        done = False
        info = {}
        
        if len(candidates) == 0:
            # Mất kết nối toàn bộ (Outage)
            reward = -10.0
            self.current_sat_id = -1
        else:
            # Map action index sang Satellite ID thực
            # Nếu action >= len(candidates), chọn cái cuối cùng (fallback)
            selected_idx = min(action, len(candidates)-1)
            target_sat = candidates[selected_idx]
            target_id = target_sat['id']
            
            # Tính Reward
            # Base reward: Chất lượng kết nối (Cosine góc ngẩng càng cao càng tốt)
            # Feature thứ 1 trong candidate là cosine
            quality_reward = target_sat['features'][1] * 2.0 # Scale lên
            
            # Handover Penalty
            ho_penalty = 0.0
            if self.current_sat_id != -1 and target_id != self.current_sat_id:
                ho_penalty = -0.5 # Phạt nhẹ để hạn chế ping-pong
                
            reward = quality_reward + ho_penalty
            
            # Cập nhật trạng thái
            self.last_sat_id = self.current_sat_id
            self.current_sat_id = target_id
            
        # Check termination
        if self.step_count >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, False, info

    def _get_candidates(self):
        """
        Lọc và sort vệ tinh tốt nhất cho User.
        """
        candidates = []
        for sat in self.constellation.satellites:
            if self.cga.check_visibility_fast(self.user_pos, sat['pos']):
                feats = self.cga.to_features(self.user_pos, sat['pos'])
                # feats: [dist_norm, cos_gamma]
                candidates.append({
                    'id': sat['id'],
                    'pos': sat['pos'],
                    'features': feats
                })
        
        # Sort theo khoảng cách (dist_norm nhỏ nhất trước)
        candidates.sort(key=lambda x: x['features'][0])
        return candidates

    def _get_obs(self):
        """
        Xây dựng vector quan sát K x 3
        """
        candidates = self._get_candidates()
        
        obs_vec = np.zeros(self.obs_dim, dtype=np.float32)
        
        for i in range(self.k_nearest):
            if i < len(candidates):
                cand = candidates[i]
                # Feat 0: Dist, Feat 1: Cos
                obs_vec[i*3] = cand['features'][0] 
                obs_vec[i*3+1] = cand['features'][1]
                # Feat 2: Is Connected? (1.0 nếu đang connect, 0.0 nếu không)
                obs_vec[i*3+2] = 1.0 if cand['id'] == self.current_sat_id else 0.0
            else:
                # Nếu không đủ K vệ tinh (vùng cực hoặc lỗ hổng), padding -1 hoặc 0
                obs_vec[i*3] = 2.0 # Dist xa
                obs_vec[i*3+1] = -1.0 # Góc xấu
                obs_vec[i*3+2] = 0.0
                
        return obs_vec
