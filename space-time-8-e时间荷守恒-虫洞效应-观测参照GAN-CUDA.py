import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from matplotlib.animation import FuncAnimation
import random
from collections import defaultdict
import math
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
import csv

# 检查CUDA可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置随机种子
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.manual_seed(42)

class SixDimensionalSpacetimeHypergraphAdvanced:
    """
    升级版六维流形时空超图计算模型
    集成深度学习启发的宇宙大尺度结构生成
    实现暗能量-地球子时空虫洞效应和高级几何演化
    """
    
    def __init__(self, n_earth=20000, n_blackhole=80, n_darkenergy=1000):
        # 添加预训练相关参数
        self.pretrain_epochs = 50  # 预训练轮数
        self.n_earth = n_earth
        self.n_blackhole = n_blackhole
        self.n_darkenergy = n_darkenergy
        
        # 深度学习启发的参数配置
        self.grid_resolution = 128  # 进一步降低分辨率以减少内存消耗
        self.box_size = 1000.0  # h⁻¹Mpc 盒子大小
        self.conv_kernel_size = 3  # 使用更小的卷积核
        self.n_filters_base = 8  # 减少基础滤波器数量
        self.n_filters_max = 256  # 减少最大滤波器数量
        self.learning_rate = 1e-4  # 降低学习率
        self.beta1 = 0.5  # Adam优化器参数
        self.beta2 = 0.9  # 调整beta2
        self.gp_weight = 10.0  # 梯度惩罚权重
        self.batch_size = 4  # 进一步减小批量大小
        self.noise_dim = 100  # 噪声维度
        self.upscale_factor = 8  # 上采样因子
        self.noise_amp = 0.1  # 噪声幅度
        self.pretrain_epochs = 5  # 预训练轮数
        self.adv_epochs = 150  # 对抗训练轮数
        self.D_steps = 1  # 判别器更新步数
        self.G_steps = 1  # 生成器更新步数
        
        # 数据重缩放参数（基于DarkAI）
        self.rescale_a = 0.02  # log变换参数
        self.rescale_b = 3000.0  # 速度归一化参数 km/s
        
        # 基础物理参数
        self.kappa_0 = 1.0
        self.alpha_quantum = 1e-36
        self.alpha_sphere = 1e-36
        self.alpha_fluid = 1.0
        
        # 时间荷守恒参数
        self.beta_t1 = 0.1
        self.lambda_ent = 0.5
        self.tau_threshold = 0.05
        
        # 虫洞效应参数
        self.wormhole_strength = 0.2
        self.negative_energy_density = -0.1
        self.traversability_factor = 0.8
        
        # 暗能量几何参数
        self.gamma_2 = 1.0
        self.gamma_3 = 1.0
        self.epsilon_2 = 0.1
        self.epsilon_3 = 0.1
        
        # Wolfram超图重写参数
        self.rewrite_rules = {
            'geometric_expansion': 1.02,
            'topological_fusion': 0.98,
            'information_diffusion': 0.15,
            'causal_propagation': 0.25
        }
        
        # 数据结构初始化
        self.hypergraph = nx.Graph()
        self.nodes = {}
        self.hyperedges = []
        self.wormhole_connections = []
        self.time_step = 0
        self.max_iterations = 5000
        
        # 历史数据存储
        self.history = {
            'positions': [],
            'connections': [],
            'energies': [],
            'masses': [],
            'wormhole_fluxes': [],
            'time_charges': []
        }
        
        # 深度学习启发的多尺度特征存储
        self.multi_scale_features = {}
        self.encoder_features = []
        self.decoder_features = []
        
        # 初始化GAN模型并移动到设备
        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)
        
        # 优化器
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), 
                                          lr=self.learning_rate, 
                                          betas=(self.beta1, self.beta2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                          lr=self.learning_rate,
                                          betas=(self.beta1, self.beta2))
        
        # 损失函数并移动到设备
        self.criterion_GAN = torch.nn.MSELoss().to(device)
        self.criterion_pixel = torch.nn.L1Loss().to(device)
        
        # 历史损失记录
        self.history['generator_loss'] = []
        self.history['discriminator_loss'] = []
        
        print(f"高级六维流形时空模型初始化完成")
        print(f"网格分辨率: {self.grid_resolution}³, 盒子大小: {self.box_size} h⁻¹Mpc")
        print(f"地球子时空: {n_earth}, 黑洞子时空: {n_blackhole}, 暗能量子时空: {n_darkenergy}")
    
    def get_total_time_charge(self):
        """
        计算三个时间维度上的总时间荷
        
        Returns:
            tuple: (total_t1, total_t2, total_t3) 三个时间维度上的总时间荷
        """
        total_t1 = 0.0
        total_t2 = 0.0
        total_t3 = 0.0
        
        for node_id, node_data in self.nodes.items():
            # 获取节点质量
            mass = node_data.get('mass', 1.0)
            
            # 获取时间坐标 (t, tau1, tau2)
            time_coords = node_data.get('time_coords', [0.0, 0.0, 0.0])
            
            # 累加三个时间维度的时间荷
            # 注意：这里假设 time_coords[0] 是 t1, time_coords[1] 是 t2, time_coords[2] 是 t3
            total_t1 += mass * time_coords[0]
            total_t2 += mass * time_coords[1]
            total_t3 += mass * time_coords[2]
        
        return total_t1, total_t2, total_t3
        
    def initialize_spacetime_nodes_advanced(self):
        """
        初始化高级六维时空节点
        集成深度学习启发的特征初始化和多尺度结构
        """
        print("初始化高级六维时空节点...")
        
        # 清空现有节点
        self.nodes = {}
        self.hypergraph = nx.Graph()
        
        # 生成宇宙大尺度结构位置
        print("生成宇宙大尺度结构...")
        positions = self.generate_cosmic_structure_positions_advanced()
        
        # 初始化地球型节点
        print(f"初始化 {self.n_earth} 个地球型节点...")
        for i in range(self.n_earth):
            node_id = f"earth_{i}"
            pos = positions['earth'][i % len(positions['earth'])] if len(positions['earth']) > 0 else np.random.rand(3) * self.box_size
            
            self.nodes[node_id] = {
                'type': 'earth',
                'position': pos,
                'mass': 1.0 + np.random.normal(0, 0.1),  # 标准质量 + 随机变化
                'energy': 0.0,
                'time_coords': [0.0, 0.0, 0.0],  # 时间坐标 (t, tau1, tau2)
                'quantum_state': np.random.normal(0, 1, 8),  # 8维量子态
                'features': np.random.normal(0, 1, 64),  # 64维特征向量
                'entanglement': [],  # 量子纠缠连接
                'last_update': 0,  # 最后更新时间步
                'stability': 1.0,  # 稳定性因子
                'dark_energy_coupling': np.random.uniform(0.8, 1.2),  # 暗能量耦合强度
            }
            self.hypergraph.add_node(node_id)
        
        # 初始化黑洞节点
        print(f"初始化 {self.n_blackhole} 个黑洞节点...")
        blackhole_positions = []
        
        # 确定黑洞集群数量 (6-8个)
        n_clusters = np.random.randint(6, 9)
        # 确定主要集群数量 (2-3个)
        n_major_clusters = np.random.randint(2, 4)
        # 主要集群占60%的黑洞数量
        major_blackhole_count = int(self.n_blackhole * 0.6)
        # 剩余黑洞数量分配给次要集群
        minor_blackhole_count = self.n_blackhole - major_blackhole_count
        
        # 确保主要集群数量不超过总集群数
        n_major_clusters = min(n_major_clusters, n_clusters)
        n_minor_clusters = n_clusters - n_major_clusters
        
        # 计算每个主要和次要集群分配的黑洞数量
        blackholes_per_major = major_blackhole_count // n_major_clusters
        blackholes_per_minor = minor_blackhole_count // n_minor_clusters if n_minor_clusters > 0 else 0
        
        # 生成集群中心位置，确保它们之间有足够的距离
        min_cluster_dist = self.box_size / (n_clusters ** (1/3)) * 0.8  # 确保集群之间有足够距离
        cluster_centers = []
        
        # 生成主要集群中心
        for _ in range(n_clusters):
            while True:
                # 生成新的集群中心
                new_center = np.random.uniform(0.1 * self.box_size, 0.9 * self.box_size, 3)
                
                # 检查与现有集群中心的最小距离
                valid_position = True
                for center in cluster_centers:
                    if np.linalg.norm(new_center - center) < min_cluster_dist:
                        valid_position = False
                        break
                
                if valid_position:
                    cluster_centers.append(new_center)
                    break
        
        # 为每个集群分配黑洞
        blackhole_count = 0
        
        # 主要集群（大质量）
        for i in range(n_major_clusters):
            cluster_center = cluster_centers[i]
            # 主要集群有更大的影响范围
            cluster_radius = min_cluster_dist * 0.6
            
            # 计算这个集群应该有的黑洞数量
            n_in_cluster = blackholes_per_major
            if i == n_major_clusters - 1:  # 最后一个主要集群处理余数
                n_in_cluster = major_blackhole_count - (n_major_clusters - 1) * blackholes_per_major
            
            # 在集群内生成黑洞
            for _ in range(n_in_cluster):
                # 使用高斯分布在集群中心周围生成黑洞
                offset = np.random.normal(0, cluster_radius/3, 3)
                pos = np.clip(cluster_center + offset, 0, self.box_size)
                blackhole_positions.append(pos)
                
                # 创建大质量黑洞节点（质量更大）
                node_id = f"bh_{len(self.nodes)}"
                self.nodes[node_id] = {
                    'type': 'blackhole',
                    'position': pos,
                    'mass': np.random.uniform(1e8, 1e10),  # 大质量黑洞
                    'event_horizon': np.random.uniform(1e-2, 3e-1) * self.box_size,
                    'spin': np.random.uniform(0.8, 0.99),  # 高速自旋
                    'quantum_state': np.random.normal(0, 1, 3),
                    'time_charge': np.random.normal(0, 1, 3),
                    'stability': np.random.uniform(0.8, 1.0),
                    'creation_time': 0,
                    'velocity': np.zeros(3),
                    'energy': 0.0,
                    'cluster_id': i  # 标记所属集群
                }
                blackhole_count += 1
        
        # 次要集群（小质量）
        for i in range(n_major_clusters, n_clusters):
            cluster_center = cluster_centers[i]
            # 次要集群有更小的影响范围
            cluster_radius = min_cluster_dist * 0.4
            
            # 计算这个集群应该有的黑洞数量
            n_in_cluster = blackholes_per_minor
            if i == n_clusters - 1:  # 最后一个次要集群处理余数
                n_in_cluster = minor_blackhole_count - (n_clusters - n_major_clusters - 1) * blackholes_per_minor
            
            # 在集群内生成黑洞
            for _ in range(n_in_cluster):
                # 使用高斯分布在集群中心周围生成黑洞
                offset = np.random.normal(0, cluster_radius/3, 3)
                pos = np.clip(cluster_center + offset, 0, self.box_size)
                blackhole_positions.append(pos)
                
                # 创建较小质量黑洞节点
                node_id = f"bh_{len(self.nodes)}"
                self.nodes[node_id] = {
                    'type': 'blackhole',
                    'position': pos,
                    'mass': np.random.uniform(1e6, 1e8),  # 较小质量黑洞
                    'event_horizon': np.random.uniform(1e-3, 1e-2) * self.box_size,
                    'spin': np.random.uniform(0.1, 0.9),
                    'quantum_state': np.random.normal(0, 1, 3),
                    'time_charge': np.random.normal(0, 1, 3),
                    'stability': np.random.uniform(0.6, 0.9),
                    'creation_time': 0,
                    'velocity': np.zeros(3),
                    'energy': 0.0,
                    'cluster_id': i  # 标记所属集群
                }
                blackhole_count += 1
        
        # 添加一个超大质量黑洞作为类星体核心
        quasar_center = np.array([self.box_size/2, self.box_size/2, self.box_size/2])
        node_id = f"bh_quasar"
        self.nodes[node_id] = {
            'type': 'quasar',
            'position': quasar_center,
            'mass': 1e12,  # 超大质量
            'event_horizon': 0.05 * self.box_size,
            'spin': 0.99,  # 接近最大自旋
            'quantum_state': np.zeros(3),
            'time_charge': np.ones(3) * 10,  # 强时间荷
            'stability': 0.95,
            'creation_time': 0,
            'velocity': np.zeros(3),
            'energy': 0.0,
            'cluster_id': -1  # 特殊标记
        }
        
        # 初始化暗能量节点
        print(f"初始化 {self.n_darkenergy} 个暗能量节点...")
        for i in range(self.n_darkenergy):
            node_id = f"darkenergy_{i}"
            pos = positions['darkenergy'][i % len(positions['darkenergy'])] if len(positions['darkenergy']) > 0 else np.random.rand(3) * self.box_size
            
            self.nodes[node_id] = {
                'type': 'darkenergy',
                'position': pos,
                'mass': 0.0,  # 暗能量质量可以为零或负
                'energy': 10.0 + np.random.normal(0, 1),  # 高能量
                'time_coords': [0.0, 0.0, 0.0],
                'quantum_state': np.random.normal(0, 0.5, 8),
                'features': np.random.normal(0, 1, 64),
                'entanglement': [],
                'last_update': 0,
                'stability': 0.8,
                'expansion_rate': np.random.uniform(0.8, 1.2),  # 膨胀率
                'coupling_F': 0.0,  # 流体耦合函数F
                'coupling_G': 0.0,   # 流体耦合函数G
            }
            self.hypergraph.add_node(node_id)
        
        # 初始化多尺度特征
        print("初始化多尺度特征...")
        self._initialize_multi_scale_features()
        
        # 初始化时间荷
        print("初始化时间荷...")
        self._initialize_time_charges()
        
        print("高级六维时空节点初始化完成！")
    
    def _initialize_multi_scale_features(self):
        """初始化多尺度特征"""
        self.multi_scale_features = {
            'scale_1': [],  # 最大尺度特征
            'scale_2': [],  # 中等尺度特征
            'scale_3': [],  # 小尺度特征
        }
        
        # 为每个尺度生成随机特征
        for node_id in self.nodes:
            self.multi_scale_features['scale_1'].append(np.random.normal(0, 1, 16))
            self.multi_scale_features['scale_2'].append(np.random.normal(0, 1, 32))
            self.multi_scale_features['scale_3'].append(np.random.normal(0, 1, 64))
        
        # 转换为numpy数组以便处理
        for scale in self.multi_scale_features:
            self.multi_scale_features[scale] = np.array(self.multi_scale_features[scale])
    
    def _initialize_time_charges(self):
        """初始化时间荷"""
        total_charge = 0.0
        
        # 为每个节点分配初始时间荷
        for node_id, node in self.nodes.items():
            if node['type'] == 'earth':
                # 地球型节点有正时间荷
                node['time_charge'] = np.random.uniform(0.1, 1.0, 3)  # 3D时间荷
            elif node['type'] == 'blackhole':
                # 黑洞有强负时间荷
                node['time_charge'] = np.random.uniform(-5.0, -1.0, 3)
            else:  # darkenergy
                # 暗能量节点时间荷接近零
                node['time_charge'] = np.random.uniform(-0.1, 0.1, 3)
            
            # 更新总时间荷
            total_charge += np.sum(node['time_charge'])
        
        # 记录初始总时间荷（用于守恒检查）
        self.initial_total_charge = total_charge
        print(f"初始总时间荷: {total_charge:.4f}")
    
    def generate_cosmic_structure_positions_advanced(self):
        """
        高级宇宙大尺度结构位置生成
        集成深度学习多尺度特征提取思想
        """
        print("生成高级宇宙大尺度结构...")
        
        # 使用分层方法生成位置
        positions = {
            'earth': self._generate_hierarchical_matter_distribution(self.n_earth, self.box_size),
            'blackhole': self._generate_filament_network(self.n_blackhole, self.box_size),
            'darkenergy': self._generate_void_structure(self.n_darkenergy, self.box_size)
        }
        
        return positions
    
    def _generate_hierarchical_matter_distribution(self, n_points, box_size):
        """
        生成分层物质分布，模拟宇宙大尺度结构
        """
        print(f"生成分层物质分布 ({n_points}个点)...")
        
        # 使用分形噪声生成密度场
        scale = 100.0
        points = np.zeros((n_points, 3))
        
        # 生成初始随机点
        for i in range(n_points):
            # 使用分形坐标
            x = np.random.uniform(0, box_size)
            y = np.random.uniform(0, box_size)
            z = np.random.uniform(0, box_size)
            
            # 添加分形噪声
            noise = np.random.normal(0, 0.1, 3)
            points[i] = [x + noise[0], y + noise[1], z + noise[2]]
        
        return points
    
    def _generate_filament_network(self, n_points, box_size):
        """
        生成纤维网络结构，模拟暗物质纤维
        """
        print(f"生成纤维网络结构 ({n_points}个点)...")
        
        # 在盒子内生成随机点
        points = np.random.uniform(0, box_size, (n_points, 3))
        
        # 使用Delaunay三角剖分
        tri = Delaunay(points)
        
        # 从三角剖分中提取边
        edges = set()
        for simplex in tri.simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        
        # 构建图并找到最小生成树
        G = nx.Graph()
        for i, j in edges:
            dist = np.linalg.norm(points[i] - points[j])
            G.add_edge(i, j, weight=dist)
        
        mst = nx.minimum_spanning_tree(G)
        
        # 添加一些额外的边以创建更丰富的结构
        extra_edges = list(edges - set(mst.edges()))
        np.random.shuffle(extra_edges)
        for i, j in extra_edges[:len(extra_edges)//4]:  # 添加25%的额外边
            mst.add_edge(i, j)
        
        # 使用图布局算法使结构更清晰
        pos = nx.spring_layout(mst, dim=3)
        filament_points = np.array([pos[i] for i in range(n_points)])
        
        # 缩放到原始盒子大小
        filament_points = filament_points - np.min(filament_points, axis=0)
        filament_points = filament_points / np.max(filament_points) * box_size
        
        return filament_points
    
    def prepare_training_data(self, batch_size=4):
        """
        准备训练数据
        生成低分辨率和高分辨率的数据对用于训练
        
        返回:
            lr_batch: 低分辨率批次数据 [batch, 1, D, H, W]
            hr_batch: 对应的高分辨率批次数据 [batch, 1, D*scale, H*scale, W*scale]
        """
        # 确保缩放因子已设置
        if not hasattr(self, 'upscale_factor'):
            self.upscale_factor = 4  # 默认上采样因子
            
        # 生成3D密度场
        density_field = self._generate_density_field()
        
        # 创建批次
        batch_lr = []
        batch_hr = []
        
        for _ in range(batch_size):
            # 随机裁剪一个3D块
            crop_size = 16  # 低分辨率块大小
            hr_size = crop_size * self.upscale_factor  # 高分辨率块大小
            
            # 确保裁剪区域在边界内
            max_offset = np.array(density_field.shape) - np.array([hr_size]*3)
            max_offset = np.maximum(1, max_offset)  # 确保至少为1
            offset = [np.random.randint(0, d) for d in max_offset]
            
            # 裁剪高分辨率块
            hr_crop = density_field[
                offset[0]:offset[0]+hr_size,
                offset[1]:offset[1]+hr_size,
                offset[2]:offset[2]+hr_size
            ]
            
            # 下采样得到低分辨率块
            lr_crop = hr_crop[::self.upscale_factor, ::self.upscale_factor, ::self.upscale_factor]
            
            # 归一化到[-1, 1]
            lr_crop = (lr_crop - lr_crop.min()) / (lr_crop.max() - lr_crop.min() + 1e-8) * 2 - 1
            hr_crop = (hr_crop - hr_crop.min()) / (hr_crop.max() - hr_crop.min() + 1e-8) * 2 - 1
            
            # 添加通道维度
            lr_crop = lr_crop[np.newaxis, np.newaxis, ...]  # [1, 1, D, H, W]
            hr_crop = hr_crop[np.newaxis, np.newaxis, ...]  # [1, 1, D*scale, H*scale, W*scale]
            
            batch_lr.append(lr_crop)
            batch_hr.append(hr_crop)
        
        # 堆叠批次
        lr_batch = torch.FloatTensor(np.concatenate(batch_lr, axis=0))
        hr_batch = torch.FloatTensor(np.concatenate(batch_hr, axis=0))
        
        return lr_batch, hr_batch
    
    def update_physics_advanced(self, dt=0.1):
        """
        高级物理更新方法
        更新所有节点的物理状态，包括位置、速度、能量等
        
        参数:
            dt (float): 时间步长
        """
        if not hasattr(self, 'nodes') or not self.nodes:
            return
            
        # 初始化物理参数
        G = 6.67430e-11  # 万有引力常数
        c = 2.99792458e8  # 光速
        
        # 获取所有节点ID和位置
        node_ids = list(self.nodes.keys())
        positions = np.array([self.nodes[node_id]['position'] for node_id in node_ids])
        
        # 计算节点间距离矩阵
        dist_matrix = squareform(pdist(positions))
        
        # 避免除零错误
        np.fill_diagonal(dist_matrix, np.inf)
        inv_dist_matrix = 1.0 / (dist_matrix + 1e-8)
        inv_dist_cubed = inv_dist_matrix ** 3
        
        # 初始化加速度
        accelerations = np.zeros_like(positions, dtype=np.float64)
        
        # 计算每个节点的加速度
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            mass = node.get('mass', 1.0)
            
            # 计算引力加速度
            for j, other_id in enumerate(node_ids):
                if i == j:
                    continue
                    
                other = self.nodes[other_id]
                other_mass = other.get('mass', 1.0)
                
                # 计算引力方向
                direction = positions[j] - positions[i]
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                
                # 计算引力大小 (F = G * m1 * m2 / r^2)
                force_magnitude = G * mass * other_mass * inv_dist_matrix[i, j] ** 2
                
                # 计算加速度 (a = F / m)，添加质量检查避免除零错误
                if mass > 1e-10:  # 使用一个小的阈值来避免数值不稳定
                    acceleration = force_magnitude / mass * direction
                else:
                    acceleration = np.zeros_like(direction)
                
                # 添加暗能量斥力 (与距离成正比)
                if 'dark_energy_repulsion' in node and 'dark_energy_repulsion' in other:
                    repulsion_strength = node['dark_energy_repulsion'] * other['dark_energy_repulsion']
                    acceleration += direction * repulsion_strength * dist_matrix[i, j] * 1e-8
                
                accelerations[i] += acceleration
            
            # 添加阻尼项 (模拟能量耗散)
            if 'velocity' in node:
                accelerations[i] -= 0.1 * np.array(node['velocity'])
        
        # 更新节点状态
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            
            # 更新速度 (v = v0 + a*dt)
            if 'velocity' not in node:
                node['velocity'] = np.zeros(3)
            
            node['velocity'] += accelerations[i] * dt
            
            # 限制最大速度 (防止数值不稳定)
            max_velocity = 0.1 * self.box_size  # 最大速度为盒子大小的10%
            velocity_norm = np.linalg.norm(node['velocity'])
            if velocity_norm > max_velocity:
                node['velocity'] = node['velocity'] / velocity_norm * max_velocity
            
            # 更新位置 (x = x0 + v*dt)
            node['position'] += node['velocity'] * dt
            
            # 确保位置在边界内
            node['position'] = np.clip(node['position'], 0, self.box_size)
            
            # 更新能量 (E = 0.5 * m * v^2 + potential_energy)
            if 'energy' not in node:
                node['energy'] = 0.0
            
            kinetic_energy = 0.5 * node.get('mass', 1.0) * np.sum(node['velocity'] ** 2)
            node['energy'] = kinetic_energy  # 简化为仅考虑动能
            
            # 更新时间步
            node['last_update'] = getattr(self, 'current_time', 0)
        
        # 更新时间
        self.current_time = getattr(self, 'current_time', 0) + dt
        
        # 更新虫洞连接
        self._update_wormholes(dt)
        
        # 更新时间荷守恒
        self._update_time_charges()
    
    def _update_wormholes(self, dt):
        """更新虫洞连接状态"""
        if not hasattr(self, 'wormholes'):
            return
            
        for wormhole in self.wormholes:
            # 更新虫洞稳定性
            if 'stability' in wormhole:
                # 随机波动
                stability_change = np.random.normal(0, 0.01)
                wormhole['stability'] = np.clip(wormhole['stability'] + stability_change, 0.1, 1.0)
            
            # 更新虫洞能量
            if 'energy' in wormhole:
                # 能量衰减
                wormhole['energy'] *= (1.0 - 0.01 * dt)
                
    def _update_time_charges(self):
        """更新时间荷守恒"""
        if not hasattr(self, 'nodes'):
            return
            
        # 计算当前总时间荷
        total_charge = 0.0
        for node_id, node in self.nodes.items():
            if 'time_charge' in node:
                total_charge += np.sum(node['time_charge'])
        
        # 计算与初始总时间荷的差异
        if hasattr(self, 'initial_total_charge') and self.initial_total_charge != 0:
            charge_ratio = total_charge / self.initial_total_charge
            
            # 如果差异超过1%，则进行调整
            if abs(charge_ratio - 1.0) > 0.01:
                # 按比例调整所有节点的时间荷
                for node_id, node in self.nodes.items():
                    if 'time_charge' in node:
                        node['time_charge'] = node['time_charge'] / charge_ratio
                        
    def record_evolution_statistics_advanced(self):
        """
        记录高级演化统计信息
        包括节点位置、连接、能量、质量、虫洞通量和时间荷等
        """
        # 记录位置
        positions = np.array([node.get('position', np.zeros(6)) for node in self.nodes.values()])
        self.history['positions'].append(positions.copy())
        
        # 记录连接
        connections = []
        for edge in self.hyperedges:
            if isinstance(edge, dict) and 'nodes' in edge:
                # 对于字典类型的超边，提取节点对
                nodes = edge['nodes']
                if len(nodes) >= 2:
                    # 只添加第一个和第二个节点的连接
                    connections.append((nodes[0], nodes[1]))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                # 对于列表/元组类型的超边，直接取前两个元素
                connections.append((edge[0], edge[1]))
        self.history['connections'].append(connections)
        
        # 记录能量和质量
        energies = [node.get('energy', 0.0) for node in self.nodes.values()]
        masses = [node.get('mass', 1.0) for node in self.nodes.values()]
        self.history['energies'].append(np.array(energies))
        self.history['masses'].append(np.array(masses))
        
        # 记录虫洞通量
        if hasattr(self, 'wormholes'):
            wormhole_fluxes = [conn.get('flux', 0.0) for conn in self.wormholes]
            self.history['wormhole_fluxes'].append(np.array(wormhole_fluxes))
        
        # 记录时间荷
        time_charges = [node.get('time_charge', 0.0) for node in self.nodes.values()]
        self.history['time_charges'].append(np.array(time_charges))
    
    def _generate_density_field(self):
        """
        生成3D密度场
        基于节点位置和质量生成密度场
        """
        # 初始化密度场
        grid_size = 64  # 密度场分辨率
        density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        
        if not hasattr(self, 'nodes') or not self.nodes:
            return density
        
        # 获取所有节点的位置和质量
        positions = []
        masses = []
        for node_id, node in self.nodes.items():
            positions.append(node['position'])
            masses.append(node.get('mass', 1.0))  # 默认质量为1.0
        
        if not positions:
            return density
        
        positions = np.array(positions)
        masses = np.array(masses)
        
        # 将位置缩放到[0, grid_size-1]范围
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        scale = (grid_size - 1) / (max_pos - min_pos + 1e-8)
        normalized_pos = (positions - min_pos) * scale
        
        # 将质量分布到网格上
        for (x, y, z), mass in zip(normalized_pos, masses):
            # 找到最近的网格点
            i = min(int(round(x)), grid_size-1)
            j = min(int(round(y)), grid_size-1)
            k = min(int(round(z)), grid_size-1)
            
            # 添加质量到最近的网格点
            density[i, j, k] += mass
        
        # 应用高斯平滑
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=1.0)
        
        return density
    
    def pretrain_generator(self, num_epochs=100, batch_size=4):
        """
        预训练生成器网络
        使用L1损失和感知损失来预训练生成器，不涉及判别器
        
        参数:
            num_epochs (int): 预训练轮数
            batch_size (int): 批量大小
        """
        print(f"开始预训练生成器，共 {num_epochs} 轮...")
        
        # 设置优化器
        optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        
        # 损失函数
        l1_loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        
        # 训练循环
        for epoch in range(num_epochs):
            total_l1_loss = 0.0
            total_perceptual_loss = 0.0
            num_batches = 0
            
            # 创建训练数据批次
            for _ in range(10):  # 每轮10个批次
                # 准备训练数据
                lr_batch, hr_batch = self.prepare_training_data()
                
                # 确保数据在正确的设备上
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                
                # 生成高分辨率图像
                fake_hr = self.generator(lr_batch)
                
                # 计算L1损失
                l1_l = l1_loss(fake_hr, hr_batch)
                
                # 计算感知损失（使用VGG特征）
                if hasattr(self, 'vgg'):
                    fake_features = self.vgg(fake_hr)
                    real_features = self.vgg(hr_batch)
                    perceptual_loss = mse_loss(fake_features, real_features)
                else:
                    perceptual_loss = torch.tensor(0.0, device=device)
                
                # 总损失
                loss = l1_l + 0.1 * perceptual_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_l1_loss += l1_l.item()
                total_perceptual_loss += perceptual_loss.item() if hasattr(self, 'vgg') else 0.0
                num_batches += 1
            
            # 打印训练统计信息
            avg_l1 = total_l1_loss / num_batches
            avg_perceptual = total_perceptual_loss / num_batches if hasattr(self, 'vgg') else 0.0
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
                print(f"预训练轮次 [{epoch+1}/{num_epochs}], "
                      f"L1损失: {avg_l1:.6f}, "
                      f"感知损失: {avg_perceptual:.6f}")
        
        print("生成器预训练完成！")
    
    def _generate_void_structure(self, n_points, box_size):
        """
        生成空洞结构，模拟宇宙空洞
        """
        print(f"生成空洞结构 ({n_points}个点)...")
        
        # 使用泊松圆盘采样生成均匀分布的点
        points = []
        cell_size = box_size / (n_points ** (1/3))
        grid_size = int(np.ceil(box_size / cell_size))
        
        # 初始化网格
        grid = np.full((grid_size, grid_size, grid_size), -1, dtype=int)
        active = []
        
        # 添加第一个随机点
        first_point = np.random.uniform(0, box_size, 3)
        points.append(first_point)
        grid_coord = (first_point / cell_size).astype(int)
        grid[grid_coord[0], grid_coord[1], grid_coord[2]] = 0
        active.append(0)
        
        # 使用Bridson算法生成泊松圆盘采样
        while active:
            # 随机选择一个活跃点
            idx = np.random.randint(len(active))
            point_idx = active[idx]
            point = points[point_idx]
            found = False
            
            # 尝试生成新点
            for _ in range(30):  # 最多尝试30次
                # 在环形区域内生成随机点
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.arccos(2*np.random.uniform(0, 1) - 1)
                r = np.random.uniform(cell_size, 2*cell_size)
                
                dx = r * np.sin(phi) * np.cos(theta)
                dy = r * np.sin(phi) * np.sin(theta)
                dz = r * np.cos(phi)
                
                new_point = point + np.array([dx, dy, dz])
                
                # 检查是否在边界内
                if np.any(new_point < 0) or np.any(new_point >= box_size):
                    continue
                    
                # 检查是否与其他点太近
                grid_coord = (new_point / cell_size).astype(int)
                min_i = max(0, grid_coord[0]-2)
                max_i = min(grid_size-1, grid_coord[0]+2)
                min_j = max(0, grid_coord[1]-2)
                max_j = min(grid_size-1, grid_coord[1]+2)
                min_k = max(0, grid_coord[2]-2)
                max_k = min(grid_size-1, grid_coord[2]+2)
                
                too_close = False
                for i in range(min_i, max_i+1):
                    for j in range(min_j, max_j+1):
                        for k in range(min_k, max_k+1):
                            if grid[i,j,k] != -1:
                                dist = np.linalg.norm(new_point - points[grid[i,j,k]])
                                if dist < cell_size:
                                    too_close = True
                                    break
                        if too_close:
                            break
                    if too_close:
                        break
                
                if not too_close:
                    # 添加新点
                    points.append(new_point)
                    grid[grid_coord[0], grid_coord[1], grid_coord[2]] = len(points) - 1
                    active.append(len(points) - 1)
                    found = True
                    break
            
            # 如果无法生成新点，则移除当前活跃点
            if not found:
                active.pop(idx)
            
            # 如果达到目标点数，停止
            if len(points) >= n_points:
                break
        
        # 如果生成的点数不足，添加随机点
        while len(points) < n_points:
            points.append(np.random.uniform(0, box_size, 3))
        
        return np.array(points[:n_points])
        
    def create_hyperedges_and_wormholes_advanced(self):
        """
        创建高级超边和虫洞连接
        基于节点属性和空间关系创建复杂的网络结构
        """
        print("创建高级超边和虫洞连接...")
        
        # 清空现有超边和虫洞
        self.hyperedges = []
        self.wormholes = []
        
        # 获取所有节点ID和位置
        node_ids = list(self.nodes.keys())
        positions = np.array([self.nodes[node_id]['position'] for node_id in node_ids])
        
        # 计算节点之间的距离矩阵
        dist_matrix = squareform(pdist(positions))
        
        # 为每种节点类型创建索引
        earth_indices = [i for i, node_id in enumerate(node_ids) if self.nodes[node_id]['type'] == 'earth']
        blackhole_indices = [i for i, node_id in enumerate(node_ids) if self.nodes[node_id]['type'] == 'blackhole']
        darkenergy_indices = [i for i, node_id in enumerate(node_ids) if self.nodes[node_id]['type'] == 'darkenergy']
        
        # 1. 创建基于空间邻近的超边（局部连接）
        print("创建空间邻近超边...")
        for i, node_id in enumerate(node_ids):
            # 找到最近的k个节点
            k = min(10, len(node_ids) - 1)  # 每个节点最多连接10个邻居
            if k == 0:
                continue
                
            # 获取距离最近的k个节点（排除自身）
            nearest = np.argpartition(dist_matrix[i], k+1)[1:k+1]
            
            # 创建超边
            hyperedge = [node_id] + [node_ids[j] for j in nearest]
            self.hyperedges.append({
                'nodes': hyperedge,
                'type': 'spatial',
                'strength': 1.0,
                'creation_time': 0
            })
        
        # 2. 创建基于节点类型的超边（功能连接）
        print("创建功能超边...")
        # 地球节点之间的连接（社交网络）
        for i in earth_indices:
            if len(earth_indices) < 2:
                continue
                
            # 随机选择几个其他地球节点
            k = min(5, len(earth_indices) - 1)
            others = np.random.choice(earth_indices, size=k, replace=False)
            others = [idx for idx in others if idx != i]
            
            if others:
                hyperedge = [node_ids[i]] + [node_ids[j] for j in others]
                self.hyperedges.append({
                    'nodes': hyperedge,
                    'type': 'social',
                    'strength': 0.8,
                    'creation_time': 0
                })
        
        # 黑洞与周围节点的连接（引力场）
        for i in blackhole_indices:
            # 找到黑洞附近的所有节点
            radius = self.nodes[node_ids[i]].get('event_horizon_radius', 20.0)
            nearby = [j for j in range(len(node_ids)) 
                     if dist_matrix[i, j] < radius and j != i]
            
            if nearby:
                hyperedge = [node_ids[i]] + [node_ids[j] for j in nearby]
                self.hyperedges.append({
                    'nodes': hyperedge,
                    'type': 'gravitational',
                    'strength': 1.5,
                    'creation_time': 0
                })
        
        # 3. 创建虫洞连接（远程连接）
        print("创建虫洞连接...")
        # 在黑洞和暗能量节点之间创建虫洞
        for bh_idx in blackhole_indices[:min(10, len(blackhole_indices))]:  # 最多创建10个虫洞
            # 找到最远的暗能量节点
            if not darkenergy_indices:
                continue
                
            # 计算到所有暗能量节点的距离
            de_dists = dist_matrix[bh_idx, darkenergy_indices]
            furthest_de_idx = darkenergy_indices[np.argmax(de_dists)]
            
            # 创建虫洞
            self.wormholes.append({
                'source': node_ids[bh_idx],
                'target': node_ids[furthest_de_idx],
                'length': dist_matrix[bh_idx, furthest_de_idx] * 0.1,  # 虫洞缩短距离
                'stability': 0.8,
                'creation_time': 0
            })
        
        # 4. 创建暗能量网络（全局连接）
        print("创建暗能量网络...")
        if len(darkenergy_indices) > 1:
            # 使用Delaunay三角剖分创建暗能量网络
            de_positions = positions[darkenergy_indices]
            try:
                tri = Delaunay(de_positions)
                
                # 从三角剖分中提取边
                edges = set()
                for simplex in tri.simplices:
                    for i in range(4):
                        for j in range(i+1, 4):
                            edge = tuple(sorted([simplex[i], simplex[j]]))
                            edges.add(edge)
                
                # 创建超边
                for i, j in edges:
                    if i < len(darkenergy_indices) and j < len(darkenergy_indices):
                        source = node_ids[darkenergy_indices[i]]
                        target = node_ids[darkenergy_indices[j]]
                        
                        self.hyperedges.append({
                            'nodes': [source, target],
                            'type': 'dark_energy',
                            'strength': 0.6,
                            'creation_time': 0
                        })
            except Exception as e:
                print(f"创建暗能量网络时出错: {e}")
        
        print(f"创建完成: {len(self.hyperedges)} 条超边, {len(self.wormholes)} 个虫洞")

    def build_generator(self):
        """
        构建生成器网络 (SRResNet架构)
        """
        # 计算上采样后的尺寸
        upsampled_size = self.grid_resolution // self.upscale_factor
        
        model = torch.nn.Sequential(
            # 输入: [batch, channels, H, W, D]
            torch.nn.Conv3d(1, 64, kernel_size=9, padding=4, padding_mode='replicate'),
            torch.nn.PReLU(),
            
            # 残差块
            *[self._residual_block(64) for _ in range(5)],
            
            # 上采样 (使用插值+卷积避免棋盘伪影)
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            torch.nn.Conv3d(64, 64, kernel_size=3, padding=1, padding_mode='replicate'),
            torch.nn.PReLU(),
            
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            torch.nn.Conv3d(64, 64, kernel_size=3, padding=1, padding_mode='replicate'),
            torch.nn.PReLU(),
            
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            torch.nn.Conv3d(64, 64, kernel_size=3, padding=1, padding_mode='replicate'),
            torch.nn.PReLU(),
            
            # 最终输出层
            torch.nn.Conv3d(64, 1, kernel_size=9, padding=4, padding_mode='replicate'),
            torch.nn.Tanh()
        )
        return model.to(device)
    
    def build_discriminator(self):
        """
        构建判别器网络 (PatchGAN架构)
        修改为更简单的架构，避免小尺寸输入的问题
        """
        def discriminator_block(in_filters, out_filters, kernel_size=3, stride=1, padding=1, normalization=True):
            layers = [
                torch.nn.Conv3d(in_filters, out_filters, kernel_size, 
                              stride=stride, padding=padding, padding_mode='replicate')
            ]
            if normalization:
                layers.append(torch.nn.InstanceNorm3d(out_filters, affine=True))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        model = torch.nn.Sequential(
            # 输入: [batch, 1, H, W, D]
            # 第一层使用较大的核和步长来快速降维
            *discriminator_block(1, 64, kernel_size=4, stride=2, padding=1, normalization=False),
            *discriminator_block(64, 128, kernel_size=4, stride=2, padding=1),
            *discriminator_block(128, 256, kernel_size=4, stride=2, padding=1),
            *discriminator_block(256, 512, kernel_size=4, stride=2, padding=1),
            
            # 全局平均池化替代自适应池化，更稳定
            torch.nn.AdaptiveAvgPool3d(1),
            
            # 展平
            torch.nn.Flatten(),
            
            # 全连接层
            torch.nn.Linear(512 * 4 * 4 * 4, 1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            # 输出1维的判别结果
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )
        return model.to(device)
    
    def _residual_block(self, channels):
        """构建残差块"""
        return torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate'),
            torch.nn.BatchNorm3d(channels),
            torch.nn.PReLU(),
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate'),
            torch.nn.BatchNorm3d(channels)
        )
    
    def apply_unet_inspired_rescaling(self, value, field_type='density'):
        """
        应用U-Net启发的重缩放
        简化版本，直接返回输入值，避免GAN处理
        """
        # 直接返回输入值，避免GAN处理
        return float(value)

    def apply_gan_rescaling(self, density_field, field_type='density'):
        """
        应用基于GAN的超分辨率重缩放
        """
        # 转换为PyTorch张量并添加批次和通道维度
        input_tensor = torch.FloatTensor(density_field).unsqueeze(0).unsqueeze(0).to(device)
        
        # 生成高分辨率场
        with torch.no_grad():
            sr_field = self.generator(input_tensor)
        
        # 后处理
        if field_type == 'density':
            sr_field = torch.log10(1 + sr_field + self.rescale_a)
        elif field_type == 'velocity':
            sr_field = sr_field / self.rescale_b
        
        # 返回numpy数组
        return sr_field.squeeze().cpu().numpy()
    
def train_gan_step(self, lr_batch, hr_batch):
    """
    GAN训练步骤
    """
    # 确保输入在[-1, 1]范围内
    lr_batch = torch.clamp(lr_batch, -1.0, 1.0)
    hr_batch = torch.clamp(hr_batch, -1.0, 1.0)
    
    # 确保输入张量在正确的设备上
    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)
    
    # 确保输入是5D张量 [batch, channels, depth, height, width]
    if len(lr_batch.shape) == 2:  # [batch, features]
        # 将特征重塑为3D体积
        # 使用固定的空间维度，确保足够大以通过判别器
        target_size = 16  # 目标空间维度
        features = lr_batch.shape[1]
        
        # 计算最接近的立方体尺寸
        cube_side = max(2, int(round((features) ** (1/3))))
        if cube_side ** 3 < features:
            cube_side += 1
        
        # 确保最小尺寸为16
        cube_side = max(cube_side, 16)
        
        # 重塑为5D [batch, 1, d, h, w]
        lr_batch = lr_batch.view(-1, 1, cube_side, cube_side, cube_side)
    
    # 对hr_batch进行类似处理，但使用更大的目标尺寸
    if len(hr_batch.shape) == 2:
        features = hr_batch.shape[1]
        target_size = 32  # 高分辨率目标尺寸
        
        # 计算最接近的立方体尺寸
        cube_side = max(2, int(round((features) ** (1/3))))
        if cube_side ** 3 < features:
            cube_side += 1
        
        # 确保最小尺寸为32
        cube_side = max(cube_side, 32)
        
        # 重塑为5D [batch, 1, d, h, w]
        hr_batch = hr_batch.view(-1, 1, cube_side, cube_side, cube_side)
    
    # 确保张量是float类型
    lr_batch = lr_batch.float()
    hr_batch = hr_batch.float()
    
    # 检查并调整空间维度
    min_spatial_dim = 16  # 判别器有4个下采样层，每个层减半
    
    def ensure_min_dim(tensor, min_dim):
        if tensor.size(2) < min_dim or tensor.size(3) < min_dim or tensor.size(4) < min_dim:
            # 计算需要的缩放因子
            scale_factors = [
                min_dim / tensor.size(2) if tensor.size(2) < min_dim else 1.0,
                min_dim / tensor.size(3) if tensor.size(3) < min_dim else 1.0,
                min_dim / tensor.size(4) if tensor.size(4) < min_dim else 1.0
            ]
            # 使用三线性插值调整大小
            return torch.nn.functional.interpolate(
                tensor,
                scale_factor=scale_factors,
                mode='trilinear',
                align_corners=False
            )
        return tensor
    
    lr_batch = ensure_min_dim(lr_batch, min_spatial_dim)
    hr_batch = ensure_min_dim(hr_batch, min_spatial_dim)
    
    # 真实和假标签
    valid = torch.ones((lr_batch.size(0), 1), device=device)
    fake = torch.zeros((lr_batch.size(0), 1), device=device)
    
    # ---------------------
    #  训练判别器
    # ---------------------
    self.optimizer_D.zero_grad()
    
    try:
        # 真实样本的损失
        real_loss = self.criterion_GAN(self.discriminator(hr_batch), valid)
        
        # 生成假样本
        gen_hr = self.generator(lr_batch)
        fake_loss = self.criterion_GAN(self.discriminator(gen_hr.detach()), fake)
        
        # 梯度惩罚
        alpha = torch.rand(hr_batch.size(0), 1, 1, 1, 1, device=device)
        interpolates = (alpha * hr_batch + ((1 - alpha) * gen_hr.detach())).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_weight
        
        # 总判别器损失
        d_loss = real_loss + fake_loss + gradient_penalty
        d_loss.backward()
        self.optimizer_D.step()
        
        # -----------------
        #  训练生成器
        # -----------------
        if self.time_step % 3 == 0:  # 每3步更新一次生成器
            self.optimizer_G.zero_grad()
            
            # 生成器损失
            g_loss = self.criterion_GAN(self.discriminator(gen_hr), valid)
            
            # 像素级损失
            pixel_loss = self.criterion_pixel(gen_hr, hr_batch)
            
            # 总生成器损失
            g_total_loss = g_loss + 100 * pixel_loss  # 像素损失的权重
            g_total_loss.backward()
            self.optimizer_G.step()
        else:
            g_total_loss = torch.tensor(0.0, device=device)
            pixel_loss = torch.tensor(0.0, device=device)
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_total_loss.item() if isinstance(g_total_loss, torch.Tensor) else 0.0,
            'pixel_loss': pixel_loss.item() if isinstance(pixel_loss, torch.Tensor) else 0.0,
            'gradient_penalty': gradient_penalty.item()
        }
        
    except RuntimeError as e:
        print(f"训练步骤中发生错误: {str(e)}")
        print(f"输入张量形状 - lr_batch: {lr_batch.shape}, hr_batch: {hr_batch.shape}")
        if 'gen_hr' in locals():
            print(f"生成器输出形状: {gen_hr.shape}")
        raise e

    def generate_multi_scale_positions(self, n_objects, object_type, scale_level=0):
        """
        基于U-Net多尺度思想生成位置
        模拟编码器-解码器的多尺度特征提取
        """
        # 定义多尺度网格
        scales = [1.0, 0.5, 0.25, 0.125, 0.0625]  # 5个尺度层级
        current_scale = scales[min(scale_level, len(scales)-1)]
        effective_box_size = self.box_size * current_scale
        
        # 基于对象类型调整分布参数
        if object_type == 'blackhole':
            # 黑洞：集中分布，模拟星系团中心
            n_clusters = np.random.randint(5, 8)
            cluster_centers = np.random.randn(n_clusters, 3) * effective_box_size * 0.3
            
            positions = []
            objects_per_cluster = n_objects // n_clusters
            
            for i, center in enumerate(cluster_centers):
                n_in_cluster = objects_per_cluster + (1 if i < n_objects % n_clusters else 0)
                
                # 使用指数分布模拟深度特征
                r = np.random.exponential(scale=effective_box_size * 0.02, size=n_in_cluster)
                theta = np.random.uniform(0, 2*np.pi, n_in_cluster)
                phi = np.random.uniform(0, np.pi, n_in_cluster)
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                
                cluster_positions = np.column_stack([x, y, z]) + center
                positions.extend(cluster_positions)
                
        elif object_type == 'darkenergy':
            # 暗能量：纤维网络分布
            positions = self._generate_filament_network(n_objects, effective_box_size)
            
        elif object_type == 'earth':
            # 地球型物质：围绕结构分布
            positions = self._generate_hierarchical_matter_distribution(n_objects, effective_box_size)
            
        return np.array(positions[:n_objects])

    def _generate_filament_network(self, n_points, box_size):
        """
        生成纤维网络结构，模拟暗能量分布
        基于Delaunay三角剖分和最小生成树
        """
        # 首先生成节点
        n_nodes = max(10, n_points // 100)
        nodes = np.random.randn(n_nodes, 3) * box_size * 0.4
        
        try:
            # Delaunay三角剖分
            tri = Delaunay(nodes)
            
            # 构建连接
            edges = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
            
            # 沿边生成纤维点
            filament_points = []
            for edge in edges:
                start, end = nodes[edge[0]], nodes[edge[1]]
                n_segments = np.random.randint(10, 30)
                
                # 生成贝塞尔曲线路径
                t_values = np.linspace(0, 1, n_segments)
                for t in t_values:
                    pos = (1-t) * start + t * end
                    
                    # 添加扰动模拟非线性特征
                    if 0.1 < t < 0.9:
                        perturbation = np.random.normal(0, box_size * 0.01, 3)
                        pos += perturbation * np.sin(t * np.pi)
                    
                    filament_points.append(pos)
                    
                    if len(filament_points) >= n_points:
                        break
                        
                if len(filament_points) >= n_points:
                    break
                    
        except Exception as e:
            print(f"三角剖分失败: {e}, 使用随机分布")
            filament_points = [np.random.randn(3) * box_size * 0.5 for _ in range(n_points)]
            
        # 填充剩余点
        while len(filament_points) < n_points:
            if len(filament_points) > 0:
                base = random.choice(filament_points)
                pos = base + np.random.normal(0, box_size * 0.05, 3)
            else:
                pos = np.random.randn(3) * box_size * 0.5
            filament_points.append(pos)
            
        return filament_points

    def _generate_hierarchical_matter_distribution(self, n_points, box_size):
        """
        生成分层物质分布，模拟地球型物质
        """
        positions = []
        
        # 60%的点形成星系结构
        n_galaxy = int(n_points * 0.6)
        n_galaxy_centers = max(5, n_galaxy // 1000)
        galaxy_centers = np.random.randn(n_galaxy_centers, 3) * box_size * 0.3
        
        for _ in range(n_galaxy):
            center_idx = np.random.randint(0, n_galaxy_centers)
            center = galaxy_centers[center_idx]
            
            # 指数分布径向距离
            r = np.random.exponential(scale=box_size * 0.03)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            
            # 添加盘状结构
            if np.random.random() < 0.7:
                phi = np.pi/2 + np.random.normal(0, 0.1)
                
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) * (0.1 if np.random.random() < 0.7 else 1.0)
            
            pos = np.array([x, y, z]) + center
            positions.append(pos)
            
        # 30%的点在中等密度区域
        n_medium = int(n_points * 0.3)
        for _ in range(n_medium):
            pos = np.random.randn(3) * box_size * 0.4
            positions.append(pos)
            
        # 10%的点在低密度区域
        n_low = n_points - len(positions)
        for _ in range(n_low):
            pos = np.random.randn(3) * box_size * 0.8
            positions.append(pos)
            
        return positions

    def generate_cosmic_structure_positions_advanced(self):
        """
        高级宇宙大尺度结构位置生成
        集成深度学习多尺度特征提取思想
        """
        print("正在生成高级宇宙大尺度结构...")
        
        # 多尺度生成，模拟U-Net的编码器路径
        scales = ['coarse', 'medium', 'fine', 'ultra_fine']
        
        all_positions = {}
        
        for scale_idx, scale_name in enumerate(scales):
            print(f"生成 {scale_name} 尺度特征...")
            
            # 黑洞位置生成
            blackhole_positions = self.generate_multi_scale_positions(
                self.n_blackhole, 'blackhole', scale_idx
            )
            
            # 暗能量纤维网络生成
            darkenergy_positions = self.generate_multi_scale_positions(
                self.n_darkenergy, 'darkenergy', scale_idx
            )
            
            # 地球型物质分布生成
            earth_positions = self.generate_multi_scale_positions(
                self.n_earth, 'earth', scale_idx
            )
            
            # 存储多尺度特征
            self.multi_scale_features[scale_name] = {
                'blackhole': blackhole_positions,
                'darkenergy': darkenergy_positions,
                'earth': earth_positions
            }
            
        # 融合多尺度特征，模拟U-Net的跳跃连接
        final_blackhole_pos = self._fuse_multi_scale_features('blackhole')
        final_darkenergy_pos = self._fuse_multi_scale_features('darkenergy')
        final_earth_pos = self._fuse_multi_scale_features('earth')
        
        print(f"多尺度特征融合完成")
        
        return final_blackhole_pos, final_darkenergy_pos, final_earth_pos

    def _fuse_multi_scale_features(self, object_type):
        """
        融合多尺度特征，模拟U-Net的跳跃连接
        """
        # 权重分配：精细尺度权重更高
        weights = {'coarse': 0.1, 'medium': 0.2, 'fine': 0.3, 'ultra_fine': 0.4}
        
        fused_positions = None
        total_weight = 0
        
        for scale_name, weight in weights.items():
            if scale_name in self.multi_scale_features:
                positions = self.multi_scale_features[scale_name][object_type]
                
                if fused_positions is None:
                    fused_positions = positions * weight
                else:
                    # 确保形状匹配
                    min_len = min(len(fused_positions), len(positions))
                    fused_positions = fused_positions[:min_len] + positions[:min_len] * weight
                    
                total_weight += weight
                
        if total_weight > 0:
            fused_positions = fused_positions / total_weight
            
        return fused_positions

    def calculate_fluid_coupling_functions(self, x, y):
        """
        计算暗能量流体耦合势函数F(x,y)和G(x,y)
        增强版本，集成深度学习的非线性特征
        """
        # F(x,y): 多层非线性变换
        F_base = np.exp(-(x**2 + y**2)/10) * np.cos(0.5*np.sqrt(x**2 + y**2))
        F_nonlinear = np.tanh(F_base * 2) * np.sin(x * y * 0.1)
        F = F_base + 0.3 * F_nonlinear
        
        # G(x,y): 复杂的双连峰函数
        G_base = (np.exp(-((x-3)**2 + y**2)/5) + np.exp(-((x+3)**2 + y**2)/5)) * np.sin(np.arctan2(y, x))
        G_modulation = np.cos(x * 0.2) * np.sin(y * 0.2)
        G = G_base * (1 + 0.2 * G_modulation)
        
        return F, G

    def calculate_spacetime_metric_advanced(self, position, node_type):
        """
        计算增强的六维时空度规张量
        集成深度学习的非线性映射思想
        """
        x, y, z = position
        
        # 添加非线性修正项
        nonlinear_correction = 0.1 * np.sin(np.linalg.norm(position) * 0.1)
        
        if node_type == 'darkenergy':
            # 暗能量子时空度规
            F, G = self.calculate_fluid_coupling_functions(x, y)
            
            # 增强的度规张量
            metric = np.array([
                [1 + nonlinear_correction, 0, -self.epsilon_3 * G],
                [0, 1 + nonlinear_correction, -self.epsilon_2 * F],
                [-self.epsilon_3 * G, -self.epsilon_2 * F, 1 + nonlinear_correction],
                [0, 0, 0],
                [0, -self.epsilon_2 * F, -self.gamma_2 * (1 + nonlinear_correction)],
                [-self.epsilon_3 * G, 0, -self.gamma_3 * (1 + nonlinear_correction)]
            ])
            
        elif node_type == 'earth':
            # 地球子时空度规
            causal_correction = self.beta_t1 * np.exp(-np.linalg.norm(position)/10)
            causal_correction *= (1 + nonlinear_correction)
            
            metric = np.array([
                [1 + nonlinear_correction, 0, 0],
                [0, 1 + nonlinear_correction, 0],
                [0, 0, 1 + nonlinear_correction],
                [-1 - causal_correction, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
            
        else:  # blackhole
            # 黑洞子时空度规
            r = np.linalg.norm(position)
            sphere_factor = 1.0 / (1.0 + r**2/100) * (1 + nonlinear_correction)
            
            metric = np.eye(6)
            metric[3, 3] = -sphere_factor
            metric[4, 4] = -sphere_factor
            metric[5, 5] = -sphere_factor
            
        return metric

    def initialize_spacetime_nodes_advanced(self):
        """
        初始化高级六维时空节点
        """
        print("正在生成高级宇宙大尺度结构...")
        blackhole_positions, darkenergy_positions, earth_positions = \
            self.generate_cosmic_structure_positions_advanced()
        
        node_id = 0
        
        # 创建黑洞节点
        for i, pos in enumerate(blackhole_positions):
            # 应用重缩放
            rescaled_mass = self.apply_unet_inspired_rescaling(
                np.random.uniform(50, 200) * (1 + np.exp(-np.linalg.norm(pos)/5)), 'density'
            )
            
            self.nodes[node_id] = {
                'type': 'blackhole',
                'position': pos,
                'time_coords': [np.random.uniform(0, 1) for _ in range(3)],
                'mass': rescaled_mass,
                'energy': 0.0,
                'kappa': self.kappa_0 * (1 + self.alpha_sphere),
                'info_storage': 0.0,
                'sphere_radius': abs(np.random.normal(2, 0.5)) * (1 + np.random.random()),
                'unet_features': np.random.randn(self.n_filters_base)  # 模拟深度特征
            }
            self.hypergraph.add_node(node_id, **self.nodes[node_id])
            node_id += 1
            
        # 创建暗能量节点
        for i, pos in enumerate(darkenergy_positions):
            rescaled_energy = self.apply_unet_inspired_rescaling(
                np.random.uniform(0.5, 1.0), 'density'
            )
            
            self.nodes[node_id] = {
                'type': 'darkenergy',
                'position': pos,
                'time_coords': [np.random.uniform(0, 1) for _ in range(2)],
                'mass': np.random.uniform(0.01, 0.1),
                'energy': rescaled_energy,
                'kappa': self.kappa_0 * (1 + self.alpha_fluid),
                'fluid_velocity': np.random.uniform(0.5, 0.99),
                'control_info': np.random.uniform(0, 1),
                'coupling_F': 0.0,
                'coupling_G': 0.0,
                'unet_features': np.random.randn(self.n_filters_base)
            }
            self.hypergraph.add_node(node_id, **self.nodes[node_id])
            node_id += 1
            
        # 创建地球节点
        for i, pos in enumerate(earth_positions):
            if len(blackhole_positions) > 0:
                distances = [np.linalg.norm(pos - bh_pos) for bh_pos in blackhole_positions]
                min_dist = min(distances)
            else:
                min_dist = 1.0
                
            rescaled_mass = self.apply_unet_inspired_rescaling(
                np.random.uniform(0.1, 1.0) * (1 + np.exp(-min_dist/5)), 'density'
            )
            
            self.nodes[node_id] = {
                'type': 'earth',
                'position': pos,
                'time_coords': [np.random.uniform(0, 1)],
                'mass': rescaled_mass,
                'energy': 0.0,
                'kappa': self.kappa_0 * (1 + self.alpha_quantum),
                'causal_potential': 0.0,
                'unet_features': np.random.randn(self.n_filters_base)
            }
            self.hypergraph.add_node(node_id, **self.nodes[node_id])
            node_id += 1
            
        print(f"高级节点初始化完成，总计 {len(self.nodes)} 个节点")

    def calculate_time_charge_density(self, node_id):
        """
        计算节点的三维时间荷密度
        """
        node = self.nodes[node_id]
        mass = node.get('mass', 1.0)
        tc = node.get('time_coords', [0.0])
        
        t1 = tc[0] if len(tc) >= 1 else 0.0
        t2 = tc[1] if len(tc) >= 2 else 0.0
        t3 = tc[2] if len(tc) >= 3 else 0.0
        
        if node['type'] == 'earth':
            return np.array([mass * t1, 0.0, 0.0])
        elif node['type'] == 'blackhole':
            return np.array([mass * t1, mass * t2, mass * t3])
        elif node['type'] == 'darkenergy':
            return np.array([0.0, mass * t2, mass * t3])
        else:
            return np.zeros(3)

    def get_total_time_charge(self):
        """
        计算系统总时间荷
        """
        total_charge = np.zeros(3)
        for node_id, node in self.nodes.items():
            total_charge += self.calculate_node_time_charge(node_id)
        return total_charge

    def create_3d_visualization_advanced(self):
        """
        创建高级3D可视化
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 14), facecolor='black')
        
        # 主视图
        ax_main = fig.add_subplot(221, projection='3d', facecolor='black')
        ax_main.set_title('高级六维流形时空-深度学习增强虫洞效应', color='white', fontsize=16)
        
        # 设置坐标轴
        for ax in [ax_main]:
            ax.set_facecolor('black')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('0.8')
            ax.yaxis.pane.set_edgecolor('0.8')
            ax.zaxis.pane.set_edgecolor('0.8')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='z', colors='white')
            
        # 收集节点数据
        earth_nodes = [n for n, d in self.nodes.items() if d['type'] == 'earth']
        blackhole_nodes = [n for n, d in self.nodes.items() if d['type'] == 'blackhole']
        darkenergy_nodes = [n for n, d in self.nodes.items() if d['type'] == 'darkenergy']
        
        # 绘制地球节点（蓝色）
        if earth_nodes:
            earth_pos = np.array([self.nodes[n]['position'] for n in earth_nodes])
            ax_main.scatter(earth_pos[:, 0], earth_pos[:, 1], earth_pos[:, 2], 
                          c='blue', s=1, alpha=0.3, label='地球子时空')
            
        # 绘制黑洞节点（红色）
        if blackhole_nodes:
            bh_pos = np.array([self.nodes[n]['position'] for n in blackhole_nodes])
            ax_main.scatter(bh_pos[:, 0], bh_pos[:, 1], bh_pos[:, 2], 
                          c='red', s=50, alpha=0.8, label='黑洞子时空')
            
        # 绘制暗能量节点（紫色）
        if darkenergy_nodes:
            de_pos = np.array([self.nodes[n]['position'] for n in darkenergy_nodes])
            ax_main.scatter(de_pos[:, 0], de_pos[:, 1], de_pos[:, 2], 
                          c='purple', s=0.5, alpha=0.1, label='暗能量子时空')
            
        # 绘制虫洞连接（绿色）
        for wh in self.wormhole_connections:
            if wh['stability'] > 0.1:  # 只显示稳定的虫洞连接
                n1, n2 = wh['nodes']
                pos1 = self.nodes[n1]['position']
                pos2 = self.nodes[n2]['position']
                ax_main.plot([pos1[0], pos2[0]], 
                           [pos1[1], pos2[1]], 
                           [pos1[2], pos2[2]], 
                           'g-', alpha=0.3 * wh['stability'],
                           linewidth=1.5 * wh['flux_capacity'])
                
        # 添加图例
        legend = ax_main.legend(loc='upper right')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_facecolor('black')
        for text in legend.get_texts():
            text.set_color('white')
            
        # 子图：时间荷演化（增强版）
        if hasattr(self, 'history') and 'time_charges' in self.history and len(self.history['time_charges']) > 0:
            ax_charge = fig.add_subplot(222, facecolor='black')
            ax_charge.set_title('时间荷演化 (深度学习增强)', color='white', fontsize=14)
            charges = np.array(self.history['time_charges'])
            times = range(len(charges))
            ax_charge.plot(times, charges[:, 0], 'r-', label='τ₁荷', linewidth=2.5)
            ax_charge.plot(times, charges[:, 1], 'g-', label='τ₂荷', linewidth=2.5)
            ax_charge.plot(times, charges[:, 2], 'b-', label='τ₃荷', linewidth=2.5)
            ax_charge.set_xlabel('时间步', color='white')
            ax_charge.set_ylabel('时间荷密度', color='white')
            ax_charge.tick_params(colors='white')
            ax_charge.legend()
            
        plt.tight_layout()
        return fig
        
    def apply_wolfram_rewrite_rules_advanced(self):
        """
        应用Wolfram风格的高级重写规则
        """
        # 规则1: 地球节点间的三角化规则
        earth_nodes = [n for n, d in self.nodes.items() if d['type'] == 'earth']
        if len(earth_nodes) >= 3:
            # 随机选择一些地球节点应用三角化规则
            for _ in range(min(10, len(earth_nodes) // 100)):
                selected = random.sample(earth_nodes, 3)
                # 确保它们之间没有连接
                if not (self.hypergraph.has_edge(selected[0], selected[1]) or 
                        self.hypergraph.has_edge(selected[1], selected[2]) or 
                        self.hypergraph.has_edge(selected[0], selected[2])):
                    # 创建三角形连接
                    self.hypergraph.add_edge(selected[0], selected[1])
                    self.hypergraph.add_edge(selected[1], selected[2])
                    self.hypergraph.add_edge(selected[0], selected[2])
                    
        # 规则2: 黑洞和暗能量节点之间的信息交换
        blackhole_nodes = [n for n, d in self.nodes.items() if d['type'] == 'blackhole']
        darkenergy_nodes = [n for n, d in self.nodes.items() if d['type'] == 'darkenergy']
        
        if blackhole_nodes and darkenergy_nodes:
            for bh in blackhole_nodes:
                # 每个黑洞连接到最近的暗能量节点
                if darkenergy_nodes:
                    bh_pos = self.nodes[bh]['position']
                    closest_de = min(darkenergy_nodes, 
                                    key=lambda x: np.linalg.norm(self.nodes[x]['position'] - bh_pos))
                    
                    # 添加或更新虫洞连接
                    wh_exists = any(set(wh['nodes']) == {bh, closest_de} 
                                  for wh in self.wormhole_connections)
                    
                    if not wh_exists:
                        self.wormhole_connections.append({
                            'nodes': (bh, closest_de),
                            'stability': 1.0,
                            'flux_capacity': 1.0,
                            'created_at': self.time_step
                        })
                        
        # 规则3: 暗能量节点的膨胀效应
        for node_id in darkenergy_nodes:
            node = self.nodes[node_id]
            t_coords = node.get('time_coords', [0.0] * 3)
            expansion_factor = 1.0 + 0.01 * np.sqrt(sum(t**2 for t in t_coords))
            node['position'] *= expansion_factor
            
        # 规则4: 更新虫洞稳定性
        for wh in self.wormhole_connections:
            # 虫洞稳定性随时间衰减，但可以通过量子涨落暂时增强
            wh['stability'] = max(0.1, min(1.0, wh['stability'] * 0.99 + random.uniform(-0.05, 0.05)))
            
        # 规则5: 移除不稳定的虫洞
        self.wormhole_connections = [wh for wh in self.wormhole_connections 
                                   if wh['stability'] > 0.1]
                                   
        # 更新历史记录
        if not hasattr(self, 'history'):
            self.history = {'time_charges': []}
            
        # 记录当前时间荷状态
        self.history['time_charges'].append(self.get_total_time_charge())
        
        # 保持历史记录大小合理
        if len(self.history['time_charges']) > 1000:
            self.history['time_charges'] = self.history['time_charges'][-1000:]

    def detect_wormhole_conditions(self, earth_node, de_node):
        """
        检测虫洞形成条件
        """
        earth_pos = self.nodes[earth_node]['position']
        de_pos = self.nodes[de_node]['position']
        
        # 计算空间距离
        spatial_distance = np.linalg.norm(earth_pos - de_pos)
        
        # 计算时间荷差异
        earth_charge = self.calculate_time_charge_density(earth_node)
        de_charge = self.calculate_time_charge_density(de_node)
        charge_gradient = np.linalg.norm(earth_charge - de_charge)
        
        # 计算度规曲率
        earth_metric = self.calculate_spacetime_metric_advanced(earth_pos, 'earth')
        de_metric = self.calculate_spacetime_metric_advanced(de_pos, 'darkenergy')
        metric_curvature = np.linalg.norm(earth_metric - de_metric)
        
        # 虫洞形成判据
        wormhole_probability = (charge_gradient * metric_curvature) / (spatial_distance + 1e-6)
        
        return wormhole_probability > self.wormhole_strength

    def create_wormhole_connection(self, earth_node, de_node):
        """
        创建虫洞连接
        """
        wormhole = {
            'earth_node': earth_node,
            'de_node': de_node,
            'throat_radius': np.random.uniform(0.5, 2.0),
            'negative_energy': self.negative_energy_density,
            'flux_capacity': np.random.uniform(0.1, 0.5),
            'stability': self.traversability_factor,
            'creation_time': self.time_step
        }
        self.wormhole_connections.append(wormhole)
        
        # 在超图中添加虫洞连接
        self.hypergraph.add_edge(earth_node, de_node,
                                 edge_type='wormhole',
                                 weight=wormhole['flux_capacity'])
        return wormhole

    def create_hyperedges_and_wormholes_advanced(self):
        """
        创建高级超边连接和虫洞
        """
        # 获取不同类型节点
        bh_nodes = [n for n, d in self.nodes.items() if d['type'] == 'blackhole']
        earth_nodes = [n for n, d in self.nodes.items() if d['type'] == 'earth']
        de_nodes = [n for n, d in self.nodes.items() if d['type'] == 'darkenergy']
        
        # 1. 引力更新超边 (黑洞→地球)
        for _ in range(min(25, len(bh_nodes))):  # 增加连接数
            if not bh_nodes or not earth_nodes:
                break
            bh_node = random.choice(bh_nodes)
            earth_group = random.sample(earth_nodes, min(10, len(earth_nodes)))
            
            hyperedge = {
                'type': 'gravity_update',
                'nodes': [bh_node] + earth_group,
                'strength': np.random.uniform(0.2, 1.0),
                'unet_weight': np.random.uniform(0.5, 1.5)  # 深度学习权重
            }
            self.hyperedges.append(hyperedge)
            
            for earth_node in earth_group:
                self.hypergraph.add_edge(bh_node, earth_node,
                                         edge_type='gravity',
                                         weight=hyperedge['strength'])
        
        # 2. 量子纠缠超边 (黑洞↔暗能量)
        for _ in range(min(20, len(bh_nodes))):
            if not bh_nodes or not de_nodes:
                break
            bh_node = random.choice(bh_nodes)
            de_group = random.sample(de_nodes, min(6, len(de_nodes)))
            
            hyperedge = {
                'type': 'quantum_entanglement',
                'nodes': [bh_node] + de_group,
                'strength': np.random.uniform(0.5, 1.0),
                'unet_weight': np.random.uniform(0.5, 1.5)
            }
            self.hyperedges.append(hyperedge)
            
            for de_node in de_group:
                self.hypergraph.add_edge(bh_node, de_node,
                                         edge_type='entanglement',
                                         weight=hyperedge['strength'])
        
        # 3. 粒子控制超边 (暗能量→地球)
        for _ in range(min(35, len(de_nodes))):
            if not de_nodes or not earth_nodes:
                break
            de_node = random.choice(de_nodes)
            earth_group = random.sample(earth_nodes, min(20, len(earth_nodes)))
            
            hyperedge = {
                'type': 'particle_control',
                'nodes': [de_node] + earth_group,
                'strength': np.random.uniform(0.3, 0.8),
                'unet_weight': np.random.uniform(0.5, 1.5)
            }
            self.hyperedges.append(hyperedge)
            
            for earth_node in earth_group:
                if not self.hypergraph.has_edge(de_node, earth_node):
                    self.hypergraph.add_edge(de_node, earth_node,
                                             edge_type='control',
                                             weight=hyperedge['strength'])
        
        # 4. 高级虫洞检测和创建
        wormhole_count = 0
        max_wormholes = min(60, len(earth_nodes) // 80, len(de_nodes) // 8)
        
        for earth_node in random.sample(earth_nodes, min(300, len(earth_nodes))):
            if wormhole_count >= max_wormholes:
                break
            for de_node in random.sample(de_nodes, min(12, len(de_nodes))):
                if self.detect_wormhole_conditions(earth_node, de_node):
                    self.create_wormhole_connection(earth_node, de_node)
                    wormhole_count += 1
                    break
        
        print(f"高级超边创建完成: {len(self.hyperedges)} 个超边")
        print(f"虫洞连接创建: {len(self.wormhole_connections)} 个虫洞")

    def update_wormhole_dynamics_advanced(self):
        """
        更新高级虫洞动力学
        """
        active_wormholes = []
        
        for wormhole in self.wormhole_connections:
            earth_node = wormhole['earth_node']
            de_node = wormhole['de_node']
            
            # 检查节点是否仍存在
            if earth_node not in self.nodes or de_node not in self.nodes:
                continue
            
            # 计算虫洞稳定性（增强版）
            age = self.time_step - wormhole['creation_time']
            stability_decay = np.exp(-age * 0.008)  # 更慢的衰减
            wormhole['stability'] *= stability_decay
            
            # 如果稳定性过低，虫洞坍缩
            if wormhole['stability'] < 0.08:
                if self.hypergraph.has_edge(earth_node, de_node):
                    self.hypergraph.remove_edge(earth_node, de_node)
                continue
            
            # 计算质量-能量流（增强版）
            earth_mass = self.nodes[earth_node]['mass']
            de_energy = self.nodes[de_node]['energy']
            
            # 虫洞传输效应
            transfer_rate = wormhole['flux_capacity'] * wormhole['stability']
            
            # 地球 → 暗能量的质量转换
            mass_transfer = earth_mass * transfer_rate * 0.015
            energy_transfer = mass_transfer * (3e8)**2  # E = mc²
            
            self.nodes[earth_node]['mass'] *= (1 - transfer_rate * 0.015)
            self.nodes[de_node]['energy'] += energy_transfer
            
            # 时间荷通过虫洞传输
            earth_charge = self.calculate_time_charge_density(earth_node)
            de_charge = self.calculate_time_charge_density(de_node)
            charge_flow = (earth_charge - de_charge) * transfer_rate * 0.08
            
            # 更新时间坐标
            if len(self.nodes[earth_node]['time_coords']) >= 1:
                self.nodes[earth_node]['time_coords'][0] -= charge_flow[0] / (earth_mass + 1e-6)
            if len(self.nodes[de_node]['time_coords']) >= 2:
                self.nodes[de_node]['time_coords'][0] += charge_flow[1] / (de_energy + 1e-6)
                self.nodes[de_node]['time_coords'][1] += charge_flow[2] / (de_energy + 1e-6)
            
            active_wormholes.append(wormhole)
        
        self.wormhole_connections = active_wormholes

    def prepare_training_data(self):
        """
        准备GAN训练数据
        生成低分辨率和高分辨率数据对用于训练
        确保最小尺寸满足判别器要求
        
        Returns:
            tuple: (低分辨率张量, 高分辨率张量)
        """
        # 1. 收集所有节点的位置和属性
        positions = np.array([node['position'] for node in self.nodes.values()])
        
        # 2. 创建密度场 - 确保最小尺寸为16x16x16
        min_size = 16
        grid_res = max(min_size, self.grid_resolution // self.upscale_factor)
        density_field = np.zeros((grid_res, grid_res, grid_res))
        
        # 3. 将节点分布到网格中
        for pos in positions:
            # 归一化到[0,1]范围
            norm_pos = (pos - np.min(positions, axis=0)) / (np.ptp(positions, axis=0) + 1e-6)
            # 映射到网格索引
            grid_pos = (norm_pos * (grid_res - 1)).astype(int)
            try:
                density_field[grid_pos[0], grid_pos[1], grid_pos[2]] += 1.0
            except IndexError:
                continue
        
        # 4. 创建低分辨率和高分辨率对
        lr_field = density_field / (np.max(density_field) + 1e-6)
        
        # 5. 生成高分辨率目标（确保尺寸是upscale_factor的整数倍）
        hr_size = grid_res * self.upscale_factor
        hr_field = np.zeros((hr_size, hr_size, hr_size))
        
        # 使用插值放大
        from scipy.ndimage import zoom
        hr_field = zoom(lr_field, self.upscale_factor, order=1)
        
        return torch.FloatTensor(lr_field).unsqueeze(0).unsqueeze(0), \
               torch.FloatTensor(hr_field).unsqueeze(0).unsqueeze(0)

    def pretrain_generator(self, num_epochs=5):
        """
        预训练生成器（仅使用像素级损失）
        优化内存使用：
        - 使用更小的批处理大小
        - 添加梯度累积
        - 添加内存清理
        
        Args:
            num_epochs (int): 预训练轮数
        """
        self.generator.train()
        
        # 减小批处理大小并添加梯度累积
        batch_size = 4  # 减小批处理大小
        accumulation_steps = 4  # 梯度累积步数
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            
            # 准备训练数据
            lr_batch, hr_batch = self.prepare_training_data()
            
            # 确保数据在CPU上
            lr_batch = lr_batch.cpu()
            hr_batch = hr_batch.cpu()
            
            # 计算批次数
            num_batches = (lr_batch.size(0) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                # 获取当前批次数据
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, lr_batch.size(0))
                
                batch_lr = lr_batch[start_idx:end_idx].to(device)
                batch_hr = hr_batch[start_idx:end_idx].to(device)
                
                # 前向传播
                with torch.cuda.amp.autocast():  # 使用混合精度训练
                    fake_hr = self.generator(batch_lr)
                    loss = self.criterion_pixel(fake_hr, batch_hr) / accumulation_steps
                
                # 反向传播（累积梯度）
                loss.backward()
                
                # 每 accumulation_steps 步更新一次参数
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    self.optimizer_G.step()
                    self.optimizer_G.zero_grad()
                
                total_loss += loss.item() * (end_idx - start_idx)
                
                # 清理中间变量
                del batch_lr, batch_hr, fake_hr
                
            # 计算平均损失
            avg_loss = total_loss / lr_batch.size(0)
            
            if (epoch + 1) % 1 == 0:
                print(f'预训练生成器 Epoch [{epoch+1}/{num_epochs}], 平均像素损失: {avg_loss:.6f}')
                
            # 清理
            del lr_batch, hr_batch
            torch.cuda.empty_cache()

    def update_physics_advanced(self):
        """
        高级物理量更新
        集成深度学习启发的特征更新和GAN超分辨率
        """
        # 1. 记录初始时间荷
        initial_tau = self.get_total_time_charge()
        
        # 2. 更新暗能量耦合函数
        for node_id, node in self.nodes.items():
            if node['type'] == 'darkenergy':
                x, y, _ = node['position']
                F, G = self.calculate_fluid_coupling_functions(x, y)
                node['coupling_F'] = F
                node['coupling_G'] = G
        
        # 3. 基础物理量更新
        self._update_gravity_effects_advanced()
        self._update_quantum_entanglement_advanced()
        self._update_particle_control_advanced()
        self._update_darkenergy_expansion_advanced()
        
        # 4. 高级虫洞动力学更新
        self.update_wormhole_dynamics_advanced()
        
        # 5. 时间荷守恒校正
        self._enforce_time_charge_conservation(initial_tau)
        
        # 6. 应用Wolfram重写规则
        if self.time_step % 8 == 0:
            self.apply_wolfram_rewrite_rules_advanced()
        
        # 7. 更新深度特征
        self._update_unet_features()
        
        # 8. 更新度规和几何
        self._update_spacetime_geometry_advanced()
        
        # 9. 训练GAN（每10个时间步）
        if self.time_step % 10 == 0 and self.time_step > 0:
            # 生成训练数据（限制数量以减少内存使用）
            earth_positions = np.array([self.nodes[n]['position'] for n in self.nodes 
                                     if self.nodes[n]['type'] == 'earth'])
            
            # 限制样本数量
            max_samples = min(32, len(earth_positions))  # 进一步减少样本数
            if max_samples >= 4:  # 确保有足够的样本
                indices = np.random.choice(len(earth_positions), max_samples, replace=False)
                hr_batch = torch.tensor(earth_positions[indices], dtype=torch.float32, device='cpu')
                
                # 创建低分辨率版本
                lr_batch = hr_batch / self.upscale_factor
                
                # 分批处理
                batch_size = 4  # 更小的批处理大小
                for i in range(0, max_samples, batch_size):
                    batch_end = min(i + batch_size, max_samples)
                    batch_lr = lr_batch[i:batch_end].to(device)
                    batch_hr = hr_batch[i:batch_end].to(device)
                    
                    # 训练一个批次
                    self.train_gan_step(batch_lr, batch_hr)
                    
                    # 清理
                    del batch_lr, batch_hr
                    torch.cuda.empty_cache()

    def _update_gravity_effects_advanced(self):
        """
        更新高级引力效应
        """
        for hyperedge in self.hyperedges:
            if hyperedge['type'] == 'gravity_update':
                bh_node = hyperedge['nodes'][0]
                earth_nodes = hyperedge['nodes'][1:]
                
                if bh_node not in self.nodes:
                    continue
                
                bh_mass = self.nodes[bh_node]['mass']
                bh_pos = self.nodes[bh_node]['position']
                unet_weight = hyperedge.get('unet_weight', 1.0)
                
                for earth_node in earth_nodes:
                    if earth_node not in self.nodes:
                        continue
                    
                    earth_pos = self.nodes[earth_node]['position']
                    distance = np.linalg.norm(bh_pos - earth_pos) + 1e-6
                    
                    # 引力效应（增强版）
                    gravity_effect = hyperedge['strength'] * bh_mass / (distance**2)
                    gravity_effect *= unet_weight  # 深度学习权重调制
                    
                    self.nodes[earth_node]['energy'] += gravity_effect * 0.12
                    
                    # 时空弯曲引起的位置变化
                    direction = (bh_pos - earth_pos) / distance
                    displacement = direction * gravity_effect * 0.0015
                    self.nodes[earth_node]['position'] += displacement
                    
                    # 因果势能更新
                    causal_potential = self.beta_t1 * (self.nodes[bh_node]['kappa'] - 
                                                       self.nodes[earth_node]['kappa'])
                    self.nodes[earth_node]['causal_potential'] = causal_potential

    def _update_quantum_entanglement_advanced(self):
        """
        更新高级量子纠缠效应
        """
        for hyperedge in self.hyperedges:
            if hyperedge['type'] == 'quantum_entanglement':
                bh_node = hyperedge['nodes'][0]
                de_nodes = hyperedge['nodes'][1:]
                
                if bh_node not in self.nodes:
                    continue
                
                bh_info = self.nodes[bh_node].get('info_storage', 0)
                unet_weight = hyperedge.get('unet_weight', 1.0)
                
                for de_node in de_nodes:
                    if de_node not in self.nodes:
                        continue
                    
                    de_info = self.nodes[de_node]['control_info']
                    
                    # 量子信息交换（增强版）
                    exchange_rate = hyperedge['strength'] * self.lambda_ent * 0.12 * unet_weight
                    new_bh_info = (bh_info + de_info * exchange_rate) / 2
                    new_de_info = (de_info + bh_info * exchange_rate) / 2
                    
                    self.nodes[bh_node]['info_storage'] = new_bh_info
                    self.nodes[de_node]['control_info'] = new_de_info
                    
                    # 时间荷交换
                    if (len(self.nodes[bh_node]['time_coords']) >= 2 and 
                        len(self.nodes[de_node]['time_coords']) >= 2):
                        
                        # t2分量交换
                        t2_flow = (self.lambda_ent * 
                                   (self.nodes[bh_node]['time_coords'][1] - 
                                    self.nodes[de_node]['time_coords'][0]) * 0.012)
                        
                        self.nodes[bh_node]['time_coords'][1] -= t2_flow
                        self.nodes[de_node]['time_coords'][0] += t2_flow
                        
                        # t3分量交换
                        if (len(self.nodes[bh_node]['time_coords']) >= 3 and 
                            len(self.nodes[de_node]['time_coords']) >= 2):
                            
                            t3_flow = (self.lambda_ent * 
                                       (self.nodes[bh_node]['time_coords'][2] - 
                                        self.nodes[de_node]['time_coords'][1]) * 0.012)
                            
                            self.nodes[bh_node]['time_coords'][2] -= t3_flow
                            self.nodes[de_node]['time_coords'][1] += t3_flow

    def _update_particle_control_advanced(self):
        """
        更新高级粒子控制效应
        """
        for hyperedge in self.hyperedges:
            if hyperedge['type'] == 'particle_control':
                de_node = hyperedge['nodes'][0]
                earth_nodes = hyperedge['nodes'][1:]
                
                if de_node not in self.nodes:
                    continue
                
                control_strength = self.nodes[de_node]['control_info']
                unet_weight = hyperedge.get('unet_weight', 1.0)
                
                for earth_node in earth_nodes:
                    if earth_node not in self.nodes:
                        continue
                    
                    # 粒子质量控制（增强版）
                    mass_correction = (hyperedge['strength'] * control_strength * 
                                       0.025 * unet_weight)
                    self.nodes[earth_node]['mass'] *= (1 + mass_correction - 0.012)
                    self.nodes[earth_node]['mass'] = max(0.01, self.nodes[earth_node]['mass'])
                    
                    # 时间荷传输 (暗能量→地球)
                    if (len(self.nodes[de_node]['time_coords']) >= 1 and 
                        len(self.nodes[earth_node]['time_coords']) >= 1):
                        
                        charge_transfer = control_strength * hyperedge['strength'] * 0.008
                        self.nodes[earth_node]['time_coords'][0] += charge_transfer

    def _update_darkenergy_expansion_advanced(self):
        """
        更新高级暗能量膨胀效应
        """
        de_nodes = [n for n, d in self.nodes.items() if d['type'] == 'darkenergy']
        
        for de_node in de_nodes:
            # 计算膨胀驱动（增强版）
            time_coords = self.nodes[de_node]['time_coords']
            expansion_factor = 1.0 + 0.0012 * np.sqrt(sum(t**2 for t in time_coords))
            
            # 位置膨胀
            self.nodes[de_node]['position'] *= expansion_factor
            
            # 能量增长
            self.nodes[de_node]['energy'] *= expansion_factor**0.6
            
            # 流体速度演化
            if 'fluid_velocity' in self.nodes[de_node]:
                self.nodes[de_node]['fluid_velocity'] = min(0.99,
                    self.nodes[de_node]['fluid_velocity'] * expansion_factor**0.12)

    def _enforce_time_charge_conservation(self, initial_tau):
        """
        强制时间荷守恒
        """
        final_tau = self.get_total_time_charge()
        drift = np.linalg.norm(final_tau - initial_tau)
        
        if drift > self.tau_threshold:
            # 计算校正比例
            ratios = np.divide(initial_tau, final_tau + 1e-12,
                               out=np.ones_like(initial_tau), where=final_tau!=0)
            
            # 应用校正
            for node_id, node in self.nodes.items():
                tc = node['time_coords']
                if node['type'] == 'earth' and len(tc) >= 1:
                    tc[0] *= ratios[0]
                elif node['type'] == 'blackhole':
                    if len(tc) >= 1:
                        tc[0] *= ratios[0]
                    if len(tc) >= 2:
                        tc[1] *= ratios[1]
                    if len(tc) >= 3:
                        tc[2] *= ratios[2]
                elif node['type'] == 'darkenergy':
                    if len(tc) >= 1:
                        tc[0] *= ratios[1]
                    if len(tc) >= 2:
                        tc[1] *= ratios[2]

    def apply_wolfram_rewrite_rules_advanced(self):
        """
        应用高级Wolfram超图重写规则
        """
        # 规则1: 几何膨胀重写（增强版）
        de_nodes = [n for n, d in self.nodes.items() if d['type'] == 'darkenergy']
        for de_node in de_nodes:
            expansion_factor = self.rewrite_rules['geometric_expansion']
            self.nodes[de_node]['position'] *= expansion_factor
            
            # 更新流体速度
            if 'fluid_velocity' in self.nodes[de_node]:
                self.nodes[de_node]['fluid_velocity'] *= expansion_factor**0.6
        
        # 规则2: 拓扑融合重写（增强版）
        earth_nodes = [n for n, d in self.nodes.items() if d['type'] == 'earth']
        fusion_candidates = []
        
        for i, node1 in enumerate(earth_nodes):
            for node2 in earth_nodes[i+1:]:
                if node1 in self.nodes and node2 in self.nodes:
                    pos1 = self.nodes[node1]['position']
                    pos2 = self.nodes[node2]['position']
                    distance = np.linalg.norm(pos1 - pos2)
                    if distance < 0.6:  # 稍微放宽融合条件
                        fusion_candidates.append((node1, node2, distance))
        
        # 执行融合
        fusion_candidates.sort(key=lambda x: x[2])
        fused_pairs = set()
        
        for node1, node2, _ in fusion_candidates[:min(12, len(fusion_candidates))]:
            if (node1 not in fused_pairs and node2 not in fused_pairs and 
                node1 in self.nodes and node2 in self.nodes):
                
                # 融合节点
                combined_mass = self.nodes[node1]['mass'] + self.nodes[node2]['mass']
                combined_energy = self.nodes[node1]['energy'] + self.nodes[node2]['energy']
                combined_pos = (self.nodes[node1]['position'] + 
                                self.nodes[node2]['position']) / 2
                
                # 更新第一个节点
                self.nodes[node1]['mass'] = combined_mass
                self.nodes[node1]['energy'] = combined_energy
                self.nodes[node1]['position'] = combined_pos
                
                # 移除第二个节点
                if self.hypergraph.has_node(node2):
                    self.hypergraph.remove_node(node2)
                del self.nodes[node2]
                
                fused_pairs.add(node1)
                fused_pairs.add(node2)
        
        # 规则3: 信息扩散重写（增强版）
        for hyperedge in self.hyperedges:
            if hyperedge['type'] == 'quantum_entanglement':
                diffusion_rate = self.rewrite_rules['information_diffusion']
                for node in hyperedge['nodes']:
                    if node in self.nodes and 'info_storage' in self.nodes[node]:
                        self.nodes[node]['info_storage'] += diffusion_rate * 1.2
        
        # 规则4: 因果传播重写（增强版）
        causal_edges = [(u, v) for u, v, d in self.hypergraph.edges(data=True)
                        if d.get('edge_type') == 'gravity']
        
        for u, v in causal_edges:
            if u in self.nodes and v in self.nodes:
                # 增强因果连接
                current_weight = self.hypergraph[u][v].get('weight', 0.1)
                new_weight = current_weight * (1 + self.rewrite_rules['causal_propagation'])
                self.hypergraph[u][v]['weight'] = min(new_weight, 1.2)

    def _update_unet_features(self):
        """
        更新深度学习特征
        模拟U-Net的特征传播
        """
        for node_id, node in self.nodes.items():
            if 'unet_features' in node:
                # 非线性激活函数
                features = node['unet_features']
                
                # ReLU激活
                features = np.maximum(0, features)
                
                # 批量归一化
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)
                
                # 添加噪声正则化
                features += np.random.normal(0, 0.01, features.shape)
                
                node['unet_features'] = features

    def _update_spacetime_geometry_advanced(self):
        """
        更新高级时空几何
        """
        for node_id, node in self.nodes.items():
            # 计算时空曲率
            metric = self.calculate_spacetime_metric_advanced(node['position'], node['type'])
            curvature = np.trace(metric @ metric.T)
            
            # 更新有效耦合常数
            if node['type'] == 'earth':
                geometric_correction = 1.0 + curvature * 1.2e-6
                node['kappa'] = (self.kappa_0 * (1 + self.alpha_quantum) * 
                                 geometric_correction)
            elif node['type'] == 'darkenergy':
                fluid_correction = 1.0 + node['fluid_velocity'] * curvature * 1.5e-4
                node['kappa'] = (self.kappa_0 * (1 + self.alpha_fluid) * 
                                 fluid_correction)
            elif node['type'] == 'blackhole':
                sphere_correction = 1.0 + node['sphere_radius'] * curvature * 1.8e-5
                node['kappa'] = (self.kappa_0 * (1 + self.alpha_sphere) * 
                                 sphere_correction)

    def record_evolution_statistics_advanced(self):
        """
        记录高级演化统计数据
        """
        # 计算质量-能量统计
        total_mass_bh = sum(self.nodes[n]['mass'] for n in self.nodes
                            if self.nodes[n]['type'] == 'blackhole')
        total_mass_earth = sum(self.nodes[n]['mass'] for n in self.nodes
                               if self.nodes[n]['type'] == 'earth')
        total_energy_de = sum(self.nodes[n]['energy'] for n in self.nodes
                              if self.nodes[n]['type'] == 'darkenergy')
        
        # 计算虫洞通量
        total_wormhole_flux = sum(wh['flux_capacity'] * wh['stability']
                                  for wh in self.wormhole_connections)
        
        # 计算时间荷
        total_time_charge = self.get_total_time_charge()
        
        # 计算深度特征统计
        unet_feature_stats = []
        for node_id, node in self.nodes.items():
            if 'unet_features' in node:
                unet_feature_stats.append(np.mean(np.abs(node['unet_features'])))
        
        avg_unet_features = np.mean(unet_feature_stats) if unet_feature_stats else 0
        
        # 记录历史
        self.history['masses'].append({
            'blackhole_total': total_mass_bh,
            'earth_total': total_mass_earth,
            'darkenergy_total': total_energy_de,
            'mass_ratio': total_mass_bh / max(total_mass_earth, 1e-6),
            'unet_features_avg': avg_unet_features
        })
        
        self.history['wormhole_fluxes'].append(total_wormhole_flux)
        self.history['time_charges'].append(total_time_charge.copy())

    def create_3d_visualization_advanced(self):
        """
        创建高级3D可视化
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 14), facecolor='black')
        
        # 主视图
        ax_main = fig.add_subplot(221, projection='3d', facecolor='black')
        ax_main.set_title('高级六维流形时空-深度学习增强虫洞效应', color='white', fontsize=16)
        
        # 设置坐标轴
        for ax in [ax_main]:
            ax.set_facecolor('black')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.tick_params(colors='white')
            ax.set_xlabel('X (h⁻¹Mpc)', color='white')
            ax.set_ylabel('Y (h⁻¹Mpc)', color='white')
            ax.set_zlabel('Z (h⁻¹Mpc)', color='white')
        
        # 获取位置数据
        positions = {node_id: node['position'] for node_id, node in self.nodes.items()}
        
        # 分离不同类型节点
        earth_pos = np.array([pos for node_id, pos in positions.items()
                              if self.nodes[node_id]['type'] == 'earth'])
        bh_pos = np.array([pos for node_id, pos in positions.items()
                           if self.nodes[node_id]['type'] == 'blackhole'])
        de_pos = np.array([pos for node_id, pos in positions.items()
                           if self.nodes[node_id]['type'] == 'darkenergy'])
        
        # 绘制暗能量纤维网络（增强效果）
        if len(de_pos) > 0:
            ax_main.scatter(de_pos[:, 0], de_pos[:, 1], de_pos[:, 2],
                            c='cyan', s=20, alpha=0.4, label=f'暗能量网络 ({len(de_pos)})')
        
        # 绘制地球型物质（增强采样）
        if len(earth_pos) > 0:
            sample_size = min(4000, len(earth_pos))
            indices = np.random.choice(len(earth_pos), sample_size, replace=False)
            earth_sample = earth_pos[indices]
            
            # 根据密度着色
            colors = plt.cm.plasma(np.linspace(0.1, 0.9, sample_size))
            ax_main.scatter(earth_sample[:, 0], earth_sample[:, 1], earth_sample[:, 2],
                            c=colors, s=4, alpha=0.8, label=f'地球子时空 ({len(earth_pos)})')
        
        # 绘制黑洞（增强效果）
        if len(bh_pos) > 0:
            ax_main.scatter(bh_pos[:, 0], bh_pos[:, 1], bh_pos[:, 2],
                            c='gold', s=300, alpha=1.0, label=f'黑洞子时空 ({len(bh_pos)})',
                            edgecolors='orange', linewidths=3)
            
            # 多层光晕效果
            for i, alpha in enumerate([0.4, 0.2, 0.1]):
                ax_main.scatter(bh_pos[:, 0], bh_pos[:, 1], bh_pos[:, 2],
                                c='yellow', s=500 + i*200, alpha=alpha)
        
        # 绘制虫洞连接（增强效果）
        wormhole_count = 0
        for wormhole in self.wormhole_connections:
            if wormhole_count >= 20:  # 限制显示数量
                break
            earth_node = wormhole['earth_node']
            de_node = wormhole['de_node']
            if earth_node in self.nodes and de_node in self.nodes:
                pos1 = self.nodes[earth_node]['position']
                pos2 = self.nodes[de_node]['position']
                
                # 虫洞连线（脉冲效果）
                stability = wormhole['stability']
                ax_main.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                             'magenta', alpha=stability, linewidth=3)
                wormhole_count += 1
        
        # 添加增强统计信息
        stats_text = f'时间步: {self.time_step}\n虫洞数: {len(self.wormhole_connections)}\n'
        stats_text += f'网格分辨率: {self.grid_resolution}³\n'
        if self.history['masses']:
            latest = self.history['masses'][-1]
            stats_text += f'总质量比: {latest["mass_ratio"]:.2f}\n'
            if 'unet_features_avg' in latest:
                stats_text += f'深度特征: {latest["unet_features_avg"]:.3f}'
        
        ax_main.text2D(0.05, 0.95, stats_text, transform=ax_main.transAxes,
                       bbox=dict(facecolor='black', alpha=0.9, edgecolor='cyan'),
                       color='white', fontsize=11, verticalalignment='top')
        
        # 设置图例
        legend = ax_main.legend(loc='upper right', fontsize=11)
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_facecolor('black')
        for text in legend.get_texts():
            text.set_color('white')
        
        # 子图：时间荷演化（增强版）
        if len(self.history['time_charges']) > 0:
            ax_charge = fig.add_subplot(222, facecolor='black')
            ax_charge.set_title('时间荷演化 (深度学习增强)', color='white', fontsize=14)
            charges = np.array(self.history['time_charges'])
            times = range(len(charges))
            ax_charge.plot(times, charges[:, 0], 'r-', label='τ₁荷', linewidth=2.5)
            ax_charge.plot(times, charges[:, 1], 'g-', label='τ₂荷', linewidth=2.5)
            ax_charge.plot(times, charges[:, 2], 'b-', label='τ₃荷', linewidth=2.5)
            ax_charge.set_xlabel('时间步', color='white')
            ax_charge.set_ylabel('时间荷密度', color='white')
            ax_charge.tick_params(colors='white')
            ax_charge.legend()
            ax_charge.grid(True, alpha=0.4, color='gray')
        
        # 子图：虫洞通量演化（增强版）
        if len(self.history['wormhole_fluxes']) > 0:
            ax_flux = fig.add_subplot(223, facecolor='black')
            ax_flux.set_title('虫洞通量演化 (深度学习优化)', color='white', fontsize=14)
            fluxes = self.history['wormhole_fluxes']
            times = range(len(fluxes))
            ax_flux.plot(times, fluxes, 'magenta', linewidth=2.5)
            ax_flux.fill_between(times, fluxes, alpha=0.3, color='magenta')
            ax_flux.set_xlabel('时间步', color='white')
            ax_flux.set_ylabel('总虫洞通量', color='white')
            ax_flux.tick_params(colors='white')
            ax_flux.grid(True, alpha=0.4, color='gray')
        
        # 子图：深度特征演化
        if len(self.history['masses']) > 0:
            ax_ml = fig.add_subplot(224, facecolor='black')
            ax_ml.set_title('深度学习特征演化', color='white', fontsize=14)
            masses = self.history['masses']
            times = range(len(masses))
            
            # 绘制多个指标
            mass_ratios = [m['mass_ratio'] for m in masses]
            ax_ml.plot(times, mass_ratios, 'yellow', linewidth=2.5, label='质量比')
            
            if 'unet_features_avg' in masses[0]:
                unet_features = [m['unet_features_avg'] for m in masses]
                ax_ml_twin = ax_ml.twinx()
                ax_ml_twin.plot(times, unet_features, 'cyan', linewidth=2.5, label='U-Net特征')
                ax_ml_twin.set_ylabel('平均特征强度', color='cyan')
                ax_ml_twin.tick_params(axis='y', colors='cyan')
            
            ax_ml.set_xlabel('时间步', color='white')
            ax_ml.set_ylabel('质量比', color='yellow')
            ax_ml.tick_params(colors='white')
            ax_ml.legend(loc='upper left')
            ax_ml.grid(True, alpha=0.4, color='gray')
        
        plt.tight_layout()
        return fig


def evolve_advanced_hypergraph_ml(model, n_iterations=1000):
    """
    演化高级超图模型（GAN增强版）
    """
    print(f"开始演化高级六维流形时空模型（GAN增强），共 {n_iterations} 个时间步...")
    
    # 初始化CSV记录
    csv_file = 'advanced_spacetime_evolution_gan.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iteration', 'earth_nodes', 'blackhole_nodes', 'darkenergy_nodes',
            'wormhole_count', 'total_time_charge_t1', 'total_time_charge_t2', 'total_time_charge_t3',
            'total_wormhole_flux', 'earth_total_mass', 'blackhole_total_mass', 'darkenergy_total_energy',
            'generator_loss', 'discriminator_loss', 'grid_resolution', 'learning_rate'
        ])
    
    # 预训练生成器
    print("预训练生成器...")
    model.pretrain_generator(num_epochs=model.pretrain_epochs)
    
    for iteration in range(n_iterations):
        model.time_step = iteration
        
        # 更新物理量（GAN增强版）
        model.update_physics_advanced()
        
        # 记录统计（GAN增强版）
        model.record_evolution_statistics_advanced()
        
        # 每30步生成可视化和记录详细数据
        if iteration % 30 == 0:
            print(f"时间步 {iteration}: 生成GAN增强可视化和记录数据")
            
            # 统计当前状态
            earth_count = sum(1 for n in model.nodes if model.nodes[n]['type'] == 'earth')
            bh_count = sum(1 for n in model.nodes if model.nodes[n]['type'] == 'blackhole')
            de_count = sum(1 for n in model.nodes if model.nodes[n]['type'] == 'darkenergy')
            
            total_time_charge = model.get_total_time_charge()
            total_wormhole_flux = sum(wh['flux_capacity'] * wh['stability']
                                    for wh in model.wormhole_connections)
            
            earth_mass = sum(model.nodes[n]['mass'] for n in model.nodes
                           if model.nodes[n]['type'] == 'earth')
            bh_mass = sum(model.nodes[n]['mass'] for n in model.nodes
                        if model.nodes[n]['type'] == 'blackhole')
            de_energy = sum(model.nodes[n]['energy'] for n in model.nodes
                          if model.nodes[n]['type'] == 'darkenergy')
            
            # 获取GAN损失
            gen_loss = model.history['generator_loss'][-1] if model.history['generator_loss'] else 0
            disc_loss = model.history['discriminator_loss'][-1] if model.history['discriminator_loss'] else 0
            
            # 写入CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    iteration, earth_count, bh_count, de_count,
                    len(model.wormhole_connections),
                    total_time_charge[0], total_time_charge[1], total_time_charge[2],
                    total_wormhole_flux, earth_mass, bh_mass, de_energy,
                    gen_loss, disc_loss, model.grid_resolution, model.learning_rate
                ])
            
            print(f"  地球子时空: {earth_count} 节点")
            print(f"  黑洞子时空: {bh_count} 节点")
            print(f"  暗能量子时空: {de_count} 节点")
            print(f"  虫洞连接: {len(model.wormhole_connections)} 个")
            print(f"  GAN生成器损失: {gen_loss:.4f}, 判别器损失: {disc_loss:.4f}")
            
            # 生成高级可视化
            try:
                fig = model.create_3d_visualization_advanced()
                os.makedirs('gan_enhanced_spacetime_frames', exist_ok=True)
                plt.savefig(f'gan_enhanced_spacetime_frames/frame_{iteration:04d}.png',
                          dpi=120, bbox_inches='tight', facecolor='black')
                plt.close(fig)
                
                # 保存GAN生成样本
                if iteration > 0 and iteration % 100 == 0:
                    model.visualize_gan_results(iteration)
                    
            except Exception as e:
                print(f"  可视化生成失败: {e}")
        
        # 每80步应用重写规则
        if iteration % 80 == 0:
            print(f"  应用GAN增强Wolfram重写规则...")
            model.apply_wolfram_rewrite_rules_advanced()
    
    print("GAN增强演化完成！")
    return model

    def create_3d_visualization_advanced(self):
        """
        创建高级3D可视化
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 14), facecolor='black')
        
        # 主视图
        ax_main = fig.add_subplot(221, projection='3d', facecolor='black')
        ax_main.set_title('高级六维流形时空-深度学习增强虫洞效应', color='white', fontsize=16)
        
        # 设置坐标轴
        for ax in [ax_main]:
            ax.set_xlabel('X (kpc)', color='white')
            ax.set_ylabel('Y (kpc)', color='white')
            ax.set_zlabel('Z (kpc)', color='white')
            ax.tick_params(colors='white')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('0.8')
            ax.yaxis.pane.set_edgecolor('0.8')
            ax.zaxis.pane.set_edgecolor('0.8')
        
        # 绘制节点
        earth_nodes = [n for n, d in self.nodes.items() if d.get('type') == 'earth']
        bh_nodes = [n for n, d in self.nodes.items() if d.get('type') == 'blackhole']
        de_nodes = [n for n, d in self.nodes.items() if d.get('type') == 'darkenergy']
        
        # 绘制地球节点（蓝色）
        if earth_nodes:
            x = [self.nodes[n]['position'][0] for n in earth_nodes]
            y = [self.nodes[n]['position'][1] for n in earth_nodes]
            z = [self.nodes[n]['position'][2] for n in earth_nodes]
            ax_main.scatter(x, y, z, c='blue', s=1, alpha=0.3, label='地球子时空')
        
        # 绘制黑洞节点（红色）
        if bh_nodes:
            x = [self.nodes[n]['position'][0] for n in bh_nodes]
            y = [self.nodes[n]['position'][1] for n in bh_nodes]
            z = [self.nodes[n]['position'][2] for n in bh_nodes]
            ax_main.scatter(x, y, z, c='red', s=10, alpha=0.7, label='黑洞子时空')
        
        # 绘制暗能量节点（绿色）
        if de_nodes:
            x = [self.nodes[n]['position'][0] for n in de_nodes]
            y = [self.nodes[n]['position'][1] for n in de_nodes]
            z = [self.nodes[n]['position'][2] for n in de_nodes]
            ax_main.scatter(x, y, z, c='green', s=5, alpha=0.5, label='暗能量子时空')
        
        # 绘制虫洞连接
        for conn in self.wormhole_connections:
            node1 = conn['nodes'][0]
            node2 = conn['nodes'][1]
            if node1 in self.nodes and node2 in self.nodes:
                x = [self.nodes[node1]['position'][0], self.nodes[node2]['position'][0]]
                y = [self.nodes[node1]['position'][1], self.nodes[node2]['position'][1]]
                z = [self.nodes[node1]['position'][2], self.nodes[node2]['position'][2]]
                ax_main.plot(x, y, z, 'm-', alpha=0.5, linewidth=0.5)
        
        # 添加图例
        legend = ax_main.legend(loc='upper right')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_facecolor('black')
        for text in legend.get_texts():
            text.set_color('white')
        
        # 子图：时间荷演化（增强版）
        if len(self.history['time_charges']) > 0:
            ax_charge = fig.add_subplot(222, facecolor='black')
            ax_charge.set_title('时间荷演化 (深度学习增强)', color='white', fontsize=14)
            charges = np.array(self.history['time_charges'])
            times = range(len(charges))
            ax_charge.plot(times, charges[:, 0], 'r-', label='τ₁荷', linewidth=2.5)
            ax_charge.plot(times, charges[:, 1], 'g-', label='τ₂荷', linewidth=2.5)
            ax_charge.plot(times, charges[:, 2], 'b-', label='τ₃荷', linewidth=2.5)
            ax_charge.set_xlabel('时间步', color='white')
            ax_charge.set_ylabel('时间荷密度', color='white')
            ax_charge.tick_params(colors='white')
            ax_charge.legend()
            ax_charge.grid(True, color='0.3', alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        return fig

    def visualize_gan_results(self, iteration):
        """
        可视化GAN生成结果
        
        Args:
            iteration (int): 当前迭代次数
        """
        try:
            # 准备测试数据
            with torch.no_grad():
                self.generator.eval()
                lr_test, hr_test = self.prepare_training_data()
                fake_hr = self.generator(lr_test)
                
                # 转换为numpy数组
                lr_img = lr_test[0, 0].cpu().numpy()
                hr_img = hr_test[0, 0].cpu().numpy()
                fake_img = fake_hr[0, 0].cpu().numpy()
                
                # 创建可视化
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 低分辨率输入
                axes[0].imshow(lr_img[:, :, lr_img.shape[2]//2], cmap='viridis')
                axes[0].set_title('低分辨率输入')
                axes[0].axis('off')
                
                # 生成的高分辨率结果
                axes[1].imshow(fake_img[:, :, fake_img.shape[2]//2], cmap='viridis')
                axes[1].set_title('GAN生成结果')
                axes[1].axis('off')
                
                # 真实高分辨率（如果可用）
                axes[2].imshow(hr_img[:, :, hr_img.shape[2]//2], cmap='viridis')
                axes[2].set_title('真实高分辨率')
                axes[2].axis('off')
                
                # 保存图像
                os.makedirs('gan_results', exist_ok=True)
                plt.savefig(f'gan_results/gan_comparison_{iteration:04d}.png', 
                          bbox_inches='tight', dpi=150)
                plt.close(fig)
                
        except Exception as e:
            print(f"GAN结果可视化失败: {e}")


def main():
    """
    主程序（深度学习增强版）
    """
    print("="*90)
    print("高级六维流形时空超图计算模拟")
    print("集成深度学习算法、虫洞效应、时间荷守恒和Wolfram重写规则")
    print("基于U-Net架构思想的宇宙大尺度结构生成")
    print("="*90)
    
    # 创建高级模型
    print("\n1. 初始化深度学习增强模型...")
    model = SixDimensionalSpacetimeHypergraphAdvanced(
        n_earth=20000,   # 地球子时空节点数
        n_blackhole=80,  # 黑洞子时空节点数
        n_darkenergy=1000 # 暗能量子时空节点数
    )
    
    # 初始化高级节点
    print("\n2. 初始化深度学习增强六维时空节点...")
    model.initialize_spacetime_nodes_advanced()
    
    # 创建高级超边和虫洞
    print("\n3. 创建深度学习增强超边连接和虫洞...")
    model.create_hyperedges_and_wormholes_advanced()
    
    # 运行高级演化
    print("\n4. 开始深度学习增强演化...")
    model = evolve_advanced_hypergraph_ml(model, n_iterations=5000)
    
    # 最终可视化
    print("\n5. 生成最终深度学习增强可视化...")
    final_fig = model.create_3d_visualization_advanced()
    plt.show()
    
    # 输出最终统计
    print("\n6. 最终深度学习增强统计结果:")
    print("="*70)
    
    earth_count = sum(1 for n in model.nodes if model.nodes[n]['type'] == 'earth')
    bh_count = sum(1 for n in model.nodes if model.nodes[n]['type'] == 'blackhole')
    de_count = sum(1 for n in model.nodes if model.nodes[n]['type'] == 'darkenergy')
    
    earth_mass = sum(model.nodes[n]['mass'] for n in model.nodes
                     if model.nodes[n]['type'] == 'earth')
    bh_mass = sum(model.nodes[n]['mass'] for n in model.nodes
                  if model.nodes[n]['type'] == 'blackhole')
    de_energy = sum(model.nodes[n]['energy'] for n in model.nodes
                    if model.nodes[n]['type'] == 'darkenergy')
    
    print(f"最终节点统计:")
    print(f"  地球子时空: {earth_count} 个节点")
    print(f"  黑洞子时空: {bh_count} 个节点")
    print(f"  暗能量子时空: {de_count} 个节点")
    
    print(f"\n质量-能量分布:")
    print(f"  地球总质量: {earth_mass:.2e} kg")
    print(f"  黑洞总质量: {bh_mass:.2e} kg")
    print(f"  暗能量总能量: {de_energy:.2e} J")
    print(f"  质量比例 (黑洞/地球): {bh_mass/max(earth_mass, 1e-6):.2f}")
    
    print(f"\n虫洞效应统计:")
    print(f"  活跃虫洞数: {len(model.wormhole_connections)}")
    total_flux = sum(wh['flux_capacity'] * wh['stability'] for wh in model.wormhole_connections)
    print(f"  总虫洞通量: {total_flux:.4f}")
    
    print(f"\n时间荷守恒:")
    final_charge = model.get_total_time_charge()
    print(f"  τ₁总荷: {final_charge[0]:.4f}")
    print(f"  τ₂总荷: {final_charge[1]:.4f}")
    print(f"  τ₃总荷: {final_charge[2]:.4f}")
    
    print(f"\n深度学习参数:")
    print(f"  网格分辨率: {model.grid_resolution}³")
    print(f"  盒子大小: {model.box_size} h⁻¹Mpc")
    print(f"  学习率参数: {model.learning_rate}")
    print(f"  批量大小: {model.batch_size}")
    
    print(f"\n网络拓扑:")
    print(f"  总节点数: {len(model.nodes)}")
    print(f"  总边数: {len(model.hypergraph.edges())}")
    print(f"  总超边数: {len(model.hyperedges)}")
    
    return model

if __name__ == "__main__":
    model = main()
