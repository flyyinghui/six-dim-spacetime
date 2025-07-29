import os
import math
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
    
    def __init__(self, n_earth=50000, n_blackhole=120, n_darkenergy=8000, max_iterations=2000, record_interval=50):
        # 节点数量
        self.n_earth = n_earth
        self.n_blackhole = n_blackhole
        self.n_darkenergy = n_darkenergy
        self.n_total = n_earth + n_blackhole + n_darkenergy
        self.max_iterations = max_iterations
        self.record_interval = record_interval
        
        # 计算记录步数
        self.max_history_steps = max_iterations // record_interval + 1
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 深度学习启发的参数配置
        self.grid_resolution = 512  # 基于DarkAI的512³网格
        self.box_size = 1000.0  # h⁻¹Mpc 盒子大小
        
        # 物理参数
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
        
        # 初始化节点数据
        self._init_nodes()
        
        # 初始化历史记录缓冲区
        self._init_history_buffers()
        
        # 使用稀疏矩阵存储虫洞连接以节省内存
        # 只存储非零元素
        self.wormhole_connections = {}  # 使用字典存储非零连接
        self.wormhole_indices = []     # 存储非零连接的索引
        
        # 初始化时间步
        self.time_step = 0
        
        # 设置随机种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        print(f"模型初始化完成，使用设备: {self.device}")
        print(f"总节点数: {self.n_total} (地球: {n_earth}, 黑洞: {n_blackhole}, 暗能量: {n_darkenergy})")
    
    def _init_nodes(self):
        """初始化节点数据"""
        # 初始化位置 - 使用GPU张量
        pos_e = (torch.randn(self.n_earth, 3, device=self.device) * self.box_size * 0.4)
        pos_b = (torch.randn(self.n_blackhole, 3, device=self.device) * self.box_size * 0.3)
        pos_d = (torch.randn(self.n_darkenergy, 3, device=self.device) * self.box_size * 0.5)
        
        # 合并所有位置
        all_pos = torch.cat([pos_e, pos_b, pos_d], dim=0)
        
        # 节点类型 (0: 地球, 1: 黑洞, 2: 暗能量)
        node_types = torch.cat([
            torch.zeros(self.n_earth, device=self.device, dtype=torch.int32),
            torch.ones(self.n_blackhole, device=self.device, dtype=torch.int32),
            torch.full((self.n_darkenergy,), 2, device=self.device, dtype=torch.int32)
        ])
        
        # 节点质量
        mass = torch.cat([
            torch.rand(self.n_earth, device=self.device) * 0.9 + 0.1,  # 地球质量: 0.1-1.0
            torch.rand(self.n_blackhole, device=self.device) * 150 + 50,  # 黑洞质量: 50-200
            torch.rand(self.n_darkenergy, device=self.device) * 0.09 + 0.01  # 暗能量: 0.01-0.1
        ])
        
        # 时间荷坐标
        tau = torch.rand((self.n_total, 3), device=self.device)
        
        # 能量
        energy = torch.zeros(self.n_total, device=self.device)
        
        # kappa值 (根据节点类型)
        kappa = torch.ones(self.n_total, device=self.device) * self.kappa_0
        kappa = kappa * torch.where(
            node_types == 0, 1 + self.alpha_quantum,
            torch.where(node_types == 1, 1 + self.alpha_sphere, 1 + self.alpha_fluid)
        )
        
        # 信息量 (初始为0)
        info = torch.zeros(self.n_total, device=self.device)
        
        # 速度 (初始为0)
        velocity = torch.zeros_like(all_pos)
        
        # 存储节点数据
        self.nodes = {
            'pos': all_pos,      # 位置 (n_total, 3)
            'type': node_types,  # 类型 (n_total,)
            'mass': mass,        # 质量 (n_total,)
            'tau': tau,          # 时间荷 (n_total, 3)
            'energy': energy,    # 能量 (n_total,)
            'kappa': kappa,      # kappa值 (n_total,)
            'info': info,        # 信息量 (n_total,)
            'velocity': velocity # 速度 (n_total, 3)
        }
    
    def _init_history_buffers(self):
        """初始化历史记录缓冲区"""
        # 预分配GPU缓冲区
        self.history = {
            'positions': torch.zeros((self.max_history_steps, self.n_total, 3), 
                                   device=self.device),
            'energies': torch.zeros((self.max_history_steps, self.n_total), 
                                  device=self.device),
            'masses': torch.zeros((self.max_history_steps, 3),  # 按类型统计
                                device=self.device),
            'wormhole_flux': torch.zeros(self.max_history_steps, 
                                       device=self.device),
            'time_charges': torch.zeros((self.max_history_steps, 3), 
                                      device=self.device)
        }
        
        print(f"高级六维流形时空模型初始化完成")
        print(f"网格分辨率: {self.grid_resolution}³, 盒子大小: {self.box_size} h⁻¹Mpc")
        print(f"地球子时空: {self.n_earth}, 黑洞子时空: {self.n_blackhole}, 暗能量子时空: {self.n_darkenergy}")
    
    def _update_physics(self, nodes, dt=0.1):
        """
        更新物理状态
        """
        with torch.cuda.amp.autocast():
            # 1. 计算时间荷守恒
            time_charge = self._compute_time_charge(nodes)
            
            # 2. 计算各种力
            forces = self._compute_forces(nodes)
            
            # 3. 更新位置和速度
            nodes = self._update_positions_velocities(nodes, forces, dt)
            
            # 4. 应用虫洞效应
            nodes = self._apply_wormhole_effects(nodes)
            
            # 5. 更新能量
            nodes = self._update_energy(nodes, forces, dt)
            
            # 更新时间步
            self.time_step += 1
            
            return nodes, time_charge
    
    def _compute_time_charge(self, nodes):
        """计算时间荷"""
        mass = nodes['mass']
        tau = nodes['tau']
        return (mass.unsqueeze(-1) * tau).sum(dim=0)
    
    def _compute_forces(self, nodes, batch_size=500):
        """计算所有节点间的力 (更高效的内存使用)"""
        pos = nodes['pos']
        mass = nodes['mass']
        node_types = nodes['type']
        n_nodes = len(pos)
        
        # 初始化力
        forces = torch.zeros_like(pos)
        
        # 引力常数
        G = 6.67430e-11
        
        # 分批处理引力计算
        for i in range(0, n_nodes, batch_size):
            end_i = min(i + batch_size, n_nodes)
            current_batch_size = end_i - i
            
            # 计算当前批次的力
            batch_forces = torch.zeros((current_batch_size, 3), device=pos.device)
            
            # 计算当前批次节点到所有其他节点的力
            for j in range(0, n_nodes, batch_size):
                end_j = min(j + batch_size, n_nodes)
                
                # 计算位置差和距离
                r_ij = pos[j:end_j].unsqueeze(1) - pos[i:end_i].unsqueeze(0)  # (batch_j, batch_i, 3)
                dist = torch.norm(r_ij, dim=2, keepdim=True) + 1e-6  # 避免除零
                
                # 计算质量乘积
                mass_ij = mass[j:end_j].unsqueeze(1) * mass[i:end_i].unsqueeze(0)  # (batch_j, batch_i)
                
                # 计算引力
                f_gravity = -G * (mass_ij.unsqueeze(-1) * r_ij / (dist**3 + 1e-10))  # (batch_j, batch_i, 3)
                
                # 累加力（排除自相互作用）
                if i == j:  # 如果是同一批，需要排除对角线
                    mask = 1 - torch.eye(current_batch_size, device=pos.device).unsqueeze(-1)
                    f_gravity = f_gravity * mask
                
                batch_forces += f_gravity.sum(dim=0)
            
            forces[i:end_i] = batch_forces
        
        # 计算暗能量斥力 (仅暗能量节点施加)
        de_mask = (node_types == 2).float()  # 暗能量节点
        if de_mask.sum() > 0:  # 只在有暗能量节点时计算
            forces.addcmul_(pos, de_mask.unsqueeze(-1), value=0.1)
        
        # 清理内存
        torch.cuda.empty_cache()
        
        return forces
    
    def _update_positions_velocities(self, nodes, forces, dt):
        """更新位置和速度"""
        nodes = nodes.copy()
        
        # 更新速度 (v = v0 + a*dt)
        acceleration = forces / (nodes['mass'].unsqueeze(-1) + 1e-10)
        nodes['velocity'] += acceleration * dt
        
        # 更新位置 (x = x0 + v*dt)
        nodes['pos'] += nodes['velocity'] * dt
        
        # 应用周期性边界条件
        nodes['pos'] = nodes['pos'] % self.box_size
        
        return nodes
    
    def _apply_wormhole_effects(self, nodes):
        """应用虫洞效应"""
        # 这里简化处理，实际应根据虫洞连接矩阵计算
        # 1. 更新虫洞连接
        self._update_wormhole_connections(nodes)
        
        # 2. 应用虫洞传输
        nodes = self._apply_wormhole_transport(nodes)
        
        return nodes
    
    def _update_wormhole_connections(self, nodes, batch_size=200):
        """
        更新虫洞连接 (稀疏矩阵版本)
        只存储黑洞节点之间的连接，大大减少内存使用
        """
        pos = nodes['pos']
        node_types = nodes['type']
        n_nodes = len(pos)
        
        # 清空现有连接
        self.wormhole_connections = {}
        self.wormhole_indices = []
        
        # 获取黑洞节点索引
        bh_indices = torch.where(node_types == 1)[0].cpu().numpy()
        n_bh = len(bh_indices)
        
        if n_bh == 0:
            return  # 没有黑洞节点，无需更新
            
        # 分批处理黑洞节点对
        for i in range(n_bh):
            idx_i = bh_indices[i]
            pos_i = pos[idx_i]
            
            # 计算当前黑洞节点到其他黑洞节点的距离
            for j in range(i, n_bh):  # 从i开始避免重复计算
                idx_j = bh_indices[j]
                if idx_i == idx_j:
                    continue  # 不自连
                    
                # 计算距离
                dist = torch.norm(pos_i - pos[idx_j]).item()
                
                # 计算连接强度
                strength = math.exp(-dist / (self.box_size * 0.1))
                
                if strength > 0.01:  # 只保留足够强的连接
                    # 存储连接（双向）
                    key = (min(idx_i, idx_j), max(idx_i, idx_j))
                    self.wormhole_connections[key] = strength
                    self.wormhole_indices.append(key)
        
        # 清理内存
        torch.cuda.empty_cache()
    
    def _apply_wormhole_transport(self, nodes):
        """应用虫洞传输效应"""
        # 这里简化处理，实际应根据虫洞连接矩阵计算
        # 随机选择一些节点对进行位置交换
        if self.time_step % 100 == 0:  # 每100步执行一次
            n_swaps = min(10, self.n_total // 100)  # 交换少量节点
            if n_swaps > 1:
                idx1 = torch.randperm(self.n_total)[:n_swaps]
                idx2 = torch.randperm(self.n_total)[:n_swaps]
                
                # 交换位置
                nodes['pos'][idx1], nodes['pos'][idx2] = \
                    nodes['pos'][idx2].clone(), nodes['pos'][idx1].clone()
        
        return nodes
    
    def _update_energy(self, nodes, forces, dt):
        """更新能量"""
        nodes = nodes.copy()
        
        # 动能: 0.5 * m * v^2
        kinetic = 0.5 * nodes['mass'] * torch.norm(nodes['velocity'], dim=1)**2
        
        # 势能: -G * m1 * m2 / r (简化处理)
        potential = -torch.norm(forces, dim=1) * nodes['mass'] * 1e-10
        
        # 总能量
        nodes['energy'] = kinetic + potential
        
        return nodes
    
    def step(self):
        """执行一个时间步的模拟"""
        # 更新物理状态
        self.nodes, time_charge = self._update_physics(self.nodes)
        
        # 记录数据
        if self.time_step % self.record_interval == 0:
            self._record_step(self.time_step // self.record_interval, time_charge)
        
        self.time_step += 1
    
    def _record_step(self, step_idx, time_charge):
        """记录当前步骤的数据"""
        if step_idx >= self.max_history_steps:
            return
            
        # 记录位置
        self.history['positions'][step_idx] = self.nodes['pos'].detach()
        
        # 记录能量
        self.history['energies'][step_idx] = self.nodes['energy'].detach()
        
        # 记录质量 (按类型统计)
        for i in range(3):  # 0:地球, 1:黑洞, 2:暗能量
            mask = (self.nodes['type'] == i)
            if mask.any():
                self.history['masses'][step_idx, i] = self.nodes['mass'][mask].mean()
        
        # 记录虫洞通量 (简化处理)
        self.history['wormhole_flux'][step_idx] = self.wormhole_connections.mean()
        
        # 记录时间荷
        self.history['time_charges'][step_idx] = time_charge.detach()
        
        if step_idx % 10 == 0:
            print(f"Step {self.time_step}: "
                  f"Avg Energy = {self.nodes['energy'].mean().item():.2f}, "
                  f"Time Charge = {time_charge.tolist()}")

    def apply_unet_inspired_rescaling(self, density_field, field_type='density'):
        """
        应用基于U-Net启发的数据重缩放
        参考DarkAI论文中的重缩放方法
        """
        if field_type == 'density':
            # 对密度场应用log变换
            rescaled = np.log10(1 + density_field + self.rescale_a)
        elif field_type == 'velocity':
            # 对速度场进行线性归一化
            rescaled = density_field / self.rescale_b
        else:
            rescaled = density_field
            
        return rescaled

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
            # 黑洞：8-10个集群，其中3个大型集群占60%以上黑洞
            n_clusters = np.random.randint(8, 11)  # 8-10个集群
            cluster_centers = np.random.randn(n_clusters, 3) * effective_box_size * 0.8
            
            # 确保集群之间有一定的最小距离 (扩大5-8倍)
            min_dist = effective_box_size * np.random.uniform(5, 8)
            for i in range(1, n_clusters):
                while True:
                    new_center = np.random.randn(3) * effective_box_size * 0.3
                    distances = [np.linalg.norm(new_center - cluster_centers[j]) for j in range(i)]
                    if all(d > min_dist for d in distances):
                        cluster_centers[i] = new_center
                        break
            
            positions = []
            
            # 分配黑洞到集群，确保3个大型集群占60%以上
            cluster_weights = np.ones(n_clusters)
            large_clusters = np.random.choice(n_clusters, 3, replace=False)
            cluster_weights[large_clusters] = 3.0  # 大型集群权重更高
            cluster_weights = cluster_weights / cluster_weights.sum()
            
            # 确保大型集群至少有总黑洞数的60%
            min_large_ratio = 0.6
            large_cluster_ratio = np.random.uniform(min_large_ratio, 0.8)
            n_large = int(n_objects * large_cluster_ratio)
            
            # 分配黑洞数量到集群
            n_per_cluster = np.zeros(n_clusters, dtype=int)
            n_per_cluster[large_clusters] = np.random.multinomial(n_large, [1/3, 1/3, 1/3])
            
            # 分配剩余的黑洞
            remaining = n_objects - n_large
            if remaining > 0:
                small_clusters = [i for i in range(n_clusters) if i not in large_clusters]
                if small_clusters:  # 如果有小型集群
                    small_weights = cluster_weights[small_clusters]
                    small_weights = small_weights / small_weights.sum()
                    n_per_cluster[small_clusters] = np.random.multinomial(remaining, small_weights)
            
            for i, (center, n_in_cluster) in enumerate(zip(cluster_centers, n_per_cluster)):
                if n_in_cluster == 0:
                    continue
                    
                # 对于大型集群，使用更紧凑的分布
                cluster_scale = effective_box_size * (0.01 if i in large_clusters else 0.02)
                
                # 使用指数分布模拟深度特征
                r = np.random.exponential(scale=cluster_scale, size=n_in_cluster)
                theta = np.random.uniform(0, 2*np.pi, n_in_cluster)
                phi = np.random.uniform(0, np.pi, n_in_cluster)
                
                # 添加一些径向分布
                r = r * (1 + 0.5 * np.random.rand(n_in_cluster))
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                
                # 添加一些随机扰动
                if i in large_clusters:
                    # 大型集群添加一些子结构
                    sub_centers = center + np.random.randn(3, 3) * cluster_scale * 0.5
                    for sc in sub_centers:
                        mask = np.random.rand(n_in_cluster) < 0.3
                        if np.any(mask):
                            offset = (sc - center) * np.random.rand(np.sum(mask), 1)
                            x[mask] += offset[:, 0]
                            y[mask] += offset[:, 1]
                            z[mask] += offset[:, 2]
                
                cluster_positions = np.column_stack([x, y, z]) + center
                positions.extend(cluster_positions)
            
            # 确保返回正确数量的位置
            positions = positions[:n_objects]
            if len(positions) < n_objects:
                # 如果需要，添加一些随机位置
                n_needed = n_objects - len(positions)
                random_pos = np.random.randn(n_needed, 3) * effective_box_size * 0.1
                positions.extend(random_pos)
                
        elif object_type == 'darkenergy':
            # 暗能量：纤维网络分布，倾向于连接黑洞集群
            positions = self._generate_filament_network(n_objects, effective_box_size)
            
            # 调整暗能量分布，使其更集中在黑洞集群之间
            if hasattr(self, 'blackhole_positions') and len(self.blackhole_positions) > 0:
                # 计算每个暗能量点到最近黑洞集群的距离
                from scipy.spatial import cKDTree
                tree = cKDTree(self.blackhole_positions)
                distances, _ = tree.query(positions, k=1)
                
                # 根据距离调整位置，使其更倾向于分布在黑洞集群之间
                for i in range(len(positions)):
                    if distances[i] > effective_box_size * 0.2 and np.random.rand() < 0.3:
                        # 将部分远离黑洞的暗能量点拉向最近的黑洞
                        direction = self.blackhole_positions[tree.query(positions[i])[1]] - positions[i]
                        positions[i] += direction * np.random.uniform(0.1, 0.5)
            
        elif object_type == 'earth':
            # 地球型物质：围绕黑洞集群分布
            positions = self._generate_hierarchical_matter_distribution(n_objects, effective_box_size)
            
            # 调整地球分布，使其更集中在黑洞集群周围
            if hasattr(self, 'blackhole_positions') and len(self.blackhole_positions) > 0:
                for i in range(len(positions)):
                    # 对于每个地球位置，找到最近的黑洞集群
                    distances = [np.linalg.norm(positions[i] - bh_pos) for bh_pos in self.blackhole_positions]
                    min_dist_idx = np.argmin(distances)
                    min_dist = distances[min_dist_idx]
                    
                    if min_dist > effective_box_size * 0.1 and np.random.rand() < 0.5:
                        # 将部分远离黑洞的地球拉向最近的黑洞
                        direction = self.blackhole_positions[min_dist_idx] - positions[i]
                        positions[i] += direction * np.random.uniform(0.1, 0.3)
        
        # 确保位置在有效范围内
        positions = np.clip(positions, -effective_box_size/2, effective_box_size/2)
        
        # 如果是黑洞位置，保存供其他分布参考
        if object_type == 'blackhole' and scale_level == 0:
            self.blackhole_positions = positions[:n_objects].copy()
            
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
        计算全局时间荷总量
        """
        total = np.zeros(3)
        for nid in self.nodes:
            total += self.calculate_time_charge_density(nid)
        return total

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

    def update_physics_advanced(self):
        """
        高级物理量更新
        集成深度学习启发的特征更新
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
    演化高级超图模型（深度学习增强版）
    """
    print(f"开始演化高级六维流形时空模型（深度学习增强），共 {n_iterations} 个时间步...")
    
    # 初始化CSV记录
    csv_file = 'advanced_spacetime_evolution_ml.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iteration', 'earth_nodes', 'blackhole_nodes', 'darkenergy_nodes',
            'wormhole_count', 'total_time_charge_t1', 'total_time_charge_t2', 'total_time_charge_t3',
            'total_wormhole_flux', 'earth_total_mass', 'blackhole_total_mass', 'darkenergy_total_energy',
            'unet_features_avg', 'grid_resolution', 'learning_rate'
        ])
    
    for iteration in range(n_iterations):
        model.time_step = iteration
        
        # 更新物理量（高级版）
        model.update_physics_advanced()
        
        # 记录统计（高级版）
        model.record_evolution_statistics_advanced()
        
        # 每30步生成可视化和记录详细数据
        if iteration % 30 == 0:
            print(f"时间步 {iteration}: 生成深度学习增强可视化和记录数据")
            
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
            
            # 计算深度特征统计
            unet_feature_stats = []
            for node_id, node in model.nodes.items():
                if 'unet_features' in node:
                    unet_feature_stats.append(np.mean(np.abs(node['unet_features'])))
            avg_unet_features = np.mean(unet_feature_stats) if unet_feature_stats else 0
            
            # 写入CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    iteration, earth_count, bh_count, de_count,
                    len(model.wormhole_connections),
                    total_time_charge[0], total_time_charge[1], total_time_charge[2],
                    total_wormhole_flux, earth_mass, bh_mass, de_energy,
                    avg_unet_features, model.grid_resolution, model.learning_rate
                ])
            
            print(f"  地球子时空: {earth_count} 节点")
            print(f"  黑洞子时空: {bh_count} 节点")
            print(f"  暗能量子时空: {de_count} 节点")
            print(f"  虫洞连接: {len(model.wormhole_connections)} 个")
            print(f"  虫洞总通量: {total_wormhole_flux:.4f}")
            print(f"  深度特征强度: {avg_unet_features:.4f}")
            
            # 生成高级可视化
            try:
                fig = model.create_3d_visualization_advanced()
                os.makedirs('advanced_spacetime_frames_ml', exist_ok=True)
                plt.savefig(f'advanced_spacetime_frames_ml/frame_{iteration:04d}.png',
                            dpi=120, bbox_inches='tight', facecolor='black')
                plt.close(fig)
            except Exception as e:
                print(f"  可视化生成失败: {e}")
        
        # 每80步应用重写规则
        if iteration % 80 == 0:
            print(f"  应用深度学习增强Wolfram重写规则...")
            model.apply_wolfram_rewrite_rules_advanced()
    
    print("高级深度学习增强演化完成！")
    return model


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
        n_earth=50000,   # 地球子时空节点数
        n_blackhole=120,  # 黑洞子时空节点数
        n_darkenergy=8000 # 暗能量子时空节点数
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

def run_simulation():
    """运行模拟的主函数"""
    # 初始化模型
    model = SixDimensionalSpacetimeHypergraphAdvanced(
        n_earth=50000,
        n_blackhole=120,
        n_darkenergy=8000,
        max_iterations=2000,  # 减少迭代次数用于测试
        record_interval=50    # 每10步记录一次
    )
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 初始化混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 主模拟循环
    print("开始模拟...")
    start_time = time.time()
    
    try:
        with torch.cuda.amp.autocast():
            for step in range(model.max_iterations):
                model.step()
                
                # 定期保存检查点
                if step > 0 and step % 100 == 0:
                    model.save_checkpoint(f'output/checkpoint_step_{step}.pt')
                    print(f"已保存检查点: step {step}")
    
    except KeyboardInterrupt:
        print("\n模拟被用户中断。")
    
    # 计算总运行时间
    total_time = time.time() - start_time
    print(f"模拟完成! 总耗时: {total_time:.2f} 秒")
    
    # 保存最终结果
    model.save_results('output/final_results.pt')
    
    # 可视化结果
    model.visualize()
    
    return model

# 保存和加载方法
class ModelIO:
    @staticmethod
    def save_checkpoint(model, filename):
        """保存模型检查点"""
        checkpoint = {
            'model_state': model.nodes,
            'history': model.history,
            'time_step': model.time_step,
            'wormhole_connections': model.wormhole_connections,
            'params': {
                'n_earth': model.n_earth,
                'n_blackhole': model.n_blackhole,
                'n_darkenergy': model.n_darkenergy,
                'max_iterations': model.max_iterations,
                'record_interval': model.record_interval,
                'box_size': model.box_size
            }
        }
        torch.save(checkpoint, filename)
    
    @staticmethod
    def load_checkpoint(filename, device='cuda'):
        """加载模型检查点"""
        checkpoint = torch.load(filename, map_location=device)
        
        # 创建新模型
        model = SixDimensionalSpacetimeHypergraphAdvanced(
            n_earth=checkpoint['params']['n_earth'],
            n_blackhole=checkpoint['params']['n_blackhole'],
            n_darkenergy=checkpoint['params']['n_darkenergy'],
            max_iterations=checkpoint['params']['max_iterations'],
            record_interval=checkpoint['params']['record_interval']
        )
        
        # 恢复状态
        model.nodes = checkpoint['model_state']
        model.history = checkpoint['history']
        model.time_step = checkpoint['time_step']
        model.wormhole_connections = checkpoint['wormhole_connections']
        
        return model

# 添加保存和加载方法到主类
SixDimensionalSpacetimeHypergraphAdvanced.save_checkpoint = ModelIO.save_checkpoint.__get__(None, SixDimensionalSpacetimeHypergraphAdvanced)
SixDimensionalSpacetimeHypergraphAdvanced.load_checkpoint = classmethod(ModelIO.load_checkpoint)

def visualize_simulation(model, save_path='output/simulation_visualization.gif'):
    """可视化模拟结果"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D
        
        print("准备可视化...")
        
        # 准备数据
        positions = model.history['positions'].cpu().numpy()
        node_types = model.nodes['type'].cpu().numpy()
        
        # 创建图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置颜色映射
        colors = ['blue', 'black', 'red']  # 地球、黑洞、暗能量
        
        # 初始化散点图
        scats = []
        for i in range(3):  # 三种节点类型
            mask = (node_types == i)
            if mask.any():
                scat = ax.scatter([], [], [], c=colors[i], s=1, alpha=0.5, 
                                label=['地球', '黑洞', '暗能量'][i])
                scats.append(scat)
        
        # 设置图形属性
        ax.set_xlim(0, model.box_size)
        ax.set_ylim(0, model.box_size)
        ax.set_zlim(0, model.box_size)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('时空超图模拟')
        ax.legend()
        
        # 动画更新函数
        def update(frame):
            pos = positions[frame]
            for i, scat in enumerate(scats):
                mask = (node_types == i)
                if mask.any():
                    scat._offsets3d = (pos[mask, 0], pos[mask, 1], pos[mask, 2])
            return scats
        
        # 创建动画
        print("创建动画中...")
        ani = FuncAnimation(fig, update, frames=len(positions), 
                          interval=100, blit=False)
        
        # 保存为GIF
        print(f"保存动画到 {save_path}...")
        ani.save(save_path, writer='pillow', fps=10)
        plt.close()
        print("可视化完成!")
        
    except ImportError as e:
        print(f"可视化需要matplotlib: {e}")

# 添加可视化方法到主类
SixDimensionalSpacetimeHypergraphAdvanced.visualize = visualize_simulation

# 添加保存结果方法
def save_results(self, filename):
    """保存模拟结果"""
    results = {
        'history': {k: v.cpu() for k, v in self.history.items()},
        'final_state': {k: v.cpu() for k, v in self.nodes.items()},
        'params': {
            'n_earth': self.n_earth,
            'n_blackhole': self.n_blackhole,
            'n_darkenergy': self.n_darkenergy,
            'max_iterations': self.max_iterations,
            'record_interval': self.record_interval,
            'box_size': self.box_size
        }
    }
    torch.save(results, filename)
    print(f"结果已保存到 {filename}")

# 将保存结果方法添加到主类
SixDimensionalSpacetimeHypergraphAdvanced.save_results = save_results

if __name__ == "__main__":
    import time
    # 运行模拟
    model = run_simulation()
