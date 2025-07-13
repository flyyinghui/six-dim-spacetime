import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
import torch
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from matplotlib.animation import FuncAnimation
import random
from collections import defaultdict
import math

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)

class SixDimensionalSpacetimeHypergraph:
    """
    六维流形时空超图计算模型
    实现地球子时空、黑洞子时空和暗能量子时空的相互作用
    """
    
    def __init__(self, n_earth=200000, n_blackhole=100, n_darkenergy=4000):
        self.n_earth = n_earth
        self.n_blackhole = n_blackhole
        self.n_darkenergy = n_darkenergy
        
        # 初始化超图结构
        self.hypergraph = nx.Graph()
        self.nodes = {}
        self.hyperedges = []
        
        # 物理参数
        self.kappa_0 = 1.0  # 基准耦合常数
        self.alpha_quantum = 1e-36  # 量子修正
        self.alpha_sphere = 1e-36   # 球体修正
        self.alpha_fluid = 1.0      # 流体修正
        
        # 时间演化参数
        self.time_step = 0
        self.max_iterations = 2000
        
        # 存储演化历史
        self.history = {
            'positions': [],
            'connections': [],
            'energies': [],
            'masses': []
        }
        
        print(f"初始化完成: 地球子时空 {n_earth}, 黑洞子时空 {n_blackhole}, 暗能量子时空 {n_darkenergy}")
        
    def _generate_blackhole_positions(self, n_blackholes, scale=20.0):  # 扩大时空范围
        """生成黑洞位置，作为宇宙网中的节点 (GPU加速版本)"""
        # 生成5-6个主要星系团中心
        n_clusters = np.random.randint(5, 7)  # 5-6个星系团
        # 增加星系团中心之间的间距，扩大10倍
        cluster_centers = torch.randn(n_clusters, 3, device=device) * scale * 5.0  # 扩大10倍
        
        # 确保每个星系团有10-20个黑洞
        min_bh_per_cluster = 10
        max_bh_per_cluster = 20
        
        # 计算每个星系团的黑洞数量
        cluster_sizes = torch.randint(min_bh_per_cluster, max_bh_per_cluster + 1, (n_clusters,))
        total_bh = cluster_sizes.sum().item()
        
        # 调整黑洞数量以匹配n_blackholes
        while total_bh < n_blackholes:
            # 随机选择一个星系团增加一个黑洞
            cluster_idx = np.random.randint(0, n_clusters)
            if cluster_sizes[cluster_idx] < max_bh_per_cluster * 2:  # 允许超过最大值但有限制
                cluster_sizes[cluster_idx] += 1
                total_bh += 1
        
        # 生成黑洞位置 (GPU加速)
        positions = []
        for i in range(n_clusters):
            center = cluster_centers[i]
            cluster_size = min(cluster_sizes[i].item(), n_blackholes - len(positions))
            
            # 在星系团内生成黑洞位置 (批量生成)
            if cluster_size > 0:
                # 使用指数分布使黑洞更集中在中心
                r = torch.rand(cluster_size, device=device).exponential_() * scale * 0.15
                theta = torch.rand(cluster_size, device=device) * 2 * np.pi
                phi = torch.acos(2 * torch.rand(cluster_size, device=device) - 1)
                
                # 转换为笛卡尔坐标 (向量化操作)
                x = r * torch.sin(phi) * torch.cos(theta)
                y = r * torch.sin(phi) * torch.sin(theta)
                z = r * torch.cos(phi)
                
                # 组合坐标并加上中心偏移
                pos = torch.stack([x, y, z], dim=1) + center
                positions.extend(pos.cpu().numpy())
                
                if len(positions) >= n_blackholes:
                    break
                    
        return np.array(positions[:n_blackholes])
    
    def _generate_darkenergy_filaments(self, n_points, blackhole_positions, scale=20.0):  # 扩大时空范围
        """生成暗能量纤维，连接黑洞 (GPU加速版本)"""
        n_blackholes = len(blackhole_positions)
        if n_blackholes < 2:
            return np.array([])
        
        # 将黑洞位置转换为PyTorch张量
        bh_pos_tensor = torch.tensor(blackhole_positions, device=device)
        
        # 使用k-means聚类将黑洞分组 (GPU加速)
        n_clusters = min(6, max(2, n_blackholes // 10))  # 确保至少有2个组
        
        # 初始化聚类中心
        indices = torch.randperm(n_blackholes, device=device)[:n_clusters]
        centers = bh_pos_tensor[indices]
        
        # 简单k-means迭代 (5次)
        for _ in range(5):
            # 计算距离矩阵
            dists = torch.cdist(bh_pos_tensor, centers)
            # 分配聚类
            cluster_ids = torch.argmin(dists, dim=1)
            # 更新中心
            for i in range(n_clusters):
                mask = cluster_ids == i
                if mask.any():
                    centers[i] = bh_pos_tensor[mask].mean(dim=0)
        
            # 构建Delaunay三角剖分 (使用PyTorch实现)
            if n_clusters >= 4:  # 3D Delaunay需要至少4个点
                # 计算凸包
                hull = torch.tensor(blackhole_positions, device=device)
                # 简单实现：连接所有中心点
                edges = set()
                for i in range(n_clusters):
                    for j in range(i+1, n_clusters):
                        edges.add((i, j))
            else:
                # 少于4个点直接全连接
                edges = set()
                for i in range(n_clusters):
                    for j in range(i+1, n_clusters):
                        edges.add((i, j))
        
            cluster_centers = centers.cpu().numpy()
        
            # 生成连接黑洞的纤维
            filaments = []
        
            # 1. 首先连接聚类中心之间的纤维
            for i, j in edges:
                if i < len(cluster_centers) and j < len(cluster_centers):
                    start = cluster_centers[i]
                    end = cluster_centers[j]
                    
                    # 生成两点之间的曲线路径
                    direction = end - start
                    length = np.linalg.norm(direction)
                    if length > 1e-6:  # 避免除以零
                        direction = direction / length
                        
                        # 生成曲线路径（贝塞尔曲线）
                        t = np.linspace(0, 1, 20)
                        for k in range(len(t)-1):
                            # 添加一些随机弯曲
                            t_val = t[k]
                            # 线性插值
                            pos = (1-t_val) * start + t_val * end
                            # 添加垂直方向的随机扰动
                            if k > 0 and k < len(t)-2:  # 不在端点添加扰动
                                perpendicular = np.cross(direction, np.random.randn(3))
                                perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)
                                pos += perpendicular * np.sin(t_val * np.pi) * length * 0.2 * np.random.randn()
                            
                            filaments.append(pos)
        
            # 2. 在每个聚类内部连接黑洞
            if n_blackholes > 1:
                # 使用之前计算的cluster_ids作为聚类标签
                cluster_labels = cluster_ids.cpu().numpy()
            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                cluster_points = blackhole_positions[cluster_mask]
                
                # 构建最小生成树连接聚类内的黑洞
                if len(cluster_points) > 1:
                    from scipy.spatial.distance import pdist, squareform
                    from scipy.sparse.csgraph import minimum_spanning_tree
                    
                    # 计算距离矩阵
                    dist_matrix = squareform(pdist(cluster_points))
                    
                    # 获取最小生成树
                    mst = minimum_spanning_tree(dist_matrix)
                    mst = mst.toarray()
                    
                    # 为最小生成树中的边生成纤维
                    for i in range(len(cluster_points)):
                        for j in range(i+1, len(cluster_points)):
                            if mst[i,j] > 0:
                                start = cluster_points[i]
                                end = cluster_points[j]
                                
                                # 生成两点之间的曲线路径
                                t = np.linspace(0, 1, 10)
                                for k in range(len(t)-1):
                                    pos = (1-t[k]) * start + t[k] * end
                                    filaments.append(pos)
        
        # 如果没有足够的点，用随机点填充
        while len(filaments) < n_points:
            # 在现有纤维附近添加点
            if len(filaments) > 0:
                base = random.choice(filaments)
                # 在纤维附近添加点，形成更真实的纤维结构
                if random.random() < 0.7:  # 70%的概率在纤维附近
                    pos = base + np.random.normal(0, 0.1, 3) * scale * 0.15  # 调整纤维密度
                else:  # 30%的概率在纤维的延长方向上
                    direction = np.random.normal(0, 1, 3)
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    pos = base + direction * np.random.uniform(0, 0.5) * scale * 0.1  # 调整纤维延伸范围
            else:
                # 如果没有纤维，在黑洞附近生成
                if len(blackhole_positions) > 0:
                    base = random.choice(blackhole_positions)
                    pos = base + np.random.normal(0, 0.3, 3) * scale * 0.25  # 调整随机分布范围
                else:
                    pos = np.random.normal(0, 1, 3) * scale * 2  # 扩大随机分布范围
            
            filaments.append(pos)
        
        # 对纤维点进行轻微平滑处理
        filaments = np.array(filaments[:n_points])
        if len(filaments) > 10:
            # 使用滑动窗口平均进行平滑
            window_size = 5
            for _ in range(2):  # 平滑2次
                for i in range(window_size, len(filaments)-window_size):
                    filaments[i] = np.mean(filaments[i-window_size:i+window_size+1], axis=0)
        
        return filaments
    
    def _generate_earth_positions(self, n_points, blackhole_positions, darkenergy_positions, scale=20.0):  # 扩大时空范围
        """生成地球位置，分布在黑洞和暗能量纤维周围 (GPU加速版本)"""
        positions = []
        n_blackholes = len(blackhole_positions)
        n_darkenergy = len(darkenergy_positions)
        
        # 将黑洞和暗能量位置转换为PyTorch张量
        bh_pos_tensor = torch.tensor(blackhole_positions, device=device) if n_blackholes > 0 else None
        de_pos_tensor = torch.tensor(darkenergy_positions, device=device) if n_darkenergy > 0 else None
        
        # 1. 在黑洞周围生成点（形成星系）
        if n_blackholes > 0:
            n_galaxy = int(n_points * 0.6)  # 60%的点在黑洞周围形成星系
            # 批量生成星系点
            bh_indices = torch.randint(0, n_blackholes, (n_galaxy,), device=device)
            centers = bh_pos_tensor[bh_indices]
            
            # 使用指数分布生成径向距离
            r = torch.rand(n_galaxy, device=device).exponential_() * scale * 0.2
            # 生成立体角
            theta = torch.rand(n_galaxy, device=device) * 2 * np.pi
            phi = torch.acos(2 * torch.rand(n_galaxy, device=device) - 1)
            
            # 转换为笛卡尔坐标
            x = r * torch.sin(phi) * torch.cos(theta)
            y = r * torch.sin(phi) * torch.sin(theta)
            z = r * torch.cos(phi)
            
            # 添加盘状结构 (70%概率)
            mask = torch.rand(n_galaxy, device=device) < 0.7
            z[mask] *= 0.1
            
            # 组合坐标并加上中心偏移
            pos = torch.stack([x, y, z], dim=1) + centers
            positions.extend(pos.cpu().numpy())
        
        # 2. 在暗能量纤维附近生成点
        if n_darkenergy > 0:
            n_filament = int(n_points * 0.3)  # 30%的点在暗能量纤维附近
            for _ in range(n_filament):
                # 随机选择一个暗能量点
                de_idx = np.random.randint(0, n_darkenergy)
                base = darkenergy_positions[de_idx]
                
                # 在纤维附近生成点
                if random.random() < 0.8:  # 80%的概率在纤维附近
                    # 在垂直于纤维的方向上添加小扰动
                    tangent = np.random.normal(0, 1, 3)
                    tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                    radius = np.random.exponential(scale=0.2) * scale * 0.15  # 调整纤维周围节点分布
                    pos = base + tangent * radius
                else:  # 20%的概率在纤维的延长方向上
                    direction = np.random.normal(0, 1, 3)
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    pos = base + direction * np.random.uniform(0, 0.5) * scale * 0.1  # 调整纤维延伸范围
                
                positions.append(pos)
        
        # 3. 在宇宙空间中随机分布剩余的点（星系间介质）
        n_remaining = n_points - len(positions)
        if n_remaining > 0:
            for _ in range(n_remaining):
                # 在空间中以较低密度随机分布
                if random.random() < 0.3 and len(positions) > 0:  # 30%的概率在现有结构附近
                    base = random.choice(positions)
                    pos = base + np.random.normal(0, 1, 3) * scale * 0.25  # 调整随机分布范围
                else:  # 70%的概率完全随机
                    pos = np.random.normal(0, 1, 3) * scale * 2  # 扩大随机分布范围 * 3  # 扩大随机分布范围
                positions.append(pos)
        
        # 确保返回正确数量的点
        return np.array(positions[:n_points])
    
    def initialize_spacetime_nodes(self):
        """初始化三种子时空的节点，基于宇宙大尺度结构"""
        node_id = 0
        
        # 1. 首先生成黑洞位置（星系团中心）
        blackhole_positions = self._generate_blackhole_positions(
            self.n_blackhole, scale=10.0
        )
        
        # 2. 生成暗能量纤维，连接黑洞
        darkenergy_positions = self._generate_darkenergy_filaments(
            self.n_darkenergy, blackhole_positions, scale=10.0
        )
        
        # 3. 生成地球位置，分布在黑洞和暗能量纤维周围
        earth_positions = self._generate_earth_positions(
            self.n_earth, blackhole_positions, darkenergy_positions, scale=10.0
        )
        
        # 4. 创建黑洞节点
        blackhole_nodes = []
        for i, pos in enumerate(blackhole_positions):
            self.nodes[node_id] = {
                'type': 'blackhole',
                'position': pos,
                'time_coords': [np.random.uniform(0, 1) for _ in range(3)],  # t1, t2, t3
                'mass': np.random.uniform(50, 200) * (1 + np.exp(-np.linalg.norm(pos)/5)),
                'energy': 0.0,
                'kappa': self.kappa_0 * (1 + self.alpha_sphere),
                'info_storage': 0.0,
                'sphere_radius': abs(np.random.normal(2, 0.5)) * (1 + np.random.random())
            }
            self.hypergraph.add_node(node_id, **self.nodes[node_id])
            blackhole_nodes.append(node_id)
            node_id += 1
        
        # 5. 创建暗能量节点
        darkenergy_nodes = []
        for i, pos in enumerate(darkenergy_positions):
            self.nodes[node_id] = {
                'type': 'darkenergy',
                'position': pos,
                'time_coords': [np.random.uniform(0, 1) for _ in range(2)],  # t1, t2
                'mass': np.random.uniform(0.01, 0.1),
                'energy': np.random.uniform(0.5, 1.0),  # 暗能量有较高能量
                'kappa': self.kappa_0 * (1 + self.alpha_fluid),
                'fluid_velocity': np.random.uniform(0.5, 0.99),  # 接近光速
                'control_info': np.random.uniform(0, 1)  # 添加控制信息初始化
            }
            self.hypergraph.add_node(node_id, **self.nodes[node_id])
            darkenergy_nodes.append(node_id)
            node_id += 1
        
        # 6. 创建地球节点
        for i, pos in enumerate(earth_positions):
            # 计算到最近的黑洞的距离
            if len(blackhole_positions) > 0:
                distances = [np.linalg.norm(pos - bh_pos) for bh_pos in blackhole_positions]
                min_dist = min(distances)
            else:
                min_dist = 1.0
                
            self.nodes[node_id] = {
                'type': 'earth',
                'position': pos,
                'time_coords': [np.random.uniform(0, 1)],  # t1
                'mass': np.random.uniform(0.1, 1.0) * (1 + np.exp(-min_dist/5)),
                'energy': 0.0,
                'kappa': self.kappa_0 * (1 + self.alpha_quantum)
            }
            self.hypergraph.add_node(node_id, **self.nodes[node_id])
            node_id += 1
            
        # 黑洞子时空节点 (1+3维) - 集中在纤维交汇处
        for i in range(self.n_blackhole):
            pos = blackhole_positions[i]
            sphere_radius = abs(np.random.normal(2, 0.5)) * (1 + np.random.random())  # 更大的变化范围
            
            self.nodes[node_id] = {
                'type': 'blackhole',
                'position': pos,
                'sphere_radius': sphere_radius,
                'time_coords': [np.random.uniform(0, 1) for _ in range(3)],  # t1, t2, t3
                'mass': np.random.uniform(50, 200) * (1 + np.exp(-np.linalg.norm(pos)/5)),  # 质量随距离增加而增大
                'energy': 0.0,
                'kappa': self.kappa_0 * (1 + self.alpha_sphere),
                'info_storage': 0.0
            }
            self.hypergraph.add_node(node_id, **self.nodes[node_id])
            node_id += 1
            
        # 暗能量子时空节点 (2+2维) - 填充在纤维之间的空隙
        for i in range(self.n_darkenergy):
            pos = darkenergy_positions[i]
            fluid_velocity = np.random.uniform(0.5, 0.99)  # 接近光速
            
            self.nodes[node_id] = {
                'type': 'darkenergy',
                'position': pos,
                'fluid_velocity': fluid_velocity,
                'time_coords': [np.random.uniform(0, 1) for _ in range(2)],  # t1, t2
                'mass': np.random.uniform(0.01, 0.1),
                'energy': 0.0,
                'kappa': self.kappa_0 * (1 + self.alpha_fluid),
                'control_info': np.random.uniform(0, 1)
            }
            self.hypergraph.add_node(node_id, **self.nodes[node_id])
            node_id += 1
            
        print(f"节点初始化完成，总计 {len(self.nodes)} 个节点")
        
    def create_hyperedges(self):
        """创建超边连接，实现子时空间的相互作用"""
        
        # 引力更新超边 (黑洞->地球)
        bh_nodes = [n for n, d in self.nodes.items() if d['type'] == 'blackhole']
        earth_nodes = [n for n, d in self.nodes.items() if d['type'] == 'earth']
        
        for _ in range(10):  # 10个因果超边
            bh_node = random.choice(bh_nodes)
            earth_group = random.sample(earth_nodes, min(5, len(earth_nodes)))
            
            hyperedge = {
                'type': 'gravity_update',
                'nodes': [bh_node] + earth_group,
                'strength': np.random.uniform(0.1, 1.0)
            }
            self.hyperedges.append(hyperedge)
            
            # 在图中添加连接
            for earth_node in earth_group:
                self.hypergraph.add_edge(bh_node, earth_node, 
                                       edge_type='gravity', 
                                       weight=hyperedge['strength'])
        
        # 量子纠缠超边 (黑洞<->暗能量)
        de_nodes = [n for n, d in self.nodes.items() if d['type'] == 'darkenergy']
        
        for _ in range(10):  # 10个纠缠超边
            bh_node = random.choice(bh_nodes)
            de_group = random.sample(de_nodes, min(3, len(de_nodes)))
            
            hyperedge = {
                'type': 'quantum_entanglement',
                'nodes': [bh_node] + de_group,
                'strength': np.random.uniform(0.5, 1.0)
            }
            self.hyperedges.append(hyperedge)
            
            # 在图中添加连接
            for de_node in de_group:
                self.hypergraph.add_edge(bh_node, de_node, 
                                       edge_type='entanglement', 
                                       weight=hyperedge['strength'])
        
        # 粒子控制超边 (暗能量->地球)
        for _ in range(20):  # 20个控制超边
            de_node = random.choice(de_nodes)
            earth_group = random.sample(earth_nodes, min(10, len(earth_nodes)))
            
            hyperedge = {
                'type': 'particle_control',
                'nodes': [de_node] + earth_group,
                'strength': np.random.uniform(0.2, 0.8)
            }
            self.hyperedges.append(hyperedge)
            
            # 在图中添加连接（较弱的连接）
            for earth_node in earth_group:
                if not self.hypergraph.has_edge(de_node, earth_node):
                    self.hypergraph.add_edge(de_node, earth_node, 
                                           edge_type='control', 
                                           weight=hyperedge['strength'])
        
        print(f"超边创建完成，总计 {len(self.hyperedges)} 个超边")

    def update_physics(self):
        """更新物理量：引力、量子纠缠、粒子控制"""
        
        # 将节点数据转换为PyTorch张量并移到GPU
        node_ids = list(self.nodes.keys())
        node_types = [self.nodes[n]['type'] for n in node_ids]
        positions = torch.tensor([self.nodes[n]['position'] for n in node_ids], 
                               device=device, dtype=torch.float32)
        masses = torch.tensor([self.nodes[n]['mass'] for n in node_ids],
                            device=device, dtype=torch.float32)
        energies = torch.tensor([self.nodes[n]['energy'] for n in node_ids],
                              device=device, dtype=torch.float32)
        
        # 计算总质量-能量 (GPU加速)
        bh_mask = torch.tensor([t == 'blackhole' for t in node_types], device=device)
        earth_mask = torch.tensor([t == 'earth' for t in node_types], device=device)
        de_mask = torch.tensor([t == 'darkenergy' for t in node_types], device=device)
        
        total_mass_bh = torch.sum(masses * bh_mask.float()).item()
        total_mass_earth = torch.sum(masses * earth_mask.float()).item()
        total_energy_de = torch.sum(energies * de_mask.float()).item()
        
        # 更新引力效应
        for hyperedge in self.hyperedges:
            if hyperedge['type'] == 'gravity_update':
                bh_node = hyperedge['nodes'][0]
                earth_nodes = hyperedge['nodes'][1:]
                
                # 黑洞向地球传递引力信息
                bh_mass = self.nodes[bh_node]['mass']
                for earth_node in earth_nodes:
                    # 计算距离
                    bh_pos = self.nodes[bh_node]['position']
                    earth_pos = self.nodes[earth_node]['position']
                    distance = np.linalg.norm(bh_pos - earth_pos) + 1e-6
                    
                    # 引力修正
                    gravity_effect = hyperedge['strength'] * bh_mass / (distance**2)
                    self.nodes[earth_node]['energy'] += gravity_effect * 0.1
                    
                    # 更新位置（微小的引力拖拽）
                    direction = (bh_pos - earth_pos) / distance
                    self.nodes[earth_node]['position'] += direction * gravity_effect * 0.001
        
        # 更新量子纠缠效应
        for hyperedge in self.hyperedges:
            if hyperedge['type'] == 'quantum_entanglement':
                bh_node = hyperedge['nodes'][0]
                de_nodes = hyperedge['nodes'][1:]
                
                # 信息同步
                bh_info = self.nodes[bh_node].get('info_storage', 0)
                for de_node in de_nodes:
                    de_info = self.nodes[de_node]['control_info']
                    
                    # 量子纠缠信息交换
                    exchange_rate = hyperedge['strength'] * 0.1
                    self.nodes[bh_node]['info_storage'] = (bh_info + de_info * exchange_rate) / 2
                    self.nodes[de_node]['control_info'] = (de_info + bh_info * exchange_rate) / 2
        
        # 更新粒子控制效应
        for hyperedge in self.hyperedges:
            if hyperedge['type'] == 'particle_control':
                de_node = hyperedge['nodes'][0]
                earth_nodes = hyperedge['nodes'][1:]
                
                # 暗能量控制地球粒子类型
                control_strength = self.nodes[de_node]['control_info']
                for earth_node in earth_nodes:
                    # 粒子质量修正
                    mass_correction = hyperedge['strength'] * control_strength * 0.01
                    self.nodes[earth_node]['mass'] *= (1 + mass_correction - 0.005)
                    
                    # 确保质量不为负
                    self.nodes[earth_node]['mass'] = max(0.01, self.nodes[earth_node]['mass'])

        # 黑洞吸收地球转换模型
        bh_nodes = [n for n, d in self.nodes.items() if d['type'] == 'blackhole']
        earth_nodes = [n for n, d in self.nodes.items() if d['type'] == 'earth']
        
        for bh_node in bh_nodes:
            bh_pos = self.nodes[bh_node]['position']
            bh_radius = self.nodes[bh_node]['sphere_radius']
            
            # 查找在黑洞影响范围内的地球节点
            for earth_node in earth_nodes[:]:  # 创建副本以便修改
                if earth_node not in self.nodes:  # 可能已被其他黑洞吸收
                    continue
                    
                earth_pos = self.nodes[earth_node]['position']
                distance = np.linalg.norm(bh_pos - earth_pos)
                
                # 如果在黑洞影响范围内
                if distance < bh_radius * 2:  # 2倍半径作为影响范围
                    earth_mass = self.nodes[earth_node]['mass']
                    
                    # 70%质量转为黑洞质量
                    bh_mass_gain = earth_mass * 0.7
                    self.nodes[bh_node]['mass'] += bh_mass_gain
                    
                    # 30%质量转为附近暗能量
                    de_mass_gain = earth_mass * 0.3
                    
                    # 找到最近的暗能量节点
                    closest_de = None
                    min_de_distance = float('inf')
                    
                    for node_id, node in self.nodes.items():
                        if node['type'] == 'darkenergy':
                            de_distance = np.linalg.norm(earth_pos - node['position'])
                            if de_distance < min_de_distance:
                                min_de_distance = de_distance
                                closest_de = node_id
                    
                    if closest_de:
                        self.nodes[closest_de]['energy'] += de_mass_gain
                    
                    # 从超图中移除被吸收的地球节点
                    self.hypergraph.remove_node(earth_node)
                    del self.nodes[earth_node]
                    earth_nodes.remove(earth_node)
                    
                    # 更新超边连接
                    for hyperedge in self.hyperedges[:]:
                        if earth_node in hyperedge['nodes']:
                            self.hyperedges.remove(hyperedge)
        
        # 更新时空结构
        for node_id, node in self.nodes.items():
            # 计算合力
            total_force = np.zeros(3)
            
            # 黑洞引力影响
            if node['type'] != 'blackhole':  # 黑洞位置固定
                for bh_node in [n for n, d in self.nodes.items() if d['type'] == 'blackhole']:
                    bh_pos = self.nodes[bh_node]['position']
                    distance = np.linalg.norm(node['position'] - bh_pos) + 1e-6
                    direction = (bh_pos - node['position']) / distance
                    
                    # 引力大小与黑洞质量成正比，与距离平方成反比
                    gravity = 0.1 * self.nodes[bh_node]['mass'] / (distance**2)
                    total_force += direction * gravity
            
            # 暗能量膨胀力
            if node['type'] == 'earth':  # 主要影响地球
                for de_node in [n for n, d in self.nodes.items() if d['type'] == 'darkenergy']:
                    de_pos = self.nodes[de_node]['position']
                    distance = np.linalg.norm(node['position'] - de_pos) + 1e-6
                    direction = (node['position'] - de_pos) / distance
                    
                    # 膨胀力与暗能量强度成正比
                    expansion = 0.01 * self.nodes[de_node]['energy'] * distance
                    total_force += direction * expansion
            
            # 更新位置 (考虑惯性)
            if node['type'] != 'blackhole':  # 黑洞位置固定
                node['position'] += total_force * 0.01  # 小步长确保稳定
                
                # 限制位置范围防止数值爆炸
                node['position'] = np.clip(node['position'], -100, 100)
        
        # 更新有效引力常数
        for node_id, node in self.nodes.items():
            if node['type'] == 'earth':
                # 动态引力常数
                neighbors = list(self.hypergraph.neighbors(node_id))
                bh_influence = sum(1 for n in neighbors if self.nodes[n]['type'] == 'blackhole')
                de_influence = sum(1 for n in neighbors if self.nodes[n]['type'] == 'darkenergy')
                
                # 基线1.0, 黑洞增强至1.2, 暗能量调节至1.1
                kappa_multiplier = 1.0 + bh_influence * 0.02 + de_influence * 0.01
                node['kappa'] = self.kappa_0 * kappa_multiplier
        
        # 记录统计信息
        self.history['masses'].append({
            'blackhole_total': total_mass_bh,
            'earth_total': total_mass_earth,
            'darkenergy_total': total_energy_de,
            'mass_ratio': total_mass_bh / max(total_mass_earth, 1e-6)
        })

def apply_rewrite_rules(hypergraph_model):
    """应用Wolfram超图重写规则"""
    
    # 地球子时空重写规则：{{x,y},{x,z},{t1}} → {{x,y},{x,z},{y,z},{t1}}
    earth_nodes = [n for n, d in hypergraph_model.nodes.items() if d['type'] == 'earth']
    
    for _ in range(min(10, len(earth_nodes)//100)):
        if len(earth_nodes) >= 3:
            selected_nodes = random.sample(earth_nodes, 3)
            # 创建三角形连接
            for i in range(3):
                for j in range(i+1, 3):
                    if not hypergraph_model.hypergraph.has_edge(selected_nodes[i], selected_nodes[j]):
                        hypergraph_model.hypergraph.add_edge(selected_nodes[i], selected_nodes[j], 
                                                           edge_type='spacetime', weight=0.1)
    
    # 黑洞子时空重写规则：{{x,y},{z,t1}} → {{sphere_compact},{z,t1},{info_storage}}
    bh_nodes = [n for n, d in hypergraph_model.nodes.items() if d['type'] == 'blackhole']
    
    for bh_node in bh_nodes:
        # 更新球体半径
        t_coords = hypergraph_model.nodes[bh_node]['time_coords']
        new_radius = np.sqrt(sum(t**2 for t in t_coords))
        hypergraph_model.nodes[bh_node]['sphere_radius'] = new_radius
        
        # 更新信息存储
        hypergraph_model.nodes[bh_node]['info_storage'] += 0.1
    
    # 暗能量子时空重写规则：{{x,y},{t2,t3}} → {{x,y},{t2,t3},{expansion_driver}}
    de_nodes = [n for n, d in hypergraph_model.nodes.items() if d['type'] == 'darkenergy']
    
    for de_node in de_nodes:
        # 更新膨胀驱动效应
        t_coords = hypergraph_model.nodes[de_node]['time_coords']
        expansion_factor = 1.0 + 0.01 * np.sqrt(sum(t**2 for t in t_coords))
        
        # 轻微膨胀暗能量节点的位置
        hypergraph_model.nodes[de_node]['position'] *= expansion_factor
        
        # 更新控制信息
        hypergraph_model.nodes[de_node]['control_info'] += 0.05

def evolve_hypergraph(hypergraph_model, n_iterations=2000):
    """演化超图模型"""
    print(f"开始演化超图，共 {n_iterations} 个时间步...")
    
    # 初始化CSV文件，写入表头
    import csv
    csv_file = 'subspace_statistics.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'earth_nodes', 'blackhole_nodes', 'darkenergy_nodes',
                        'earth_hyperedges', 'blackhole_hyperedges', 'darkenergy_hyperedges',
                        'earth_energy', 'blackhole_energy', 'darkenergy_energy'])
    
    for iteration in range(n_iterations):
        # 更新物理量
        hypergraph_model.update_physics()
        
        # 记录当前状态
        positions = {}
        connections = []
        energies = {}
        
        for node_id, node in hypergraph_model.nodes.items():
            positions[node_id] = node['position'].copy()
            energies[node_id] = node['energy']
        
        # 记录连接信息
        for edge in hypergraph_model.hypergraph.edges():
            connections.append(edge)
        
        # 每50步记录一次详细状态并生成可视化
        if iteration % 50 == 0:
            hypergraph_model.history['positions'].append(positions)
            hypergraph_model.history['connections'].append(connections)
            hypergraph_model.history['energies'].append(energies)
            
            print(f"时间步 {iteration}: 记录状态并生成可视化")
            
            # 统计各子时空的节点数量、超边数量和能量值
            earth_nodes = sum(1 for n in hypergraph_model.nodes if hypergraph_model.nodes[n]['type'] == 'earth')
            blackhole_nodes = sum(1 for n in hypergraph_model.nodes if hypergraph_model.nodes[n]['type'] == 'blackhole')
            darkenergy_nodes = sum(1 for n in hypergraph_model.nodes if hypergraph_model.nodes[n]['type'] == 'darkenergy')
            
            # 统计各子时空的超边数量（连接中至少有一个节点属于该子时空）
            earth_hyperedges = 0
            blackhole_hyperedges = 0
            darkenergy_hyperedges = 0
            
            for edge in hypergraph_model.hypergraph.edges():
                edge_types = [hypergraph_model.nodes[n]['type'] for n in edge]
                if 'earth' in edge_types:
                    earth_hyperedges += 1
                if 'blackhole' in edge_types:
                    blackhole_hyperedges += 1
                if 'darkenergy' in edge_types:
                    darkenergy_hyperedges += 1
            
            # 计算各子时空的总能量
            earth_energy = sum(hypergraph_model.nodes[n]['energy'] for n in hypergraph_model.nodes 
                             if hypergraph_model.nodes[n]['type'] == 'earth')
            blackhole_energy = sum(hypergraph_model.nodes[n]['energy'] for n in hypergraph_model.nodes 
                                 if hypergraph_model.nodes[n]['type'] == 'blackhole')
            darkenergy_energy = sum(hypergraph_model.nodes[n]['energy'] for n in hypergraph_model.nodes 
                                  if hypergraph_model.nodes[n]['type'] == 'darkenergy')
            
            # 将统计信息写入CSV文件
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    iteration,
                    earth_nodes,
                    blackhole_nodes,
                    darkenergy_nodes,
                    earth_hyperedges,
                    blackhole_hyperedges,
                    darkenergy_hyperedges,
                    earth_energy,
                    blackhole_energy,
                    darkenergy_energy
                ])
            
            print(f"  地球子时空: {earth_nodes} 节点, {earth_hyperedges} 超边, 能量: {earth_energy:.2f}")
            print(f"  黑洞子时空: {blackhole_nodes} 节点, {blackhole_hyperedges} 超边, 能量: {blackhole_energy:.2f}")
            print(f"  暗能量子时空: {darkenergy_nodes} 节点, {darkenergy_hyperedges} 超边, 能量: {darkenergy_energy:.2f}")
            
            # 计算当前统计量
            total_mass_bh = sum(hypergraph_model.nodes[n]['mass'] for n in hypergraph_model.nodes 
                           if hypergraph_model.nodes[n]['type'] == 'blackhole')
            total_mass_earth = sum(hypergraph_model.nodes[n]['mass'] for n in hypergraph_model.nodes 
                              if hypergraph_model.nodes[n]['type'] == 'earth')
            total_energy_de = sum(hypergraph_model.nodes[n]['energy'] for n in hypergraph_model.nodes 
                             if hypergraph_model.nodes[n]['type'] == 'darkenergy')
            mass_ratio = total_mass_bh / max(total_mass_earth, 1e-6)
            
            # 创建3D可视化图表
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(15, 10), facecolor='black')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            ax = fig.add_subplot(111, projection='3d', facecolor='black')
            
            # 设置背景为深空色
            ax.set_facecolor('black')
            
            # 设置坐标轴颜色为浅灰色
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('0.8')
            ax.yaxis.pane.set_edgecolor('0.8')
            ax.zaxis.pane.set_edgecolor('0.8')
            
            # 设置坐标轴标签颜色
            ax.tick_params(axis='x', colors='0.8')
            ax.tick_params(axis='y', colors='0.8')
            ax.tick_params(axis='z', colors='0.8')
            
            # 设置坐标轴标签
            ax.set_xlabel('X', color='0.8')
            ax.set_ylabel('Y', color='0.8')
            ax.set_zlabel('Z', color='0.8')
            
            # 分离不同类型的节点
            earth_pos = []
            bh_pos = []
            de_pos = []
            
            for node_id, pos in positions.items():
                if hypergraph_model.nodes[node_id]['type'] == 'earth':
                    earth_pos.append(pos)
                elif hypergraph_model.nodes[node_id]['type'] == 'blackhole':
                    bh_pos.append(pos)
                elif hypergraph_model.nodes[node_id]['type'] == 'darkenergy':
                    de_pos.append(pos)
            
            # 转换为numpy数组
            earth_pos = np.array(earth_pos)
            bh_pos = np.array(bh_pos)
            de_pos = np.array(de_pos)
            
            # 绘制暗能量纤维（先绘制，使其在底层）
            if len(de_pos) > 0:
                # 对暗能量纤维使用半透明的青色，使其更柔和
                ax.scatter(de_pos[:, 0], de_pos[:, 1], de_pos[:, 2], 
                          c='cyan', s=8, alpha=0.2, label=f'暗能量纤维 ({len(de_pos)})')
            
            # 绘制地球节点（中间层）
            if len(earth_pos) > 0:
                # 对地球节点进行下采样以提高性能
                sample_size = min(2000, len(earth_pos))  # 增加采样数以获得更好的视觉效果
                indices = np.random.choice(len(earth_pos), sample_size, replace=False)
                earth_sample = earth_pos[indices]
                
                # 使用渐变色表示地球节点
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, sample_size))
                np.random.shuffle(colors)
                
                ax.scatter(earth_sample[:, 0], earth_sample[:, 1], earth_sample[:, 2], 
                          c=colors, s=2, alpha=0.6, label=f'地球子时空 ({len(earth_pos)})')
            
            # 绘制黑洞（最后绘制，使其在最上层）
            if len(bh_pos) > 0:
                # 为黑洞添加发光效果
                ax.scatter(bh_pos[:, 0], bh_pos[:, 1], bh_pos[:, 2], 
                          c='yellow', s=150, alpha=1.0, label=f'黑洞子时空 ({len(bh_pos)})',
                          edgecolors='orange', linewidths=1.5)
                
                # 添加黑洞光晕效果
                ax.scatter(bh_pos[:, 0], bh_pos[:, 1], bh_pos[:, 2], 
                          c='yellow', s=300, alpha=0.3, edgecolors='none')
                
                # 添加坐标轴范围，确保所有元素可见
                all_pos = np.vstack([bh_pos, earth_pos, de_pos]) if len(earth_pos) > 0 and len(de_pos) > 0 else bh_pos
                max_extent = max(np.ptp(all_pos, axis=0)) * 0.6  # 使用60%的范围
                mid_x, mid_y, mid_z = np.mean(all_pos, axis=0)
                ax.set_xlim(mid_x - max_extent, mid_x + max_extent)
                ax.set_ylim(mid_y - max_extent, mid_y + max_extent)
                ax.set_zlim(mid_z - max_extent, mid_z + max_extent)
            
            # 添加半透明背景的统计信息文本
            stats_text = (f'时间步: {iteration}\n'
                         f'黑洞总质量: {total_mass_bh:.2e}\n'
                         f'地球总质量: {total_mass_earth:.2e}\n'
                         f'暗能量总能量: {total_energy_de:.2e}\n'
                         f'质量比例: {mass_ratio:.2f}')
            
            # 使用半透明黑色背景的文本框
            ax.text2D(0.05, 0.95, stats_text, transform=ax.transAxes,
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='0.8'),
                     color='white', fontsize=9, verticalalignment='top')
            
            # 设置标题和图例
            ax.set_title('六维流形时空超图3D可视化 (时间步 {})'.format(iteration), 
                        color='white', pad=20)
            
            # 添加图例并设置样式
            legend = ax.legend(loc='upper right', fontsize=9)
            legend.get_frame().set_alpha(0.7)
            legend.get_frame().set_facecolor('black')
            legend.get_frame().set_edgecolor('0.8')
            
            # 设置图例文本颜色
            for text in legend.get_texts():
                text.set_color('white')
            
            # 调整布局，确保所有元素都可见
            plt.tight_layout()
            
            # 添加网格（浅灰色，半透明）
            ax.grid(True, color='0.3', alpha=0.3, linestyle='--')
            
            # 保存图像
            os.makedirs('cosmic_expansion_frames/X1', exist_ok=True)
            plt.savefig(f'cosmic_expansion_frames/X1/step_{iteration:04d}.png')
            plt.close()
        
        # 应用Wolfram式重写规则
        if iteration % 100 == 0:
            apply_rewrite_rules(hypergraph_model)
        
        hypergraph_model.time_step = iteration
    
    print("演化完成！")
    return hypergraph_model

def create_3d_visualization(hypergraph_model):
    """创建3D可视化"""
    print("创建3D可视化...")
    
    # 设置深色主题
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # 设置背景为深空色
    ax.set_facecolor('black')
    
    # 设置坐标轴颜色为浅灰色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('0.8')
    ax.yaxis.pane.set_edgecolor('0.8')
    ax.zaxis.pane.set_edgecolor('0.8')
    
    # 设置坐标轴标签颜色
    ax.tick_params(axis='x', colors='0.8')
    ax.tick_params(axis='y', colors='0.8')
    ax.tick_params(axis='z', colors='0.8')
    
    # 设置坐标轴标签
    ax.set_xlabel('X', color='0.8')
    ax.set_ylabel('Y', color='0.8')
    ax.set_zlabel('Z', color='0.8')
    
    # 获取最后一个时间步的位置
    if hypergraph_model.history['positions']:
        positions = hypergraph_model.history['positions'][-1]
    else:
        positions = {node_id: node['position'] for node_id, node in hypergraph_model.nodes.items()}
    
    # 分离不同类型的节点
    earth_pos = []
    bh_pos = []
    de_pos = []
    
    for node_id, pos in positions.items():
        if hypergraph_model.nodes[node_id]['type'] == 'earth':
            earth_pos.append(pos)
        elif hypergraph_model.nodes[node_id]['type'] == 'blackhole':
            bh_pos.append(pos)
        elif hypergraph_model.nodes[node_id]['type'] == 'darkenergy':
            de_pos.append(pos)
    
    # 转换为numpy数组
    earth_pos = np.array(earth_pos)
    bh_pos = np.array(bh_pos)
    de_pos = np.array(de_pos)
    
    # 绘制节点
    if len(earth_pos) > 0:
        # 对地球节点进行下采样以提高性能
        sample_size = min(1000, len(earth_pos))
        indices = np.random.choice(len(earth_pos), sample_size, replace=False)
        earth_sample = earth_pos[indices]
        
        ax.scatter(earth_sample[:, 0], earth_sample[:, 1], earth_sample[:, 2], 
                  c='lightgreen', s=1, alpha=0.6, label=f'地球子时空 ({len(earth_pos)})')
    
    if len(bh_pos) > 0:
        ax.scatter(bh_pos[:, 0], bh_pos[:, 1], bh_pos[:, 2], 
                  c='red', s=100, alpha=0.8, label=f'黑洞子时空 ({len(bh_pos)})')
    
    if len(de_pos) > 0:
        # 对暗能量纤维使用淡蓝色
        ax.scatter(de_pos[:, 0], de_pos[:, 1], de_pos[:, 2], 
                  c='lightblue', s=10, alpha=0.5, label=f'暗能量纤维 ({len(de_pos)})')
    
    # 绘制连接（选择性绘制以提高性能）
    connections = hypergraph_model.history['connections'][-1] if hypergraph_model.history['connections'] else []
    
    # 只绘制部分连接以提高性能
    connection_sample = random.sample(connections, min(200, len(connections)))
    
    for edge in connection_sample:
        node1, node2 = edge
        if node1 in positions and node2 in positions:
            pos1 = positions[node1]
            pos2 = positions[node2]
            
            # 根据连接类型设置颜色和透明度
            type1 = hypergraph_model.nodes[node1]['type']
            type2 = hypergraph_model.nodes[node2]['type']
            
            if type1 == 'blackhole' or type2 == 'blackhole':
                # 黑洞连接的线条更粗，更明显
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                       'yellow', alpha=0.4, linewidth=1.0, zorder=1)
            else:
                # 其他连接使用半透明蓝色
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                       'cyan', alpha=0.2, linewidth=0.5, zorder=0)
    
    # 设置标题和图例
    ax.set_title('六维流形时空超图3D可视化\n（Wolfram超图计算模型）', 
                color='white', pad=20)
    
    # 添加图例并设置样式
    legend = ax.legend(loc='upper right', fontsize=9)
    legend.get_frame().set_alpha(0.7)
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('0.8')
    
    # 设置图例文本颜色
    for text in legend.get_texts():
        text.set_color('white')
    
    # 调整视角以获得更好的3D效果
    ax.view_init(elev=30, azim=45)
    
    # 添加网格（浅灰色，半透明）
    ax.grid(True, color='0.3', alpha=0.3, linestyle='--')
    
    # 调整布局，确保所有元素都可见
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    return fig

def plot_evolution_statistics(hypergraph_model):
    """绘制演化统计图"""
    print("绘制演化统计...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    if not hypergraph_model.history['masses']:
        print("没有足够的历史数据用于统计")
        return
    
    masses = hypergraph_model.history['masses']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 质量演化
    times = range(len(masses))
    bh_masses = [m['blackhole_total'] for m in masses]
    earth_masses = [m['earth_total'] for m in masses]
    
    axes[0, 0].plot(times, bh_masses, 'r-', label='黑洞总质量')
    axes[0, 0].plot(times, earth_masses, 'b-', label='地球总质量')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('质量')
    axes[0, 0].set_title('质量演化')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 质量比例
    mass_ratios = [m['mass_ratio'] for m in masses]
    axes[0, 1].plot(times, mass_ratios, 'g-')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('质量比例')
    axes[0, 1].set_title('黑洞/地球质量比例')
    axes[0, 1].grid(True)
    
    # 最终质量分布
    final_masses = masses[-1]
    categories = ['黑洞', '地球', '暗能量']
    values = [final_masses['blackhole_total'], 
              final_masses['earth_total'],
              final_masses['darkenergy_total']]
    
    axes[1, 0].bar(categories, values, color=['red', 'blue', 'purple'])
    axes[1, 0].set_ylabel('总质量/能量')
    axes[1, 0].set_title('最终质量-能量分布')
    
    # 显示数值
    for i, v in enumerate(values):
        axes[1, 0].text(i, v, f'{v:.2e}', ha='center', va='bottom')
    
    # 网络统计
    n_nodes = len(hypergraph_model.nodes)
    n_edges = len(hypergraph_model.hypergraph.edges())
    n_hyperedges = len(hypergraph_model.hyperedges)
    
    stats = ['节点数', '边数', '超边数']
    counts = [n_nodes, n_edges, n_hyperedges]
    
    axes[1, 1].bar(stats, counts, color=['orange', 'green', 'cyan'])
    axes[1, 1].set_ylabel('数量')
    axes[1, 1].set_title('网络结构统计')
    
    # 显示数值
    for i, v in enumerate(counts):
        axes[1, 1].text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """主程序：运行六维流形时空超图计算模拟"""
    print("="*60)
    print("六维流形时空超图计算模拟")
    print("基于Wolfram超图计算模型")
    print("="*60)
    
    # 创建超图模型
    print("\n1. 初始化超图模型...")
    model = SixDimensionalSpacetimeHypergraph(
        n_earth=200000,      # 地球子时空节点数
        n_blackhole=100,    # 黑洞子时空节点数
        n_darkenergy=4000   # 暗能量子时空节点数
    )
    
    # 初始化节点
    print("\n2. 初始化时空节点...")
    model.initialize_spacetime_nodes()
    
    # 创建超边
    print("\n3. 创建超边连接...")
    model.create_hyperedges()
    
    # 运行演化
    print("\n4. 开始超图演化...")
    model = evolve_hypergraph(model, n_iterations=2000)
    
    # 创建3D可视化
    print("\n5. 创建3D可视化...")
    fig_3d = create_3d_visualization(model)
    
    # 创建演化统计图
    print("\n6. 创建演化统计图...")
    fig_stats = plot_evolution_statistics(model)
    
    # 输出最终统计
    print("\n7. 最终统计结果:")
    print("="*40)
    
    # 计算最终质量分布
    total_mass_bh = sum(model.nodes[n]['mass'] for n in model.nodes 
                       if model.nodes[n]['type'] == 'blackhole')
    total_mass_earth = sum(model.nodes[n]['mass'] for n in model.nodes 
                          if model.nodes[n]['type'] == 'earth')
    total_energy_de = sum(model.nodes[n]['energy'] for n in model.nodes 
                         if model.nodes[n]['type'] == 'darkenergy')
    
    print(f"黑洞总质量: {total_mass_bh:.2e} kg")
    print(f"地球型物质总质量: {total_mass_earth:.2e} kg")
    print(f"暗能量总能量: {total_energy_de:.2e} J")
    print(f"质量比例 (黑洞/地球型): {total_mass_bh/max(total_mass_earth, 1e-6):.1f}")
    
    # 网络统计
    print(f"\n网络结构统计:")
    print(f"总节点数: {len(model.nodes)}")
    print(f"总边数: {len(model.hypergraph.edges())}")
    print(f"总超边数: {len(model.hyperedges)}")
    
    return model

# 运行主程序
if __name__ == "__main__":
    model = main()


# ============================================================
#  时间荷守恒扩展版
#  添加 SixDimensionalSpacetimeHypergraphTau 子类
#  实现三维时间荷守恒 (t1, t2, t3)
# ============================================================
class SixDimensionalSpacetimeHypergraphTau(SixDimensionalSpacetimeHypergraph):
    """
    继承原始六维流形时空超图模型, 加入时间荷守恒机制
    - 在三个时间维度 (t1, t2, t3) 上保持总时间荷不变
    - 通过超边在不同子时空之间传输时间荷
    """

    # --------------------------------------------------------
    # 初始化: 增加时间荷守恒相关参数
    # --------------------------------------------------------
    def __init__(self, n_earth=50000, n_blackhole=100, n_darkenergy=2000):
        super().__init__(n_earth=n_earth, n_blackhole=n_blackhole, n_darkenergy=n_darkenergy)

        # 时间荷传输与守恒参数
        self.beta_t1: float = 0.1     # t1 因果传输系数
        self.lambda_ent: float = 0.5  # t2,t3 纠缠传输系数
        self.tau_threshold: float = 0.05  # 守恒阈值 (L2 范数)

    # --------------------------------------------------------
    # 计算单个节点的时间荷密度 (tau_1, tau_2, tau_3)
    # --------------------------------------------------------
    def calculate_time_charge_density(self, node_id):
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
            # 暗能量节点 time_coords = [t2, t3]
            return np.array([0.0, mass * t1, mass * t2])
        else:
            return np.zeros(3)

    # --------------------------------------------------------
    # 计算全局时间荷总量
    # --------------------------------------------------------
    def get_total_time_charge(self):
        total = np.zeros(3)
        for nid in self.nodes:
            total += self.calculate_time_charge_density(nid)
        return total

    # --------------------------------------------------------
    # 覆写 update_physics : 先执行基类逻辑, 再处理时间荷流动与守恒
    # --------------------------------------------------------
    def update_physics(self):
        # 记录初始时间荷
        initial_tau = self.get_total_time_charge()

        # 调用基类的物理量更新
        super().update_physics()

        # ----------------------------------------------------
        # 1. 通过超边在节点间传输时间荷
        # ----------------------------------------------------
        for hyperedge in self.hyperedges:
            htype = hyperedge['type']
            strength = hyperedge.get('strength', 1.0)

            # 黑洞 -> 地球 (仅 t1)
            if htype == 'gravity_update':
                bh = hyperedge['nodes'][0]
                for earth in hyperedge['nodes'][1:]:
                    tau_bh = self.calculate_time_charge_density(bh)[0]
                    tau_e  = self.calculate_time_charge_density(earth)[0]
                    # 时间荷流量 (正值表示从 BH 流向 Earth)
                    flow = self.beta_t1 * (tau_bh - tau_e) * strength * 0.01
                    # 更新两节点 time_coords[0]
                    self.nodes[bh]['time_coords'][0]  -= flow / (self.nodes[bh]['mass']    + 1e-12)
                    self.nodes[earth]['time_coords'][0] += flow / (self.nodes[earth]['mass'] + 1e-12)

            # 黑洞 <-> 暗能量 (t2, t3)
            elif htype == 'quantum_entanglement':
                bh = hyperedge['nodes'][0]
                for de in hyperedge['nodes'][1:]:
                    # t2 分量
                    tau_bh_t2 = self.calculate_time_charge_density(bh)[1]
                    tau_de_t2 = self.calculate_time_charge_density(de)[1]
                    flow_t2 = self.lambda_ent * (tau_bh_t2 - tau_de_t2) * strength * 0.01
                    self.nodes[bh]['time_coords'][1] -= flow_t2 / (self.nodes[bh]['mass'] + 1e-12)
                    # de time_coords index 0 对应 t2
                    self.nodes[de]['time_coords'][0] += flow_t2 / (self.nodes[de]['mass'] + 1e-12)

                    # t3 分量
                    tau_bh_t3 = self.calculate_time_charge_density(bh)[2]
                    tau_de_t3 = self.calculate_time_charge_density(de)[2]
                    flow_t3 = self.lambda_ent * (tau_bh_t3 - tau_de_t3) * strength * 0.01
                    self.nodes[bh]['time_coords'][2] -= flow_t3 / (self.nodes[bh]['mass'] + 1e-12)
                    # de time_coords index 1 对应 t3
                    self.nodes[de]['time_coords'][1] += flow_t3 / (self.nodes[de]['mass'] + 1e-12)

        # ----------------------------------------------------
        # 2. 守恒性校正 (若数值漂移超过阈值)
        # ----------------------------------------------------
        final_tau = self.get_total_time_charge()
        drift = np.linalg.norm(final_tau - initial_tau)
        if drift > self.tau_threshold:
            # 比例系数
            ratios = np.divide(initial_tau, final_tau + 1e-12)
            for nid, node in self.nodes.items():
                tc = node['time_coords']
                if node['type'] == 'earth':
                    tc[0] *= ratios[0]
                elif node['type'] == 'blackhole':
                    if len(tc) >= 1:
                        tc[0] *= ratios[0]
                    if len(tc) >= 2:
                        tc[1] *= ratios[1]
                    if len(tc) >= 3:
                        tc[2] *= ratios[2]
                elif node['type'] == 'darkenergy':
                    # de: [t2, t3]
                    if len(tc) >= 1:
                        tc[0] *= ratios[1]
                    if len(tc) >= 2:
                        tc[1] *= ratios[2]

            # 更新 drift (可选打印)
            # print(f"[Tau-Correction] drift={drift:.3e} -> ratios={ratios}")

# ============================================================
#   重写 main() : 使用 Tau 扩展类
# ============================================================

def main():
    """主程序：运行六维流形时空超图计算模拟 (含时间荷守恒)"""
    print("="*60)
    print("六维流形时空超图计算模拟 (时间荷守恒版)")
    print("基于Wolfram超图计算模型 + Tau 守恒")
    print("="*60)

    # 创建 Tau 模型
    print("\n1. 初始化 Tau 超图模型...")
    model = SixDimensionalSpacetimeHypergraphTau(
        n_earth=50000,
        n_blackhole=100,
        n_darkenergy=2000
    )

    # 初始化节点
    print("\n2. 初始化时空节点...")
    model.initialize_spacetime_nodes()

    # 创建超边
    print("\n3. 创建超边连接...")
    model.create_hyperedges()

    # 运行演化
    print("\n4. 开始超图演化...")
    model = evolve_hypergraph(model, n_iterations=500)

    # 创建3D可视化
    print("\n5. 创建3D可视化...")
    fig_3d = create_3d_visualization(model)

    # 创建演化统计图
    print("\n6. 创建演化统计图...")
    fig_stats = plot_evolution_statistics(model)

    # 输出最终统计
    print("\n7. 最终统计结果:")
    print("="*40)
    # 计算最终质量分布
    total_mass_bh = sum(model.nodes[n]['mass'] for n in model.nodes if model.nodes[n]['type'] == 'blackhole')
    total_mass_earth = sum(model.nodes[n]['mass'] for n in model.nodes if model.nodes[n]['type'] == 'earth')
    total_energy_de = sum(model.nodes[n]['energy'] for n in model.nodes if model.nodes[n]['type'] == 'darkenergy')
    print(f"黑洞总质量: {total_mass_bh:.2e} kg")
    print(f"地球型物质总质量: {total_mass_earth:.2e} kg")
    print(f"暗能量总能量: {total_energy_de:.2e} J")
    print(f"质量比例 (黑洞/地球型): {total_mass_bh / max(total_mass_earth, 1e-6):.1f}")

    # 网络统计
    print(f"\n网络结构统计:")
    print(f"总节点数: {len(model.nodes)}")
    print(f"总边数: {len(model.hypergraph.edges())}")
    print(f"总超边数: {len(model.hyperedges)}")

    return model

# -------------------------------------------------------------
#  若作为脚本执行, 运行 main()
# -------------------------------------------------------------
if __name__ == "__main__":
    model = main()
