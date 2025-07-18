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
    
    def __init__(self, n_earth=80000, n_blackhole=80, n_darkenergy=2500):
        self.n_earth = n_earth
        self.n_blackhole = n_blackhole
        self.n_darkenergy = n_darkenergy
        
        # 深度学习启发的参数配置
        self.grid_resolution = 256  # 降低分辨率以减少内存消耗
        self.box_size = 1000.0  # h⁻¹Mpc 盒子大小
        self.conv_kernel_size = 4  # 4³卷积核
        self.n_filters_base = 16  # 基础滤波器数量
        self.n_filters_max = 512  # 最大滤波器数量
        self.learning_rate = 2e-4  # 学习率参数
        self.beta1 = 0.5  # Adam优化器参数
        self.beta2 = 0.99  # Adam优化器参数
        self.gp_weight = 10.0  # 梯度惩罚权重
        self.batch_size = 16  # 减小批量大小以减少内存消耗
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

    def build_generator(self):
        """
        构建生成器网络 (SRResNet架构)
        """
        model = torch.nn.Sequential(
            # 输入: [batch, channels, H, W, D]
            torch.nn.Conv3d(1, 64, kernel_size=9, padding=4, padding_mode='replicate'),
            torch.nn.PReLU(),
            
            # 残差块
            *[self._residual_block(64) for _ in range(5)],
            
            # 上采样 (使用ConvTranspose3d)
            # 2x upsampling
            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.PReLU(),
            
            # 4x upsampling
            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.PReLU(),
            
            # 8x upsampling
            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.PReLU(),
            
            # 最终输出层
            torch.nn.Conv3d(64, 1, kernel_size=9, padding=4, padding_mode='replicate'),
            torch.nn.Tanh()
        )
        return model.to(device)
    
    def build_discriminator(self):
        """
        构建判别器网络 (PatchGAN架构)
        """
        def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
            layers = [
                torch.nn.Conv3d(in_filters, out_filters, 4, stride=stride, padding=1, padding_mode='replicate')
            ]
            if normalization:
                layers.append(torch.nn.InstanceNorm3d(out_filters, affine=True))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        model = torch.nn.Sequential(
            # 输入: [batch, 1, H, W, D]
            *discriminator_block(1, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            
            # 输出1x1x1的判别结果
            torch.nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='replicate'),
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
        执行单步GAN训练
        """
        # 转换为PyTorch张量
        lr_batch = torch.FloatTensor(lr_batch).unsqueeze(1).to(device)  # [B, 1, H, W, D]
        hr_batch = torch.FloatTensor(hr_batch).unsqueeze(1).to(device)  # [B, 1, H, W, D]
        
        # 真实和假标签
        valid = torch.ones((lr_batch.size(0), 1, 1, 1, 1), device=device)
        fake = torch.zeros((lr_batch.size(0), 1, 1, 1, 1), device=device)
        
        # ---------------------
        #  训练判别器
        # ---------------------
        self.optimizer_D.zero_grad()
        
        # 真实损失
        pred_real = self.discriminator(hr_batch)
        loss_real = self.criterion_GAN(pred_real, valid)
        
        # 生成假样本
        noise = torch.randn(lr_batch.size(0), self.noise_dim, *lr_batch.shape[2:], device=device) * self.noise_amp
        gen_hr = self.generator(torch.cat([lr_batch, noise], dim=1))
        
        # 假损失
        pred_fake = self.discriminator(gen_hr.detach())
        loss_fake = self.criterion_GAN(pred_fake, fake)
        
        # 梯度惩罚 (WGAN-GP)
        alpha = torch.rand(lr_batch.size(0), 1, 1, 1, 1, device=device)
        interpolated = (alpha * hr_batch + (1 - alpha) * gen_hr).requires_grad_(True)
        pred_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=pred_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(pred_interpolated).to(device),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_weight
        
        # 总判别器损失
        loss_D = (loss_real + loss_fake) * 0.5 + gradient_penalty
        loss_D.backward()
        self.optimizer_D.step()
        
        # ---------------------
        #  训练生成器
        # ---------------------
        self.optimizer_G.zero_grad()
        
        # 对抗损失
        pred_fake = self.discriminator(gen_hr)
        loss_GAN = self.criterion_GAN(pred_fake, valid)
        
        # 像素级损失
        loss_pixel = self.criterion_pixel(gen_hr, hr_batch)
        
        # 总生成器损失
        loss_G = loss_GAN + 100 * loss_pixel  # 平衡对抗损失和像素级损失
        loss_G.backward()
        self.optimizer_G.step()
        
        return {
            'G_loss': loss_G.item(),
            'D_loss': loss_D.item(),
            'G_loss_GAN': loss_GAN.item(),
            'G_loss_pixel': loss_pixel.item(),
            'D_loss_real': loss_real.item(),
            'D_loss_fake': loss_fake.item(),
            'gradient_penalty': gradient_penalty.item()
        }

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

    def prepare_training_data(self):
        """
        准备GAN训练数据
        生成低分辨率和高分辨率数据对用于训练
        
        Returns:
            tuple: (低分辨率张量, 高分辨率张量)
        """
        # 1. 收集所有节点的位置和属性
        positions = np.array([node['position'] for node in self.nodes.values()])
        
        # 2. 创建密度场
        grid_res = self.grid_resolution // self.upscale_factor
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
        
        # 5. 生成高分辨率目标（这里可以替换为真实的高分辨率数据）
        hr_field = np.repeat(np.repeat(np.repeat(lr_field, self.upscale_factor, axis=0), 
                                      self.upscale_factor, axis=1), 
                            self.upscale_factor, axis=2)
        
        return torch.FloatTensor(lr_field).unsqueeze(0).unsqueeze(0), \
               torch.FloatTensor(hr_field).unsqueeze(0).unsqueeze(0)

    def pretrain_generator(self, num_epochs=5):
        """
        预训练生成器（仅使用像素级损失）
        
        Args:
            num_epochs (int): 预训练轮数
        """
        self.generator.train()
        self.generator.to(device)  # Move generator to device
        self.criterion_pixel.to(device)  # Move pixel loss to device
        
        for epoch in range(num_epochs):
            # 准备训练数据并移动到设备
            lr_batch, hr_batch = self.prepare_training_data()
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            # 前向传播
            self.optimizer_G.zero_grad()
            fake_hr = self.generator(lr_batch)
            
            # 计算像素级损失
            loss_pixel = self.criterion_pixel(fake_hr, hr_batch)
            
            # 反向传播和优化
            loss_pixel.backward()
            self.optimizer_G.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'预训练生成器 Epoch [{epoch+1}/{num_epochs}], 像素损失: {loss_pixel.item():.4f}')

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
            self.train_gan_step()

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
        n_earth=80000,   # 地球子时空节点数
        n_blackhole=80,  # 黑洞子时空节点数
        n_darkenergy=2500 # 暗能量子时空节点数
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
