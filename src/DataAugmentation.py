import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from ctgan import CTGAN
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance, ks_2samp
import umap.umap_ as umap

class DataAugmentation:
    """支持多种数据增强方法及评价相关性的类"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_aug = None
        self.y_aug = None
        self.x_combined = None
        self.y_combined = None

    def show_data(self):
        """显示一下当前的数据"""
        print("\n" + "="*50)
        print("原始特征数据 (x):")
        print("="*50)
        print(f"形状: {self.x.shape}")
        print(self.x.head())
        
        print("\n" + "="*50)
        print("原始目标变量 (y):")
        print("="*50)
        print(f"形状: {self.y.shape}")
        print(self.y.head())
        
        if self.x_aug is not None:
            print("\n" + "="*50)
            print("增强后的特征数据 (x_aug):")
            print("="*50)
            print(f"形状: {self.x_aug.shape}")
            print(self.x_aug.head())
            
        if self.y_aug is not None:
            print("\n" + "="*50)
            print("增强后的目标变量 (y_aug):")
            print("="*50)
            print(f"形状: {self.y_aug.shape}")
            print(self.y_aug.head())
        
        if self.x_combined is not None:
            print("\n" + "="*50)
            print("合并后的特征数据 (x_combined):")
            print("="*50)
            print(f"形状: {self.x_combined.shape}")
            print("前5行原始数据 + 后5行增强数据:")
            print(pd.concat([self.x_combined.head(), self.x_combined.tail()]))
            
        if self.y_combined is not None:
            print("\n" + "="*50)
            print("合并后的目标变量 (y_combined):")
            print("="*50)
            print(f"形状: {self.y_combined.shape}")
            print("前5行原始数据 + 后5行增强数据:")
            print(pd.concat([self.y_combined.head(), self.y_combined.tail()]))

    def apply_smogn(self):
        """SMOGN"""
        print("应用SMOGN数据增强...")
        print("该方法基本行不通")
        return None, None

    def apply_bootstrapping(self, n_samples=100, noise_scale=0.1, random_state=42):
        """Bootstrapping数据增强"""
        print("应用 Bootstrap重采样 + 高斯噪声 进行数据增强...")

        x, y = self.x, self.y
        
        # 记录原始列名
        x_cols = x.columns.tolist()
        y_cols = y.columns.tolist()
    
        # 标准化数据
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_scaled = scaler_x.fit_transform(x)
        y_scaled = scaler_y.fit_transform(y)
        
        # Bootstrap重采样
        x_resampled, y_resampled = resample(
            x_scaled, y_scaled, 
            n_samples=n_samples, 
            replace=True,
            random_state=random_state
        )
        
        # 添加高斯噪声
        np.random.seed(random_state)
        x_noise = np.random.normal(scale=noise_scale, size=x_resampled.shape)
        y_noise = np.random.normal(scale=noise_scale, size=y_resampled.shape)
        
        # 合并噪声
        x_aug = x_resampled + x_noise
        y_aug = y_resampled + y_noise
        
        # 逆标准化
        x_aug = scaler_x.inverse_transform(x_aug)
        y_aug = scaler_y.inverse_transform(y_aug)
        
        # 转回DataFrame
        x_aug = pd.DataFrame(x_aug, columns=x_cols)
        y_aug = pd.DataFrame(y_aug, columns=y_cols)
        
        print(f"数据增强完成: {len(x)} -> {len(x_aug)} 样本")
        
        # 更新类属性
        self.x_aug = x_aug
        self.y_aug = y_aug
        self.x_combined = pd.concat([x, x_aug], ignore_index=True)
        self.y_combined = pd.concat([y, y_aug], ignore_index=True)
        
        return x_aug, y_aug

    def apply_gans(self, n_samples=100, epochs=100, batch_size=100):
        """GANs (CTGAN) 数据增强"""
        print("应用全维度CTGAN数据增强...")

        x, y = self.x, self.y
        
        # 合并特征和目标
        data = pd.concat([x, y], axis=1)
        original_cols = data.columns.tolist()
        
        # 识别连续列
        continuous_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        
        # 归一化
        scaler = MinMaxScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns
        )
        
        # 配置CTGAN
        ctgan = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            pac=5,
            cuda=False,
            verbose=True
        )
        
        print("开始训练GAN...")
        ctgan.fit(data_scaled, continuous_cols)
        
        # 生成新样本
        print(f"生成 {n_samples} 个样本...")
        synthetic_data = ctgan.sample(n_samples)
        
        # 逆归一化
        synthetic_data = pd.DataFrame(
            scaler.inverse_transform(synthetic_data),
            columns=original_cols
        )
        
        # 分离特征和目标
        x_cols = x.columns.tolist()
        y_cols = y.columns.tolist()
        x_aug = synthetic_data[x_cols]
        y_aug = synthetic_data[y_cols]
        
        print(f"生成完成: 原始数据 {len(x)} -> 合成数据 {len(x_aug)}")
        print(f"目标变量维度: {y_aug.shape[1]}")
        
        # 更新类属性
        self.x_aug = x_aug
        self.y_aug = y_aug
        self.x_combined = pd.concat([x, x_aug], ignore_index=True)
        self.y_combined = pd.concat([y, y_aug], ignore_index=True)

        return x_aug, y_aug

    def direct_quality_assessment(self):
        """直接评估生成数据的质量"""
        if self.y_aug is None:
            print("请先进行数据增强！")
            return None
            
        y_original = self.y.values if hasattr(self.y, 'values') else self.y
        y_synthetic = self.y_aug.values if hasattr(self.y_aug, 'values') else self.y_aug
        
        print("生成数据质量直接评估")
        print("="*50)
        
        # 1. 均值相似性
        mean_original = np.mean(y_original, axis=0)
        mean_synthetic = np.mean(y_synthetic, axis=0)
        mean_correlation = np.corrcoef(mean_original, mean_synthetic)[0, 1]
        print(f"均值相关性: {mean_correlation:.4f}")
        
        # 2. 标准差相似性  
        std_original = np.std(y_original, axis=0)
        std_synthetic = np.std(y_synthetic, axis=0)
        std_correlation = np.corrcoef(std_original, std_synthetic)[0, 1]
        print(f"标准差相关性: {std_correlation:.4f}")
        
        # 3. 分布距离评估
        n_vars = min(10, y_original.shape[1])
        wasserstein_distances = []
        ks_pvalues = []
        
        for i in range(n_vars):
            w_dist = wasserstein_distance(y_original[:, i], y_synthetic[:, i])
            wasserstein_distances.append(w_dist)
            
            ks_stat, ks_pval = ks_2samp(y_original[:, i], y_synthetic[:, i])
            ks_pvalues.append(ks_pval)
        
        print(f"平均Wasserstein距离: {np.mean(wasserstein_distances):.4f}")
        print(f"KS检验平均p值: {np.mean(ks_pvalues):.4f}")
        
        # 可视化
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # 均值一致性
        ax1.scatter(mean_original, mean_synthetic, alpha=0.6)
        ax1.plot([mean_original.min(), mean_original.max()], 
                [mean_original.min(), mean_original.max()], 'r--')
        ax1.set_xlabel('原始数据均值')
        ax1.set_ylabel('生成数据均值')
        ax1.set_title('均值一致性')
        
        # 方差一致性
        ax2.scatter(std_original, std_synthetic, alpha=0.6)
        ax2.plot([std_original.min(), std_original.max()], 
                [std_original.min(), std_original.max()], 'r--')
        ax2.set_xlabel('原始数据标准差')
        ax2.set_ylabel('生成数据标准差')
        ax2.set_title('方差一致性')
        
        # 分布距离
        ax3.bar(range(n_vars), wasserstein_distances)
        ax3.set_xlabel('目标变量索引')
        ax3.set_ylabel('Wasserstein距离')
        ax3.set_title('分布距离评估')
        
        plt.tight_layout()
        plt.show()

        return {
            'mean_correlation': mean_correlation,
            'std_correlation': std_correlation,
            'wasserstein_distances': wasserstein_distances,
            'ks_pvalues': ks_pvalues
        }

    def comprehensive_visualization(self):
        """综合可视化分析"""
        if self.y_aug is None:
            print("请先进行数据增强！")
            return None
            
        y_original = self.y.values if hasattr(self.y, 'values') else self.y
        y_synthetic = self.y_aug.values if hasattr(self.y_aug, 'values') else self.y_aug
        
        y_combined = np.vstack([y_original, y_synthetic])
        labels = ['原始y'] * len(y_original) + ['生成y'] * len(y_synthetic)
        
        # 三种降维方法
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
        y_umap = reducer.fit_transform(y_combined)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        y_tsne = tsne.fit_transform(y_combined)
        
        pca = PCA(n_components=2)
        y_pca = pca.fit_transform(y_combined)
        
        # 可视化
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # UMAP
        for label, color in zip(['原始y', '生成y'], ['blue', 'red']):
            mask = np.array(labels) == label
            ax1.scatter(y_umap[mask, 0], y_umap[mask, 1], c=color, alpha=0.7, label=label, s=50)
        ax1.set_title('UMAP - 全局结构评估')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # t-SNE
        for label, color in zip(['原始y', '生成y'], ['blue', 'red']):
            mask = np.array(labels) == label
            ax2.scatter(y_tsne[mask, 0], y_tsne[mask, 1], c=color, alpha=0.7, label=label, s=50)
        ax2.set_title('t-SNE - 局部结构分析')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # PCA
        for label, color in zip(['原始y', '生成y'], ['blue', 'red']):
            mask = np.array(labels) == label
            ax3.scatter(y_pca[mask, 0], y_pca[mask, 1], c=color, alpha=0.7, label=label, s=50)
        ax3.set_title(f'PCA (解释方差: {np.sum(pca.explained_variance_ratio_):.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return y_umap, y_tsne, y_pca

    # def save_to_csv(self, save_dir=".", prefix=None, include_timestamp=True):
    #     """
    #     保存增强数据和合并数据到CSV文件
        
    #     参数:
    #         save_dir: 保存目录，默认为当前目录
    #         prefix: 文件名前缀
    #         include_timestamp: 是否在文件名中包含时间戳
    #     """
    #     if self.x_aug is None or self.y_aug is None:
    #         print("警告：尚未进行数据增强，无法保存数据！")
    #         return False
        
    #     try:
    #         # 创建保存目录（如果不存在）
    #         os.makedirs(save_dir, exist_ok=True)
            
    #         # 生成文件名前缀
    #         if prefix is None:
    #             prefix = "augmented_data"
            
    #         # 生成时间戳（可选）
    #         timestamp = ""
    #         if include_timestamp:
    #             if self.augmentation_time is None:
    #                 self.augmentation_time = datetime.now()
    #             timestamp = f"_{self.augmentation_time.strftime('%Y%m%d_%H%M%S')}"
            
    #         # 生成文件名
    #         files_info = {
    #             'original_x': f"{prefix}_original_x{timestamp}.csv",
    #             'original_y': f"{prefix}_original_y{timestamp}.csv",
    #             'augmented_x': f"{prefix}_augmented_x{timestamp}.csv", 
    #             'augmented_y': f"{prefix}_augmented_y{timestamp}.csv",
    #             'combined_x': f"{prefix}_combined_x{timestamp}.csv",
    #             'combined_y': f"{prefix}_combined_y{timestamp}.csv",
    #             'metadata': f"{prefix}_metadata{timestamp}.txt"
    #         }
            
    #         # 保存原始数据
    #         self.x.to_csv(os.path.join(save_dir, files_info['original_x']), index=False)
    #         self.y.to_csv(os.path.join(save_dir, files_info['original_y']), index=False)
    #         print(f"✓ 原始特征数据保存至: {files_info['original_x']}")
    #         print(f"✓ 原始目标数据保存至: {files_info['original_y']}")
            
    #         # 保存增强数据
    #         self.x_aug.to_csv(os.path.join(save_dir, files_info['augmented_x']), index=False)
    #         self.y_aug.to_csv(os.path.join(save_dir, files_info['augmented_y']), index=False)
    #         print(f"✓ 增强特征数据保存至: {files_info['augmented_x']}")
    #         print(f"✓ 增强目标数据保存至: {files_info['augmented_y']}")
            
    #         # 保存合并数据
    #         self.x_combined.to_csv(os.path.join(save_dir, files_info['combined_x']), index=False)
    #         self.y_combined.to_csv(os.path.join(save_dir, files_info['combined_y']), index=False)
    #         print(f"✓ 合并特征数据保存至: {files_info['combined_x']}")
    #         print(f"✓ 合并目标数据保存至: {files_info['combined_y']}")
            
    #         # 保存元数据信息
    #         self._save_metadata(os.path.join(save_dir, files_info['metadata']))
    #         print(f"✓ 元数据信息保存至: {files_info['metadata']}")
            
    #         # 保存文件列表信息
    #         self._save_file_list(save_dir, files_info, prefix)
            
    #         print(f"\n🎉 所有数据已成功保存到目录: {os.path.abspath(save_dir)}")
    #         return True
            
    #     except Exception as e:
    #         print(f"❌ 保存数据时出错: {e}")
    #         return False

    # def save_visualization(self, save_dir=".", prefix=None, dpi=300):
    #     """
    #     保存可视化图表到文件
        
    #     参数:
    #         save_dir: 保存目录
    #         prefix: 文件名前缀
    #         dpi: 图片分辨率
    #     """
    #     if self.y_aug is None:
    #         print("警告：尚未进行数据增强，无法保存可视化！")
    #         return False
        
    #     try:
    #         os.makedirs(save_dir, exist_ok=True)
            
    #         if prefix is None:
    #             prefix = "visualization"
            
    #         timestamp = ""
    #         if self.augmentation_time:
    #             timestamp = f"_{self.augmentation_time.strftime('%Y%m%d_%H%M%S')}"
            
    #         # 生成质量评估图表
    #         plt.figure(figsize=(12, 4))
            
    #         y_original = self.y.values if hasattr(self.y, 'values') else self.y
    #         y_synthetic = self.y_aug.values if hasattr(self.y_aug, 'values') else self.y_aug
            
    #         # 均值一致性
    #         mean_original = np.mean(y_original, axis=0)
    #         mean_synthetic = np.mean(y_synthetic, axis=0)
            
    #         plt.subplot(1, 3, 1)
    #         plt.scatter(mean_original, mean_synthetic, alpha=0.6)
    #         plt.plot([mean_original.min(), mean_original.max()], 
    #                 [mean_original.min(), mean_original.max()], 'r--')
    #         plt.xlabel('原始数据均值')
    #         plt.ylabel('生成数据均值')
    #         plt.title('均值一致性')
            
    #         # 方差一致性
    #         std_original = np.std(y_original, axis=0)
    #         std_synthetic = np.std(y_synthetic, axis=0)
            
    #         plt.subplot(1, 3, 2)
    #         plt.scatter(std_original, std_synthetic, alpha=0.6)
    #         plt.plot([std_original.min(), std_original.max()], 
    #                 [std_original.min(), std_original.max()], 'r--')
    #         plt.xlabel('原始数据标准差')
    #         plt.ylabel('生成数据标准差')
    #         plt.title('方差一致性')
            
    #         # 分布距离
    #         n_vars = min(10, y_original.shape[1])
    #         wasserstein_distances = []
    #         for i in range(n_vars):
    #             w_dist = wasserstein_distance(y_original[:, i], y_synthetic[:, i])
    #             wasserstein_distances.append(w_dist)
            
    #         plt.subplot(1, 3, 3)
    #         plt.bar(range(n_vars), wasserstein_distances)
    #         plt.xlabel('目标变量索引')
    #         plt.ylabel('Wasserstein距离')
    #         plt.title('分布距离评估')
            
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(save_dir, f"{prefix}_quality_assessment{timestamp}.png"), 
    #                    dpi=dpi, bbox_inches='tight')
    #         plt.close()
            
    #         print(f"✓ 质量评估图表保存至: {prefix}_quality_assessment{timestamp}.png")
            
    #         # 保存综合可视化（如果已生成）
    #         try:
    #             # 这里可以调用comprehensive_visualization并保存结果
    #             pass
    #         except:
    #             pass
                
    #         return True
            
    #     except Exception as e:
    #         print(f"❌ 保存可视化时出错: {e}")
    #         return False

    # def quick_save(self, description=""):
    #     """
    #     快速保存方法（使用默认设置）
        
    #     参数:
    #         description: 数据描述，用于文件名
    #     """
    #     if description:
    #         prefix = f"augmented_{description}"
    #     else:
    #         prefix = "augmented_data"
        
    #     # 设置增强时间
    #     self.augmentation_time = datetime.now()
        
    #     # 保存数据
    #     success1 = self.save_to_csv(prefix=prefix)
        
    #     # 保存可视化
    #     success2 = self.save_visualization(prefix=prefix)
        
    #     return success1 and success2


# def main():
#     """主函数示例"""
#     print("=== 数据增强系统演示 ===")
    
#     # 示例：创建模拟数据（实际使用时替换为您的数据）
#     np.random.seed(42)

#     n_samples = 20
#     n_features = 2
#     n_targets = 58
    
#     # 创建示例数据
#     x = pd.DataFrame(np.random.randn(n_samples, n_features), columns=['feature1', 'feature2'])
#     y = pd.DataFrame(np.random.randn(n_samples, n_targets), columns=[f'target_{i}' for i in range(n_targets)])
    
#     print(f"原始数据形状: x={x.shape}, y={y.shape}")
    
#     # 初始化数据增强类
#     da = DataAugmentation(x, y)
    
#     # 显示原始数据
#     print("\n1. 原始数据概览:")
#     da.show_data()
    
#     # 方法1: 使用GANs进行数据增强
#     print("\n2. 使用GANs进行数据增强:")
#     x_aug, y_aug = da.apply_gans(n_samples=50, epochs=50, batch_size=20)
    
#     # 显示增强后的数据
#     print("\n3. 增强后数据概览:")
#     da.show_data()
    
#     # 质量评估
#     print("\n4. 数据质量评估:")
#     quality_results = da.direct_quality_assessment()
    
#     # 可视化分析
#     print("\n5. 综合可视化分析:")
#     da.comprehensive_visualization()
    
#     print("\n=== 演示完成 ===")

def main():
    """简化版主函数用于测试"""
    print("=== 数据增强系统测试 ===")
    
    # 创建更小的测试数据
    np.random.seed(42)
    n_samples = 10  # 减少样本数
    n_features = 2
    n_targets = 5   # 减少目标变量数
    
    x = pd.DataFrame(np.random.randn(n_samples, n_features), columns=['feature1', 'feature2'])
    y = pd.DataFrame(np.random.randn(n_samples, n_targets), columns=[f'target_{i}' for i in range(n_targets)])
    
    print(f"原始数据形状: x={x.shape}, y={y.shape}")
    
    # 初始化数据增强类
    da = DataAugmentation(x, y)
    
    # 先尝试简单的 bootstrapping 方法
    print("\n1. 使用 Bootstrapping 进行数据增强:")
    x_aug, y_aug = da.apply_bootstrapping(n_samples=5)  # 这个方法更快
    
    # 显示数据
    print("\n2. 增强后数据概览:")
    da.show_data()
    
    # 质量评估
    print("\n3. 数据质量评估:")
    quality_results = da.direct_quality_assessment()
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    print(1)
    main()