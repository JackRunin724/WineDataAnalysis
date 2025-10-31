import pandas as pd
from sklearn.model_selection import train_test_split

def getData(path = './data/raw/SEM.csv'):
    """加载数据"""
    # 加载数据
    data = pd.read_csv(path)
    
    # 划分特征和目标变量
    x = data.iloc[:, :2]
    y = data.iloc[:, 2:]

    return x, y, data

def splitData(x, y):
    """划分训练集/测试集"""    
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    return x_train, x_test, y_train, y_test

def gatSplitData(path = './data/raw/SEM.csv'):
    """加载数据并划分训练集/测试集"""
    # 加载数据
    x, y, _ = getData(path)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = splitData(x, y)

    return x_train, x_test, y_train, y_test

def analysisData(x, y, data, x_train, x_test, y_train, y_test):
    """分析数据的基本信息"""
    # 先直接打印一下
    print(data)

    # 显示数据基本信息
    print("数据形状:", data.shape)
    
    # 检查缺失值
    print("\n缺失值统计(仅显示有缺失的项):")
    missing_stats = data.isnull().sum()
    missing_stats_nonzero = missing_stats[missing_stats > 0]
    print(missing_stats_nonzero.to_dict())
    
    # 显示特征和目标变量的列名
    print("\n特征列名:", x.columns.tolist())
    print("目标变量列名:", y.columns.tolist())
    
    print(f"\n训练集大小: x_train{x_train.shape}, y_train{y_train.shape}")
    print(f"测试集大小: x_test{x_test.shape}, y_test{y_test.shape}")


# 使用示例
if __name__ == "__main__":
    x, y, data = getData()
    x_train, x_test, y_train, y_test = splitData(x, y)
    analysisData(x, y, data, x_train, x_test, y_train, y_test)