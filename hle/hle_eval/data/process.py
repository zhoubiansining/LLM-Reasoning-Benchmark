import pandas as pd


def process_dataset(input_file: str = "0000.parquet", output_file: str = "0001.parquet"):
    """
    读取数据并构建一个不含图片的子数据集。
    对每一项数据，如果"image"字段非空，则舍弃，否则加入子数据集。
    最后用同样的格式（parquet）存储子数据集至指定文件。

    Args:
        input_file (str): 输入的parquet文件路径
        output_file (str): 输出的parquet文件路径
    """
    # 读取parquet文件
    df = pd.read_parquet(input_file)

    # 过滤掉"image"字段非空的行，只保留"image"字段为空的行
    # 注意：需要同时处理NaN值和空字符串
    filtered_df = df[df['image'].isnull() | (df['image'] == '')]

    # 重置索引
    filtered_df = filtered_df.reset_index(drop=True)

    # 保存为parquet格式
    filtered_df.to_parquet(output_file, index=False)

    print(f"原始数据集大小: {len(df)}")
    print(f"过滤后数据集大小: {len(filtered_df)}")
    print(f"已保存到: {output_file}")


def sample_dataset(input_file: str = "0001.parquet", output_file: str = "0002.parquet", sample_size: int = 200):
    """
    从数据集中随机采样指定数量的样本并保存到新文件。

    Args:
        input_file (str): 输入的parquet文件路径
        output_file (str): 输出的parquet文件路径
        sample_size (int): 采样的样本数量
    """
    # 读取parquet文件
    df = pd.read_parquet(input_file)

    # 随机采样
    if len(df) <= sample_size:
        sampled_df = df
        print(f"数据集大小({len(df)})小于等于采样数量({sample_size})，将使用全部数据")
    else:
        sampled_df = df.sample(n=sample_size, random_state=42)  # 使用固定随机种子确保结果可重现
        sampled_df = sampled_df.reset_index(drop=True)

    # 保存为parquet格式
    sampled_df.to_parquet(output_file, index=False)

    print(f"原始数据集大小: {len(df)}")
    print(f"采样后数据集大小: {len(sampled_df)}")
    print(f"已保存到: {output_file}")


if __name__ == "__main__":
    process_dataset()
    sample_dataset()
