# 导入numpy库，用于向量/矩阵计算
import numpy as np
import json


class CoordinateSystem:
    """
    坐标系类
    属性说明：
    - base_vectors: 基准坐标系的坐标轴向量（如直角坐标系是[[1,0],[0,1]]）
    - vectors: 需要计算的原始向量列表（如[[1,1],[2,3]]）
    """

    def __init__(self, base_vectors, vectors):
        """
        初始化坐标系
        :param base_vectors: 基准坐标系坐标轴，二维列表，如[[1,0],[0,1]]（直角坐标系）
        :param vectors: 待计算的原始向量，二维列表，如[[1,1],[2,3]]
        """
        # 将列表转为numpy数组（方便后续计算），dtype=float确保是浮点数运算
        self.base_vectors = np.array(base_vectors, dtype=float)
        self.vectors = np.array(vectors, dtype=float)

        # 检查输入维度是否匹配（大一新生要注意：维度不匹配会报错，必须先校验）
        self.dim = len(self.base_vectors)  # 坐标系维度（二维=2，三维=3）
        for vec in self.base_vectors:
            if len(vec) != self.dim:
                raise ValueError("坐标轴向量维度不匹配！")
        for vec in self.vectors:
            if len(vec) != self.dim:
                raise ValueError("原始向量维度与坐标系不匹配！")

    def is_valid_coordinate_system(self, target_vectors):
        """
        判断目标向量是否能构成有效坐标系（核心：线性无关）
        线性无关=坐标轴不共线（二维）/不共面（三维），用行列式判断
        :param target_vectors: 目标坐标系坐标轴向量，二维列表
        :return: True（有效）/False（无效）
        """
        target_arr = np.array(target_vectors, dtype=float)
        # 第一步：检查维度是否一致
        if target_arr.shape != (self.dim, self.dim):
            print(" 坐标轴数量与维度不匹配，无法构成坐标系")
            return False
        # 第二步：计算行列式，行列式≠0则线性无关（有效坐标系）
        det = np.linalg.det(target_arr)
        if abs(det) < 1e-8:  # 浮点数精度问题，不能直接判0
            print("坐标轴线性相关，无法构成有效坐标系")
            return False
        print("目标坐标系有效")
        return True

    def vector_transfer(self, target_vectors):
        """
        向量从当前基准坐标系转移到目标坐标系
        转移公式=原始向量 × 目标坐标系矩阵的逆矩阵
        :param target_vectors: 目标坐标系坐标轴向量
        :return: 转移后的向量列表
        """
        # 先校验目标坐标系是否有效
        if not self.is_valid_coordinate_system(target_vectors):
            return None

        target_arr = np.array(target_vectors, dtype=float)
        # 计算目标坐标系矩阵的逆矩阵
        target_inv = np.linalg.inv(target_arr)

        # 对每个原始向量进行转移计算
        transferred_vectors = []
        for vec in self.vectors:
            # 向量转移：原始向量 · 逆矩阵（点乘）
            new_vec = np.dot(vec, target_inv)
            transferred_vectors.append(new_vec.round(4).tolist())  # 保留4位小数，转列表

        # 更新基准坐标系为目标坐标系（后续运算基于新坐标系）
        self.base_vectors = target_arr
        print(f" 坐标系转移完成，新基准坐标系：{target_vectors}")
        return transferred_vectors

    def vector_projection(self, target_vectors):
        """
        计算原始向量在目标坐标系各坐标轴上的投影长度
        投影公式=（向量·坐标轴）/ 坐标轴长度
        :param target_vectors: 目标坐标系坐标轴向量
        :return: 投影结果字典
        """
        if not self.is_valid_coordinate_system(target_vectors):
            return None

        target_arr = np.array(target_vectors, dtype=float)
        projection_results = {}

        for i, axis in enumerate(target_arr):
            axis_name = f"坐标轴{i + 1}"  # 坐标轴1、坐标轴2...
            axis_length = np.linalg.norm(axis)  # 计算坐标轴向量的长度（范数）
            projections = []

            for j, vec in enumerate(self.vectors):
                vec_name = f"原始向量{j + 1}"
                # 点乘：向量·坐标轴
                dot_product = np.dot(vec, axis)
                # 投影长度=点乘结果 / 坐标轴长度
                proj_length = dot_product / axis_length
                projections.append({vec_name: round(proj_length, 4)})

            projection_results[axis_name] = projections

        print("投影计算完成")
        return projection_results

    def vector_angle(self, target_vectors):
        """
        计算原始向量与目标坐标系各坐标轴的夹角（弧度）
        夹角公式=arccos(（向量·坐标轴）/(向量长度×坐标轴长度))
        :param target_vectors: 目标坐标系坐标轴向量
        :return: 夹角结果字典
        """
        if not self.is_valid_coordinate_system(target_vectors):
            return None

        target_arr = np.array(target_vectors, dtype=float)
        angle_results = {}

        for i, axis in enumerate(target_arr):
            axis_name = f"坐标轴{i + 1}"
            axis_length = np.linalg.norm(axis)
            angles = []

            for j, vec in enumerate(self.vectors):
                vec_name = f"原始向量{j + 1}"
                vec_length = np.linalg.norm(vec)
                # 避免除以0
                if vec_length < 1e-8 or axis_length < 1e-8:
                    angles.append({vec_name: "无效（向量长度为0）"})
                    continue
                # 点乘计算
                dot_product = np.dot(vec, axis)
                # 计算余弦值（注意范围：-1到1）
                cos_theta = dot_product / (vec_length * axis_length)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                # 计算弧度值
                theta = np.arccos(cos_theta)
                angles.append({vec_name: round(theta, 4)})

            angle_results[axis_name] = angles

        print("夹角计算完成（单位：弧度）")
        return angle_results

    def calculate_area_scale(self, target_vectors):
        """
        计算目标坐标系相对直角坐标系的面积/体积缩放倍数
        缩放倍数=目标坐标系矩阵的行列式绝对值
        :param target_vectors: 目标坐标系坐标轴向量
        :return: 缩放倍数
        """
        if not self.is_valid_coordinate_system(target_vectors):
            return None

        target_arr = np.array(target_vectors, dtype=float)
        # 计算行列式的绝对值
        det = abs(np.linalg.det(target_arr))
        scale = round(det, 4)

        # 维度说明
        dim_desc = "面积" if self.dim == 2 else "体积" if self.dim == 3 else "超体积"
        print(f" 目标坐标系相对直角坐标系的{dim_desc}缩放倍数：{scale}")
        return scale


def load_task_from_json(json_file):
    """
    从JSON文件读取任务数据
    :param json_file: JSON文件路径
    :return: 解析后的任务数据
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f" 找不到文件：{json_file}")
        return None
    except json.JSONDecodeError:
        print(f" JSON文件格式错误：{json_file}")
        return None


if __name__ == "__main__":
    # 测试
    test_task = {
        "base_coordinate": [[1, 0], [0, 1]],  # 直角坐标系（基准）
        "original_vectors": [[1, 1], [2, 3]],  # 两个原始向量
        "target_coordinate": [[1, 0], [1, 1]],  # 目标坐标系
        "tasks": ["transfer", "projection", "angle", "area"]  # 要执行的任务
    }

    try:
        # 创建坐标系对象（把基准坐标系和原始向量传进去）
        cs = CoordinateSystem(
            base_vectors=test_task["base_coordinate"],
            vectors=test_task["original_vectors"]
        )
        print(" 坐标系初始化成功！")
        print(f" 基准坐标系：{test_task['base_coordinate']}")
        print(f" 原始向量：{test_task['original_vectors']}\n")

        target = test_task["target_coordinate"]
        for task in test_task["tasks"]:
            print(f"\n执行任务：{task}")
            if task == "transfer":
                result = cs.vector_transfer(target)
                print(f"转移结果：{result}")
            elif task == "projection":
                result = cs.vector_projection(target)
                print(f"投影结果：{result}")
            elif task == "angle":
                result = cs.vector_angle(target)
                print(f"夹角结果：{result}")
            elif task == "area":
                result = cs.calculate_area_scale(target)
                print(f"缩放倍数：{result}")
            else:
                print(f" 未知任务：{task}")

    except Exception as e:
        print(f" 程序运行出错：{e}")