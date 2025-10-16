#!/usr/bin/env python3
"""
6D旋转转换工具函数
基于RDT官方实现，提供稳定和精确的6D旋转转换
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_vector(v):
    """归一化向量，确保数值稳定性"""
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def cross_product(u, v):
    """计算叉积"""
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)
        
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = np.stack((i, j, k), axis=1)
    return out.squeeze() if out.shape[0] == 1 else out


def compute_rotation_matrix_from_ortho6d(ortho6d):
    """
    从6D正交表示重构旋转矩阵
    基于RDT官方实现，确保数值稳定性
    
    Args:
        ortho6d: (N, 6) 或 (6,) 6D旋转表示
        
    Returns:
        matrix: (N, 3, 3) 或 (3, 3) 旋转矩阵
    """
    if ortho6d.ndim == 1:
        ortho6d = ortho6d.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
        
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
        
    # 归一化第一列
    x = normalize_vector(x_raw)
    
    # 计算第三列
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    
    # 重新计算第二列确保正交性
    y = cross_product(z, x)
    
    # 重构旋转矩阵
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    
    if squeeze_output:
        matrix = matrix.squeeze(0)
    
    return matrix


def compute_ortho6d_from_rotation_matrix(matrix):
    """
    从旋转矩阵提取6D正交表示
    基于RDT官方实现
    
    Args:
        matrix: (N, 3, 3) 或 (3, 3) 旋转矩阵
        
    Returns:
        ortho6d: (N, 6) 或 (6,) 6D旋转表示
    """
    if matrix.ndim == 2:
        matrix = matrix.reshape(1, 3, 3)
        squeeze_output = True
    else:
        squeeze_output = False
        
    # 6D表示使用旋转矩阵的前两列
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    
    if squeeze_output:
        ortho6d = ortho6d.squeeze(0)
    
    return ortho6d


def convert_euler_to_6d_rotation(euler_angles):
    """
    将欧拉角转换为6D旋转表示
    基于RDT官方实现
    
    Args:
        euler_angles: (N, 3) 或 (3,) 欧拉角 (roll, pitch, yaw)
        
    Returns:
        6d_rotation: (N, 6) 或 (6,) 6D旋转表示
    """
    if euler_angles.ndim == 1:
        euler_angles = euler_angles.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 转换为旋转矩阵
    rotation_matrices = R.from_euler('xyz', euler_angles).as_matrix()
    
    # 提取6D表示
    ortho6d = compute_ortho6d_from_rotation_matrix(rotation_matrices)
    
    if squeeze_output:
        ortho6d = ortho6d.squeeze(0)
    
    return ortho6d


def convert_6d_rotation_to_euler(ortho6d):
    """
    将6D旋转表示转换为欧拉角
    基于RDT官方实现，确保数值稳定性
    
    Args:
        ortho6d: (N, 6) 或 (6,) 6D旋转表示
        
    Returns:
        euler: (N, 3) 或 (3,) 欧拉角 (roll, pitch, yaw)
    """
    if ortho6d.ndim == 1:
        ortho6d = ortho6d.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 重构旋转矩阵
    rotation_matrix = compute_rotation_matrix_from_ortho6d(ortho6d)
    
    # 转换为欧拉角
    r = R.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=False)
    
    if squeeze_output:
        euler = euler.squeeze(0)
    
    return euler


def convert_quaternion_to_6d_rotation(quat):
    """
    将四元数转换为6D旋转表示
    
    Args:
        quat: (N, 4) 或 (4,) 四元数 (x, y, z, w)
        
    Returns:
        6d_rotation: (N, 6) 或 (6,) 6D旋转表示
    """
    if quat.ndim == 1:
        quat = quat.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 转换为旋转矩阵
    rotation_matrices = R.from_quat(quat).as_matrix()
    
    # 提取6D表示
    ortho6d = compute_ortho6d_from_rotation_matrix(rotation_matrices)
    
    if squeeze_output:
        ortho6d = ortho6d.squeeze(0)
    
    return ortho6d


def test_6d_rotation_conversion():
    """测试6D旋转转换的往返一致性"""
    print("🧪 测试6D旋转转换的往返一致性...")
    
    # 生成随机欧拉角
    euler = np.random.rand(3) * 2 * np.pi - np.pi
    print(f"原始欧拉角: {euler}")
    
    # 转换为6D旋转
    ortho6d = convert_euler_to_6d_rotation(euler)
    print(f"6D旋转表示: {ortho6d}")
    
    # 转换回欧拉角
    euler_recovered = convert_6d_rotation_to_euler(ortho6d)
    print(f"恢复的欧拉角: {euler_recovered}")
    
    # 计算误差
    error = np.abs(euler - euler_recovered)
    print(f"转换误差: {error}")
    print(f"最大误差: {error.max():.6f}")
    
    if error.max() < 1e-3:
        print("✅ 6D旋转转换测试通过！")
    else:
        print("❌ 6D旋转转换测试失败！")
    
    return error.max() < 1e-3


if __name__ == "__main__":
    # 运行测试
    test_6d_rotation_conversion()
