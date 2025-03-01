#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import time
import sys

plt.rcParams["font.family"] = ["STSong"]

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 导入自定义模块
print("导入demmath和cell模块...")
try:
    from demmath import (
        Vector2i,
        Vector2r,
        Vector3i,
        Vector3r,
        Vector4r,
        Vector6r,
        Matrix3r,
        Matrix3i,
        Matrix4r,
        Matrix6r,
        MatrixXr,
        VectorXr,
        Quaternionr,
        AngleAxisr,
        AlignedBox3r,
        AlignedBox2r,
        matrix_eigen_decomposition,
        matrix_from_euler_angles_xyz,
        Math,
        Mathr,
        sgn,
        pow2,
        pow3,
        pow4,
        pow5,
        pown,
        voigt_to_symm_tensor,
        tensor_to_voigt,
        levi_civita,
        PI,
        HALF_PI,
        TWO_PI,
        NAN,
        INF,
        EPSILON,
    )
    from Cell import Cell, DeformationMode, CompUtils

    from Object import Object

    print("模块导入成功！")
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)


def separator(title):
    """打印分隔线和标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_basic_types():
    """测试基本数学类型"""
    separator("测试基本数学类型")

    # 测试向量创建
    print("创建各种向量:")
    v2i = Vector2i(1, 2)
    v2r = Vector2r(1.5, 2.5)
    v3i = Vector3i(1, 2, 3)
    v3r = Vector3r(1.5, 2.5, 3.5)
    v4r = Vector4r(1.5, 2.5, 3.5, 4.5)
    v6r = Vector6r(1, 2, 3, 4, 5, 6)

    print(f"Vector2i: {v2i}")
    print(f"Vector2r: {v2r}")
    print(f"Vector3i: {v3i}")
    print(f"Vector3r: {v3r}")
    print(f"Vector4r: {v4r}")
    print(f"Vector6r: {v6r}")

    # 测试矩阵创建
    print("\n创建各种矩阵:")
    m3i = Matrix3i(1, 0, 0, 0, 1, 0, 0, 0, 1)
    m3r = Matrix3r(1, 0, 0, 0, 1, 0, 0, 0, 1)
    m4r = Matrix4r()  # 零矩阵
    m6r = Matrix6r()  # 零矩阵

    print(f"Matrix3i:\n{m3i}")
    print(f"Matrix3r:\n{m3r}")
    print(f"Matrix4r:\n{m4r}")
    print(f"Matrix6r (前3行3列):\n{m6r[:3,:3]}")

    # 测试动态大小矩阵和向量
    print("\n创建动态大小矩阵和向量:")
    mxr = MatrixXr(2, 3)
    vxr = VectorXr(4)

    print(f"MatrixXr(2,3):\n{mxr}")
    print(f"VectorXr(4): {vxr}")

    return v3r, m3r  # 返回用于后续测试的对象


def test_quaternion_and_rotations(v3r):
    """测试四元数和旋转"""
    separator("测试四元数和旋转操作")

    # 创建四元数
    q1 = Quaternionr(1, 0, 0, 0)  # 单位四元数
    q2 = Quaternionr(0.7071, 0.7071, 0, 0)  # 绕x轴旋转90度

    print(f"q1: w={q1.w}, x={q1.x}, y={q1.y}, z={q1.z}")
    print(f"q2: w={q2.w}, x={q2.x}, y={q2.y}, z={q2.z}")

    # 四元数乘法
    q3 = q1 * q2
    print(f"q1 * q2: w={q3.w}, x={q3.x}, y={q3.y}, z={q3.z}")

    # 四元数与标量乘法
    q4 = 2.0 * q1
    print(f"2 * q1: w={q4.w}, x={q4.x}, y={q4.y}, z={q4.z}")

    # 四元数加法
    q5 = q1 + q2
    print(f"q1 + q2: w={q5.w}, x={q5.x}, y={q5.y}, z={q5.z}")

    # 四元数到旋转矩阵的转换
    r_mat = q2.toRotationMatrix()
    print(f"四元数q2的旋转矩阵:\n{r_mat}")

    # 角轴表示
    aa = AngleAxisr(PI / 4, Vector3r(0, 0, 1))  # 绕z轴旋转45度
    print(f"角轴: angle={aa.angle}, axis={aa.axis}")

    # 角轴到旋转矩阵的转换
    aa_mat = aa.toRotationMatrix()
    print(f"角轴的旋转矩阵:\n{aa_mat}")

    # 角轴到四元数的转换
    aa_quat = aa.toQuaternion()
    print(f"角轴的四元数: w={aa_quat.w}, x={aa_quat.x}, y={aa_quat.y}, z={aa_quat.z}")

    # 应用旋转到向量
    print(f"原始向量: {v3r}")
    rotated_v = r_mat @ v3r
    print(f"旋转后向量: {rotated_v}")

    # 欧拉角到旋转矩阵的转换
    euler_mat = matrix_from_euler_angles_xyz(PI / 6, PI / 4, PI / 3)
    print(f"从欧拉角(30°,45°,60°)创建的旋转矩阵:\n{euler_mat}")

    return euler_mat  # 返回用于后续测试的对象


def test_aligned_boxes():
    """测试对齐盒"""
    separator("测试对齐盒")

    # 2D盒
    box2r = AlignedBox2r()
    print(f"空的2D盒: min={box2r.min}, max={box2r.max}")

    # 扩展盒以包含点
    box2r.extend(Vector2r(1, 2))
    box2r.extend(Vector2r(-1, 3))
    print(f"扩展后的2D盒: min={box2r.min}, max={box2r.max}")

    # 检查点是否在盒内
    inside_point = Vector2r(0, 2.5)
    outside_point = Vector2r(2, 4)
    print(f"点{inside_point}在盒内? {box2r.contains(inside_point)}")
    print(f"点{outside_point}在盒内? {box2r.contains(outside_point)}")

    # 3D盒
    box3r = AlignedBox3r()
    print(f"空的3D盒: min={box3r.min}, max={box3r.max}")

    # 扩展盒以包含点
    box3r.extend(Vector3r(1, 2, 3))
    box3r.extend(Vector3r(-1, 3, -2))
    print(f"扩展后的3D盒: min={box3r.min}, max={box3r.max}")

    return box3r  # 返回用于后续测试的对象


def test_math_functions():
    """测试数学函数"""
    separator("测试数学函数和工具")

    # 基本函数
    print(f"sgn(-5) = {sgn(-5)}")
    print(f"pow2(3) = {pow2(3)}")
    print(f"pow3(2) = {pow3(2)}")
    print(f"pow4(2) = {pow4(2)}")
    print(f"pow5(2) = {pow5(2)}")
    print(f"pown(2, 6) = {pown(2, 6)}")

    # Math类的静态方法
    print(f"Math.sign(-3.5) = {Math.sign(-3.5)}")

    # 随机数生成
    print(f"Math.unit_random() = {Math.unitRandom()}")
    print(f"Math.unit_random3() = {Math.unitRandom3()}")
    print(f"Math.interval_random(10, 20) = {Math.intervalRandom(10, 20)}")
    print(f"Math.symmetric_random() = {Math.symmetricRandom()}")

    # 向量操作
    v = Vector3r(3, 4, 0)
    print(f"向量v = {v}")
    print(f"v的安全归一化 = {Math.safeNormalize(v)}")

    n = Vector3r(1, 0, 0)
    print(f"向量v在n上的投影 = {Math.project(v, n)}")
    print(f"向量v在n上的拒绝 = {Math.reject(v, n)}")

    # 正交基
    normal = Vector3r(0, 0, 1)
    b1, b2 = Math.createOrthonormalBasis(normal)
    print(f"从法向量{normal}创建的正交基:")
    print(f"  b1 = {b1}")
    print(f"  b2 = {b2}")
    print(f"检查正交性: b1·b2 = {np.dot(b1, b2)}")
    print(f"检查正交性: b1·normal = {np.dot(b1, normal)}")
    print(f"检查正交性: b2·normal = {np.dot(b2, normal)}")

    # 随机旋转
    q_rand = Math.uniformRandomRotation()
    print(f"均匀随机旋转: w={q_rand.w}, x={q_rand.x}, y={q_rand.y}, z={q_rand.z}")

    # 矩阵特征分解
    m = Matrix3r(2, 1, 0, 1, 2, 0, 0, 0, 1)
    print(f"矩阵m:\n{m}")
    eigenvectors, eigenvalues = matrix_eigen_decomposition(m)
    print(f"特征向量:\n{eigenvectors}")
    print(f"特征值:\n{eigenvalues}")

    return normal, b1, b2  # 返回用于后续测试的对象


def test_voigt_notation(m3r):
    """测试Voigt表示法"""
    separator("测试Voigt表示法")

    # 创建一个对称张量
    symm_tensor = Matrix3r(1.0, 0.5, 0.3, 0.5, 2.0, 0.4, 0.3, 0.4, 3.0)
    print(f"对称张量:\n{symm_tensor}")

    # 转换为Voigt表示
    voigt_vector = tensor_to_voigt(symm_tensor)
    print(f"Voigt向量: {voigt_vector}")

    # 转换回对称张量
    recovered_tensor = voigt_to_symm_tensor(voigt_vector)
    print(f"恢复的对称张量:\n{recovered_tensor}")

    # 计算Levi-Civita乘积
    antisymm_tensor = m3r - m3r.T
    levi_civ_result = levi_civita(antisymm_tensor)
    print(f"反对称张量:\n{antisymm_tensor}")
    print(f"Levi-Civita乘积: {levi_civ_result}")

    return symm_tensor, voigt_vector  # 返回用于后续测试的对象


def test_cell_basics():
    """测试Cell类的基本功能"""
    separator("测试Cell类基本功能")

    # 创建一个单元格
    cell = Cell()
    print("创建了默认Cell实例")

    # 获取大小
    size = cell.getSize()
    print(f"初始Cell大小: {size}")

    # 设置成一个非立方体盒子
    cell.setBox(Vector3r(10.0, 5.0, 7.0))
    print(f"设置为非立方体后的大小: {cell.getSize()}")
    print(f"体积: {cell.getVolume()}")

    # 测试点的包装和变换
    test_point = Vector3r(12.0, 6.0, 8.0)  # 超出单元格的点
    print(f"测试点: {test_point}")

    wrapped_point = cell.wrapPt(test_point)
    print(f"包装后的点: {wrapped_point}")

    # 使用周期性信息包装点
    period = Vector3i(0, 0, 0)
    wrapped_point, period = cell.wrapPt(test_point, return_period=True)
    print(f"包装后的点: {wrapped_point}, 周期: {period}")

    # 测试规范化
    canonical_point = cell.canonicalizePt(test_point)
    print(f"规范化后的点: {canonical_point}")

    # 检查点是否规范化
    is_canonical = cell.isCanonical(canonical_point)
    print(f"点是否规范? {is_canonical}")

    return cell  # 返回用于后续测试的对象


def test_cell_deformation(cell):
    """测试Cell变形功能"""
    separator("测试Cell变形功能")

    # 设置速度梯度
    gradV = Matrix3r(0.01, 0.002, 0, 0, 0.005, 0, 0, 0, -0.015)
    print(f"设置速度梯度:\n{gradV}")
    cell.setCurrGradV(gradV)

    # 打印初始状态
    print(f"初始大小: {cell.getSize()}")
    print(f"初始体积: {cell.getVolume()}")

    # 模拟变形100个小时间步
    dt = 0.01
    step_count = 10
    volumes = []
    sizes = []

    for i in range(step_count):
        cell.integrateAndUpdate(dt)
        volumes.append(cell.getVolume())
        sizes.append(cell.getSize())

        if i % 2 == 0:
            print(f"步骤 {i}, 大小: {cell.getSize()}, 体积: {cell.getVolume()}")

    print(f"最终大小: {cell.getSize()}")
    print(f"最终体积: {cell.getVolume()}")

    # 测试不同变形模式
    print("\n测试不同的变形模式:")
    for mode in DeformationMode:
        cell_test = Cell()
        cell_test.setBox(Vector3r(10.0, 5.0, 7.0))
        cell_test.setCurrGradV(gradV)
        cell_test.homoDeform = mode

        cell_test.integrateAndUpdate(dt * 10)  # 更大的时间步以显示差异
        print(f"{mode.name}: 体积 = {cell_test.getVolume()}")

    # 返回模拟数据用于绘图
    return volumes, sizes, step_count, dt


def test_cell_periodicity(cell):
    """测试Cell周期性功能"""
    separator("测试Cell周期性功能")

    # 设置一个具有剪切的单元格
    shear_matrix = Matrix3r(10.0, 1.0, 0.5, 0.0, 5.0, 0.3, 0.0, 0.0, 7.0)
    cell.setHSize(shear_matrix)
    print(f"设置具有剪切的单元格形状:\n{cell.getHSize()}")
    print(f"单元格是否有剪切? {cell.hasShear()}")

    # 测试周期性位移
    cell_dist = Vector3i(1, 0, 0)  # 在x方向移动一个单元格
    pos_shift = cell.intrShiftPos(cell_dist)
    print(f"周期性位移 {cell_dist}: {pos_shift}")

    # 使用不同的变形模式测试周期性速度
    for mode in [
        DeformationMode.HOMO_NONE,
        DeformationMode.HOMO_VEL,
        DeformationMode.HOMO_GRADV2,
    ]:
        cell.homoDeform = mode
        vel_shift = cell.intrShiftVel(cell_dist)
        print(f"模式 {mode.name} 下的周期性速度 {cell_dist}: {vel_shift}")

    # 测试波动速度计算
    curr_pos = Vector3r(5.0, 2.5, 3.5)
    prev_vel = Vector3r(0.1, 0.2, 0.3)
    dt = 0.01

    for mode in [
        DeformationMode.HOMO_NONE,
        DeformationMode.HOMO_VEL,
        DeformationMode.HOMO_GRADV2,
    ]:
        cell.homoDeform = mode
        fluct_vel = cell.pprevFluctVel(curr_pos, prev_vel, dt)
        print(f"模式 {mode.name} 下的波动速度: {fluct_vel}")

    return shear_matrix  # 返回用于后续测试的对象


def test_visualize_cell(cell, volumes, sizes, step_count, dt):
    """可视化Cell的变形过程"""
    separator("可视化Cell变形过程")

    # 绘制体积随时间的变化
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(0, step_count) * dt

    plt.subplot(2, 1, 1)
    plt.plot(time_steps, volumes)
    plt.title("Cell体积随时间的变化")
    plt.xlabel("时间")
    plt.ylabel("体积")
    plt.grid(True)

    # 绘制各维度大小随时间的变化
    plt.subplot(2, 1, 2)
    sizes_array = np.array(sizes)
    plt.plot(time_steps, sizes_array[:, 0], "r-", label="X")
    plt.plot(time_steps, sizes_array[:, 1], "g-", label="Y")
    plt.plot(time_steps, sizes_array[:, 2], "b-", label="Z")
    plt.title("Cell各维度大小随时间的变化")
    plt.xlabel("时间")
    plt.ylabel("大小")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 绘制当前Cell的3D表示
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 获取当前形状
    h = cell.getHSize()

    # 定义单元格的顶点
    vertices = [
        np.array([0, 0, 0]),
        h[:, 0],
        h[:, 1],
        h[:, 0] + h[:, 1],
        h[:, 2],
        h[:, 0] + h[:, 2],
        h[:, 1] + h[:, 2],
        h[:, 0] + h[:, 1] + h[:, 2],
    ]

    # 定义单元格的边
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]

    # 绘制边
    for edge in edges:
        ax.plot3D(
            [vertices[edge[0]][0], vertices[edge[1]][0]],
            [vertices[edge[0]][1], vertices[edge[1]][1]],
            [vertices[edge[0]][2], vertices[edge[1]][2]],
            "b-",
        )

    # 绘制坐标轴
    ax.quiver(0, 0, 0, 2, 0, 0, color="r", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 2, 0, color="g", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 2, color="b", arrow_length_ratio=0.1)

    # 设置标签和标题
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("当前Cell的3D表示")

    # 调整视角
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()


def test_object_class():
    """测试Object类"""
    separator("测试Object类")
    obj = Object()
    print(f"Class Name:", obj.toString())


def test_all():
    """运行所有测试"""
    start_time = time.time()

    # 基础数学类型测试
    v3r, m3r = test_basic_types()

    # 四元数和旋转测试
    euler_mat = test_quaternion_and_rotations(v3r)

    # 对齐盒测试
    box3r = test_aligned_boxes()

    # 数学函数测试
    normal, b1, b2 = test_math_functions()

    # Voigt表示法测试
    symm_tensor, voigt_vector = test_voigt_notation(m3r)

    # Cell基本功能测试
    cell = test_cell_basics()

    # Cell变形功能测试
    volumes, sizes, step_count, dt = test_cell_deformation(cell)

    # Cell周期性功能测试
    shear_matrix = test_cell_periodicity(cell)

    # 可视化Cell变形过程
    # test_visualize_cell(cell, volumes, sizes, step_count, dt)

    # Test Object class
    test_object_class()

    end_time = time.time()
    print(f"\n所有测试完成，用时 {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    test_all()
