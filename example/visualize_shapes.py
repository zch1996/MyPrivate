import pydem
from pydem import *
from pydem.src.utils import *

# 创建默认材料
mat = defaultMaterial()

# 创建默认引擎
engines = defaultEngines(damping=0.1)

# 创建一个场景
if O.scene is None:
    O.scene = Scene()

field = DEMField()
field.gravity = Vector3r(0, 0, -10.0)

O.scene.setField(field)
O.scene.dt = 1e-3

for engine in engines:
    O.scene.addEngine(engine)

# 添加不同尺寸的球体
# 大球
sp1 = sphere(center=[0, 0, 0], radius=1.0, mat=mat, vel=[0, 0, 0])
O.scene.addParticle(sp1)

# 中等大小的球
sp2 = sphere(center=[3, 0, 0], radius=0.5, mat=mat, vel=[0, 0, 0])
O.scene.addParticle(sp2)

# # 小球
# sp3 = sphere(center=[0, 3, 0], radius=0.1, mat=mat, vel=[0, 0, 0])
# O.scene.addParticle(sp3)

# # 非常小的球
# sp4 = sphere(center=[3, 3, 0], radius=0.01, mat=mat, vel=[0, 0, 0])
# O.scene.addParticle(sp4)

# # 添加一个胶囊体
# cap = capsule(center=[0, 0, 3], radius=0.3, shaft=1.0, mat=mat)
# O.scene.addParticle(cap)

# # 添加一个椭球体
# ell = ellipsoid(center=[3, 0, 3], semiAxes=[0.5, 0.3, 0.2], mat=mat)
# O.scene.addParticle(ell)

# 启动可视化
renderer = O.startGL()

# 在后台运行模拟
O.scene.run(steps=1000, wait=False)

# 保持脚本运行以维持可视化
import time

print("可视化运行中。按 Q 退出，G 切换网格，W 切换线框，C 切换颜色模式，R 重置视图。")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("停止模拟...")
    O.scene.stop()
    O.scene.wait()
    O.stopGL()
    print("模拟已停止。")
