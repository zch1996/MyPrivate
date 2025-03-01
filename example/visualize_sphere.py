import pydem
from pydem import *
from pydem.src.utils import *

# Create a default material
mat = defaultMaterial()

# Create default engines
engines = defaultEngines(damping=0.1)

# Create a scene
if O.scene is None:
    O.scene = Scene()

field = DEMField()
field.gravity = Vector3r(0, 0, -10.0)

O.scene.setField(field)
O.scene.dt = 1e-3

for engine in engines:
    O.scene.addEngine(engine)

# Add different sized spheres
# Large sphere
sp1 = sphere(center=[0, 0, 0], radius=1.0, mat=mat, vel=[0, 0, 0])
O.scene.addParticle(sp1)

# Medium sized sphere
sp2 = sphere(center=[3, 0, 0], radius=0.5, mat=mat, vel=[0, 0, 0])
O.scene.addParticle(sp2)

# Small sphere
sp3 = sphere(center=[0, 3, 0], radius=0.1, mat=mat, vel=[0, 0, 0])
O.scene.addParticle(sp3)

# Very small sphere
sp4 = sphere(center=[3, 3, 0], radius=0.01, mat=mat, vel=[0, 0, 0])
O.scene.addParticle(sp4)

# Start visualization
renderer = O.startGL()

# Run simulation in background
O.scene.run(steps=1000, wait=False)

# Keep script running to maintain visualization
import time

print("Visualization running. Press Q to exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping simulation...")
    O.scene.stop()
    O.scene.wait()
    O.stopGL()
    print("Simulation stopped.")
