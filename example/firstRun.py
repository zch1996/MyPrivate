import pydem
from pydem import *

from pydem.src.utils import *


# Compute SpherePWaveDt
print("SpherePWaveDt: ", spherePWaveDt(1e-3, 2400, 30e9))


# Create a default material
mat = defaultMaterial()


# Create default engines
engines = defaultEngines(damping=0.1)


# Create a sphere from utils

sp = sphere(center=[0, 0, 0], radius=1e-3, mat=mat, vel=[0, 0, 0])


print(sp)
print("CheckNodes: ", sp.checkNodes())
print("Real Contacts:", sp.countRealContacts())


############################ Scene ############################
if O.scene is None:
    O.scene = Scene()


field = DEMField()
field.gravity = Vector3r(0, 0, -10.0)

O.scene.setField(field)
O.scene.dt = 1e-3

for engine in engines:
    O.scene.addEngine(engine)

O.scene.addParticle(sp)
O.scene.run(steps=100)


print(sp.getPos())


# Check Background Thread
if O.scene.isBackgroundThread():
    print("Background Thread")
else:
    print("Main Thread")

# IMPORTANT: Properly finalize the scene before exiting
print("Stopping simulation...")
O.scene.stop()
O.scene.wait()  # Wait for background thread to terminate
O.scene.finalize()
print("Simulation stopped and finalized.")


Wall()
