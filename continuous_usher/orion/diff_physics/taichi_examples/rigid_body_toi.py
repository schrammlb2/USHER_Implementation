import pybullet as p
import pybullet_data
import time
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)
max_steps = 4096
steps = 204
assert(steps*2 <= max_steps)

scalar = lambda:ti.var(dt=real)
vec = lambda:ti.Vector(2,dt=real)
sphereRadius = 0.05

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

p.resetSimulation()
p.setGravity(0, 0, 0)

loss = scalar()
x = vec()
x_bullet = vec()
v = vec()
v_bullet = vec()

n_objects = 1
dt = 0.002
p.setTimeStep(dt)
useRealTimeSim = 0
p.setRealTimeSimulation(useRealTimeSim)

@ti.layout
def place():
    ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x,x_bullet,v,v_bullet)
    ti.root.place(loss)
    ti.root.lazy_grad()

@ti.kernel
def advance_one_time_step(t: ti.i32):
    old_v = v[t-1,0]
    new_v = v_bullet[t,0]
    old_x = x[t-1,0]
    new_x = x_bullet[t,0]
    toi=0.0
    if(new_x[1] < sphereRadius and new_v[1] < 0):
        new_v[1] = -new_v[1]
        toi = -(old_x[1] - sphereRadius)/old_v[1]
    v[t,0] = new_v
    x[t,0] = x[t-1,0] + toi*old_v + (dt-toi)*new_v

@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = x[t,0][1]

def forward_bullet():
    total_steps = steps
    for t in range(1,total_steps):
        p.stepSimulation()
        pos,orn=p.getBasePositionAndOrientation(1)
        linVel,angVel=p.getBaseVelocity(1)
        x_bullet[t,0]=[pos[1],pos[2]]
        v_bullet[t,0]=[linVel[1],linVel[2]]

def forward():
    total_steps = steps
    for t in range(1,total_steps):
        advance_one_time_step(t)
    loss[None]=0
    compute_loss(steps-1)

def main():
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id,-1,rollingFriction=0.,spinningFriction=0.,lateralFriction=0.,restitution=1,linearDamping=0.,angularDamping=0.)

    useMaximalCoordinates = True
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)
    colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=[sphereRadius, sphereRadius, sphereRadius])
    
    mass = 1
    visualShapeId = -1
    
    sphereUid = p.createMultiBody(mass,colSphereId,visualShapeId,[0.,0.7,0.5],useMaximalCoordinates=useMaximalCoordinates)
    p.changeDynamics(sphereUid,-1,ccdSweptSphereRadius=0.002)
    p.changeDynamics(sphereUid,-1,rollingFriction=0.,spinningFriction=0.,lateralFriction=0.,restitution=1,linearDamping=0.,angularDamping=0.)

    losses = []
    grads = []
    y_offsets = []
    ran = np.arange(0,0.3,0.02)
    for dy in ran:
        y_offsets.append(0.5+dy)
        x[0, 0] = [0.7, 0.5 + dy]
        v[0, 0] = [-1, -2]
        p.resetBasePositionAndOrientation(sphereUid,[0.,0.7,0.5+dy],[0,0,0,1])
        p.resetBaseVelocity(sphereUid,[0.,-1.,-2.],[0.,0.,0.])

        forward_bullet()
        with ti.Tape(loss):
            forward()

        print('dy=', dy, 'Loss=', loss[None])
        grads.append(x.grad[0, 0][1])
        losses.append(loss[None])

    plt.plot(y_offsets,losses,'.',label='Loss')
    plt.plot(y_offsets,grads,label='Gradient')
    plt.show()

if __name__ == '__main__':
    main()

#while(p.isConnected()):
#    time.sleep(1./240.)
#    p.stepSimulation()
