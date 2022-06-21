import os
import pybullet as p
import pybullet_data
import numpy as np

cid = p.connect(p.SHARED_MEMORY)
if(cid < 0):
    p.connect(p.DIRECT)

p.resetSimulation()
p.setGravity(0, 0, -10)

useRealTimeSim = 0
p.setRealTimeSimulation(useRealTimeSim)
p.loadSDF(os.path.join(pybullet_data.getDataPath(), "stadium.sdf"))

car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"))
p.changeDynamics(car, -1, linearDamping=0, angularDamping=0)
dt = 0.01
p.setTimeStep(dt)
for i in range(p.getNumJoints(car)):
    p.changeDynamics(car, i, linearDamping=0, angularDamping=0)

inactive_wheels = [3, 5, 7]
wheels = [2]

for wheel in inactive_wheels:
    p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

steering = [4, 6]
state_lines = []

with open('state.txt','r') as f_state:
    for line in f_state.readlines():
        state_lines.append(line.rstrip('\n'))

value = 0
with open('control.txt','r') as f_control:
    for idx,line in enumerate(f_control.readlines()):
        line=line.rstrip('\n')
        c_list=[float(i) for i in line.split(' ')]
        s_list=[float(i) for i in state_lines[idx].split(' ')]
        target_time=c_list[0]
        targetVelocity=c_list[1]
        steeringAngle=c_list[2]
        maxForce=c_list[3]

        for i in range(p.getNumBodies()):
            position=[s_list[13*i+1],s_list[13*i+2],s_list[13*i+3]]
            orientation=[s_list[13*i+4],s_list[13*i+5],s_list[13*i+6],s_list[13*i+7]]
            lin_vel=[s_list[13*i+8],s_list[13*i+9],s_list[13*i+10]]
            ang_vel=[s_list[13*i+11],s_list[13*i+12],s_list[13*i+13]]

            p.resetBasePositionAndOrientation(i,position,orientation)
            p.resetBaseVelocity(i,lin_vel,ang_vel)

        for wheel in wheels:
            p.setJointMotorControl2(car,wheel,p.VELOCITY_CONTROL,targetVelocity=targetVelocity,force=maxForce)

        for steer in steering:
            p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)

        p.stepSimulation()

        if(target_time<10):
            s_list=[float(i) for i in state_lines[idx+1].split(' ')]
            for i in range(p.getNumBodies()):
                pos,orn=p.getBasePositionAndOrientation(i)
                linVel,angVel=p.getBaseVelocity(i)
                position=[s_list[13*i+1],s_list[13*i+2],s_list[13*i+3]]
                orientation=[s_list[13*i+4],s_list[13*i+5],s_list[13*i+6],s_list[13*i+7]]
                lin_vel=[s_list[13*i+8],s_list[13*i+9],s_list[13*i+10]]
                ang_vel=[s_list[13*i+11],s_list[13*i+12],s_list[13*i+13]]

                value+=np.linalg.norm(np.asarray(pos)-np.asarray(position),2)
                value+=np.linalg.norm(np.asarray(orn)-np.asarray(orientation),2)
                #assert(np.linalg.norm(np.asarray(pos)-np.asarray(position),2)<1e-2)
                #assert(np.linalg.norm(np.asarray(orn)-np.asarray(orientation),2)<1e-2)
print('Value: ',value)
