import os
import pybullet as p
import pybullet_data
import time

cid = p.connect(p.SHARED_MEMORY)
if(cid < 0):
    p.connect(p.GUI)

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

#targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -1, 1, 0)
maxForceSlider = p.addUserDebugParameter("maxForce", 0, 10, 10)
#steeringSlider = p.addUserDebugParameter("steering", -0.5, 0.5, 0)

target_time=0
targetVelocity=0
steeringAngle=0

f_state = open('state.txt','w')
f_control = open('control.txt','w')

while (target_time<10):
    maxForce = p.readUserDebugParameter(maxForceSlider)

    f_control.write('%f %f %f %f\n'%(target_time,targetVelocity,steeringAngle,maxForce))
    f_state.write('%f'%target_time)
    for i in range(p.getNumBodies()):
        pos,orn=p.getBasePositionAndOrientation(i)
        linVel,angVel=p.getBaseVelocity(i)
        f_state.write(' %f %f %f'%(pos[0],pos[1],pos[2]))
        f_state.write(' %f %f %f %f'%(orn[0],orn[1],orn[2],orn[3]))
        f_state.write(' %f %f %f'%(linVel[0],linVel[1],linVel[2]))
        f_state.write(' %f %f %f'%(angVel[0],angVel[1],angVel[2]))
    f_state.write('\n')

    if(target_time<2): targetVelocity=.5
    elif(target_time>=2 and target_time<4): steeringAngle=.25
    elif(target_time>=4 and target_time<6): targetVelocity=1
    elif(target_time>=6 and target_time<8): steeringAngle=.35

    for wheel in wheels:
        p.setJointMotorControl2(car,wheel,p.VELOCITY_CONTROL,targetVelocity=targetVelocity,force=maxForce)

    for steer in steering:
        p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)

    p.stepSimulation()
    target_time+=dt
    #time.sleep(0.01)

f_state.close()
f_control.close()
