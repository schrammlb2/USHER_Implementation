01/02/2020

### btMultiBodyJointMotor

Check source code of `Joint Motor`

* [btMultiBodyJointMotor.h](https://github.com/bulletphysics/bullet3/blob/master/src/BulletDynamics/Featherstone/btMultiBodyJointMotor.h)
* [btMultiBodyJointMotor.cpp](https://github.com/bulletphysics/bullet3/blob/master/src/BulletDynamics/Featherstone/btMultiBodyJointMotor.cpp)

```
[106] void btMultiBodyJointMotor::createConstraintRows()
```

* Ctrl+ F Keyword
  * maxMotorImpulse 
  * m_maxAppliedImpulse
  * velocityError `btScalar velocityError = (m_desiredVelocity - currentVelocity);`
  * rhs `btScalar rhs = m_kp * positionStabiliationTerm + currentVelocity + m_kd * velocityError;`
  * fillMultiBodyConstraint: 
    * [btMultiBodyConstraint.h](https://github.com/bulletphysics/bullet3/blob/cdd56e46411527772711da5357c856a90ad9ea67/src/BulletDynamics/Featherstone/btMultiBodyConstraint.h) 
    * **[btMultiBodyConstraint.cpp](https://github.com/bulletphysics/bullet3/blob/cdd56e46411527772711da5357c856a90ad9ea67/src/BulletDynamics/Featherstone/btMultiBodyConstraint.cpp)**

### btMultiBodyConstraint
* [btMultiBodyConstraint.h](https://github.com/bulletphysics/bullet3/blob/cdd56e46411527772711da5357c856a90ad9ea67/src/BulletDynamics/Featherstone/btMultiBodyConstraint.h) 
* **[btMultiBodyConstraint.cpp](https://github.com/bulletphysics/bullet3/blob/cdd56e46411527772711da5357c856a90ad9ea67/src/BulletDynamics/Featherstone/btMultiBodyConstraint.cpp)**


---

01/01/2020

### Note about Physical parameters tunning

After a few days of parameters tunning on the physical parameters tunning, I feel like I might have to use `p. torch_control` to make the wheels turns following the pattern "Expected velocity" not equals to "Actual velocity" s.t. to get the robot turns.

The problem is, 

1. We want to set all the wheels with the same `targetVelocity` to get a turn pattern
2. And expect the pattern comes from different force and friction parameters
3. But if one of the wheel can reach the `targetVelocity`
4. Then, there is no way (that I failed to do so in my testing in Bullet), to reduce the speed of another 3 wheels to lower velocity to get the turning pattern
5. s.t. 1) is violated....

Possible solution: 

​	Use Torch control. Calculate actual velocity and torque applied to each wheel to get "Actual velocity" of each wheel.

---

I feel like main reason that cause target velocity is not the "Expected velocity" but the "Actual Velocity"

>  Note: the actual implementation of the joint motor controller is as a constraint for POSITION_CONTROL and VELOCITY_CONTROL, and as an external force for TORQUE_CONTROL

I do some parameter tunnings on `setJointMotor` and `changeDynamics`, and it looks like to me that makes parameters (`force, lateralFriction, rollingFriction,spinningFriction`) different among 4 wheels still cannot make car turns. Only changing `targetVelocity` with large value difference will make things works. However, changing this is not what we want......

Here is what happens: If we sent `targetVelocity` to Bullet, it will try its best to reach this velocity unless `force = 0`. The torque can comes from it's motor (No torque will be accelerated from motor if `force = 0`), the friction between ground and it's link, etc.  So in Bullet, after my testing, the `targetVelocity` is in fact not the "Expected velocity" or saying control signal we sent to the motor, but the "Actual velocity" as long as force is accelerated enough. 



---

### Tuning Notes

#### SetJointMotor (Tunning velocity and force)

1. All revolute joints and prismatic joints is initialized as  a velocity motor as follow

   ```python
   p.setJointMotorControl2(objUid, jointIndex, controlMode=p.VELOCITY_CONTROL, force=0)
   ```

   It means in initial state, no force is applied from the motor itself. It the wheel turns, the force/torque is in fact comes from the friction force from ground. `targetVelocity` does not matters if force = 0.

2. >  Note: the actual implementation of the joint motor controller is as a constraint for POSITION_CONTROL and VELOCITY_CONTROL, and as an external force for TORQUE_CONTROL

3. If force is set to value other than 0 with targetVelocity set as a specific value, the motor tend to maintain the velocity with applied maximum force.

   ```python
   maxForce = 1
   p.setJointMotorControl2(objUid, jointIndex, controlMode=p.VELOCITY_CONTROL,  targetVelocity = 0, force=maxForce)
   ```

   * Setting 1: Only one wheel motor is working. Car moves directly forward with velocity 1 (Assume force is large enough to get to that velocity)

   ```
   ## 0 is p.VELOCITY_CONTROL
   {2: {'controlMode': 0, 'targetVelocity': .1, 'force': 999},
   3: {'controlMode': 0, 'targetVelocity': 0, 'force': 0},
   5: {'controlMode': 0, 'targetVelocity': 0, 'force': 0},
   7: {'controlMode': 0, 'targetVelocity': 0, 'force': 0}},
   ## linear,_ = p.getBaseVelocity()
   ## print(linear[0]) = 0.0049 ~ 0.0051
   ```

   The whole car will move forward and try its best to maintain velocity 1 with max force 1 from wheel [2]. Other wheels will still moves because of friction.

   * Setting 2: All 4 wheel motor is working. Car moves directly forward with STILL with almost the same velocity.

   ```
   ## 0 is p.VELOCITY_CONTROL
   {2: {'controlMode': 0, 'targetVelocity': .1, 'force': 999},
   3: {'controlMode': 0, 'targetVelocity': .1, 'force': 999},
   5: {'controlMode': 0, 'targetVelocity': .1, 'force': 999},
   7: {'controlMode': 0, 'targetVelocity': .1, 'force': 999}},
   ## print(linear[0]) = 0.0050 ~ 0.0052
   ```

#### changeDynamics (Tunning lateralFriction, spinning Friction and rollingFriciton)

With motor setting

```python
# Four wheel with same velocity and maxforce (--------->Moving Forward)
## Motor Status
{2: {'controlMode': 0, 'targetVelocity': .1, 'force': 0.1},
3: {'controlMode': 0, 'targetVelocity': .1, 'force': 0.1},
5: {'controlMode': 0, 'targetVelocity': .1, 'force': 0.1},
7: {'controlMode': 0, 'targetVelocity': .1, 'force': 0.1}},
## Physical Para
{2: {'lateralFriction': .5, 'rollingFriction': 0, 'spinningFriction': 0},
3: {'lateralFriction': .5, 'rollingFriction': 0, 'spinningFriction': 0},
5: {'lateralFriction': .5, 'rollingFriction': 0, 'spinningFriction': 0},
7: {'lateralFriction': .5, 'rollingFriction': 0, 'spinningFriction': 0}}
```

1. Change lateralFriction from [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]: 
   * When 0, car will be slipping. [As expected]
   * Two wheel driven pattern
   * Slow start and "Fail to move" is in fact caused by `targetVelocity=0.1`, if it's 1, still can start quickly.

|      | force_BL | force_BR | force_FL | force_RL | lateral Friction | Start Moving from                |
| ---- | -------- | -------- | -------- | -------- | ---------------- | -------------------------------- |
| 0    | .1       | .1       | 0        | 0        | .6               | 869539                           |
| 1    | .1       | .1       | 0        | 0        | .5               | 14327                            |
| 2    | .1       | .1       | 0        | 0        | .4               | 18496                            |
| 3    | .1       | .1       | 0        | 0        | .3               | **Fail to move** even after 900k |
| 4    | .1       | .1       | 0        | 0        | .2               | 10804                            |
| 5    | .1       | .1       | 0        | 0        | .1               | 25690                            |

2. Rolling Friction

   Large value will cause crazy things like turn the car around with its back attach to the ground.

3. Spinning Friciton

   No exact difference