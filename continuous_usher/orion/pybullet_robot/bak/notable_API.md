# Notable API

### Notes

1. Exceed (-100, 100) velocity limits (might have problems though). From https://github.com/bulletphysics/bullet3/issues/2513

   ```
   pybullet.changeDynamics(bodyUid, -1, maxJointVelocity=newvalue) 
   ```

2. By default, `loadURDF` not using inertia tensor from urdf file. https://github.com/bulletphysics/bullet3/issues/2260

   ```
   loadURDF(..., flags=p.URDF_USE_INERTIA_FROM_FILE)
   
   URDF_USE_INERTIA_FROM_FILE
   URDF_USE_IMPLICIT_CYLINDER
   ```

3. 



###  setJointMotorControl2  (P20)

We can control a robot by setting a desired control mode for one or more joint motors. During the stepSimulation the physics engine will simulate the motors to reach the given target value that can be reached within the maximum motor forces and other constraints. 

Important Note: by default, each revolute joint and prismatic joint is motorized using a velocity motor. You can disable those default motor by using a maximum force of 0. This will let you perform torque control. 

For example:

```python
maxForce = 0
mode = p.VELOCITY_CONTROL
p.setJointMotorControl2(objUid, jointIndex,
 	controlMode=mode, force=maxForce)
 	
## You can also use a small non-zero force to mimic joint friction.
```

 If you want a wheel to maintain a constant velocity, with a max force you can use: 

```python
maxForce = 500
p.setJointMotorControl2(bodyUniqueId=objUid, 
jointIndex=0, 
controlMode=p.VELOCITY_CONTROL,
targetVelocity = targetVel,
force = maxForce)
```

Input arguments

| required | bodyUniqueId    | int   | body unique id as returned from loadURDF etc.                |
| -------- | --------------- | ----- | ------------------------------------------------------------ |
| required | jointIndex      | int   | link index in  range [0..getNumJoints(bodyUniqueId) (note that link index == joint index) |
| required | controlMode     | int   | POSITION_CONTROL  (which is in fact CONTROL_MODE_POSITION_VELOCITY_PD), VELOCITY_CONTROL,  TORQUE_CONTROL and PD_CONTROL. |
| optional | targetPosition  | float | in  POSITION_CONTROL the targetValue is target position of the joint |
| optional | targetVelocity  | float | in  VELOCITY_CONTROL and POSITION_CONTROL the targetVelocity is the desired  velocity of the joint, see implementation note below. Note that the  targetVelocity is not the maximum joint velocity. In PD_CONTROL and  POSITION_CONTROL/CONTROL_MODE_POSITION_VELOCITY_PD, the final target velocity  is computed  using:kp*(erp*(desiredPosition-currentPosition)/dt)+currentVelocity+kd*(m_desiredVelocity  - currentVelocity). See also examples/pybullet/examples/pdControl.py |
| optional | force           | float | in POSITION_CONTROL and VELOCITY_CONTROL this is  the maximum motor force used to reach the target value. In TORQUE_CONTROL  this is the force/torque to be applied each simulation step. |
| optional | positionGain    | float | See  implementation note below                               |
| optional | velocityGain    | float | See  implementation note below                               |
| optional | maxVelocity     | float | in  POSITION_CONTROL this limits the velocity to a maximum   |
| optional | physicsClientId | int   | if you are  connected to multiple servers, you can pick one. |

 Note: the actual implementation of the joint motor controller is as a constraint for POSITION_CONTROL and VELOCITY_CONTROL, and as an external force for TORQUE_CONTROL: 

| method           | implementation | component                         | constraint error to be minimized                             |
| ---------------- | -------------- | --------------------------------- | ------------------------------------------------------------ |
| POSITION_CONTROL | constraint     | velocity and  position constraint | error =  position_gain*(desired_position-actual_position)+velocity_gain*(desired_velocity-actual_velocity) |
| VELOCITY_CONTROL | constraint     | pure velocity  constraint         | error =  desired_velocity - actual_velocity                  |
| TORQUE_CONTROL   | external force |                                   |                                                              |

Generally it is best to start with VELOCITY_CONTROL or POSITION_CONTROL. It is much harder to do TORQUE_CONTROL (force control) since simulating the correct forces relies on very accurate URDF/SDF file parameters and system identification (correct masses, inertias, center of mass location, joint friction etc).

###  getDynamicsInfo(P30)

You can get information about the mass, center of mass, friction and other properties of the base and links.

The input parameters to getDynamicsInfo are:

| required | bodyUniqueId    | int  | object unique id, as returned by loadURDF etc.               |
| -------- | --------------- | ---- | ------------------------------------------------------------ |
| required | linkIndex       | int  | link (joint)  index or -1 for the base.                      |
| optional | physicsClientId | int  | if you are  connected to multiple servers, you can pick one. |

The return information is limited, we will expose more information when we need it:

| mass                    | double                  | mass in kg                                                   |
| ----------------------- | ----------------------- | ------------------------------------------------------------ |
| lateral_friction        | double                  | friction  coefficient                                        |
| local  inertia diagonal | vec3, list of 3  floats | local inertia  diagonal. Note that links and base are centered around the center of mass and  aligned with the principal axes of inertia. |
| local  inertial pos     | vec3                    | position of  inertial frame in local coordinates of the joint frame |
| local  inertial orn     | vec4                    | orientation of  inertial frame in local coordinates of joint frame |
| restitution             | double                  | coefficient of  restitution                                  |
| mass                    | double                  | mass in kg                                                   |
| lateral_friction        | double                  | friction  coefficient                                        |
| local  inertia diagonal | vec3, list of 3  floats | local inertia  diagonal. Note that links and base are centered around the center of mass and  aligned with the principal axes of inertia. |
| local  inertial pos     | vec3                    | position of  inertial frame in local coordinates of the joint frame |
| local  inertial orn     | vec4                    | orientation of  inertial frame in local coordinates of joint frame |
| restitution             | double                  | coefficient of  restitution                                  |
| rolling  friction       | double                  | rolling friction  coefficient orthogonal to contact normal   |
| spinning  friction      | double                  | spinning  friction coefficient around contact normal         |
| contact  damping        | double                  | -1 if not  available. damping of contact constraints.        |
| contact  stiffness      | double                  | -1 if not  available. stiffness of contact constraints.      |

### changeDynamics  (P32)

You can change the properties such as mass, friction and restitution coefficients using changeDynamics.
The input parameters are:

| required | bodyUniqueId         | int    | object unique id, as returned by loadURDF etc.               |
| -------- | -------------------- | ------ | ------------------------------------------------------------ |
| required | linkIndex            | int    | link index or -1  for the base                               |
| optional | mass                 | double | change the mass  of the link (or base for linkIndex -1)      |
| optional | lateralFriction      | double | lateral (linear)  contact friction                           |
| optional | spinningFriction     | double | torsional  friction around the contact normal                |
| optional | rollingFriction      | double | torsional  friction orthogonal to contact normal             |
| optional | restitution          | double | bouncyness of  contact. Keep it a bit less than 1.           |
| optional | physicsClientId      | int    | if you are  connected to multiple servers, you can pick one. |
| optional | linearDamping        | double | linear damping  of the link (0.04 by default)                |
| optional | angularDamping       | double | angular damping  of the link (0.04 by default)               |
| optional | contactStiffness     | double | stiffness of the  contact constraints, used together with contactDamping. |
| optional | contactDamping       | double | damping of the  contact constraints for this body/link. Used together with contactStiffness.  This overrides the value if it was specified in the URDF file in the contact  section. |
| optional | frictionAnchor       | int    | enable or  disable a friction anchor: positional friction correction (disabled by  default, unless set in the URDF contact section) |
| optional | localInertiaDiagnoal | vec3   | diagonal  elements of the inertia tensor. Note that the base and links are centered  around the center of mass and aligned with the principal axes of inertia so  there are no off-diagonal elements in the inertia tensor. |
| optional | jointDamping         | double | Joint damping  coefficient applied at each joint. This coefficient is read from URDF joint  damping field. Keep the value close to 0.Joint damping force =  -damping_coefficient * joint_velocity. |