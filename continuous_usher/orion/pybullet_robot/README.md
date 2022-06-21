## Models

* CAR_v000: Fixed top motor model
* simple_mecanum: A middle cube with 4 mecanum wheel
* racecar_phy: PyBullet example racecar model



### Code

* fusion2urdf: Fusion360 exporter
* src: modularized code to run a vehicle (Specify wheel Id by a dictionary and set up their velocities separately  by PyBullet.)



## Pybullet Documentation and notable API

* PyBullet User Guide:  https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.jxof6bt5vhut 



### Get Started

1. Run corresponding `hello_bullet.py` to check if model is loaded correctly

2. Print joint Info to grab wheel motor jointID

 ```
for j in range(p.getNumJoints(car)):
    info = p.getJointInfo(car, j)
    print(info)
 ```

4. Play with p.setJointMotorControl2()

