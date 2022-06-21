```
    def _appendXY(self):
        robotPos, robotOri = p.getBasePositionAndOrientation(self.robotId)
        robotLinearVec, robotAngVEc = p.getBaseVelocity(
            self.robotId)  # [[xdot,ydot,zdot], [wx,wy,wz]]
        vec_tuple = self.getCurrentSignal()
        fric_tuple = self.getCurrentPhysicalParas()
        q = Quaternion(robotOri[3], robotOri[0], robotOri[1], robotOri[2])
        rP2 = robotPos[0:2]
        linear_xydot = robotLinearVec[0:2]
        angular_wz = robotAngVEc[2]

        dat_X = list(rP2) + [q.angle] + list(linear_xydot) + \
            [angular_wz] + list(vec_tuple) + list(fric_tuple)
        dat_Y = list(rP2) + [q.angle] + list(linear_xydot) + [angular_wz]
        self.X.append(dat_X)
        self.Y.append(dat_Y)
```
