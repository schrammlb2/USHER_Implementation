1. By default, `loadURDF` not using inertia tensor from urdf file. https://github.com/bulletphysics/bullet3/issues/2260

   ```
   loadURDF(..., flags=p.URDF_USE_INERTIA_FROM_FILE)
   
   URDF_USE_INERTIA_FROM_FILE
   URDF_USE_IMPLICIT_CYLINDER
   ```

2. If using `changeDynamics` to change mass, even though the above tag is set, the inertia tensor will still be recomputed

3. Parse `localInertiaDiagnoal`  into `p.changeDynamics()`failed. (TypeError?)

4. Looks like changing mass didn't affect the car's speed.............