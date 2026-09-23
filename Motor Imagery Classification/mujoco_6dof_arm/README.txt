MuJoCo 6-DOF arm — colored home pose

Based on the previously working home-pose model supplied in this package.

Changes:
- Robot mesh geoms are gray.
- Table/floor is light off-white.
- Environment uses a blue-gray gradient skybox.
- Home keyframe is J1=0, J2=+90 degrees, J3=0, J4=0, J5=0, J6=0.
- Gripper visual meshes have a fixed -90 degree Z visual rotation to correct their home orientation.
- The six joint definitions and existing body positions are otherwise unchanged.

Run:
    python show_arm.py

The gripper visual rotation is applied at the geom level, so it does not alter joint6's axis/frame.
