import numpy as np
import pandas as pd
from scipy.spatial.transform import *

# Global 5.30389	-8.0065	-5.60693	0.774617	0.881389	0.271791
hipRot = [5.30389, -8.0065, -5.60693]
hipPos = [0.774617, 0.881389, 0.271791]

chestRot = [8.552199, -6.29635, -3.388422]
chestPos = [0.792099, 1.134489, 0.231872]

# Local Chest Rot   22.002422	-0.007569	1.247368
hipRot_L = [5.30389, -8.0065, -5.60693]
hipPos_L = [0.774617, 0.881389, 0.271791]

# abRot_L = [-19.750607, 0.322335, 4.693333]
chestRot_L = [22.855778, -0.113972, -2.440463]

rotate_array = np.array(hipRot)
hip_origin = np.array(hipPos)
r = Rotation.from_euler('XYZ', [rotate_array[0], rotate_array[1], rotate_array[2]], degrees=True)

body_part_origin = np.array(chestPos)
new_position = r.inv().as_matrix() @ (body_part_origin - hip_origin)
# new_position = r.inv() @ (body_part_origin - hip_origin)
print(new_position)
# temp_body_part = ":".join(body_part.split(":")[0:1] + body_part.split(":")[2:])
# new_df[temp_body_part + "Position:X"][index] = new_position[0]
# new_df[temp_body_part + "Position:Y"][index] = new_position[1]
# new_df[temp_body_part + "Position:Z"][index] = new_position[2]
