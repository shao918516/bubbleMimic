import mujoco

# model = mujoco.MjModel.from_xml_path("resources/dex_v3/urdf/dex_v3_mujoco.urdf")

# mujoco.mj_saveLastXML("resources/dex_v3/urdf/dex_v3_mujoco_v2.xml",model)

model = mujoco.MjModel.from_xml_path("resources/tiangong3/urdf/tiangong3_mujoco.urdf")

mujoco.mj_saveLastXML("resources/tiangong3/urdf/tiangong3.xml",model)
