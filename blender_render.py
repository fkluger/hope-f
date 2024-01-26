import bpy 
import os
import glob
import numpy as np
import random
import mathutils

mesh_folder = "./meshes"
render_folder = "./render"

start = 0
num_images = 2000

obj_files = glob.glob(os.path.join(mesh_folder, "*.obj"))

def deleteAllObjects():
    for o in bpy.context.scene.objects:
        o.select_set(True)
    bpy.ops.object.delete() 

def object_lowest_point(obj):
    minz = 999999.0

    for vertex in obj.data.vertices:
        v_world = obj.matrix_world @ mathutils.Vector((vertex.co[0],vertex.co[1],vertex.co[2], 1))

        if v_world[2] < minz:
            minz = v_world[2]
    return minz

def get_calibration_matrix_K_from_blender(cam):
    camd = cam.data
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    print(sensor_height_in_mm, sensor_width_in_mm)
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0
    K = np.array([
        [alpha_u, skew,    u_0],
        [    0  , alpha_v, v_0],
        [    0  , 0,        1 ]])
    return K


for num_objects in range(4):
    for render_id in range(start, num_images):

        target_dir = os.path.join(render_folder, "%d" % num_objects, "%06d" % render_id)
        os.makedirs(target_dir, exist_ok=True)

        random.shuffle(obj_files)

        deleteAllObjects()

        light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
        light_height = np.random.uniform(2, 10)
        light_data.energy = np.random.uniform(300, 600) * light_height
        light_object = bpy.data.objects.new(name="my-light", object_data=light_data)
        bpy.context.collection.objects.link(light_object)

        focal_length = np.random.uniform(24, 40)
        cam_rot_x = np.random.uniform(25, 65)
        cam_rot_y = np.random.uniform(90, 270)

        cam_distance = np.random.uniform(3, 5)
        t = np.array([0, 0, -cam_distance])

        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0), rotation=(cam_rot_x * np.pi/180, 0, cam_rot_y * np.pi/180))
        cam = bpy.context.object
        cam.data.lens = focal_length
        _, quat, _ = cam.matrix_world.decompose()
        R = quat.to_matrix()
        C = -np.array(R) @ t

        cam.location = C
        cam.data.sensor_width = 30
        cam.data.sensor_height = 30

        light_object.location = (np.random.uniform(-1, 1) + C[0], np.random.uniform(-1, 1) + C[1], light_height)

        K = get_calibration_matrix_K_from_blender(cam)

        R_cam = np.array(R)
        t_cam = t.copy()

        np.savez(os.path.join(target_dir, "camera_parameters.npz"), K=K, R=R_cam, t=t_cam)

        for scene in bpy.data.scenes:
            scene.render.resolution_x = 1024
            scene.render.resolution_y = 1024
            scene.camera = cam

        bpy.ops.mesh.primitive_plane_add(size=100)

        bl = 0.5
        scale = 0.007

        obj_rotations = [[], []]
        obj_translations = [[], []]
        obj_names = []

        for obj_id in range(num_objects):
            obj_file = obj_files[obj_id].split("/")
            obj_name = obj_file[-1]
            obj_names += [obj_name]

        for ri in range(2):

            base_positions = [
                [-bl, -bl, 0],
                [bl, -bl, 0],
                [-bl, bl, 0],
                [bl, bl, 0]
            ]

            random.shuffle(base_positions)

            objects = []

            for obj_id in range(num_objects):
                obj_file = obj_files[obj_id]

                base_pos = base_positions[obj_id]

                offset = np.random.uniform(-bl/2, bl/2, size=2)

                bpy.ops.import_scene.obj(filepath=obj_file)

                obj = bpy.context.scene.objects[-1]
                obj.scale = np.array([1,1,1])*scale
                t = np.array([base_pos[0]+offset[0], base_pos[1]+offset[1], -object_lowest_point(obj)*scale])
                obj.location = t
                obj.rotation_euler.z = np.random.uniform(-10, 10) * np.pi/180.0

                rot = obj.matrix_world.to_euler('XYZ')
                obj_rotations[ri] += [-np.array(obj.rotation_euler.z)]
                obj_translations[ri] += [np.array(t)]

                objects += [obj]

            scene.render.image_settings.file_format = 'PNG'
            scene.render.engine = 'CYCLES'
            scene.render.filepath = os.path.join(target_dir, "render%d.png" % ri)
            bpy.ops.render.render(write_still = 1)

            for o in bpy.context.scene.objects:
                o.select_set(False)
            for o in objects:
                o.select_set(True)
            bpy.ops.object.delete()

        for obj_id in range(num_objects):
            np.savez(os.path.join(target_dir, "object_%d.npz" % obj_id), rot_euler_0=obj_rotations[0][obj_id], rot_euler_1=obj_rotations[1][obj_id],
                t_0=obj_translations[0][obj_id], t_1=obj_translations[1][obj_id], id=obj_names[obj_id])