# camera-ready
import open3d as o3d
import pickle
import numpy as np
import math
import cv2

import sys


def proj_to_img(corners_3d, K, T):
    """ Project 3D points onto the image plane using intrinsic and extrinsic parameters. """
    # Transform corners to camera frame using the extrinsic matrix
    ones = np.ones((corners_3d.shape[0], 1))
    homogeneous_corners = np.hstack([corners_3d, ones])
    camera_frame_corners = T.dot(homogeneous_corners.T)

    # Project onto the image plane using the intrinsic matrix
    projected_points = K.dot(camera_frame_corners[:3, :])
    projected_points /= projected_points[2, :]  # Normalize by the third (depth) coordinate
    return projected_points[:2, :].T  # Return only x and y coordinates


def create3Dbbox(center, h, w, l, r_y, type="pred"):
    if type == "pred":
        color = [1, 0.75, 0] # (normalized RGB)
        front_color = [1, 0, 0] # (normalized RGB)
    else: # (if type == "gt":)
        color = [1, 0, 0.75] # (normalized RGB)
        front_color = [0, 0.9, 1] # (normalized RGB)

    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    Rmat_90 = np.asarray([[math.cos(r_y+np.pi/2), 0, math.sin(r_y+np.pi/2)],
                          [0, 1, 0],
                          [-math.sin(r_y+np.pi/2), 0, math.cos(r_y+np.pi/2)]],
                          dtype='float32')

    Rmat_90_x = np.asarray([[1, 0, 0],
                            [0, math.cos(np.pi/2), math.sin(np.pi/2)],
                            [0, -math.sin(np.pi/2), math.cos(np.pi/2)]],
                            dtype='float32')

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())

    p0_3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, 0], dtype='float32').flatten())
    p1_2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, 0], dtype='float32').flatten())
    p4_7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, 0], dtype='float32').flatten())
    p5_6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, 0], dtype='float32').flatten())
    p0_1 = center + np.dot(Rmat, np.asarray([0, 0, w/2.0], dtype='float32').flatten())
    p3_2 = center + np.dot(Rmat, np.asarray([0, 0, -w/2.0], dtype='float32').flatten())
    p4_5 = center + np.dot(Rmat, np.asarray([0, -h, w/2.0], dtype='float32').flatten())
    p7_6 = center + np.dot(Rmat, np.asarray([0, -h, -w/2.0], dtype='float32').flatten())
    p0_4 = center + np.dot(Rmat, np.asarray([l/2.0, -h/2.0, w/2.0], dtype='float32').flatten())
    p3_7 = center + np.dot(Rmat, np.asarray([l/2.0, -h/2.0, -w/2.0], dtype='float32').flatten())
    p1_5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h/2.0, w/2.0], dtype='float32').flatten())
    p2_6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h/2.0, -w/2.0], dtype='float32').flatten())
    p0_1_3_2 = center

    length_0_3 = np.linalg.norm(p0 - p3)
    cylinder_0_3 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_0_3)
    cylinder_0_3.compute_vertex_normals()
    transform_0_3 = np.eye(4)
    transform_0_3[0:3, 0:3] = Rmat
    transform_0_3[0:3, 3] = p0_3
    cylinder_0_3.transform(transform_0_3)
    cylinder_0_3.paint_uniform_color(front_color)

    length_1_2 = np.linalg.norm(p1 - p2)
    cylinder_1_2 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_1_2)
    cylinder_1_2.compute_vertex_normals()
    transform_1_2 = np.eye(4)
    transform_1_2[0:3, 0:3] = Rmat
    transform_1_2[0:3, 3] = p1_2
    cylinder_1_2.transform(transform_1_2)
    cylinder_1_2.paint_uniform_color(color)

    length_4_7 = np.linalg.norm(p4 - p7)
    cylinder_4_7 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_4_7)
    cylinder_4_7.compute_vertex_normals()
    transform_4_7 = np.eye(4)
    transform_4_7[0:3, 0:3] = Rmat
    transform_4_7[0:3, 3] = p4_7
    cylinder_4_7.transform(transform_4_7)
    cylinder_4_7.paint_uniform_color(front_color)

    length_5_6 = np.linalg.norm(p5 - p6)
    cylinder_5_6 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_5_6)
    cylinder_5_6.compute_vertex_normals()
    transform_5_6 = np.eye(4)
    transform_5_6[0:3, 0:3] = Rmat
    transform_5_6[0:3, 3] = p5_6
    cylinder_5_6.transform(transform_5_6)
    cylinder_5_6.paint_uniform_color(color)

    # #

    length_0_1 = np.linalg.norm(p0 - p1)
    cylinder_0_1 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_0_1)
    cylinder_0_1.compute_vertex_normals()
    transform_0_1 = np.eye(4)
    transform_0_1[0:3, 0:3] = Rmat_90
    transform_0_1[0:3, 3] = p0_1
    cylinder_0_1.transform(transform_0_1)
    cylinder_0_1.paint_uniform_color(color)

    length_3_2 = np.linalg.norm(p3 - p2)
    cylinder_3_2 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_3_2)
    cylinder_3_2.compute_vertex_normals()
    transform_3_2 = np.eye(4)
    transform_3_2[0:3, 0:3] = Rmat_90
    transform_3_2[0:3, 3] = p3_2
    cylinder_3_2.transform(transform_3_2)
    cylinder_3_2.paint_uniform_color(color)

    length_4_5 = np.linalg.norm(p4 - p5)
    cylinder_4_5 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_4_5)
    cylinder_4_5.compute_vertex_normals()
    transform_4_5 = np.eye(4)
    transform_4_5[0:3, 0:3] = Rmat_90
    transform_4_5[0:3, 3] = p4_5
    cylinder_4_5.transform(transform_4_5)
    cylinder_4_5.paint_uniform_color(color)

    length_7_6 = np.linalg.norm(p7 - p6)
    cylinder_7_6 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_7_6)
    cylinder_7_6.compute_vertex_normals()
    transform_7_6 = np.eye(4)
    transform_7_6[0:3, 0:3] = Rmat_90
    transform_7_6[0:3, 3] = p7_6
    cylinder_7_6.transform(transform_7_6)
    cylinder_7_6.paint_uniform_color(color)

    # #

    length_0_4 = np.linalg.norm(p0 - p4)
    cylinder_0_4 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_0_4)
    cylinder_0_4.compute_vertex_normals()
    transform_0_4 = np.eye(4)
    transform_0_4[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_0_4[0:3, 3] = p0_4
    cylinder_0_4.transform(transform_0_4)
    cylinder_0_4.paint_uniform_color(front_color)

    length_3_7 = np.linalg.norm(p3 - p7)
    cylinder_3_7 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_3_7)
    cylinder_3_7.compute_vertex_normals()
    transform_3_7 = np.eye(4)
    transform_3_7[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_3_7[0:3, 3] = p3_7
    cylinder_3_7.transform(transform_3_7)
    cylinder_3_7.paint_uniform_color(front_color)

    length_1_5 = np.linalg.norm(p1 - p5)
    cylinder_1_5 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_1_5)
    cylinder_1_5.compute_vertex_normals()
    transform_1_5 = np.eye(4)
    transform_1_5[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_1_5[0:3, 3] = p1_5
    cylinder_1_5.transform(transform_1_5)
    cylinder_1_5.paint_uniform_color(color)

    length_2_6 = np.linalg.norm(p2 - p6)
    cylinder_2_6 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_2_6)
    cylinder_2_6.compute_vertex_normals()
    transform_2_6 = np.eye(4)
    transform_2_6[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_2_6[0:3, 3] = p2_6
    cylinder_2_6.transform(transform_2_6)
    cylinder_2_6.paint_uniform_color(color)

    # #

    length_0_1_3_2 = np.linalg.norm(p0_1 - p3_2)
    cylinder_0_1_3_2 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_0_1_3_2)
    cylinder_0_1_3_2.compute_vertex_normals()
    transform_0_1_3_2 = np.eye(4)
    transform_0_1_3_2[0:3, 0:3] = Rmat
    transform_0_1_3_2[0:3, 3] = p0_1_3_2
    cylinder_0_1_3_2.transform(transform_0_1_3_2)
    cylinder_0_1_3_2.paint_uniform_color(color)

    return [cylinder_0_1_3_2, cylinder_0_3, cylinder_1_2, cylinder_4_7, cylinder_5_6, cylinder_0_1, cylinder_3_2, cylinder_4_5, cylinder_7_6, cylinder_0_4, cylinder_3_7, cylinder_1_5, cylinder_2_6]


def conv3Dbbox_poly(center, h, w, l, r_y):
    # Rotation by Z axis 
    Rmat = np.asarray([[math.cos(r_y), -math.sin(r_y), 0],\
                       [math.sin(r_y), math.cos(r_y), 0],\
                       [0,             0,              1]], \
                       dtype='float32')
    '''
        describe eight corners 
        p1-----p5
        /|     /|
    / |    / |
    p2-p0--p6 p4
    | /    | /
    |/     |/
    p3-----p7

    '''

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, w/2.0, -h/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([l/2.0, w/2.0, h/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, w/2.0, h/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([-l/2.0, w/2.0, -h/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -w/2.0, -h/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([l/2.0, -w/2.0, h/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -w/2.0, h/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([-l/2.0, -w/2.0, -h/2.0], dtype='float32').flatten())
    p = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    return p 

# https://en.wikipedia.org/wiki/Rotation_matrix
# https://github.com/fregu856/3DOD_thesis/blob/master/visualization/visualize_eval_val.py
def create3Dbbox_poly(center, h, w, l, r_y, type, pred_bbox):
    if type == "pred":
        color = [0, 1, 0] # (RGB)
        front_color = [1, 0, 0] # (RGB)
  
    p = conv3Dbbox_poly(center, h, w, l, r_y)
    lines = [[0, 1], [1, 5], [0, 4], [4, 5], [1, 2], [2, 6], [5, 6],
        [0, 3], [4, 7], [3, 7], [2, 3],[6, 7]]

    colors = [front_color] * 4 + [color] * 8
    # colors = [[1, 0, 0] for i in range(len(lines))]
    
    # line_set = o3d.geometry.LineSet() 
    # pred_bbox.points = o3d.utility.Vector3dVector(np.array([p0, p1, p2, p3, p4, p5, p6, p7]))
    pred_bbox.points = o3d.utility.Vector3dVector(p)
    pred_bbox.lines = o3d.utility.Vector2iVector(lines) 
    pred_bbox.colors = o3d.utility.Vector3dVector(colors)
    # return line_set

def draw_3d_polys(img, polys):
    img = np.copy(img)
    for poly in polys:
        for n, line in enumerate(poly['lines']):
            if 'colors' in poly:
                bg = poly['colors'][n]
            else:
                bg = np.array([255, 0, 0], dtype='float64')

            p3d = np.vstack((poly['points'][line].T, np.ones((1, poly['points'][line].shape[0]))))
            p2d = np.dot(poly['P0_mat'], p3d)

            for m, p in enumerate(p2d[2, :]):
                p2d[:, m] = p2d[:, m]/p

            cv2.polylines(img, np.int32([p2d[:2, :].T]), False, bg, lineType=cv2.LINE_AA, thickness=2)

    return img

