import open3d as o3d
import numpy as np
import copy
import time
import os


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def color(pc,num):
    import random
    import webcolors
    color_list = random.sample(list(webcolors.CSS3_NAMES_TO_HEX.values()), num)
    if len(color_list) < num:
        num = len(color_list)
    num = random.randint(0, num-1)
    color_rgb = webcolors.hex_to_rgb(color_list[num])
    color_float = [x/255 for x in color_rgb]
    pc.paint_uniform_color(color_float)
    return pc

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 7.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)

def fgr_registration(src_down, dst_down, src_fpfh, dst_fpfh):

    dist_threshold= 0.7
    max_iter=256
    tuple_scl=0.5
    max_tuple=1500

    fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                src_down, dst_down, src_fpfh, dst_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(use_absolute_scale=0,
                    decrease_mu = True,
                    maximum_correspondence_distance=dist_threshold,
                    iteration_number=max_iter,
                    tuple_scale=tuple_scl,
                    maximum_tuple_count=max_tuple,
                    tuple_test = False))

    trans = fgr.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(src_down, dst_down, 
                dist_threshold,fgr.transformation)
    return fgr, trans, info

def ransac_registration(src_down, dst_down, src_fpfh, dst_fpfh):
    distance_threshold = 0.7
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, dst_down, src_fpfh, dst_fpfh, False,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999))
    trans = ransac.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(src_down, dst_down, 
                distance_threshold,ransac.transformation)
    return ransac, trans, info

def robust_icp_registration(src_down,dst_down):
    threshold = 1.0
    mu, sigma = 0, 0.1
    source_noisy = apply_noise(src_down, mu, sigma)
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma)

    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_noisy,dst_down, threshold, np.identity(4), p2l)
    trans = reg_p2l.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(src_down, dst_down, 
                threshold,reg_p2l.transformation)
    return reg_p2l, trans, info

def gicp_registration(src_down, dst_down):
    threshold = 0.2
    loss = o3d.pipelines.registration.HuberLoss(k=0.1)
    init = np.identity(4)

    # robust_kernel = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP.LossFunction.Huber
    estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(loss)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30)
    gicp = o3d.pipelines.registration.registration_generalized_icp(
        src_down, dst_down,threshold,init,estimation,criteria)

    trans = gicp.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(src_down, dst_down, 
                threshold,gicp.transformation)
    return gicp, trans, info

def cficp_registration(src_down, dst_down):
    max_correspondence_distance_coarse = 0.5
    max_correspondence_distance_fine = 0.06


    icp_coarse = o3d.pipelines.registration.registration_icp(
        src_down, dst_down, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    icp_result = o3d.pipelines.registration.registration_icp(
        src_down, dst_down, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    trans = icp_result.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(src_down, dst_down, 
                            max_correspondence_distance_fine,icp_result.transformation)
    return icp_result, trans, info



def full_registration(init_regis,pcds):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds): 
                voxel_size=0.05

                source_path = fr"test_data/medium_real/{pcd_list[source_id]}"
                target_path = fr"test_data/medium_real/{pcd_list[target_id]}"
                source = o3d.io.read_point_cloud(source_path)
                target = o3d.io.read_point_cloud(target_path)
                source, source_fpfh = preprocess_point_cloud(source, voxel_size)
                target, target_fpfh = preprocess_point_cloud(target, voxel_size)

                # Initial Registration
                if init_regis=="fgr":
                    future = executor.submit(fgr_registration(source,target,source_fpfh,target_fpfh))
                    future.result()
                elif init_regis=="ransac":
                    future = executor.submit(ransac_registration(source,target,source_fpfh,target_fpfh))
                    future.result()
                elif init_regis=="robust":
                    future = executor.submit(robust_icp_registration(source,target))
                    future.result()
                elif init_regis=="gicp":
                    future = executor.submit(gicp_registration(source,target))
                    future.result()
                else:
                    future = executor.submit(result=cficp_registration(source,target))
                    future.result()
                
                init_trans=result[1]
                init_info=result[2]
                print("Build o3d.pipelines.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(init_trans, odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                init_trans,
                                                                init_info,
                                                                uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                init_trans,
                                                                init_info,
                                                                uncertain=True))
    return pose_graph, init_trans

def pose_opt(pose_graph):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=1.0,
                edge_prune_threshold=0.25,
                reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

def prepare_pcd(path):
    pcds = []
    for i in range(len(path)):
        pcd = o3d.io.read_point_cloud(fr"test_data/medium_real/{path[i]}")
        pcds_down = pcd.voxel_down_sample(0.05)
        pcds_down, ind = pcds_down.remove_statistical_outlier(nb_neighbors=50,std_ratio=2.0)
        pcds.append(pcds_down)
    return pcds

pcd_list = os.listdir('test_data/medium_real/')
list_pcd_registered = []
list_target = []

if __name__ == "__main__":
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        s = time.time()
        # Choose the initial registration between ransac, fgr, robust, gicp, and cficp
        # And select the folder that contain point cloud data to register
        pose_graph, result = full_registration("cficp",pcd_list)
        pose_opt(pose_graph)

        pcds = prepare_pcd(pcd_list)
        pcd_combined = o3d.geometry.PointCloud()
        for point_id in range(len(pcds)):
            pcds[point_id].transform(pose_graph.nodes[point_id].pose)
            source_points = color(pcds[point_id],len(pcds))
            pcd_combined += pcds[point_id]
            o3d.io.write_point_cloud(fr"result/medium_real/mt_{point_id}.pts", pcd_combined)
            target_points = pcd_combined
            with open("result/medium_real/report_pose_graph_mt.txt","a") as f:
                f.write(fr"Point Cloud:{point_id}"+ os.linesep)
                f.write(fr"Pose Nodes:{pose_graph.nodes[point_id]}"+ os.linesep)
                f.write(fr"Pose Edges:{pose_graph.edges[point_id]}"+ os.linesep)
                f.write(fr"Iteration {point_id} took:{time.time()-s}s"+ os.linesep)

        o3d.io.write_point_cloud("result/medium_real/pose_graph_mt_complete.pts", pcd_combined)
        o3d.visualization.draw([pcd_combined])
