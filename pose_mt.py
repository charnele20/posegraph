import open3d as o3d
import copy
import os
import numpy as np
import time
from macpath import join
import sys



class registration():
    def __init__(self):
        self.voxel_size = 0.05 # means 5cm for this dataset
        self.source_down = None
        self.target_down = None
        self.source_fpfh = None
        self.target_fpfh = None
        self.result = None
        self.icp_result = None
        self.source = None
        self.target = None
        self.list_pcd_registered = []

    def prepare_dataset(self,source_path,target_path):
        self.source_down = o3d.io.read_point_cloud(source_path)
        self.target_down = o3d.io.read_point_cloud(target_path)
        print(":: Load two point clouds and disturb initial pose.")
        #self.source_down = self.preprocess_point_cloud(self.source)
        #self.target_down = self.preprocess_point_cloud(self.target)
    
    def write_pcd(self,path):
        for i in range(len(path)):
            pcd = o3d.io.read_point_cloud(fr"raw/{path[i]}")
            pcds_down = pcd.voxel_down_sample(self.voxel_size)
            pcds_down, ind = pcds_down.remove_statistical_outlier(nb_neighbors=50,std_ratio=2.0)
            if i < 10:
                o3d.io.write_point_cloud(fr"project/0{i}.pts", pcds_down)
            else:
                o3d.io.write_point_cloud(fr"project/{i}.pts", pcds_down)

    def prepare_pcd(self,path):
        pcds = []
        for i in range(len(path)):
            pcd = o3d.io.read_point_cloud(fr"project/{path[i]}")
            pcds_down = pcd.voxel_down_sample(self.voxel_size)
            pcds_down, ind = pcds_down.remove_statistical_outlier(nb_neighbors=50,std_ratio=2.0)
            pcds.append(pcds_down)
        return pcds

    def run_ransac(self):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            self.execute_global_registration()
        
    def run_icp(self):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            self.refine_registration()

    def preprocess_point_cloud(self, pcd):
        print(":: Downsample with a voxel size %.3f." % self.voxel_size)
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        # Remove Outlier for RANSAC
        pcd_down, ind = pcd_down.remove_statistical_outlier(nb_neighbors=50,std_ratio=2.0)
        
        # radius_normal = self.voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        # pcd_down.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))

        # radius_feature = self.voxel_size * 10
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        # pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        #     pcd_down,
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=500))
        # return pcd_down, pcd_fpfh
        return pcd_down

    def execute_global_registration(self):
        distance_threshold = self.voxel_size * 1.4
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % self.voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        self.result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.source_down, self.target_down, self.source_fpfh, self.target_fpfh, False,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999))

    def refine_registration(self):
        print(":: Point-to-plane ICP registration is applied on original point")

        radius_normal = self.voxel_size * 4
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        self.source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))
        self.target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))      

        self.max_correspondence_distance_coarse = self.voxel_size * 10
        self.max_correspondence_distance_fine = self.voxel_size * 2
        loss = o3d.pipelines.registration.HuberLoss(k=0.2)
        print("   distance threshold coarse %.3f." % self.max_correspondence_distance_coarse)
        print("   distance threshold fine %.3f." % self.max_correspondence_distance_fine)
        icp_coarse = o3d.pipelines.registration.registration_icp(
            self.source_down, self.target_down, self.max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss))

        self.icp_result = o3d.pipelines.registration.registration_icp(
            self.source_down, self.target_down, self.max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss))

        self.transformation_icp = self.icp_result.transformation
        self.information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(self.source_down, self.target_down, 
                                self.max_correspondence_distance_fine,self.icp_result.transformation)
    
    def pose_graph_opt(self,pose_graph):
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)

    def transform(self):
        self.source_down.transform(self.transformation_icp)

    def color(self,pc,num):
        import random
        import webcolors
        color_list = random.sample(list(webcolors.CSS3_NAMES_TO_HEX.values()), num)
        if len(color_list) < num:
            num = len(color_list)
        num = random.randint(0, num-1)
        color_rgb = webcolors.hex_to_rgb(color_list[num])
        color_float = [x/255 for x in color_rgb]
        pc.paint_uniform_color(color_float)

    def draw(self):
        if self.result.fitness > 0.5:
            o3d.visualization.draw([self.source_down, self.target_down])
        else:
            o3d.visualization.draw([self.target_down])

def draw_pcd(list_pcd_registered):
    o3d.visualization.draw(list_pcd_registered)
'''  
pcd_list = os.listdir('project/')
target_path = fr"project/{pcd_list[0]}"
list_pcd_registered = []
list_target = []
for i in range(1, len(pcd_list)):
    print("Iteration:",i)
    source_path = fr"project/{pcd_list[i]}"
    obj = registration()
    obj.prepare_dataset(source_path,target_path) 
    obj.run_ransac()
    obj.run_icp()
    obj.color(i)
    target = obj.target_down
    list_target.append(target)
    if (obj.result.fitness > 0.50):  
        obj.transform()
        list_pcd_registered.append(obj.source_down)
        #obj.source_down = obj.color()
        target += obj.source_down
        #obj.draw()

    o3d.io.write_point_cloud(fr"result/updated_result_{i}.pts", target)
    #obj.draw()
    target_path = fr"result/updated_result_{i}.pts"
list_pcd_registered.append(list_target[0])
draw_pcd(list_pcd_registered) '''  
# Down Sampled before Registration
obj = registration()                                   
write_list = os.listdir('raw/')
obj.write_pcd(write_list)

pcd_list = os.listdir('project/')
list_pcd_registered = []
list_target = []


def full_registration(pcds):
    with open("report.txt","w") as f:
        sys.stdout=f
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            for source_id in range(n_pcds):
                for target_id in range(source_id + 1, n_pcds): 
                    s = time.time()
                    source_path = fr"project/{pcd_list[source_id]}"
                    target_path = fr"project/{pcd_list[target_id]}"
                    obj.prepare_dataset(source_path,target_path)
                    # obj.run_ransac()
                    future = executor.submit(obj.run_icp)
                    future.result()
                    print("Build o3d.pipelines.registration.PoseGraph")
                    if target_id == source_id + 1:  # odometry case
                        odometry = np.dot(obj.transformation_icp, odometry)
                        pose_graph.nodes.append(
                            o3d.pipelines.registration.PoseGraphNode(
                                np.linalg.inv(odometry)))
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                    target_id,
                                                                    obj.transformation_icp,
                                                                    obj.information_icp,
                                                                    uncertain=False))
                    else:  # loop closure case
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                    target_id,
                                                                    obj.transformation_icp,
                                                                    obj.information_icp,
                                                                    uncertain=True))
                    print (f"Registration point cloud {source_id}-{target_id} took: ",time.time()-s)
            sys.stdout = sys.__stdout__
    return pose_graph


if __name__ == "__main__":
    start = time.time()
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcd_list)
        obj.pose_graph_opt(pose_graph)

    print("Make a combined point cloud")
    pcds = obj.prepare_pcd(pcd_list)
    pcd_combined = o3d.geometry.PointCloud()
    processing_time=[]
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        source_points = obj.color(pcds[point_id],len(pcds))
        pcd_combined += pcds[point_id]
        o3d.io.write_point_cloud(fr"result2/icp_result{point_id}.pts", pcd_combined)
        target_points = pcd_combined

        with open("result2/report_fitness.txt","a") as f:
            f.write(fr"Point Cloud:{point_id}"+ os.linesep)
            f.write(fr"RMSE Value:{obj.icp_result.inlier_rmse}"+ os.linesep)
            f.write(fr"Fitness Value:{obj.icp_result.fitness}"+ os.linesep)
            f.write(fr"Registration process took: {time.time()-start}")

    # print ("Registration process took: ",time.time()-start)
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.05)
    o3d.io.write_point_cloud("scene/1-2.pts", pcd_combined_down)
    o3d.visualization.draw([pcd_combined_down])