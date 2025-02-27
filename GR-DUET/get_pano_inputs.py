import os
import json
import random
import math
import torch
import h5py
import tqdm
import MatterSim
import numpy as np
import networkx as nx
from map_nav_src.utils.data import ImageFeaturesDB, get_all_point_angle_feature
from map_nav_src.utils.data import angle_feature as precompute_angle_feature

def init_sim(connectivity_dir):
    image_w = 640
    image_h = 480
    vfov = 60
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(image_w, image_h)
    sim.setCameraVFOV(math.radians(vfov))
    sim.setBatchSize(1)
    sim.initialize()
    return sim

connectivity_dir = 'datasets/R2R/connectivity'
sim = init_sim(connectivity_dir)
angle_feature = get_all_point_angle_feature(sim, 4)
feat_db = ImageFeaturesDB(os.path.join('datasets', 'R2R', 'features', 'clip_vit-b16_mp3d_original.hdf5'), 512)

def load_nav_graph(scan):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
            + (pose1['pose'][7]-pose2['pose'][7])**2\
            + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i,item in enumerate(data):
            if item['included']:
                for j,conn in enumerate(item['unobstructed']):
                    if conn and data[j]['included']:
                        positions[item['image_id']] = np.array([item['pose'][3],
                                item['pose'][7], item['pose'][11]]);
                        assert data[j]['unobstructed'][i], 'Graph should be undirected'
                        G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
        nx.set_node_attributes(G, values=positions, name='position')
    return G

def getStates():
    state = sim.getState()[0]
    feature = feat_db.get_image_feature(state.scanId, state.location.viewpointId)
    return feature, state

def make_candidate(feature, scanId, viewpointId, viewId):
    def _loc_distance(loc):
        return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
    base_heading = (viewId % 12) * math.radians(30)
    base_elevation = (viewId // 12 - 1) * math.radians(30)

    adj_dict = {}
    for ix in range(36):
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        # Heading and elevation for the viewpoint center
        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        visual_feat = feature[ix]

        # get adjacent locations
        for j, loc in enumerate(state.navigableLocations[1:]):
            # if a loc is visible from multiple view, use the closest
            # view (in angular distance) as its representation
            distance = _loc_distance(loc)

            # Heading and elevation for for the loc
            loc_heading = heading + loc.rel_heading
            loc_elevation = elevation + loc.rel_elevation
            angle_feat = precompute_angle_feature(loc_heading, loc_elevation, 4)
            if (loc.viewpointId not in adj_dict or
                    distance < adj_dict[loc.viewpointId]['distance']):
                adj_dict[loc.viewpointId] = {
                    'heading': loc_heading,
                    'elevation': loc_elevation,
                    "normalized_heading": state.heading + loc.rel_heading,
                    "normalized_elevation": state.elevation + loc.rel_elevation,
                    'scanId': scanId,
                    'viewpointId': loc.viewpointId, # Next viewpoint id
                    'pointId': ix,
                    'distance': distance,
                    'idx': j + 1,
                    'feature': np.concatenate((visual_feat, angle_feat), -1),
                    'position': (loc.x, loc.y, loc.z),
                }
    candidate = list(adj_dict.values())

    return candidate

def get_ob(scan, node):
    heading = random.uniform(0, 2 * math.pi)
    sim.newEpisode([scan], [node], [heading], [0])
    feature, state = getStates()
    base_view_id = state.viewIndex
    candidate = make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
    feature = np.concatenate((feature, angle_feature[base_view_id]), -1)

    ob = {
        'feature' : feature,
        'candidate': candidate,
    }

    return ob

def panorama_feature_variable(ob):
    ''' Extract precomputed features into variable. '''
    image_feat_size = 512
    view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
    # cand views
    used_viewidxs = set()
    for j, cc in enumerate(ob['candidate']):
        view_img_fts.append(cc['feature'][:image_feat_size])
        view_ang_fts.append(cc['feature'][image_feat_size:])
        nav_types.append(1)
        cand_vpids.append(cc['viewpointId'])
        used_viewidxs.add(cc['pointId'])
    # non cand views
    view_img_fts.extend([x[:image_feat_size] for k, x \
        in enumerate(ob['feature']) if k not in used_viewidxs])
    view_ang_fts.extend([x[image_feat_size:] for k, x \
        in enumerate(ob['feature']) if k not in used_viewidxs])
    nav_types.extend([0] * (36 - len(used_viewidxs)))
    # combine cand views and noncand views
    view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
    view_ang_fts = np.stack(view_ang_fts, 0)
    view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
    view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
    
    batch_view_img_fts = torch.from_numpy(view_img_fts).unsqueeze(0).cuda()
    batch_loc_fts = torch.from_numpy(view_loc_fts).unsqueeze(0).cuda()
    batch_nav_types = torch.LongTensor(nav_types).unsqueeze(0).cuda()
    batch_cand_vpids = [cand_vpids]
    batch_view_lens = torch.LongTensor([len(view_img_fts)]).cuda()

    return {
        'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
        'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
        'cand_vpids': batch_cand_vpids,
    }

def main():
    with open("datasets/R2R/connectivity/scans.txt") as f:
        scans = f.read().splitlines()

    output_file = 'datasets/R2R/features/pano_inputs.h5'
    with h5py.File(output_file, 'w') as outf:
        with tqdm.tqdm(total=len(scans), desc='Scans') as pbar_scans:
            for scan in scans:
                G = load_nav_graph(scan)
                nodes = G.nodes
                with tqdm.tqdm(total=len(nodes), desc=f'Nodes in {scan}', leave=False) as pbar_nodes:
                    for node in nodes:
                        ob = get_ob(scan, node)
                        pano_inputs = panorama_feature_variable(ob)

                        key = f"{scan}_{node}"
                        for k, v in pano_inputs.items():
                            if k == 'cand_vpids':
                                outf.create_dataset(f"{key}/{k}", data=np.array(v, dtype='S'), compression='gzip')
                            else:
                                outf.create_dataset(f"{key}/{k}", data=v.cpu().numpy(), compression='gzip')
                        pbar_nodes.update(1)
                pbar_scans.update(1)

def read_h5(file, key):
    pano_inputs = {}
    with h5py.File(file, 'r') as f:
        for k, v in f[key].items():
            data =v[()]
            if k == 'cand_vpids':
                data = [[s.decode('utf-8') for s in data[0]]]
                pano_inputs[k] = data
            else:
                pano_inputs[k] = torch.from_numpy(data).cuda()
    return pano_inputs

if __name__ == '__main__':
    main()
