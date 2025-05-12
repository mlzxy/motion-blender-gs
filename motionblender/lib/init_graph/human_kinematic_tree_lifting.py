import torch
from copy import copy
from loguru import logger as guru
import numpy as np
import motionblender.lib.animate as anim
from motionblender.lib.init_graph.coco_wholebody import dataset_info as coco_wholebody_info
from collections import Counter

def skeleton_info_to_connectivity(keypoint_names, skeleton_info):
    """ convert the coco whole body skeletion info to a connectivity integer list """
    keypoint_name_to_id = {name: i for i, name in enumerate(keypoint_names)} 
    connectivity = []
    if isinstance(skeleton_info, dict):
        skeleton_info = [a['link'] for a in skeleton_info.values()]
    for start, end in skeleton_info:
        if start not in keypoint_name_to_id or end not in keypoint_name_to_id:
            continue
        connectivity.append((keypoint_name_to_id[start], keypoint_name_to_id[end]))
    return connectivity 

def get_keypoint_names(keypoint_info):
    """ get the keypoint names from the keypoint info of coco wholebody """
    return [v['name'] for k, v in sorted(keypoint_info.items())]

coco_wholebody_keypoints = get_keypoint_names(coco_wholebody_info['keypoint_info'])

wholebody_keypoints = [
    'head', 'thorax', 'neck',  'left_shoulder', 'right_shoulder',
    'root', 'left_hip', 'right_hip',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_toe', 'right_toe',

    'right_thumb3', 'left_thumb3', 'right_thumb2', 'left_thumb2', 'right_thumb1', 'left_thumb1',
    'right_forefinger4', 'left_forefinger4', 'right_forefinger3', 'left_forefinger3', 'right_forefinger2', 'left_forefinger2', 'right_forefinger1', 'left_forefinger1',
    'right_middle_finger4', 'left_middle_finger4', 'right_middle_finger3', 'left_middle_finger3', 'right_middle_finger2', 'left_middle_finger2', 'right_middle_finger1', 'left_middle_finger1',
    'right_ring_finger4', 'left_ring_finger4', 'right_ring_finger3', 'left_ring_finger3', 'right_ring_finger2', 'left_ring_finger2', 'right_ring_finger1', 'left_ring_finger1',
    'right_pinky_finger4', 'left_pinky_finger4', 'right_pinky_finger3', 'left_pinky_finger3', 'right_pinky_finger2', 'left_pinky_finger2', 'right_pinky_finger1', 'left_pinky_finger1'
]

translation_from_coco_wholebody = {
    'left_toe': ['left_big_toe', 'left_small_toe'],
    'right_toe': ['right_big_toe', 'right_small_toe'],
    'root': ['left_hip', 'right_hip'],
    'thorax': ['left_shoulder', 'right_shoulder'],
    'head': ['nose'],
    'left_wrist': ['left_wrist', 'left_hand_root'],
    'right_wrist': ['right_wrist', 'right_hand_root'],    
    'neck': ['nose', 'thorax']
}

wholebody_connections_with_length = [
    ('root', 'thorax', 1.0),  # Thorax as reference length
    ('root', 'left_hip', 0.5),
    ('root', 'right_hip', 0.5),
    ('thorax', 'neck', 0.3),
    ('thorax', 'left_shoulder', 0.4),
    ('thorax', 'right_shoulder', 0.4),
    ('neck', 'head', 0.3),
    ('left_shoulder', 'left_elbow', 0.6),
    ('right_shoulder', 'right_elbow', 0.6),
    ('left_elbow', 'left_wrist', 0.5),
    ('right_elbow', 'right_wrist', 0.5),
    ('left_hip', 'left_knee', 0.8),
    ('right_hip', 'right_knee', 0.8),
    ('left_knee', 'left_ankle', 0.8),
    ('right_knee', 'right_ankle', 0.8),
    ('left_ankle', 'left_heel', 0.2),
    ('right_ankle', 'right_heel', 0.2),
    ('left_ankle', 'left_toe', 0.3),
    ('right_ankle', 'right_toe', 0.3),
    # Wrist to fingers and thumbs connections
    ('left_wrist', 'left_thumb1', 0.2),
    ('left_wrist', 'left_forefinger1', 0.2),
    ('left_wrist', 'left_middle_finger1', 0.2),
    ('left_wrist', 'left_ring_finger1', 0.2),
    ('left_wrist', 'left_pinky_finger1', 0.2),
    ('right_wrist', 'right_thumb1', 0.2),
    ('right_wrist', 'right_forefinger1', 0.2),
    ('right_wrist', 'right_middle_finger1', 0.2),
    ('right_wrist', 'right_ring_finger1', 0.2),
    ('right_wrist', 'right_pinky_finger1', 0.2),
    # Thumbs connections (ascending)
    ('left_thumb1', 'left_thumb2', 0.1),
    ('left_thumb2', 'left_thumb3', 0.1),
    ('right_thumb1', 'right_thumb2', 0.1),
    ('right_thumb2', 'right_thumb3', 0.1),
    # Forefinger connections (ascending)
    ('left_forefinger1', 'left_forefinger2', 0.1),
    ('left_forefinger2', 'left_forefinger3', 0.1),
    ('left_forefinger3', 'left_forefinger4', 0.1),
    ('right_forefinger1', 'right_forefinger2', 0.1),
    ('right_forefinger2', 'right_forefinger3', 0.1),
    ('right_forefinger3', 'right_forefinger4', 0.1),
    # Middle fingers connections (ascending)
    ('left_middle_finger1', 'left_middle_finger2', 0.1),
    ('left_middle_finger2', 'left_middle_finger3', 0.1),
    ('left_middle_finger3', 'left_middle_finger4', 0.1),
    ('right_middle_finger1', 'right_middle_finger2', 0.1),
    ('right_middle_finger2', 'right_middle_finger3', 0.1),
    ('right_middle_finger3', 'right_middle_finger4', 0.1),
    # Ring fingers connections (ascending)
    ('left_ring_finger1', 'left_ring_finger2', 0.1),
    ('left_ring_finger2', 'left_ring_finger3', 0.1),
    ('left_ring_finger3', 'left_ring_finger4', 0.1),
    ('right_ring_finger1', 'right_ring_finger2', 0.1),
    ('right_ring_finger2', 'right_ring_finger3', 0.1),
    ('right_ring_finger3', 'right_ring_finger4', 0.1),
    # Pinky fingers connections (ascending)
    ('left_pinky_finger1', 'left_pinky_finger2', 0.1),
    ('left_pinky_finger2', 'left_pinky_finger3', 0.1),
    ('left_pinky_finger3', 'left_pinky_finger4', 0.1),
    ('right_pinky_finger1', 'right_pinky_finger2', 0.1),
    ('right_pinky_finger2', 'right_pinky_finger3', 0.1),
    ('right_pinky_finger3', 'right_pinky_finger4', 0.1)
]

standard_wholebody_connection_lengths = {tp[:2]: tp[2] for tp in wholebody_connections_with_length}

wholebody_connections = [tp[:2] for tp in wholebody_connections_with_length]
wholebody_connections_int = skeleton_info_to_connectivity(wholebody_keypoints, wholebody_connections)

wholebody_constraints = [
    [('root', 'left_hip'), ('root', 'right_hip')],
    [('thorax', 'left_shoulder'), ('thorax', 'right_shoulder')],

    [('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow')],
    [('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist')],

    [('left_hip', 'left_knee'), ('right_hip', 'right_knee')],
    [('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')],
    [('left_ankle', 'left_heel'), ('right_ankle', 'right_heel')],
    [('left_ankle', 'left_toe'), ('right_ankle', 'right_toe')],

       [('left_wrist', 'left_thumb1'), ('right_wrist', 'right_thumb1')],
    [('left_wrist', 'left_forefinger1'), ('right_wrist', 'right_forefinger1')],
    [('left_wrist', 'left_middle_finger1'), ('right_wrist', 'right_middle_finger1')],
    [('left_wrist', 'left_ring_finger1'), ('right_wrist', 'right_ring_finger1')],
    [('left_wrist', 'left_pinky_finger1'), ('right_wrist', 'right_pinky_finger1')],
    [('left_thumb1', 'left_thumb2'), ('right_thumb1', 'right_thumb2')],
    [('left_thumb2', 'left_thumb3'), ('right_thumb2', 'right_thumb3')],
    [('left_forefinger1', 'left_forefinger2'), ('right_forefinger1', 'right_forefinger2')],
    [('left_forefinger2', 'left_forefinger3'), ('right_forefinger2', 'right_forefinger3')],
    [('left_forefinger3', 'left_forefinger4'), ('right_forefinger3', 'right_forefinger4')],
    [('left_middle_finger1', 'left_middle_finger2'), ('right_middle_finger1', 'right_middle_finger2')],
    [('left_middle_finger2', 'left_middle_finger3'), ('right_middle_finger2', 'right_middle_finger3')],
    [('left_middle_finger3', 'left_middle_finger4'), ('right_middle_finger3', 'right_middle_finger4')],
    [('left_ring_finger1', 'left_ring_finger2'), ('right_ring_finger1', 'right_ring_finger2')],
    [('left_ring_finger2', 'left_ring_finger3'), ('right_ring_finger2', 'right_ring_finger3')],
    [('left_ring_finger3', 'left_ring_finger4'), ('right_ring_finger3', 'right_ring_finger4')],
    [('left_pinky_finger1', 'left_pinky_finger2'), ('right_pinky_finger1', 'right_pinky_finger2')],
    [('left_pinky_finger2', 'left_pinky_finger3'), ('right_pinky_finger2', 'right_pinky_finger3')],
    [('left_pinky_finger3', 'left_pinky_finger4'), ('right_pinky_finger3', 'right_pinky_finger4')]
]   


def verify_constraints(constraints):
    """ this function ensure each link only appears once in the constraints """
    for k, v in Counter(sum(constraints, [])).items():
        if v != 1:  raise ValueError(f"link {k} appears more than once in the constraints")


def from_coco_wholebody_keypoints(kps, kp_scores): 
    """ convert coco wholebody keypoints/scores to our wholebody keypoints/scores """
    new_kps, new_kp_scores = [], []
    for k in wholebody_keypoints:
        pass
        if k not in translation_from_coco_wholebody:
            i = coco_wholebody_keypoints.index(k)
            v = kps[i]
            score = kp_scores[i]
        else:
            v, score = 0, 0
            for k2 in translation_from_coco_wholebody[k]:
                if k2 not in coco_wholebody_keypoints:
                    i = wholebody_keypoints.index(k2)
                    v += np.array(new_kps[i])
                    score += new_kp_scores[i]
                else:
                    i = coco_wholebody_keypoints.index(k2)
                    v += np.array(kps[i])
                    score += kp_scores[i]
            v /= len(translation_from_coco_wholebody[k])
            score /= len(translation_from_coco_wholebody[k])
            v = v.tolist()
            score = float(score)
        new_kps.append(v)
        new_kp_scores.append(score)
    return new_kps, new_kp_scores


def kps_to_joint_links(kps, kp_scores, thr=0.0, connections=wholebody_connections_int):
    """ convert the extracted keypoints / scores to format:
        joints [J, 2], links [L, 2], link_ids [L] """
    kps = np.array(kps)
    kp_scores = np.array(kp_scores)
    mask = kp_scores > thr
    result_connections, result_connection_ids = [], []
    all_used_kps = set()
    for ci, conn in enumerate(connections):
        if mask[conn[0]] and mask[conn[1]]:
            for c in conn: all_used_kps.add(int(c))
            result_connections.append(conn)
            result_connection_ids.append(ci)

    old2new = {old_id: new_id for new_id, old_id in enumerate(all_used_kps)}
    new2old = {new_id: old_id for old_id, new_id in old2new.items()}
    result_connections = [[old2new[c] for c in conn] for conn in result_connections]
    new_kps = [kps[new2old[i]] for i in range(len(new2old))]
    
    return np.array(new_kps), np.array(result_connections), np.array(result_connection_ids)

@torch.no_grad()
def cleanup_unreliable_links(track_link_id, track_link_dist, thr=0.002, simple_finger=False):
    """ input: tensor, output: tensor
    mark unreliable link to be 500 based on a set of rules 
    """
    track_link_id = track_link_id.clone()
    link_ids, link_counts = torch.unique(track_link_id, return_counts=True)
    link_counts = {int(lid): int(lc) for lid, lc in zip(link_ids, link_counts)}
    if 500 in link_counts: link_counts.pop(500)
    link2id = {link: i for i, link in enumerate(wholebody_connections)}

    neck_id = link2id[('thorax', 'neck')]
    if neck_id in link_counts: # otherwise, no neck
        shoulder_counts = (link_counts[link2id[('thorax', 'left_shoulder')]] + link_counts[link2id[('thorax', 'right_shoulder')]]) / 2
        if link_counts[neck_id] / shoulder_counts < 0.05: # not reliable
            link_counts.pop(neck_id)
            guru.info("remove link (thorax, neck) because it has too few tracks than shoulders (< 5%)")
            link_counts.pop(link2id[('neck', 'head')])
            guru.info("remove link (neck, head) because it has too few tracks than shoulders (< 5%)")
    
    
    remove_lower_body = False 
    _ = [('left_hip', 'left_knee'), ('right_hip', 'right_knee')]
    if min([link_counts.get(v, 0) for v in _]) < 100:
        remove_lower_body = True

    if remove_lower_body:
        for link_name in [('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'), ('root', 'left_hip'), ('root', 'right_hip'), ('left_ankle', 'left_toe'), ('right_ankle', 'right_toe'), ('left_ankle', 'left_heel'), ('right_ankle', 'right_heel')]:
            link_id = link2id[link_name]
            if link_id in link_counts:
                guru.info(f"remove link {link_name} because the lower body has too few tracks")
                link_counts.pop(link_id)
    
    fingers = [('right_thumb', 3), ('left_thumb', 3), ('left_forefinger', 4), ('right_forefinger', 4), ('left_middle_finger', 4), ('right_middle_finger', 4), ('left_ring_finger', 4), ('right_ring_finger', 4), ('left_pinky_finger', 4), ('right_pinky_finger', 4)] 
    for finger_name, start_id in fingers:
        init_start_id = start_id
        while start_id > 1:
            link_name = (f'{finger_name}{start_id-1}', f'{finger_name}{start_id}')
            link_id = link2id[link_name]
            if link_id in link_counts and link_counts[link_id] < 50:
                link_counts.pop(link_id)
                for i in range(start_id, init_start_id+1):
                    link_name = (f'{finger_name}{i-1}', f'{finger_name}{i}')
                    link_id = link2id[link_name]
                    if link_id in link_counts:
                        link_counts.pop(link_id)
                    guru.info(f"remove link {link_name} because it has too few tracks (<50)")
            start_id -= 1

    if simple_finger:
        for finger_name, start_id in fingers:       
            link_name = (f'{finger_name}{start_id-1}', f'{finger_name}{start_id}')
            link_id = link2id[link_name]
            if link_id in link_counts:
                link_counts.pop(link_id)
                guru.info(f"remove link {link_name} (finger-tip is usually not reliable or useful)")
    
    
    for link_id in list(link_counts.keys()):
        c = (track_link_dist[track_link_id == link_id] <= thr).sum() # NOTE: if a link is not part of the "fg" (moving part), then it will get removed here. 
        if c < 10: # too few tracks
            link_counts.pop(link_id)
            track_link_id[track_link_id == link_id] = 500
            guru.info(f"remove link {link_id} because it has too few high-quality tracks (<10)")
    
    for link_id in link_ids:
        if int(link_id) not in link_counts:
            track_link_id[track_link_id == link_id] = 500
    return track_link_id


def initialize_kinematic_tree_from_links(links=wholebody_connections, constraints=wholebody_constraints, 
                                         only_keep_link_ids=None, 
                                         init_length_dict: dict=standard_wholebody_connection_lengths):
    """ convert links/constraints to a initial kinematic tree.
        if only_keep_link_ids is provided, then only keep these links

        return hollow_chain, length_tensor (J), rot6d_tensor (Jx6), rot6d_linkid2indice (link id to tensor idx), 
                length_linkid2indice, links_tensor (J, 2), links_global_id2local_ind, joints_id_2_name
    """
    init_length = 0.0
    full_links = copy(links)
    input_only_keep_link_ids = only_keep_link_ids
    if only_keep_link_ids is not None:
        links = [links[int(i)] for i in only_keep_link_ids]
    else:
        only_keep_link_ids = list(range(len(links)))

    # NOTE: only keep the largest tree, because the detected kps can be fragmented
    root_joint, valid_joints = anim.find_root_joint_id(links, return_all_roots=True, return_largest=True) 
    only_keep_link_ids, links = [], []
    for i, link in enumerate(full_links):
        if link[0] in valid_joints and link[1] in valid_joints:
            only_keep_link_ids.append(i)
            links.append(link)

    links_global_id2local_ind = torch.zeros(len(only_keep_link_ids), 2, dtype=torch.long)
    links_global_id2local_ind[:, 0] = torch.tensor(only_keep_link_ids)
    
    adj_list = {} 
    for a, b in links: adj_list.setdefault(a, []).append(b)
    joint_name2id = {joint_name: joint_id for joint_id, joint_name in enumerate(sorted(set(sum([list(l) for l in links], []))))}

    def walk(node_name, chain):
        for c in adj_list.get(node_name, []):
            length = init_length
            if init_length_dict is not None:
                length = init_length_dict[(node_name, c)]
            child = {
                'id': joint_name2id[c],
                'name': c,
                'chain': [],
                'length': torch.as_tensor([length]).float(),
                'rot6d': anim.rmat_to_cont_6d(torch.eye(3)).float()
            }
            walk(c, child)
            chain['chain'].append(child)
        return chain
        
    chain = walk(root_joint, {'id': joint_name2id[root_joint], 'name': root_joint, 'chain': [], 'length': None, 'rot6d': None})

    length_tensor, length_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'length', return_linkid2indice=True)
    rot6d_tensor, rot6d_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'rot6d', return_linkid2indice=True)
    
    # based on constraints, sharing the same length across constrained links
    if constraints:
        verify_constraints(constraints)
        constraints = [[c for c in cs if c in links] for cs in constraints]
        constraints = [c for c in constraints if len(c) > 1]
        constraints = [[tuple([joint_name2id[n] for n in l]) for l in cs] for cs in constraints]
        for constrained_links in constraints:
            constrained_link_ids = [l[1] for l in constrained_links]
            for l in constrained_link_ids[1:]: 
                length_linkid2indice[l] = length_linkid2indice[constrained_link_ids[0]]

            all_used_indices = sorted(set(length_linkid2indice.values()))
            oldind_to_newind = {old: new for new, old in enumerate(all_used_indices)}
            length_tensor = length_tensor[all_used_indices]
            length_linkid2indice = {k: oldind_to_newind[v] for k, v in length_linkid2indice.items()}
    
    links_int = [[joint_name2id[n] for n in l] for l in links]
    link_orders = torch.argsort(torch.as_tensor([end_ind for _, end_ind in links_int]))
    links_tensor = torch.as_tensor(links_int).long()[link_orders]
    # for i, (_, end_ind) in enumerate(links_int):
    #     links_global_id2local_id[i, 1] = end_ind
    links_global_id2local_ind[:, 1] = torch.argsort(link_orders)

    hollow_chain = anim.create_hollow_chain_wo_tensor(chain)
    return hollow_chain, length_tensor, rot6d_tensor, rot6d_linkid2indice, length_linkid2indice, \
        links_tensor, links_global_id2local_ind, {v: k for k, v in joint_name2id.items()}








# ----------------------------------------------------------------------------------------------------------------------- #



# def initialize_kinematic_tree_from_links2(links=wholebody_connections, constraints=wholebody_constraints, 
#                                          only_keep_link_ids=None, 
#                                          init_length_dict: dict=standard_wholebody_connection_lengths):
#     """ convert links/constraints to a initial kinematic tree.
#         if only_keep_link_ids is provided, then only keep these links

#         return hollow_chain, length_tensor (J), rot6d_tensor (Jx6), rot6d_linkid2indice (link id to tensor idx), 
#                 length_linkid2indice, links_tensor (J, 2), links_global_id2local_ind, joints_id_2_name
#     """
#     init_length = 0.0
#     full_links = copy(links)
#     input_only_keep_link_ids = only_keep_link_ids
#     if only_keep_link_ids is not None:
#         links = [links[int(i)] for i in only_keep_link_ids]
#     else:
#         only_keep_link_ids = list(range(len(links)))

#     # NOTE: only keep the largest tree, because the detected kps can be fragmented
#     root_joint, valid_joints = anim.find_root_joint_id(links, return_all_roots=True, return_largest=True, return_smallest=False) 


#     only_keep_link_ids, links = [], []
#     for i, link in enumerate(full_links):
#         if link[0] in valid_joints and link[1] in valid_joints:
#             only_keep_link_ids.append(i)
#             links.append(link)

#     links_global_id2local_ind = torch.zeros(len(only_keep_link_ids), 2, dtype=torch.long)
#     links_global_id2local_ind[:, 0] = torch.tensor(only_keep_link_ids)
    
#     adj_list = {} 
#     for a, b in links: adj_list.setdefault(a, []).append(b)
#     joint_name2id = {joint_name: joint_id for joint_id, joint_name in enumerate(sorted(set(sum([list(l) for l in links], []))))}

#     def walk(node_name, chain):
#         for c in adj_list.get(node_name, []):
#             length = init_length
#             if init_length_dict is not None:
#                 length = init_length_dict[(node_name, c)]
#             child = {
#                 'id': joint_name2id[c],
#                 'name': c,
#                 'chain': [],
#                 'length': torch.as_tensor([length]).float(),
#                 'rot6d': anim.rmat_to_cont_6d(torch.eye(3)).float()
#             }
#             walk(c, child)
#             chain['chain'].append(child)
#         return chain
        
#     chain = walk(root_joint, {'id': joint_name2id[root_joint], 'name': root_joint, 'chain': [], 'length': None, 'rot6d': None})

#     length_tensor, length_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'length', return_linkid2indice=True)
#     rot6d_tensor, rot6d_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'rot6d', return_linkid2indice=True)
    
#     # based on constraints, sharing the same length across constrained links
#     if constraints:
#         verify_constraints(constraints)
#         constraints = [[c for c in cs if c in links] for cs in constraints]
#         constraints = [c for c in constraints if len(c) > 1]
#         constraints = [[tuple([joint_name2id[n] for n in l]) for l in cs] for cs in constraints]
#         for constrained_links in constraints:
#             constrained_link_ids = [l[1] for l in constrained_links]
#             for l in constrained_link_ids[1:]: 
#                 length_linkid2indice[l] = length_linkid2indice[constrained_link_ids[0]]

#             all_used_indices = sorted(set(length_linkid2indice.values()))
#             oldind_to_newind = {old: new for new, old in enumerate(all_used_indices)}
#             length_tensor = length_tensor[all_used_indices]
#             length_linkid2indice = {k: oldind_to_newind[v] for k, v in length_linkid2indice.items()}
    
#     links_int = [[joint_name2id[n] for n in l] for l in links]
#     link_orders = torch.argsort(torch.as_tensor([end_ind for _, end_ind in links_int]))
#     links_tensor = torch.as_tensor(links_int).long()[link_orders]
#     # for i, (_, end_ind) in enumerate(links_int):
#     #     links_global_id2local_id[i, 1] = end_ind
#     links_global_id2local_ind[:, 1] = torch.argsort(link_orders)

#     hollow_chain = anim.create_hollow_chain_wo_tensor(chain)
#     return hollow_chain, length_tensor, rot6d_tensor, rot6d_linkid2indice, length_linkid2indice, \
#         links_tensor, links_global_id2local_ind, {v: k for k, v in joint_name2id.items()}

