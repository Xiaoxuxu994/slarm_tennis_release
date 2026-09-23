import os

import numpy as np


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

opencv2waymo = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

# flu(forward-left-up) coordinate system: x-forward y-left z-up
DATASETS = {
    "driving_sim": {"opencv2dataset": np.eye(4), "canonical_to_flu": opencv2waymo},
    "waymo": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    "nuscenes": {"opencv2dataset": np.eye(4), "canonical_to_flu": opencv2waymo},
    "argoverse2": {"opencv2dataset": np.eye(4), "canonical_to_flu": opencv2waymo},
    "b2d": {"opencv2dataset": np.eye(4), "canonical_to_flu": opencv2waymo},
    "ball_catch_24cm_stereo40_stream25_nopitch": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    "ball_catch_24cm_triview": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    "ball_catch_6.5cm_triview": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    # catch45: same axis convention as 6.5cm. Rig height differs but does not appear
    # here -- rig_to_world is read per scene; these matrices only describe axis order.
    "ball_catch_6.5cm_triview_catch45": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    # v3_0829: same convention. Extrinsics vary per frame while rig_to_world is a
    # single matrix, but the ball's second difference still equals gravity, so the
    # rig frame is inertial and ballistic extrapolation holds.
    "ball_catch_triview_v3_0829": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    # Native-resolution version of the same data; the convention does not change.
    "ball_catch_triview_v3_0829_native": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    # 0902_fixed: same convention. rig_to_world is a pure translation, so the frame
    # is inertial. Verified: second difference -9.809981, position and velocity
    # satisfy the midpoint rule to 1e-5 m/s -- an exact analytic trajectory.
    "ball_catch_triview_0902_fixed": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    # 0903_2k: same convention; a_z = -9.809990 m/s^2 and rig_to_world is again a
    # pure translation. World z=0 sits at base_link here and the floor is near
    # -0.10 m, which only affects heights a human reads, not these matrices.
    "ball_catch_triview_0903_2k": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    # Assumes the 0903 rig convention; verify the new annotations before training.
    "ball_catch_triview_0908_10k": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
}

waymo_train = "scene_list/waymo_train.txt"  # NOTE: Use full data for multi-GPU
waymo_val = "scene_list/waymo_val.txt"
driving_sim_train = "scene_list/driving_sim_train.txt"
driving_sim_val = "scene_list/driving_sim_val.txt"
b2d_train = "scene_list/b2d_train.txt"
b2d_val = "scene_list/b2d_val.txt"

if os.environ.get('OVERFIT_EXP'):  # Overfitting experiment on a single dynamic scene
    # b2d scene id TBD, modify b2d_train_1.txt directly for overfitting experiment
    b2d_train = "scene_list/b2d_train_1.txt"
    b2d_val = "scene_list/b2d_train_1.txt"
    print("B2D overfitting using default scene")

    # Specify waymo scene id
    if os.environ.get('SCENE_ID_WAYMO'):
        scene_list_file_path = "data/dataset_scene_list/waymo_train_list.txt"
        if os.environ.get('USE_VALIDATION'):
            scene_list_file_path = "data/dataset_scene_list/waymo_val_list.txt"
        try:
            with open(scene_list_file_path, 'r', encoding='utf-8') as file:
                # Read all lines and remove trailing newlines
                lines = [line.rstrip('\n') for line in file]
            scene_id = int(os.environ.get('SCENE_ID_WAYMO'))
            scene_name = lines[scene_id].strip()
            scene_file_context = f"annotations/waymo/training/{scene_name}.json"
            if os.environ.get('USE_VALIDATION'):
                scene_file_context = f"annotations/waymo/validation/{scene_name}.json"
            scene_file_name = "scene_list/waymo_train_1_scene.txt"
            with open(os.path.join(os.environ.get('DATA_ROOT').strip(), scene_file_name),
                'w', encoding='utf-8') as file:
                file.write(scene_file_context)
            waymo_train = scene_file_name
            waymo_val = scene_file_name
            print(f"Waymo training on Scene {scene_id}.")
        except FileNotFoundError:
            print(f"Error: File '{scene_list_file_path}' does not exist, using default scene")
            waymo_train = "scene_list/waymo_train_1.txt"
            waymo_val = "scene_list/waymo_train_1.txt"
        except Exception as e:
            print(f"Error determining Waymo scene: {e}, using default scene")
            waymo_train = "scene_list/waymo_train_1.txt"
            waymo_val = "scene_list/waymo_train_1.txt"
    else:
        waymo_train = "scene_list/waymo_train_1.txt"
        waymo_val = "scene_list/waymo_train_1.txt"
        print("Waymo overfitting ID not specified, using default scene")

    # Specify driving_sim scene id
    if os.environ.get('SCENE_ID_DRIVING_SIM'):
        try:
            scene_id = int(os.environ.get('SCENE_ID_DRIVING_SIM'))
            scene_name = f"{scene_id:03d}"  # Convert to 3-digit string, padded with leading zeros
            scene_file_context = f"annotations/driving_sim/training/{scene_name}.json"
            if os.environ.get('USE_VALIDATION'):
                scene_file_context = f"annotations/driving_sim/validation/{scene_name}.json"
            scene_file_name = "scene_list/driving_sim_train_1_scene.txt"
            with open(os.path.join(os.environ.get('DATA_ROOT').strip(), scene_file_name),
                'w', encoding='utf-8') as file:
                file.write(scene_file_context)
            driving_sim_train = scene_file_name
            driving_sim_val = scene_file_name
            print(f"Driving_sim training on Scene {scene_id}.")
        except Exception as e:
            print(f"Error determining Driving_sim scene: {e}, using default scene")
            driving_sim_train = "scene_list/driving_sim_train_1.txt"
            driving_sim_val = "scene_list/driving_sim_train_1.txt"
    else:
        driving_sim_train = "scene_list/driving_sim_train_1.txt"
        driving_sim_val = "scene_list/driving_sim_train_1.txt"
        print("Driving_sim overfitting ID not specified, using default scene")

DATASET_DICT = {
    "b2d": {
            "size": [160, 240],
            "temporal": True,
            "num_context_timesteps": 4,
            "num_target_timesteps": 4,
            "annotation_txt_file_train": b2d_train,
            "annotation_txt_file_val": b2d_val,
            "camera_list": {
                1: ["CAM_FRONT"],
                3: ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
                5: [
                    "CAM_BACK_LEFT",
                    "CAM_FRONT_LEFT",
                    "CAM_FRONT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK_RIGHT",
                ],
                6: [
                    "CAM_FRONT_LEFT",
                    "CAM_FRONT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK_RIGHT",
                    "CAM_BACK",
                    "CAM_BACK_LEFT",
                ],
                7: [
                    "CAM_FRONT_LEFT",
                    "CAM_FRONT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK_RIGHT",
                    "CAM_BACK",
                    "CAM_BACK_LEFT",
                ],
            },
        "ref_camera": "CAM_FRONT",
        },

    "driving_sim":{
        "size": [160, 240],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": driving_sim_train,
        "annotation_txt_file_val": driving_sim_val,
        "camera_list": {
            1: ["front"],
            3: ["front_left", "front", "front_right"],
            5: ["back_left", "front_left", "front", "front_right", "back_right"],
            6: ["front_left", "front", "front_right", "back_right", "back", "back_left"],
            7: ["front_left", "front", "front_right", "back_right", "back", "back_left"],
        },
        "ref_camera": "front",
    },

    "ball_catch_24cm_stereo40_stream25_nopitch": {
        "size": [320, 240],
        "temporal": True,
        "num_context_timesteps": 6,
        "num_target_timesteps": 7,
        "annotation_txt_file_train": "scene_list/ball_catch_24cm_stereo40_stream25_nopitch_train.txt",
        "annotation_txt_file_val": "scene_list/ball_catch_24cm_stereo40_stream25_nopitch_validation.txt",
        "camera_list": {
            2: ["front_left", "front_right"],
        },
        "ref_camera": "front_left",
    },

    "ball_catch_24cm_triview": {
        "size": [320, 240],
        "temporal": True,
        "num_context_timesteps": 6,
        "num_target_timesteps": 7,
        "annotation_txt_file_train": "scene_list/ball_catch_24cm_triview_train.txt",
        "annotation_txt_file_val": "scene_list/ball_catch_24cm_triview_validation.txt",
        "camera_list": {
            2: ["front_left", "front_right"],
            3: ["front_left", "front_right", "lower_front"],
        },
        "ref_camera": "front_left",
    },

    "ball_catch_6.5cm_triview": {
        "size": [320, 240],
        "temporal": True,
        "num_context_timesteps": 6,
        "num_target_timesteps": 7,
        "annotation_txt_file_train": "scene_list/ball_catch_6.5cm_triview_train.txt",
        "annotation_txt_file_val": "scene_list/ball_catch_6.5cm_triview_validation.txt",
        "camera_list": {
            2: ["front_left", "front_right"],
            3: ["front_left", "front_right", "lower_front"],
        },
        "ref_camera": "front_left",
    },

    # catch45. This key must match the annotation JSON's "dataset" field exactly,
    # or both tables miss and raise KeyError at the canonical transform and at
    # camera_list. Check with tools/check_dataset_contract.py --json-only.
    "ball_catch_6.5cm_triview_catch45": {
        "size": [320, 240],
        "temporal": True,
        "num_context_timesteps": 6,
        "num_target_timesteps": 7,
        "annotation_txt_file_train": "scene_list/ball_catch_6.5cm_triview_catch45_train.txt",
        "annotation_txt_file_val": "scene_list/ball_catch_6.5cm_triview_catch45_validation.txt",
        "camera_list": {
            2: ["front_left", "front_right"],
            3: ["front_left", "front_right", "lower_front"],
        },
        "ref_camera": "front_left",
    },

    # v3_0829. Run check_dataset_contract.py --visibility-summary first: two of five
    # sampled scenes have a context frame blind in all three views, one of them
    # frame 15, which leaves the landing prediction with no anchor at all.
    "ball_catch_triview_v3_0829": {
        "size": [320, 240],
        "temporal": True,
        "num_context_timesteps": 6,
        "num_target_timesteps": 7,
        "annotation_txt_file_train": "scene_list/ball_catch_triview_v3_0829_train.txt",
        "annotation_txt_file_val": "scene_list/ball_catch_triview_v3_0829_validation.txt",
        "camera_list": {
            2: ["front_left", "front_right"],
            3: ["front_left", "front_right", "lower_front"],
        },
        "ref_camera": "front_left",
    },

    # v3_0829 at native resolution.
    #
    # size is (H, W), not (W, H). datasets.py uses
    #       fx_px = fx_norm * target_size[1]   (W)
    #       fy_px = fy_norm * target_size[0]   (H)
    # and transforms.Resize takes (h, w). Swapping them exchanges the two intrinsic
    # directions: the geometry is wrong everywhere and nothing raises.
    #
    # normalized_intrinsics is resolution independent and is reused as is:
    #       fx_px = 1.0723 * 480 = 514.7
    #       fy_px = 0.8042 * 640 = 514.7      still equal, so pixels stay square
    "ball_catch_triview_v3_0829_native": {
        "size": [640, 480],
        "temporal": True,
        "num_context_timesteps": 6,
        "num_target_timesteps": 7,
        "annotation_txt_file_train": "scene_list/ball_catch_triview_v3_0829_native_train.txt",
        "annotation_txt_file_val": "scene_list/ball_catch_triview_v3_0829_native_validation.txt",
        "camera_list": {
            2: ["front_left", "front_right"],
            3: ["front_left", "front_right", "lower_front"],
        },
        "ref_camera": "front_left",
    },

    # 0902_fixed, sharing a data_root with v3_0829.
    #
    # lower_front loses the ball partway through. Not field of view -- the ball
    # stays near the image centre throughout -- but the catch net occluding it. In
    # 39 of 50 scenes frame 15, the extrapolation anchor, is left with only the
    # horizontal front_left/front_right baseline, so vertical parallax is missing
    # at the frame that matters most. This is the real rig, not a data defect.
    #
    # The catch is at frame 45, not the annotation's first_contact_frame of 30:
    # that field was computed from frame 15 instead of the release frame, off by
    # exactly 15. Evaluation reads the config's stream25_catch_frame.
    "ball_catch_triview_0902_fixed": {
        "size": [320, 240],
        "temporal": True,
        "num_context_timesteps": 6,
        "num_target_timesteps": 7,
        "annotation_txt_file_train": "scene_list/ball_catch_triview_0902_fixed_train.txt",
        "annotation_txt_file_val": "scene_list/ball_catch_triview_0902_fixed_validation.txt",
        "camera_list": {
            2: ["front_left", "front_right"],
            3: ["front_left", "front_right", "lower_front"],
        },
        "ref_camera": "front_left",
    },

    # 0903_2k: the full version of 0902_fixed.
    #
    # size is (H, W); the annotation's own fields use the same order. Verify with
    # the intrinsics:
    #       fx_px = 0.6765179 * W(240) = 162.3643
    #       fy_px = 0.5073884 * H(320) = 162.3643      equal, so pixels are square
    # Writing size the other way round swaps these to 216.5 / 121.8: wrong geometry,
    # no error. The 243.5464 in provenance is the pre-downsample focal; do not use it.
    #
    # lower_front loses the ball from frame 16, occluded by the catch net as in
    # 0902_fixed, leaving only the horizontal baseline past the anchor frame.
    #
    # The catch frame is 45. Neither first_contact_frame=30 nor catch_plane_x_rig
    # is right: the latter solves to frame 29, which looks like independent
    # confirmation but is the same error on another axis -- the ball is 3.77 m up
    # there, while the rig is 1.5 m. Frame 45.15 returns to the release height.
    "ball_catch_triview_0903_2k": {
        "size": [320, 240],
        "temporal": True,
        "num_context_timesteps": 6,
        "num_target_timesteps": 7,
        "annotation_txt_file_train": "scene_list/ball_catch_triview_0903_2k_train.txt",
        "annotation_txt_file_val": "scene_list/ball_catch_triview_0903_2k_validation.txt",
        "camera_list": {
            2: ["front_left", "front_right"],
            3: ["front_left", "front_right", "lower_front"],
        },
        "ref_camera": "front_left",
    },

    "ball_catch_triview_0908_10k": {
        "size": [320, 240],
        "temporal": True,
        "num_context_timesteps": 6,
        "num_target_timesteps": 7,
        "annotation_txt_file_train": "scene_list/ball_catch_triview_0908_10k_train.txt",
        "annotation_txt_file_val": "scene_list/ball_catch_triview_0908_10k_validation.txt",
        "camera_list": {
            2: ["front_left", "front_right"],
            3: ["front_left", "front_right", "lower_front"],
        },
        "ref_camera": "front_left",
    },

    "waymo": {
        "size": [160, 240],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": waymo_train,
        "annotation_txt_file_val": waymo_val,
        "camera_list": {
            1: ["0"],
            3: ["1", "0", "2"],
            5: ["3", "1", "0", "2", "4"],
            6: ["3", "1", "0", "2", "4"],  # capped at 5
            7: ["3", "1", "0", "2", "4"],
        },
        "ref_camera": "0",
    },
    "nuscenes": {
        "size": [160, 288],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": "scene_list/nuscenes_train.txt",
        "annotation_txt_file_val": "scene_list/nuscenes_val.txt",
        "camera_list": {
            1: ["CAM_FRONT"],
            3: ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
            5: [
                "CAM_BACK_LEFT",
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
            ],
            6: [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ],
            7: [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ],
        },
        "ref_camera": "CAM_FRONT",
    },
    "argoverse2": {
        "size": [192, 256],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": "scene_list/argoverse2_train.txt",
        "annotation_txt_file_val": "scene_list/argoverse2_val.txt",
        "camera_list": {
            1: ["0"],
            3: ["1", "0", "2"],
            5: ["3", "1", "0", "2", "4"],
            6: ["3", "1", "0", "2", "4"],
            7: ["5", "3", "1", "0", "2", "4", "6"],
        },
        "ref_camera": "0",
    },
    "rel10k": {
        "size": [160, 296],
        "temporal": False,
        "num_context_timesteps": 2,
        "num_target_timesteps": 8,
        "annotation_txt_file_train": "scene_list/rel10k_train.txt",
        "annotation_txt_file_val": "scene_list/rel10k_val.txt",
        "batch_size_scale": 6,
        "camera_list": {
            1: ["0"],
            3: ["0"],
            5: ["0"],
            6: ["0"],
            7: ["0"],
        },
    },
    "dl3dv": {
        "size": [160, 288],
        "temporal": False,
        "num_context_timesteps": 2,
        "num_target_timesteps": 8,
        "annotation_txt_file_train": "scene_list/dl3dv_train.txt",
        "annotation_txt_file_val": "scene_list/dl3dv_val.txt",
        "batch_size_scale": 6,
        "camera_list": {
            1: ["0"],
            3: ["0"],
            5: ["0"],
            6: ["0"],
            7: ["0"],
        },
    },
}

FEAT_TYPES = ["pe3r", "lseg"]
ONLINE_FEAT_TYPES = ["lseg"]

# text label list
# NOTE: If you change this category, all the corresponding categories below it must also be changed.
# For example WAYMO_SEMANTIC_ID_TO_LABEL, B2D_SEMANTIC_ID_TO_LABEL, DRIVING_SIM_SEMANTIC_ID_TO_LABEL
SEMANTIC_LABEL_LIST = ['others', 'vehicle', 'bicycle', 'motorcycle', 'people', 'buildings', 'road', 'sidewalk', 'sky']

# text label to idx
SEMANTIC_LABEL_TO_IDX = {}
for i, label in enumerate(SEMANTIC_LABEL_LIST):
    SEMANTIC_LABEL_TO_IDX[label] = i

SEMANTIC_ID_TO_COLOR = [
    [99., 178., 50.],  # others
    [199., 2., 14.],  # vehicle
    [203., 56., 122.],  # bicycle
    [153., 56, 58.],  # motorcycle
    [252., 200., 0.],  # people
    [128., 128., 128.],  # buildings
    [50., 181., 199.],  # road
    [236., 110., 3.],  # sidewalk
    [34., 121., 182.],  # sky
]

def textlabel_to_id(SEMANTIC_ID_TO_LABEL):
    SEMANTIC_LABEL_TO_ID = {}
    for key, value in SEMANTIC_ID_TO_LABEL.items():
        assert value in SEMANTIC_LABEL_LIST, f"Class [{value}] not in SEMANTIC_LABEL_LIST!"
        if value not in SEMANTIC_LABEL_TO_ID:
            SEMANTIC_LABEL_TO_ID[value] = [key]
        else:
            SEMANTIC_LABEL_TO_ID[value].append(key)
    return SEMANTIC_LABEL_TO_ID

def id_to_idx(SEMANTIC_ID_TO_LABEL):
    SEMANTIC_ID_TO_IDX = {}
    for key, value in SEMANTIC_ID_TO_LABEL.items():
        SEMANTIC_ID_TO_IDX[key] = SEMANTIC_LABEL_TO_IDX[value]
    return SEMANTIC_ID_TO_IDX

# Semantic classes for the camera segmentation labels.
# ref: https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/protos/camera_segmentation.proto#L72
WAYMO_SEMANTIC_CLASSES = {
    "TYPE_UNDEFINED": 0,  # Anything that does not fit the other classes or is too ambiguous to label.
    "TYPE_EGO_VEHICLE": 1, # The Waymo vehicle.
    "TYPE_CAR": 2,  # Small vehicle such as a sedan, SUV, pickup truck, minivan or golf cart.
    "TYPE_TRUCK": 3,  # Large vehicle that carries cargo.
    "TYPE_BUS": 4,  # Large vehicle that carries more than 8 passengers.
    "TYPE_OTHER_LARGE_VEHICLE": 5,  # Large vehicle that is not a truck or a bus.
    "TYPE_BICYCLE": 6,  # Bicycle with no rider.
    "TYPE_MOTORCYCLE": 7,  # Motorcycle with no rider.
    "TYPE_TRAILER": 8,  # Trailer attached to another vehicle or horse.
    "TYPE_PEDESTRIAN": 9,  # Pedestrian. Does not include objects associated with the pedestrian, such as suitcases, strollers or cars.
    "TYPE_CYCLIST": 10,  # Bicycle with rider.
    "TYPE_MOTORCYCLIST": 11,  # Motorcycle with rider.
    "TYPE_BIRD": 12,  # Birds, including ones on the ground.
    "TYPE_GROUND_ANIMAL": 13,  # Animal on the ground such as a dog, cat, cow, etc.
    "TYPE_CONSTRUCTION_CONE_POLE": 14,  # Cone or short pole related to construction.
    "TYPE_POLE": 15,  # Permanent horizontal and vertical lamp pole, traffic sign pole, etc.
    "TYPE_PEDESTRIAN_OBJECT": 16,  # Large object carried/pushed/dragged by a pedestrian.
    "TYPE_SIGN": 17,  # Sign related to traffic, including front and back facing signs.
    "TYPE_TRAFFIC_LIGHT": 18,  # The box that contains traffic lights regardless of front or back facing.
    "TYPE_BUILDING": 19,  # Permanent building and walls, including solid fences.
    "TYPE_ROAD": 20,  # Drivable road with proper markings, including parking lots and gas stations.
    "TYPE_LANE_MARKER": 21,  # Marking on the road that is parallel to the ego vehicle and defines lanes.
    "TYPE_ROAD_MARKER": 22,  # All markings on the road other than lane markers.
    "TYPE_SIDEWALK": 23,  # Paved walkable surface for pedestrians, including curbs.
    "TYPE_VEGETATION": 24,  # Vegetation including tree trunks, tree branches, bushes, tall grasses, flowers and so on.
    "TYPE_SKY": 25,  # The sky, including clouds.
    "TYPE_GROUND": 26,  # Other horizontal surfaces that are drivable or walkable.
    "TYPE_DYNAMIC": 27,  # Object that is not permanent in its current position and does not belong to any of the above classes.
    "TYPE_STATIC": 28,  # Object that is permanent in its current position and does not belong to any of the above classes.
}

# Waymo: id to text label (others, vehicle, bicycle, motorcycle, people, buildings, road, sidewalk, sky)
WAYMO_SEMANTIC_ID_TO_LABEL = {
    0: "others",
    1: "others",
    2: "vehicle",
    3: "vehicle",
    4: "vehicle",
    5: "vehicle",
    6: "bicycle",
    7: "motorcycle",
    8: "others",
    9: "people",
    10: "bicycle",
    11: "motorcycle",
    12: "others",
    13: "others",
    14: "others",
    15: "others",
    16: "others",
    17: "others",
    18: "others",
    19: "buildings",
    20: "road",
    21: "road",  # lane marker
    22: "road",  # road_marker
    23: "sidewalk",  # merge to road?
    24: "others",  # "tree",  # vegetation: Vegetation including tree trunks, tree branches, bushes, tall grasses, flowers and so on.
    25: "sky",
    26: "road",  # ground: Other horizontal surfaces that are drivable or walkable.
    27: "others",  # dynamic: Object that is not permanent in its current position and does not belong to any of the above classes.
    28: "others",  # static: Object that is permanent in its current position and does not belong to any of the above classes.
}

# Semantic classes for the camera segmentation labels.
B2D_SEMANTIC_CLASSES = {
    "Unlabeled": 0,  # Elements that have not been categorized are considered Unlabeled. This category is meant to be empty or at least contain elements with no collisions.
    "Roads": 1, # Part of ground on which cars usually drive. E.g. lanes in any directions, and streets.
    "SideWalks": 2,  # Part of ground designated for pedestrians or cyclists. Delimited from the road by some obstacle (such as curbs or poles), not only by markings. This label includes a possibly delimiting curb, traffic islands (the walkable part), and pedestrian zones.
    "Building": 3,  # Buildings like houses, skyscrapers,... and the elements attached to them. E.g. air conditioners, scaffolding, awning or ladders and much more.
    "Wall": 4,  # Individual standing walls. Not part of a building.
    "Fence": 5,  # Barriers, railing, or other upright structures. Basically wood or wire assemblies that enclose an area of ground.
    "Pole": 6,  # Small mainly vertically oriented pole. If the pole has a horizontal part (often for traffic light poles) this is also considered pole. E.g. sign pole, traffic light poles.
    "TrafficLight": 7,  # Traffic light boxes without their poles.
    "TrafficSign": 8,  # Signs installed by the state/city authority, usually for traffic regulation. This category does not include the poles where signs are attached to. E.g. traffic- signs, parking signs, direction signs...
    "Vegetation": 9,  # Trees, hedges, all kinds of vertical vegetation. Ground-level vegetation is considered Terrain.
    "Terrain": 10,  # Grass, ground-level vegetation, soil or sand. These areas are not meant to be driven on. This label includes a possibly delimiting curb.
    "Sky": 11,  # Open sky. Includes clouds and the sun.
    "Pedestrian": 12,  # Humans that walk
    "Rider": 13,  # Humans that ride/drive any kind of vehicle or mobility system E.g. bicycles or scooters, skateboards, horses, roller-blades, wheel-chairs, etc. .
    "Car": 14,  # Cars, vans
    "Truck": 15,  # Trucks
    "Bus": 16,  # Busses
    "Train": 17,  # Trains
    "Motorcycle": 18,  # Motorcycle, Motorbike
    "Bicycle": 19,  # Bicylces
    "Static": 20,  # Elements in the scene and props that are immovable. E.g. fire hydrants, fixed benches, fountains, bus stops, etc.
    "Dynamic": 21,  # Elements whose position is susceptible to change over time. E.g. Movable trash bins, buggies, bags, wheelchairs, animals, etc.
    "Other": 22,  # Everything that does not belong to any other category.
    "Water": 23,  # Horizontal water surfaces. E.g. Lakes, sea, rivers.
    "RoadLine": 24,  # The markings on the road.
    "Ground": 25,  # Any horizontal ground-level structures that does not match any other category. For example areas shared by vehicles and pedestrians, or flat roundabouts delimited from the road by a curb.
    "Bridge": 26,  # Only the structure of the bridge. Fences, people, vehicles, an other elements on top of it are labeled separately.
    "RailTrack": 27,  # All kind of rail tracks that are non-drivable by cars. E.g. subway and train rail tracks.
    "GuardRail": 28,  # All types of guard rails/crash barriers.
}

# B2D: id to text label (others, vehicle, bicycle, motorcycle, people, buildings, road, sidewalk, sky)
B2D_SEMANTIC_ID_TO_LABEL = {
    0: "others",
    1: "road",
    2: "sidewalk",
    3: "buildings",
    4: "others",
    5: "others",
    6: "others",
    7: "others",
    8: "others",
    9: "others",
    10: "others",
    11: "sky",
    12: "people",
    13: "people",
    14: "vehicle",
    15: "vehicle",
    16: "vehicle",
    17: "vehicle",
    18: "motorcycle",
    19: "bicycle",
    20: "others",
    21: "others",
    22: "others",
    23: "others",
    24: "road",
    25: "sidewalk",
    26: "others",
    27: "others",
    28: "others",
}

# Semantic classes for the camera segmentation labels.
DRIVING_SIM_SEMANTIC_CLASSES = {
    "building": 0,
    "road": 1,
    "car": 2,  # car/truck
    "sidewalk": 3,  # sidewalk/sideway
    "tree": 4,
    "bench": 5,
    "signboard": 6,
    "hydrant": 7,
    "streetlamp": 8,
    "trafficlight": 9,
    "freeway": 10,
    "trashcan": 11,
    "fence": 12,  # fence/guardrail
    "mailbox": 13,
    "parkingmeter": 14,
    "buststop": 15,
    "trafficsign": 16,
    "other": 17,
    "sky": 18,
    "ground": 19,
    "pole": 20,
    "vegetation": 21,
    "person": 22,
    "trafficline": 23,
    "dynamic_car": 24,
}

# DRIVING_SIM: id to text label (others, vehicle, bicycle, motorcycle, people, buildings, road, sidewalk, sky)
DRIVING_SIM_SEMANTIC_ID_TO_LABEL = {
    0: "buildings",
    1: "road",
    2: "vehicle",
    3: "sidewalk",
    4: "others",
    5: "others",
    6: "others",
    7: "others",
    8: "others",
    9: "others",
    10: "road",
    11: "others",
    12: "others",
    13: "others",
    14: "others",
    15: "others",
    16: "others",
    17: "others",
    18: "sky",
    19: "road",
    20: "others",
    21: "others",
    22: "people",
    23: "others",
    24: "vehicle",
}

# Datasets mapping: text label to id
WAYMO_SEMANTIC_LABEL_TO_ID = textlabel_to_id(WAYMO_SEMANTIC_ID_TO_LABEL)
B2D_SEMANTIC_LABEL_TO_ID = textlabel_to_id(B2D_SEMANTIC_ID_TO_LABEL)
DRIVING_SIM_SEMANTIC_LABEL_TO_ID = textlabel_to_id(DRIVING_SIM_SEMANTIC_ID_TO_LABEL)

# Datasets mapping: id to idx
WAYMO_SEMANTIC_ID_TO_IDX = id_to_idx(WAYMO_SEMANTIC_ID_TO_LABEL)
B2D_SEMANTIC_ID_TO_IDX = id_to_idx(B2D_SEMANTIC_ID_TO_LABEL)
DRIVING_SIM_SEMANTIC_ID_TO_IDX = id_to_idx(DRIVING_SIM_SEMANTIC_ID_TO_LABEL)

SEMANTIC_ID_TO_IDX_DICT = {
    'waymo': WAYMO_SEMANTIC_ID_TO_IDX,
    'b2d': B2D_SEMANTIC_ID_TO_IDX,
    'driving_sim': DRIVING_SIM_SEMANTIC_ID_TO_IDX,
    'ball_catch_24cm_stereo40_stream25_nopitch': {0: 0, 1: 0, 2: 5, 3: 0},  # 4-class task semantic, 30-frame 6-context variant
    'ball_catch_24cm_triview': {0: 0, 1: 0, 2: 5, 3: 0},  # 4-class tri-view task semantic
}
