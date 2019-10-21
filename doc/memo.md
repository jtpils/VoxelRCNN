# Memos

## Dataset

### Format Converstion

The dataset is composed by several parts. original images/lidar in .jpeg and .bin files. e.g.:
`train_images/host-a004_cam0_1232815252251064006.jpeg`, `train_lidar/host-a004_lidar1_1232815252301696606.bin`. These files are all cooresponding to the `sample_data.json` file.

The connotations are in another folder what the fuck!!!!! plz dont lost connection plz.

### Json format

1. sample_data
   - dict_keys(['is_key_frame', 'prev', 'fileformat', 'token', 'timestamp', 'next', 'ego_pose_token', 'sample_token', 'filename', 'calibrated_sensor_token'])
   - length: ?
2. sample_annotation
    - dict_keys(['token', 'num_lidar_pts', 'size', 'sample_token', 'rotation', 'prev', 'translation', 'num_radar_pts', 'attribute_tokens', 'next', 'instance_token', 'visibility_token'])
    - length: ?
3. attributes
    - dict_keys(['description', 'token', 'name'])
4. calibrated_sensor
    - length: 148
    - dict_keys(['sensor_token', 'rotation', 'camera_intrinsic', 'translation', 'token'])

### Table names

* 9 category,
* 18 attribute,
* 4 visibility,
* 18421 instance,
* 10 sensor,
* 148 calibrated_sensor,
* 177789 ego_pose,
* 180 log,
* 180 scene,
* 22680 sample,
* 189504 sample_data,
* 638179 sample_annotation,
* 1 map

### Table::Sample

Sample.keys:
dict_keys(['next', 'prev', 'token', 'timestamp', 'scene_token', **'data'**, 'anns'])

Sample['data'].keys():
dict_keys(['CAM_BACK', 'CAM_FRONT_ZOOMED', 'LIDAR_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT', 'LIDAR_TOP', 'LIDAR_FRONT_LEFT'])

### Table::Sample_data

sample data as lidar, keys:
dict_keys(['is_key_frame', 'prev', 'fileformat', 'token', 'timestamp', 'next', 'ego_pose_token', 'sample_token', 'calibrated_sensor_token', 'filename', 'sensor_modality', 'channel'])

sample data as image: token itself.

lidar render time: ~6.797s
image render time: ~1.175s