from lyft_dataset_sdk.lyftdataset import LyftDataset
from config import cfg










if __name__ == "__main__":
    cfg = cfg.data
    lyft_data = LyftDataset(data_path=cfg.lyft,
                            json_path=cfg.train_path,
                            verbose=False)
    one_scene = lyft_data.scene[0]
    first_sample_token = one_scene["first_sample_token"]
    sample = lyft_data.get('sample', first_sample_token)
    sample_data = sample['data']
    lidar_top_channel = 'LIDAR_FRONT_RIGHT'
    lidar_token = sample_data[lidar_top_channel]
    lidar_data = lyft_data.get('sample_data', lidar_token)
    cs_lidar = lyft_data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    cam_front_channel = 'CAM_FRONT_ZOOMED'
    cam_token = sample_data[cam_front_channel]
    cam_data = lyft_data.get("sample_data", cam_token)
    cs_cam = lyft_data.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

    my_annotation_token = sample['anns'][10]
    my_annotation =  sample_data.get('sample_annotation', my_annotation_token)
    lyft_data.render_annotation(my_annotation_token)
    
    lyft_data.render_sample_3d_interactive(sample['token'])
    
    pass