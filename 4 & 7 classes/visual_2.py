import sed_vis
import dcase_util
import os
file_path = "/home/lijingru/3classes/data3/test"

# Load audio signal first
audio_container = dcase_util.containers.AudioContainer().load(
    os.path.join(file_path, "二胡-独奏.wav")
)

# Load event lists
reference_event_list = dcase_util.containers.MetaDataContainer().load(
    'label.ann'
)
estimated_event_list = dcase_util.containers.MetaDataContainer().load(
    'ans.ann'
)

event_lists = {
    'reference': reference_event_list, 
    'estimated': estimated_event_list
}

# Visualize the data
vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,
                                                audio_signal=audio_container.data,
                                                sampling_rate=44100)
vis.show()