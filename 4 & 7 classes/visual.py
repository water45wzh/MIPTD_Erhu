import sed_vis
import sed_vis.io
import os
file_path = "/home/lijingru/3classes/data3/test"

# Load audio signal first
audio, fs = sed_vis.io.load_audio(os.path.join(file_path, "二胡-独奏.wav"))

# Load event lists
reference_event_list = sed_vis.io.load_event_list('label.ann')
estimated_event_list = sed_vis.io.load_event_list('ans.ann')
event_lists = {'reference': reference_event_list, 'estimated': estimated_event_list}

# Visualize the data
vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,
                                                audio_signal=audio,
                                                sampling_rate=fs)
# vis.show()

vis.save("res.png")