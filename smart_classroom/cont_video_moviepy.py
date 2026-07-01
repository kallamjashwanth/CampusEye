from moviepy import VideoFileClip

videos = [
    "videos/iot_empty_classroom.mp4",
    # "videos/iot_output_students_entering.mp4",
    "videos/iot_output_students_leaving.mp4"
]

for video in videos:
    output = video.replace(".mp4", "_web.mp4")
    clip = VideoFileClip(video)
    clip.write_videofile(output, codec="libx264", audio=False)
    print("Saved:", output)