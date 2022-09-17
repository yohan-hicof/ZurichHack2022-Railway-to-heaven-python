# importing the module 
from pytube import YouTube 

list_links = ["https://www.youtube.com/watch?v=zomZywCAPTA",
             "https://www.youtube.com/watch?v=Mw9qiV7XlFs",
             "https://www.youtube.com/watch?v=nKOe2PuERD0",
             "https://www.youtube.com/watch?v=CIRlmM8wI1g",
             "https://www.youtube.com/watch?v=XPEfXF5S0wc",
              "https://www.youtube.com/watch?v=523mJGNPiH0",
              "https://www.youtube.com/watch?v=p8yspuIVR-Q",
              "https://www.youtube.com/watch?v=DNy6F7ZwX8I",
              "https://www.youtube.com/watch?v=obMiCTXoB4s",
              "https://www.youtube.com/watch?v=Tb7MSsKyAcI",
              "https://www.youtube.com/watch?v=Mx1bhQMcCX8"
]

out_path = "./download"
# idea: get the curvature locally to determine the oposite rail distance


def dl_video(path):
    yt = YouTube(path)

    print(f"Title: {yt.title}")

    streams = yt.streams.filter(file_extension='mp4', res="720p")

    for s in streams:
        print(f"Stream: {s}, {s.itag}")
        stream = yt.streams.get_by_itag(s.itag)
        stream.download(out_path)
        break


if __name__ == "__main__":
    for i, link in enumerate(list_links):
        if i < 9:
            continue
        dl_video(link)
