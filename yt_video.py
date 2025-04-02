import yt_dlp

url = "https://www.youtube.com/watch?v=CftLBPI1Ga4" 

ydl_opts = {
    'format': 'best',  # Download best single format (with video and audio combined)
    'outtmpl': '%(title)s.%(ext)s',  # Save with video title as filename
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("Download complete!")
