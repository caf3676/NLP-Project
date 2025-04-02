from pytube import YouTube
import whisper
from whisper.utils import get_writer
import tempfile
import os

#set of non-functioning youtube links (USE singleTranscriber FOR THESE)
#"https://www.youtube.com/watch?v=nRETwx1W6nM&list=PLCnlJOMhMC0OVavLPG44SmgTJdN8wpV97&index=4", 
# "https://www.youtube.com/watch?v=M2t45zei6Bc&list=PLCnlJOMhMC0OVavLPG44SmgTJdN8wpV97&index=6", 
# "https://www.youtube.com/watch?v=u2Lr2Qdfq5k&list=PLCnlJOMhMC0OVavLPG44SmgTJdN8wpV97&index=5", 
# "https://www.youtube.com/watch?v=cCifxEysbqo&list=PLCnlJOMhMC0OVavLPG44SmgTJdN8wpV97&index=7",
# "https://www.youtube.com/watch?v=CdsQXP1iUIU&list=PLCnlJOMhMC0OVavLPG44SmgTJdN8wpV97&index=8",
# "https://www.youtube.com/watch?v=6Kn64FROhGE&list=PLCnlJOMhMC0OVavLPG44SmgTJdN8wpV97&index=9"
# "https://www.youtube.com/watch?v=Pv0lWrCvGJQ&list=PLCnlJOMhMC0OVavLPG44SmgTJdN8wpV97&index=10"
# "https://www.youtube.com/watch?v=hLBvNPG6UZc&list=PLCnlJOMhMC0OVavLPG44SmgTJdN8wpV97&index=11"
# "https://www.youtube.com/watch?v=V9DphcYeYr0&list=PLCnlJOMhMC0OVavLPG44SmgTJdN8wpV97&index=12"
#"https://youtu.be/IGPayWv1BBM"
#"https://youtu.be/NSvyljWT4mw" 
#"https://youtu.be/Ppwfr8H177M" 
#"https://youtu.be/wCgdoImlLY0" 
#"https://youtu.be/w0HgVibxpMQ" 
#"https://youtu.be/rN0RKcTIVcI" 
#"https://youtu.be/IQ7Mzy0HWxc" 
#"https://youtu.be/QwxQ9xuUHlU" 
#"https://youtu.be/j0I5SGFm00c" 
#"https://youtu.be/jVRLGOsnYuw" 
#"https://youtu.be/mzPLicovE7Q" 
#"https://youtu.be/NE4xfhIHAm4" 
#"https://youtu.be/Li-Xzlu7hvs" 
#"https://youtu.be/bRj3HnEa1Z4" 
#"https://youtu.be/BCnivpSKF18" 
#"https://youtu.be/CVkmuwF8cJ8" 
#"https://youtu.be/3cLqK3lR56Y"
#"https://youtu.be/6mCfgbh7f2g"
#"https://youtu.be/mvxE0dAT38c"
#"https://youtu.be/AVms8JoUWXc" 
#"https://youtu.be/Yu9jDlqBrJE" 
#"https://youtu.be/D1UKZGOYDOg" 
#"https://youtu.be/bRj3HnEa1Z4" 
#"https://youtu.be/Y2AdJnlvuNw"

#set of functional youtube links (USE youTranscribe FOR THESE)
links = ["https://youtu.be/FLgww1kfhgQ", "https://youtu.be/wCqPCA0yCW0", "https://youtu.be/3S4NybR5RP8", "https://youtu.be/pYvsb-uhnUI", "https://youtu.be/Z9Mob1GZfKc", "https://youtu.be/urF-5L8DcqM", "https://youtu.be/1X2BEFHnfzg", "https://youtu.be/UJQZJcow6Ek", "https://youtu.be/xJxa0nDZ9lk", "https://youtu.be/REsXkfCRrbg", "https://youtu.be/qbg6zO6u0Pk", "https://youtu.be/-x2kvXEHthw", "https://youtu.be/i68PxSdCHDU", "https://youtu.be/Ik0S8CBlozM", "https://youtu.be/nfai9a3YyqI", "https://youtu.be/GhLj84tY84c", "https://youtu.be/rxW-bALKAcs", "https://youtu.be/8jmOuqVrVvY", "https://youtu.be/Fm_4szW1IQw", "https://youtu.be/sYcrlxP6z2A", "https://youtu.be/h08ncZKxWq0", "https://youtu.be/nICtNIU_uqA", "https://youtu.be/vvvmGmZYOBo", "https://youtu.be/F7UfzX_ESps", "https://youtu.be/icmjtX7Muwo", "https://youtu.be/5lYmKkZ4FEg", "https://youtu.be/_u00N-LNLDU", "https://youtu.be/QqGsPmJuwvQ", "https://youtu.be/ua5JShO9E-M", "https://youtu.be/c8K1SDQaWB4", "https://youtu.be/Y2AdJnlvuNw", "https://youtu.be/H2FnFjChujU", "https://youtu.be/96qJYaEeaks", "https://youtu.be/UcEv2u55BB8", "https://youtu.be/7nUvB-Yxqls", "https://youtu.be/WTRv9PPgDGE", "https://youtu.be/-L6qJmG8BYw", "https://youtu.be/Vzijm68Jt4k", "https://youtu.be/M-iLizq_QFM", "https://youtu.be/ywOm3bZJDK0", "https://youtu.be/abABHh5Dzsc", "https://youtu.be/8lTRdGEPbBw", "https://youtu.be/zXJZDfwtQdA", "https://youtu.be/9mlaORuPpDc", "https://youtu.be/SfAICo_wmN8", "https://youtu.be/0FFARgE_RpU", "https://youtu.be/MZssdUzOWTE", "https://youtu.be/Wrlc5kY05go", "https://youtu.be/nolR-swQqgw", "https://youtu.be/6FcPy5kyJVg", "https://youtu.be/w7epeL9X0lQ", "https://youtu.be/PEJAmI69OBo", "https://youtu.be/xiSYtmrwfSU", "https://youtu.be/rePPjmcCa54", "https://youtu.be/pjT4DMJTQLI", "https://youtu.be/indhThEotCA", "https://youtu.be/MWoS-OXRi54", "https://youtu.be/kCyl-R3utQc", "https://youtu.be/F-oENvXSBAw", "https://youtu.be/SFrVC-Sub48", "https://youtu.be/37WA_ZBXMeo", "https://youtu.be/7Ndgkokq5Q8", "https://youtu.be/lKE_y2EI5k4", "https://youtu.be/y1m3TrfNYDI", "https://youtu.be/sgb6R5vQk8U", "https://youtu.be/JW9cECna9dg", "https://youtu.be/fHgcILLymeM", "https://www.youtube.com/watch?v=DYJLQQPRjYc&list=PLTQMmRSAZ8KAYYpytmABnIuDHxBecysx6&index=31", "https://youtu.be/rg9eaDAMtTg", "https://youtu.be/PiLYfWGEUUs", "https://youtu.be/n5Yc05bsKPI", "https://youtu.be/2k1bj7Yysbc", "https://youtu.be/7ISp_7eJDOo", "https://youtu.be/igxg4BDH7x4"]

#Transcribes a list of videos and returns text files for each transcribed video
def youTranscribe(links):
    for url in links:
        yt = YouTube(url)
        youtube_title = yt.title
        
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download('INSERT DIRECTORY HERE')
        
        model = whisper.load_model("base")
        result = model.transcribe("INSERT DIRECTORY HERE\\" + str(youtube_title) + ".mp4")
        
        with open(str(youtube_title) + ".txt", "w", encoding="utf-8") as txt:
            txt.write(result["text"])
            
#Transcribes a single youtube video and returns a text file of the transcribed audio
def singleTrancriber(link):
    YOUTUBE_VIDEO =  link
    if not os.path.exists("transcription.txt"):
        youtube = YouTube(YOUTUBE_VIDEO)
        audio = youtube.streams.filter(only_audio=True).first()
        whisper_model = whisper.load_model("base")
        with tempfile.TemporaryDirectory() as tmpdir:
            file = audio.download(output_path=tmpdir)
            transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

            with open("transcription.txt", "w") as file:
                file.write(transcription)
    
#youTranscribe(links)
#singleTranscribe(link)