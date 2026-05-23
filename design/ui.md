there will be the basic screen. 
There is no point in asking if its a show or a movie, coz a movie is just a show with one episode. 

we do want to ask the user if they want moderate strict or lineant censoring

but how do you say that? you can put it in brackets, or directly ask. 


so 

Enter the complete path of your media folder (TV Show / Movie) that contains n videos with n subttile files of the same name. 

/path/to/media/folder

Enter censorship strength: 
1. Moderate - Only Explicit on screen exposed nudity is removed. 
2. Strict - Almost all on screen nudity is removed. If any story is present in said scenes, an AI generated summary is provided on a black screen later. Profane dialogues are muted and their subttiles replaced by AI generated sentences with the same meaning. 

You will have `<video_name>_censored_<moderate/strict>.<ext>` in your media folder along with subtitles of the same name. 

Detected 10 videos and 10 subtitles.
Do you want to continue? (y/n)

y/n

Processing Video 1 of 10
// dynamic description changing progress bar.











we gotta prep a series of test recordings from different shows that we gotta censor and them clump them together into cases rather than creating cases. Coz there seems to be a lot. 

we really need the ai to identify somehow that there are sex scenes or obscene dialogues going on that we wanna cut. Its gotta be possible man. at this point im ready to train some models of my own. 

so the idea is every time the image is changing a lot more than the previous one, we can call it a scene. there would be some 200 or 300 of these in a show? then we take a few pics from that scene, which will look mostly the same, and ask a descriptor model to describe what is happening. Then we pick out those words, and subtitles from a chromadb db to get us some context. finally we feed all this to a model that then tells us if we gotta cut this scene or not. You cut it, cleanly summarize it, and show it on screen. Then you move on. 

well be splitting subtitles so well put it in a pandas df. every few scenes will have one image. well save this image in a temp folder, and have its location there. well later delete this folded. a column in this df will be description of those images. 