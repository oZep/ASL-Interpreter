# HandTracking (ASL Interpreter)

[23/12/2023] - Present

- Using OpenCV and MediaPy

https://github.com/oZep/HandTracking/assets/97713154/899c24fd-c1d9-43c3-a08b-800a034008db


# Main Idea
<img width="1073" alt="hand-landmarks" src="https://github.com/oZep/HandTracking/assets/97713154/cd23c2fd-d38d-4bfe-b095-496a580b42c3">

![image](https://github.com/oZep/HandTracking/assets/97713154/a7412c7e-72fe-49c8-8981-b5f996b71e01)

This module will help interpret these symbols and enable beginners to understand ASL instantly. 

After the main ASL interpreter module is complete, I will take this project into the real world and decode ASL via Audio connected to a headset

## Progress
- Main Landmark Decoding Logic Complete
- Finished assigning Alphabet Landmarks it's correct POI (points of interest) and set up a way to have orientation of the hand be considered (ORT)
- Testing all characters, I found the program has issues distingishing between gestures that are very similar to eachother but very by lenght between landmarks, especially with C and O. I think calculating the distance between all hand landmarks will be the next step in making the system more precise. 
