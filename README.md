# Road-Lane-Line-Detection

[![Road Lane Line Detection](https://yt-embed.herokuapp.com/embed?v=kK9qB37BV6w)](https://www.youtube.com/watch?v=kK9qB37BV6w "Road Lane Line Detection")

Lane detection is a critical component of self-driving cars and autonomous vehicles, In this project i did Road Lane Line Detection using OpenCV library. I implemented this algorithm with the following main steps:

1.Video divisions for frames, pre processing for each frames, converting to gray scale and smoothing using gaussian blur.

![gray_image](https://user-images.githubusercontent.com/50642442/125351187-72f9a780-e368-11eb-91a4-1e928737a0bf.jpg)

2.Apply canny to detect the image edge.

![canny_image](https://user-images.githubusercontent.com/50642442/125351272-945a9380-e368-11eb-8e3b-d566ed4f8244.jpg)

3.Select a specific area and apply masking region to the rode lane area.

![mask](https://user-images.githubusercontent.com/50642442/125351463-ca981300-e368-11eb-9c90-9528193f11d8.jpg)

4.Apply Hough transform to detect multiple lines in area of interest.
5.Extrapolate the lines found in the hough transform to construct the left and right lane lines.

![color_lines](https://user-images.githubusercontent.com/50642442/125351670-0632dd00-e369-11eb-9b7e-d8658613e3de.jpg)

6.Blending the original frame and the lines found using weighted img function.

![weighted img](https://user-images.githubusercontent.com/50642442/125353757-a1c54d00-e36b-11eb-99c8-36c866f45048.jpg)



