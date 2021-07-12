# Road-Lane-Line-Detection

Lane detection is a critical component of self-driving cars and autonomous vehicles, In this project i did Road Lane Line Detection using OpenCV library. I implemented this algorithm with the following main steps:

1.Video divisions for frames, pre processing for each frames, converting to gray scale and smoothing using gaussian blur.
2.Apply canny to detect the image edge.
3.Select a specific area and apply masking region to the rode lane area.
4.Apply Hough transform to detect multiple lines in area of interest.
5.Extrapolate the lines found in the hough transform to construct the left and right lane lines. 
6.blending the original frame and the lines found using weighted img function.


<iframe width="560" height="315"
        src="https://www.youtube.com/embed/kK9qB37BV6w"
        title="YouTube video player"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen></iframe>



