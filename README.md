# Panorama stitching

The aim of this tutorial and program is to go over all the constituent parts, and theory contained in, panorama stitching, 
as well as providing a reference solution.
The best place to start is main.cpp, where each component is used in sequence, and then step into each function or file as necessary. 
Each component has a comment block that I will also explain here, along with reference texts.

I started this project at suggestion from a colleague to learn in-depth, from scratch, more about computer vision. Specifically, I wanted to learn, really learn, what features were, how they worked, how they were used, how you transformed between images, etc, and the various 'tool' algorithms computer vision engineers use, like RANSAC and Levenberg-Marquardt. The solution here is not the best solution. It is not optimised for efficiency, or beauty, or anything. I just coded it to work, and be understandable. But it does indeed work, and I learnt a lot doing it, and I hope anyone reading this can too. I encourage you to read the reference texts I give, learn the theory, and perhaps try to write the algorithms yourself, and compare results with what I have. Another good thing to try is to tweak the parameters I mention below, to 'retune' my solution, so to speak, and see how the results differ. I'll mention any specifically that I think would be good to try. A good dataset to work with, that was made to test panography programs, is Adobe's [panorama dataset]( https://sourceforge.net/projects/adobedatasets.adobe/files/adobe_panoramas.tgz/download).

As you read through the code, it's good to use this as a reference g

# Components
So to get a panorama, you take a photo of something, say a mountain, and then you turn a bit and take another photo, maybe of the side of the mountain and the sunset next to it. And then some magical stuff happens and la-di-da, out pops the two images made nicely into one, such that you can't even see the join (hopefully). So what happens under the hood?
I'll talk specifically about the case for two pictures, but it easily generalises - or iterates. Get the panorama from the two images, stitch in a third, and repeat.

In broad terms, we need to find what the images have in common, find a way to match those bits up, and then make sure that way works for everything else as well. More specifically, for each of the images, we need to find distinct features in them, create descriptors of those features so that we can potentially find that same feature elsewhere, try to match the features between the images, create a way of transforming from one image to the other based on how these matches line up, and then perform this transform for every pixel. 

Here is how I have broken down my code into components, based on the above:

## Feature Detection
"Features" can be a bit of a vague concept in computer vision, and it certainly was for me at the start. Just call some OpenCV function and magical dots appear on your image! But what are they? How do you get them?

There are a lot of different types of features, based on how you look for them. Here's a list of some:
- FAST features
- SIFT features
- SURF features
- ORB features


There are plenty more. Some are simple, some are ... rather complex (read the wikipedia page for SIFT features, and enjoy). They each might find slightly different things, but in general, what 'feature detectors' aim to do is find points in an image that are sufficiently distinct that you can easily find that same feature again in another image - a future one, or the other of a stereo pair, for example. Features are distinct things like corners (of a table, of an eye, of a leaf, whatever), or edges, or points in the image where there is a lot of change in more than just one direction. To give an example of what is not an image, think of a blank wall. Take a small part of it. Could you find that bit again on the wall? That exact bit? It's not very distinct, so you likely couldn't. Then take a picture of someone's face. If I gave you a small image snippet containing just a bit of the corner of an eye, you'd find where it fit very quickly. AIShack has a [rather good overview](http://aishack.in/tutorials/features/) of the general theory of features.

Here, I'm using FAST features, also called FAST corners because they are specifically designed to find corners. 

- OpenCV's [Explanation of FAST](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html)
- Here is a better [reference implementation of FAST](https://github.com/edrosten/fast-C-src), that is trained by a learner
- Here is the [original paper](https://www.edwardrosten.com/work/rosten_2006_machine.pdf)

OpenCV's explanation is pretty good, so I'll be brief with my own here, since it's ... heavily influenced and copied from that.

The idea behind FAST is that corners usually have two lines leading away from them, and that the intensity of the pixels in one of the two angles created by those lines will be either lighter or darker than the other. For example, think of the corner of a roof with the sun above. Two lines (the roof edges) come out from the corner, and below will be darker than above. The way a FAST feature detector works is that for each pixel, it scans a circle of 16 picels around it, about 3 pixels radius, and compares the intensities to the centre intensity (plus or minus a threshold). If there is a stretch of sequential pixels 12 or more in length that are all of greater intensity (plus a threshold) than the centre, or lesser intensity (minus a threshold) than the centre, this is deemed a FEATURE. (OpenCV's explanation has some better visuals)


### Tunable parameters
FAST_THRESHOLD, found in Features.h. Changing this will determine how much brighter or darker the ring of pixels around a certain point needs to be for it to be noted as a feature. Eg. a value of 1 means that too many points will be features - blank walls have a pixel-to-pixel variance that's likely bigger. A value of 100 means that you'll barely get any features at all, since you need a sequence of at least 12 pixels that are centre pixel intensity + 100, which ou tof 255 possible values is a large change. 

### Other notes:
- This could easily be parallelised. Try to figure out how, as an exercise. 

- You could calculate these features at multiple scales. This would register bigger objects in the image as features, which might help if you had large featureless areas with thick borders (a wall of a house, perhaps)


## Feature Scoring

Your average image might have over a thousand features - this is quite a lot to process later, as you'll see. I cut this list down by computing a score for each feature, and then having a cutoff point, where I keep no features with a score below a certain threshold. 
Once again, there are many methods of scoring features, and one of the most famous is the Shi-Tomasi score, invented by Shi and Tomasi in 1994. Here is their [original paper](http://www.ai.mit.edu/courses/6.891/handouts/shi94good.pdf).

AI Shack has a [good article](http://aishack.in/tutorials/shitomasi-corner-detector/) on the Shi Tomasi score, but it relies on some [background knowledge](http://aishack.in/tutorials/harris-corner-detector/), or having read the previous articles linked at the bottom (they're short and easy and good).

Essentially, for a the feature point and a small patch surrounding it, a matrix called the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) is computed. This is basically the two dimensional version of the gradient. The way we compute this is documented [here](http://aishack.in/tutorials/harris-corner-detector/). Then, we compute the eigenvalues for this matrix. Since this is a two-by-two matrix (see the previous link), the eigenvalues are just the solutions to a simple quadratic equation. The Shi-Tomasi score, then, is simply the minimum eigenvalue. 

Why? Why does this work? Well, for a two-by-two jacobian matrix, the eigenvalues define how strong the gradient is in the direction of the two eigen__vectors__ of the matrix. Basically, how much change we have in each direction. For a good corner, you should have a sharp image gradient (difference in pixel intensity) in both directions, so the minimum eigenvalue won't be that small. For just an edge, you'll have a sharp gradient in one direction but not the other, meaning one eigenvalue will be small. 

We then have a cutoff threshold for this value - another tunable parameter - and everything with a score below this - that is to say, every feature with a minimal eigenvalue of value lower than this threshold - is thrown away and ignored. Every feature with a higher score is kept. 


The final stage of the feature scoring is to perform Non-Maximal Suppresion (and unfortunately I can't find a good explanation online). The theory of Non-Maximal Suppression is that for where you have a group of points clustered in an area, like having, say, twenty features of a good score in the same little patch of the image ... you don't need all of these. You've already registered a strong feature there. So you suppress, that is to say, put aside, the weaker features within some radius of the strongest in that patch. In other words, you suppress the points that aren't maximal. Hence, non-maximal suppression. 

So we do this over our feature set that we've already cut down. For every feature, if there are any features in a 5x5 patch around it that are weaker, we suppress these too, just to reduce the amount we have to process over.


### Tunable parameters
ST_THRESH, found in Features.h. Changing this determines how many features we'll keep or throw away. Too high, and we keep very few features, since few have a super high score. Too low, and we keep too many and it just becomes noise and slows down processing unnecessarily. 

NMS_WINDOW, found in Features.h. Changing this determines how many features are suppressed around particularly strong features. Too small, and there will just be a lot of features in clusters, which we don't really need and just adds time to processing. Too large, and you risk cutting out too many important features, and lose quality for the end result. 

### Other notes
- Once again, this can easily be parallelised, since every feature is independent. The only bit that can't be is the non-maximal suppression


## Feature Description


### Tunable parameters

### Other notes


## Feature Matching

## Finding the best transform

## Composition

### Other
There could be more steps. I could do some alpha blending, to smooth the transition between images. If you want to add that, feel free. I didn't because ... well, all I was really doing was panography. Blending is nice, but ... eh. Don't need to. 

Other things could be if you have a lot of images, bundle adjustment across them all, rather than doing it pairwise. Large-scale bundle adjustment changes the task quite a bit and can be interesting to try.

You could also parallelise a lot of these - I've given it some thought, but haven't included much here. Give it a shot!

## How to read the code


## Building and running

