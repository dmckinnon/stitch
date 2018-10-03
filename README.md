# Panorama stitching

The aim of this tutorial and program is to go over all the constituent parts and theory of  panorama stitching, 
as well as providing a reference solution.
The best place to start after reading here is main.cpp, where each component is used in sequence. Then step into each function or file as necessary. 
Each component has a comment block that I will also explain here, along with reference texts.

I started this project at suggestion from a colleague to learn in-depth (from scratch) more about computer vision. Specifically, I wanted to learn - _really learn_ - what features were, how they worked, how they were used, how you transformed between images, etc, and the various 'tool' algorithms computer vision engineers use (like RANSAC and Levenberg-Marquardt). The solution here is not the best solution. It is not optimised for efficiency or beauty or anything. I just coded it to work and be understandable. But it does indeed work and I learnt a lot doing it, and I hope anyone reading this can too. I encourage you to read the reference texts I give, learn the theory, and perhaps try to write the algorithms yourself and compare results with what I have. Another good thing to try is to tweak the parameters I mention below to 'retune' my solution, so to speak, and see how the results differ. I'll mention any tunable parameters  that I think would be good to try. A good dataset to work with, that was made to test panography programs, is Adobe's [panorama dataset]( https://sourceforge.net/projects/adobedatasets.adobe/files/adobe_panoramas.tgz/download).

# Contents:
1. Components - an overview of the README
2. Feature Detection
    - Exercise 1
    - Tunable Parameters
    - Other notes
3. Feature Scoring
    - Exercise 2
    - Tunable Parameters
    - Other notes
4. Feature Description
    - Exercise 3
    - Tunable Parameters
    - Other notes
5. Feature Matching
    - Exercise 4
    - Tunable Parameters
    - Other notes   
6. Finding the best transform
    - Exercise 6
    - Tunable Parameters
    - Other notes
7. Composition
    - Exercise 6
    - Tunable Parameters
    - Other notes

# Components
So to get a panorama, you take a photo of something, say a mountain, and then you turn a bit and take another photo, maybe of the side of the mountain and the sunset next to it. And then some magical stuff happens and la-di-da, out pops the two images made nicely into one, such that you can't even see the join (hopefully). So what happens under the hood?
I'll talk specifically about the case for two pictures, but it easily generalises - or iterates. Get the panorama from the two images, stitch in a third, and repeat.

In broad terms, we need to find what the images have in common, find a way to match those bits up, and then make sure that way works for everything else as well. More specifically, for each of the images, we need to find distinct features in them, create descriptors of those features so that we can potentially find that same feature elsewhere, try to match the features between the images, create a way of transforming from one image to the other based on how these matches line up, and then perform this transform for every pixel. 

Here is how I have broken down my code into components, based on the above. Each section will have a brief header about the topic, and then I'll go into more detail. It's my explanation of my understanding, but I also have links in these explanations to the theoretical basis and better write-ups. 

## Feature Detection
"Features" can be a bit of a vague concept in computer vision, and it certainly was for me at the start. Just call some OpenCV function and magical dots appear on your image! But what are they? How do you get them? Why do we want them?

Features are basically identifiable parts of images. An image is an array of numbers. How do I know what is identifiable if I see it again in another image? How do I know what is important to track? A 'feature point' is this, and Feature Detection finds these points. These points are useful because we can find them again in the other image (see the paragraph below for a greater description of this). So we find a feature on a part of one image, and hopefully we can find the same feature in the other image. Using Feature Descriptors, the next section, we can compare features and know that we have found the same one. Multiple matched features then helps us in the later section Feature Matching, where we try to figure out how to go from one image to the other. If we have several feature points in one image, and have found the same in the other image, then we can figure out how the two images fit together ... and that, right there, is how panoramas work!


There are a lot of different types of features, based on how you look for them.

#### Some common types of features:

- FAST features
- SIFT features
- SURF features
- ORB features


There are plenty more. Some are simple, some are ... rather complex (read the wikipedia page for SIFT features, and enjoy). They each might find slightly different things, but in general, what 'feature detectors' aim to do is find points in an image that are sufficiently distinct that you can easily find that same feature again in another image - a future one, or the other of a stereo pair, for example. Features are distinct things like corners (of a table, of an eye, of a leaf, whatever), or edges, or points in the image where there is a lot of change in more than just one direction. To give an example of what is not a feature, think of a blank wall. Take a small part of it. Could you find that bit again on the wall? That exact bit? It's not very distinct, so you likely couldn't. Then take a picture of someone's face. If I gave you a small image snippet containing just a bit of the corner of an eye, you'd find where it fit very quickly. AIShack has a [rather good overview](http://aishack.in/tutorials/features/) of the general theory of features.

In this tutorial and program, I'm using FAST features. These are also called FAST corners because they are specifically designed to find corners. 

- OpenCV's [Explanation of FAST](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html)
- Here is a better [reference implementation of FAST](https://github.com/edrosten/fast-C-src), that is trained by a learner
- Here is the [original paper](https://www.edwardrosten.com/work/rosten_2006_machine.pdf)

OpenCV's explanation is pretty good, so I'll be brief with my own here, since it's ... heavily influenced and copied from that.

The idea behind FAST is that corners usually have two lines leading away from them, and that the intensity of the pixels in one of the two angles created by those lines will be either lighter or darker than the other. For example, think of the corner of a roof with the sun above. Two lines (the roof edges) come out from the corner, and below will be darker than above. The way a FAST feature detector works is that for each pixel, it scans a circle of 16 pixels around it, about 3 pixels radius, and compares the intensities to the centre intensity (plus or minus a threshold). If there is a stretch of sequential pixels 12 or more in length that are all of greater intensity (plus a threshold) than the centre, or lesser intensity (minus a threshold) than the centre, this is deemed a FEATURE. (OpenCV's explanation has some better visuals)

### Stop - Exercise 1
At this point you should know enough theory to make at least a good attempt at Feature Detection - if you are trying to implement this yourself. The next section is for if you are compiling and playing around with *my* code, and you want to experiment. 

Once you are done with this, move on to **Feature Scoring**.


### Tunable parameters
FAST_THRESHOLD, found in Features.h. Changing this will determine how much brighter or darker the ring of pixels around a certain point needs to be for it to be noted as a feature. Eg. a value of 1 means that too many points will be features - blank walls have a pixel-to-pixel variance that's likely bigger. A value of 100 means that you'll barely get any features at all, since you need a sequence of at least 12 pixels that are centre pixel intensity + 100, which ou tof 255 possible values is a large change. 

### Other notes:
- This could easily be parallelised. Try to figure out how, as an exercise. 

- You could calculate these features at multiple scales. This would register bigger objects in the image as features, which might help if you had large featureless areas with thick borders (a wall of a house, perhaps)


## Feature Scoring

Your average image might have over a thousand features - this is quite a lot to process later, as you'll see. We don't need that many features to do figure out how the panorama fits together (100 feature points per image is more than enough). So we should remove some features. How do we know which ones to remove? We compute a 'score' for each feature, that measures how strong that feature is, and we get rid of all features below a certain score. A 'strong' feature here means a feature point that is really clear and distinct, and easy to find again. A 'weak' feature is one that is vague, and would easily be mismatched. 

Once again, there are many methods of scoring features, and one of the most famous is the Shi-Tomasi score, invented by Shi and Tomasi in 1994. Here is their [original paper](http://www.ai.mit.edu/courses/6.891/handouts/shi94good.pdf).

AI Shack has a [good article](http://aishack.in/tutorials/shitomasi-corner-detector/) on the Shi Tomasi score, but it relies on some [background knowledge](http://aishack.in/tutorials/harris-corner-detector/), or having read the previous articles linked at the bottom (they're short and easy and good).

Essentially, for a the feature point and a small patch surrounding it, a matrix called the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) is computed. This is basically the two dimensional version of the gradient. The way we compute this is documented [here](http://aishack.in/tutorials/harris-corner-detector/). Then, we compute the eigenvalues for this matrix. Since this is a two-by-two matrix (see the previous link), the eigenvalues are just the solutions to a simple quadratic equation. The Shi-Tomasi score, then, is simply the minimum eigenvalue. 

Why? Why does this work? Well, for a two-by-two jacobian matrix, the eigenvalues define how strong the gradient is in the direction of the two eigen__vectors__ of the matrix. Basically, how much change we have in each direction. For a good corner, you should have a sharp image gradient (difference in pixel intensity) in both directions, so the minimum eigenvalue won't be that small. For just an edge, you'll have a sharp gradient in one direction but not the other, meaning one eigenvalue will be small. 

We then have a cutoff threshold for this value - another tunable parameter - and everything with a score below this - that is to say, every feature with a minimal eigenvalue of value lower than this threshold - is thrown away and ignored. Every feature with a higher score is kept. 


The final stage of the feature scoring is to perform Non-Maximal Suppresion (and unfortunately I can't find a good explanation online). The theory of Non-Maximal Suppression is that for where you have a group of points clustered in an area, like having, say, twenty features of a good score in the same little patch of the image ... you don't need all of these. You've already registered a strong feature there. So you suppress, that is to say, put aside, the weaker features within some radius of the strongest in that patch. In other words, you suppress the points that aren't maximal. Hence, non-maximal suppression. 

So we do this over our feature set that we've already cut down. For every feature, if there are any features in a 5x5 patch around it that are weaker, we suppress these too, just to reduce the amount we have to process over.

### Stop - Exercise 2
At this point you should know enough theory to make at least a good attempt at Feature Scoring - if you are trying to implement this yourself. The next section is for if you are compiling and playing around with *my* code, and you want to experiment. 

Once you've got this down, try **Feature Description**.

### Tunable parameters
ST_THRESH, found in Features.h. Changing this determines how many features we'll keep or throw away. Too high, and we keep very few features, since few have a super high score. Too low, and we keep too many and it just becomes noise and slows down processing unnecessarily. 

NMS_WINDOW, found in Features.h. Changing this determines how many features are suppressed around particularly strong features. Too small, and there will just be a lot of features in clusters, which we don't really need and just adds time to processing. Too large, and you risk cutting out too many important features, and lose quality for the end result. 

### Other notes
- Once again, this can easily be parallelised, since every feature is independent. The only bit that can't be is the non-maximal suppression


## Feature Description
I'm now going to start referring to these feature points we've been talking about as 'keypoints'. Because they are points that are key to matching the two images ... anyway. We now want to create keypoint descriptors, which are unique numbers for each keypoint/feature so that if we ever saw this again, we could easily identify it, or at least say "look, this one is very similar to that one, as their identifying numbers are close". This is what I mentioned before in Feature Detection. We found feature points in each image. Now we want to try to see which ones are the same, to find matching points in each image. How do we know if two features are actually referring to the same image patch? We need some distinct identifier - a descriptor. Each feature point will have its own descriptor, and when we try to figure out which of the first image's features are also seen in the second image, we try to see which features actually have the same descriptor, or very similar descriptors. This works because the descriptor is based on the patch of the image around the feature point - so if we see this again, it should register as the same. 

So, for now, how do we make these 'keypoint IDs'? What identifies a keypoint uniquely? Instead of saying ID, I'm going to use the word 'descriptor', because all the literature does that, but all that means is 'something that describes', and you can replace it with ID and you lose little. 
As always, there are many ways of making descriptors for keypoints. BRIEF descriptors, SURF descriptors ... different descriptors have different strengths and weaknesses, like robustness to various conditions, or small size but similar identifying power ... but here, because I wanted to learn them, I chose SIFT descriptors. This stands for the [Scale-Invariant Feature Transform](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform). 

Once again, AI Shack has [quite a good description of SIFT descriptors](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-features/) - in fact, a good description of the entire [SIFT feature detector](http://www.aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/) too. I encourage reading both (the second is the whole series, the first link is just the last page in the series. It's not a long series).


I'll attempt to write my own briefer description here. So what's unique about a feature? Well, the patch around and including it, up to a certain size. On an image of a face, the corner of an eye is pretty unique, if you get a decent-sized patch around it - a bit of eye, some crinkles if the face is smiling, etc. What makes these things unique? Or rather, how do we capture this uniqueness?

Once again, gradients. The change in intensity - going from mild skin (say. The skin could be darker, but this example works best on a really pale person), to the darker shadow of a crinkle, back to mild skin - can be measured best by the gradient of a line through that area. If we get a little patch over a face crinkle, then we could find a dominant gradient vector for that patch - pointing perpendicular to the crinkle since that's the direction of the greatest change (light to dark). This dominant gradient vector, both in angle and magnitude, can then be used to identify such a patch. 

Ok, so we can identify a little patch with one edge. But aren't features corners? So, let's do more patches. SIFT creates a 4x4 grid of patches, with the feature at the centre of it all. Each patch is 4x4 pixels (so 16x16 pixels total to scan over). We find the dominant gradient vector for each patch - the vector angles are bucketed into one of 8 buckets, just to simplify things - and then we list all these in sequence. This string of bits - magnitude, angle, magnitude, angle, etc, for 16 gradient vectors around the feature, is the unqiue descriptor that defines the feature. 

There are a couple of things worth mentioning. The angle of each gradient is taken relative to the angle of the feature (when we create the feature, we measure the angle of the gradient in a smaller patch centred on the feature), to make this descriptor rotationally invariant. Another thing is that all the magnitudes are normalised, capped at 0.2, and normalised again, so as to make the vector invariant to drastic illumination. This all means that if we look at the feature again from a different angle, or under mildly different lighting, we should still be able to uniquely match it to the original. 

All of this is explained with nice diagrams in the AI Shack link above. 

### Stop - Exercise 3
At this point you should know enough theory to make at least a good attempt at Feature Description - if you are trying to implement this yourself. The next section is for if you are compiling and playing around with *my* code, and you want to experiment. 

Once this is working (Admittedly, this is hard to test without the next section), go on to **Feature Matching**. I actually recommend doing them together, but up to you. Feature matching is a good way to test Feature Description. 

### Tunable parameters
Technically, ILLUMINANCE_BOUND and NN_RATIO, both in Features.h, are tunable parameters, but Lowe (the inventor of SIFT) tuned these parameters already and found the values they have to be experimentally pretty good. Still, feel free to change them. 

### Other notes
- Strictly speaking, SIFT features are captured at multiple scales, and thus the descriptors are created at different scales too, but I was just simplifying things and got features at only one scale.

## Feature Matching
So far we have found features, cut out the ones we don't want, and then made unique descriptors, or IDs, for the remainder. What's next? Matching them! This is so that we know which parts of the first image are in the second image, and vice versa - basically, where do the images overlap and how do they overlap? Does a feature in a corner of image 1 match with a feature in the middle of image 2? Yes? Is that how the images should line up then? Feature Matching answers this question - which bits of the first image are in the second and where?

Now we have to get the features from the left image and features from the right image and ask "which of these are the same thing?" and then pair them up. There are some complicated ways to do this, that work fast and optimise certain scenarios (look up k-d trees, for example) but what I have done here is, by and large, pretty simple. I have two lists of features. For each feature in one list (I use the list from the left image), search through all features in the second list, and find the closest and second closest. 'Closest' here is defined by the norm of the vector difference between the descriptors. 

When we have found the closest and second closest right-image features for a particular left-image feature, we take the ratio of their distances to the left-image feature to compare them. If DistanceToClosestRightImageFeature / DistanceToSecondClosestRightImageFeature < 0.8, this is considered a strong enough match, and we store these matching left and right features together. What is this ratio test? This was developed by Lowe, who also invented SIFT. His reasoning was that for this match to be considered strong, the feature closest in the descriptor space must be the closest feature by a clear bound - that is, it stands out and is obviously the closest, and not like "oh, it's a tiny bit closer than this other one". Mathematically, the closest feature should be less than 80 percent of the next closest feature. 

### Stop - Exercise 4
At this point you should know enough theory to make at least a good attempt at Feature Matching - if you are trying to implement this yourself. The next section is for if you are compiling and playing around with *my* code, and you want to experiment. 

Once this is working - to test, see how many of your features between images match up -  go on to **Finding the best transform**. This is probably the hardest and most complicated section. 

### Tunable Parameters
You can try to tune Lowe's ratio, which is NN_RATIO, defined in Features.h. Changing this determines how "strong matches" are made. 

## Finding the best transform
Now that we know which left-image features correspond to which right-image features, we have to find a mathematical operation that accurately transforms each right point into the correct left-image point. This is further answering the questions raised in the intro to Feature Matching - how do these images line up together? Here, we'll find a way of lining the second image up to the first, and then from there we just need to stick them together. Think of having two printed photos, and trying to align them - making a panorama by hand, with scissors and tape. You would hold one photo straight, and turn the other, trying to see how it best fits over the first. You might wish you could stretch certain bits, or shrink them. When you were satisfied, you would tape the photo down. That taping is the next section, Composition. This section is the turning and alignment. 


The operation we want is a 3D transform called a [Homography](https://en.wikipedia.org/wiki/Homography_(computer_vision)). Unfortunately, the wikipedia page for homographies is ... lacking ... although those more mathematically inclined can try the [mathematical definition](https://en.wikipedia.org/wiki/Homography). OpenCV [tries](https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html) but in classic OpenCV style still manages to be rather vague and unclear unless you're already an expert. You can try [Youtube](https://www.youtube.com/watch?v=MlaIWymLCD8). Perhaps the best place is Multiple View Geometry In Computer Vision, by Hartley and Zisserman. A textbook, yes, but an excellent one. 

Failing all that, I'm going to give a simple explanation here. I'm drawing from [this article](http://www.corrmap.com/features/homography_transformation.php), which is the simplest, best visualised, clearest explanation I've managed to find. Visualise your two images as two planes in space - think of the images sitting as giant rectangles overlaid in the air where you took the photos. A homography transforms points in planes to points in planes. Let's call these planes *P* and *P*'. We're going to normalise all our 3D points - that is, divide by the third coordinate to make it 1. This makes the mathematics simpler, and has some geometric connotations as well. So, for a homography H, and points **x**' in *P*' and **x** in *P*, 

(*x*', *y*', 1) = H * (*x*, *y*, 1).



So H is a 3x3 matrix. After we do this computation, we tend to normalise **x**' again, resulting in the fractional coordinates seen [here](http://www.corrmap.com/features/homography_transformation.php). Each of the parameters of H has a particular function:

     ( scale the x coordinate      skew the x coordinate      translate x coordinate  )
     
`H =  ( skew the y coordinate          scale the y coordinate        translate y coordinate  )`
     
     ( x total scale               y total scale                        1             )
     
     
I realise this is not a comprehensive overview of how homographies work, but alas, this is not that place. Just assume for now that they do work - if we tack a 1 on to the `(x,y)` coords from the images, there is a 3x3 matrix that will take you from one image to the other. 

So how do we compute this matrix?
I know I keep referring to it, but [this](http://www.corrmap.com/features/homography_transformation.php) sort-of explains it at the bottom. I also have a brief explanation above GetHomographyFromMatches in Estimation.cpp, so I won't parrot it here. Suffice to say that we assume that **x**' - H**x** = 0, for (at least) four pairs of matching points (**x**', **x**), form a 9-vector **h** from the elements of H and rework this equation to be A**h** = 0, for some matrix A (see Estimation.cpp). Then we use some estimation methods to solve this. The thing is ... there may be no **h** that precisely solves this equation, but there may be one that _approximately_ solves it. So we get the smallest possible solution that does so. But how are we getting these solutions **h** to A**h** = 0?

[Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular-value_decomposition). For more reference on this, I list a lot of links relating to the computation of this process, the theory behind it, the practical use, etc, above GetHomographyFromMatches in Estimation.cpp. This breaks A down into its 'singular values - these are the analogy of eigenvalues, but for non-square matrices. The matrix A is split into three matrices, U, D, and V transpose. D contains, listed from left to right descending along the diagonal, the singular values of A. The columns of V are the singular vectors of A. If we set **h** to be the vector corresponding to the smallest singular value of A, this gives us an approximate solution to A**h** = 0. 

> If you don't understand this part - that's perfectly ok! I have a degree in mathematics and this took me a while and a lot of reading to figure things out, and even then I struggle with it. This is complicated stuff. If you want, you can just think "there's some magical method of getting this matrix H" and leave it at that. 

So we now have a tool to estimate our homography H, using four pairs of matching points (**x**', **x**). Which four do we pick? How do we know if we picked the best? 

The answer to this is we try a lot of combinations. Not all, cos that's stupidly many. Let's say we have 50 matches (a relatively low number). There are 50!/(50-4)!4! = 230300 combinations. Have fun waiting for that to finish calculating, since we need to do SVD each time to get the homography, and then test the homography on each matching pair to evaluate it. Nope. 

The advantage we have is that most fours we pick should be good - and so if we pick just a few, then we'll likely hit a decent solution, and from there we have some tricks to refine it further - but that's later. By 'a few' I mean several hundred or so, which when you think about it is less than a percent of the full number of possibilities. But we still hit the question "how do we choose which fours to pick?". RANSAC is the answer, and in my opinion, one of the loveliest algorithms. 

RANSAC stands for [RANdom SAmple Consensus](https://en.wikipedia.org/wiki/Random_sample_consensus). I won't go into this algorithm too much here, but the basics of it are that we pick X number of points randomly from our pool, and then evaluate every other point according to some metric (here, how well the matches work under a homography constructed from these four points). We classify these into inliers and outliers according to some bound, and we keep the best set of points we get over a certain number of iterations. 

The way I use it here is as follows: 
1. Choose four matches at random from the pool of matched points (GetRandomFourIndices, Estimation.cpp). 

2. Compute the homography for these points (GetHomographyFromMatches, Estiamtion.cpp). 

3. Evaluate the homography and get the inlier set, to see how many inliers we get (EvaluateHomography, Estimation.cpp). 

4. Refine this homography on just this inlier set (This is known as Bundle Adjustment or Optimisation, and is mentioned below).

5. Repeat 3 and 4 until we get no new inliers

6. Repeat 1-5 MAX_RANSAC_ITERATIONS number of times and choose the homography from the set that finishes with the most inliers


After all this ... The homography that remains after all this should be a really good one. By "really good homography", I mean that we have tested how well it fits the matches we have from before - this is how EvaluateHomography works - and it has a total error below some bound. We test the homography by looping over each matching pair of features, and using H to transform one feature point of the match into the coordinate space of the other feature point. We then take the euclidean distance between the transformed point and the other feature point. Adding this up over all feature matches, we get the total error for a given H. Each individual error should be less than a pixel off - far less, for a good H - so we can use this to asses what a reasonable H is, what a good H is, and what a terrible H is. 

So now we have a homography. It's understandable that a lot of that theory went in one ear and out the other. Like I said, feel free to black box all this for now and just assume we found some magical way of turning the matched features into figuring out how to match every pixel in the overlap. 

So this figured out how we turn and bend and align the second photo to fit on to the first, given a collection of matching points that we know we can line up. The final step now is to stick them together. 

N.B. I mentioned Optimisation above, and if you want, you can black box this too and ignore it. If you are interested in the theory, read on:


**Optimsation**:

The idea behind [optimisation](http://ethaneade.com/optimization.pdf) is that we have some function, f(**x**), that we are comparing to some data, **y**, and it's off by some error **e** = |**y** - f(**x**)|. It'd be nice if we could make that error smaller by tweaking the function f. Well, we can. Remember that if you want to find the local minimum of a function, you find the derivative and set that to zero? And solve for the point, and that point is the minimal point of the function?

Well, same theory applies here. I'm using a variant of [Gauss-Newton Optimisation](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) called [Levenberg-Marquardt Optimisation](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm). These both do approximately the same thing. We want to fine-tune our estimate of H, right? Make it as good as possible. Since the homography computation can only get so good on four points, we use optimisation across all the points to tweak H, which in this section is represented by f, to reduce the error **e** = |**x**' - H**x**|.

Unfortunately it's hard to write a lot of the math here so I'm going to try my best without equations. Following the steps mentioned in the solution section of the [Levenberg-Marquardt wikipedia page](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm), we want to solve for a small update to H to tweak it in the right direction. So we want to replace H with H + *w*, where *w* is some 3x3 matrix of small numbers. The idea is that we'll use the derivative of this function to find the direction of the gradient, and then make a change in that direction. This means that we move down towards the minimum (This is so much easier with diagrams).

To achieve this we approximate the equation **e** = |**x**' - (H + *w*)**x**| by the first order Taylor series approximation, giving **e** = |**x**' - H**x** - **J***w*|, where J is the Jacobian (multidimensional derivative) of H evaluated at **x**. We then differentiate this and solve for **e** = **0**. This leads to **J**t * **J** * *h* = **J**t * e; in words, this is the Jacobian of H (transposed) times the Jacobian of H times the small update to H all equal to the Jacobian of H (transposed) times the original error vector. If the math doesn't make sense ... well ... that's ok. It's not simple, and I'm sorry I can't explain it well. A good explanation is [here, written by Ethan Eade](http://ethaneade.com/optimization.pdf).

We solve this equation for *w* since we can compute everything else, and then apply the update to H by H(new) = H(old) + *w*. We then test how good this is by seeing if the error now is lower than the error from last time, and repeat. If the error at all increases, we know we have hit the bottom and started going up again, or we're just doing the wrong thing. In either case, we quit. If the error gets below a certain tiny threshold - this is good enough, no point computing more, and we quit. 

### Stop - Exercise 5
At this point you should know enough theory to make at least a good attempt at finding the best transform - if you are trying to implement this yourself. If this takes you a lot of tries, and is full of bugs, don't worry! It took me _ages_ to get right. The next section is for if you are compiling and playing around with *my* code, and you want to experiment. 

Once this is working - you can display the images to test this, cos if you transform the other image to the first's coordinate frame they should sorta line up - then finish it off with **Composition**.

### Tunable parameters
- MAX_RANSAC_ITERATIONS, in Estimation.h. This is the number of RANSAC iterations. Too few and we risk not finding a good enough solution. Too many and we are wasting processor time. 
- There are other parameters to tune for the various thresholds mentioned, but they are harder to tune obviously and see the effect of.

### Other notes
- There are other ways to do optimisation, and this is by no means the best or worst or whatever. But that's getting a bit deep and too much for here. 


## Composition
Last, but not least, we have to actually put the two images into the same frame. We have the original left image. We now have a way to transform pixels in the right image to the left image coordinate space. Think of the analogy in the Transform introduction - you've figured out how to sit one photo over the other, and now you just need to tape it down. That's this step. How do we do this? 

At first, this seems obvious: for each pixel in the right image, transform it by the homography H from the previous stage, and then, say, average the transformed right pixel, and the left pixel it overlaps with. Simple. 

Except not quite. 

When I have a vector **x** = (x,y,1) and transform it to **x**' = H**x** ... well, **x**' isn't going to have nice integer coordinates. they'll be ... 36.8 or whatever. Point is, they won't be nicely pixel-aligned. To solve this, we actually use the inverse of H, which I'm going to call G, since I don't know how to do superscripts in markdown. 

So we use H inverse = G to transform a _left-image_ pixel into _right image_ space, by **y**' = G**y**, and it's going to not be pixel-aligned either. We then use [Bilinear Interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation) to figure out the correct sub-pixel value for this point, and then we take that value as the right image pixel value to stitch at **y** in the left image coordinate space. Interpolation is how we approximate a value between two known points, when we can't sample any finer than those points. For example, if you know that for some function f(x), f(0) = 0 and f(1) = 1, and you want to guess the value at x=0.8, then you can use [linear interpolation](https://en.wikipedia.org/wiki/Linear_interpolation), which says it's going to be 0.8 times the value at x = 1 averaged with 0.2 times the value at x = 0. Bilinear interpolation is simply the two dimensional version of this. 

Now that we have the correct pixel values from each image in the same coordinate space, I simply average the corresponding ones together, and BAM, we're done!

### Stop - Exercise 6
At this point you should know enough theory to make at least a good attempt at Composition - if you are trying to implement this yourself. The next section is for if you are compiling and playing around with *my* code, and you want to experiment. 

Once this is working .... be proud!! You did it!

### Tunable Parameters
None, really. This is probably the simplest step.

### Other notes
- You could do something other than averaging here when stitching the images. One thing to try is as the pixels get deeper into the right image and more to the edge of the left image, using a greater percentage of right pixels than 50%. 

- Another technique is that of **Alpha Blending**. This involves more complicated ways of merging the two, for example [Poisson Blending](http://eric-yuan.me/poisson-blending/). This gets more necessary/useful when your two images are under different lighting conditions; when they are very similar in terms of saturation, etc, then it may not have a huge effect.

- Sometimes stretching is needed, when the scales of the images or objects don't quite match up and you have a close object that needs to be stretch across a long space to line up with several distant objects across the pictures. 

- Finally, this step can be parallelised, and may be the easiest spot to do so. Every pixel can be computed independently of every other. Try to figure out the optimal method of parallelising this! I've given one solution in the code comments already. 

## How to read the code
I've included explanations and links above each major function, as well as the helper functions they use and the purpose each serves. I've also commented the code heavily. The way I read it is you go through main, and then dive into each component as it is used, and then back out to main, to retain context. 

## Building and running
I've included in the repo the .exe and the necessary dlls to just run this straight out of the box. It takes in two command-line arguments - the absolute paths of the two images, ordered left (first arg) and right (second arg) (the order shouldn't actually matter), and spits out panorama.jpg, which is the stitched image. If something fails, it prints error messages. 

I developed this in Visual Studio on Windows, but nothing is platform-dependent. The only dependencies it has are [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) and [OpenCV](https://opencv.org/), just for a few things like the Mat types, Gaussian blur, the Sobel operator, etc. If you don't want to do the whole download, build OpenCV yourself thing, just use the dlls and libs I supplied, and download the source code and be sure to link the headers the right way. Installing OpenCV was complicated enough for me that this is a topic for another day, and unfortunately I can't find the link yet that I used. If I find it, I'll add it. 

Thanks for reading - enjoy!
