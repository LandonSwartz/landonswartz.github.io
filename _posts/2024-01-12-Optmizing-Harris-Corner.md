---
title: "Optimizing the Harris Corner Detector"
date: 2024-01-12
permalink: /posts/2024/1/OptHarrisCorner/
tags:
  - ComputerVision
  - Corners
  - Harris Corner Detector
  - Optimization
---

<!-- # Optimizing the Harris Corner Detector -->

## Brief Introduction

![](/images/P2_optharris/optimization_xkcd.png)

One of the reasons that I started this blog was to start doing project-based learning on computer vision concepts outside of my graduate classes. That was why I chose the Harris Corner Detector as one of my first posts - it was one of the first projects in my Computer Vision graduate class. But that lead me into one of the greatest problems of project-based learning outside of the classroom. After a class project, one moves on to new work with little regard for state of the project they just did. But, as I found with this project, that mentality doesn't always translate to the real world. 

The corner detector we implemented in the last posts was, at the time, not too far off from being a finished product. It was quite similar to my original implementation I submitted. But then I started working on implement feature descriptors and matching them. It was a long time, like a REALLY long time. That is what led to this blog post. This post is all about optimizing through vectorizing, parallelization, and generally smarter ways to implement things. It was a great learning experience for learning that a project is never truly finished but just reaches steps of maturity.

## How to Optimize Python Corner Detection

Python is notorious for being slow. It is for a variety of reasons (such as Global Interpreter Lock) but at the end of the day, it boils down to the fact that it is a high-level language. Python was designed to abstract away the complexities and annoyances of memory management, types, and all of the nuances that can drive one crazy in low level languages like C/C++. The trade off is less development time for slower speeds. It is perfect for prototyping but not always the best for implementation onto a hardware system. But there are several different methods we can implement to shave off a lot of time by using libraries implemented in C with python wrappers.

Another thing to discuss before we begin is the idea of scale. When I first implemented the corner detector, I chose images from the hpatches dataset for ease of accessibility. But like the MNIST dataset of machine learning, the dataset is saturated. While revolutionary at the time of the release, hpatches has been eclipsed by the needs of modern computer vision. For example, our corner detector took around $25-30$ seconds for images in hpatches. That is a great start for just implementing the detector. But then when one starts to implement experiments where you do an entire scene of six images and find matches between them, that is now $3$ minutes per scene. If we use an image that is $5000x3000$ pixels, we now have to wait several minutes just for one image! In a world where paradigms change almost daily for computer scientists, knowing how to scale your project for the future is vital. So now let us scale first with vectorization.

A quick note: I changed our original implementation to detect a given number of keypoints instead of the top 10% of responses to make later matching evaluation more reproducible and comparable.

### Vectorization

Vectorization is the process of applying operations across an entire array at once rather than going one array element at a time and performing the operation. I think one of the best ways to explain it is to see it in practice. Below is our original implementation for the harris corner response calculations using the second derivatives of the image gradients:

```python
    def detect_corners(self):
        Ix2 = self.Ix**2
        Iy2 = self.Iy**2
        IxIy = self.Ix * self.Iy

        offset = self.window_size // 2
        height, width = self.gray_image.shape
        self.R = np.zeros((height, width), dtype=np.float64)

        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                Sx2 = np.sum(
                    Ix2[y - offset : y + offset + 1, x - offset : x + offset + 1]
                )
                Sy2 = np.sum(
                    Iy2[y - offset : y + offset + 1, x - offset : x + offset + 1]
                )
                Sxy = np.sum(
                    IxIy[y - offset : y + offset + 1, x - offset : x + offset + 1]
                )

                detM = (Sx2 * Sy2) - (Sxy**2)
                traceM = Sx2 + Sy2

                self.R[y, x] = detM - self.k * (traceM**2)
```

There is a lot of big no-no's for python implementations in this method: for loops. For loops are the bread and butter of C/C++ programs. But python struggles using them because of the mechanisms of being high-level language. Python will carry all of the additional overhead to each operation on the array. But many different libraries utilize vectorization to turn this for loops into singular operations.

For instance, instead of iterating over every point, we can use a convolution to achieve the same method of iteration without the overhead. 

```python
from scipy.ndimage import convolve

def optimized_detect_corners(self):
    """
    Optimized method to detect corners using vectorized operations.
    """
    Ix2 = self.Ix ** 2
    Iy2 = self.Iy ** 2
    IxIy = self.Ix * self.Iy

    # Define a window for convolution
    window = np.ones((self.window_size, self.window_size))

    # Use convolution to replace the nested loops for summing in a window
    Sx2 = convolve(Ix2, window, mode='constant', cval=0)
    Sy2 = convolve(Iy2, window, mode='constant', cval=0)
    Sxy = convolve(IxIy, window, mode='constant', cval=0)

    # Calculate the determinant and trace of M for each pixel
    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2

    # Calculate R for all pixels simultaneously
    self.R = detM - self.k * (traceM ** 2)
```

Now our corner detectors speeds through the harris response calculation. The local response calculations are done all at once using the convolutions. We can see another example in our Non-Maximal Suppression method:

```python
def apply_non_maximal_suppression(self, neighborhood_size=3):
    height, width = self.R.shape
    offset = neighborhood_size // 2
    suppressed_R = np.zeros((height, width), dtype=np.float64)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            local_max = np.max(
                self.R[y - offset : y + offset + 1, x - offset : x + offset + 1]
            )
            if self.R[y, x] == local_max:
                suppressed_R[y, x] = self.R[y, x]

    self.R = suppressed_R
```

Lots of loops going element by element. But in our new method, we utilize the maximum filter from scipy.ndimage to achieve the same result across the entire array.

```python
from scipy.ndimage import maximum_filter

def optimized_apply_non_maximal_suppression(self, neighborhood_size=3): 
    """
    Optimized method to apply non-maximal suppression using maximum filter.
    """
    # Apply maximum filter
    max_filtered = maximum_filter(self.R, size=neighborhood_size)

    # Create a mask where original R values match the maximum filter result
    local_max_mask = (self.R == max_filtered)

    # Zero out non-local-maxima
    self.R *= local_max_mask
```

The other benefit of vectorization is cleaner code. The downside to vectorization is that the cleaner code can be very hand-wavey in an implementation sense. In the original function, I felt very little need to document anything because the code was the explanation of the method. In the new method, I wanted to document every line because the inherent understanding of what was happening was not there in the code. Be very careful to observe what a built-in vectorization function does and document it.

### Parallelization

Parallelization is simple in theory but complicated in execution. In theory, parallelization is dividing a collection of operations that are usually performed sequentially into operations performed concurrently with multiple threads/processes/etc. Parallelization is particularly effective at helping with large data processing, asynchronous I/O operations (such as file reading/writing), and, most importantly for our work, CPU-intensive computations.

But, there are also several limitations to parallelization - especially for Python. The first and more obvious is computational overhead. If one wants to divide three CPU-intensive tasks into separate threads (as we will do in our case) then you will need a CPU that can not only support three threads running at full speed at the time but also be able to perform faster than one CPU core doing it sequentially. That is where our interactions with scale will come into play. 

Another major pitfall for parallelization in python is the Global Interpreter Lock (GIL). The GIL is responsible for limiting CPU-bound tasks inherently in Python using a mutex (or a lock for computer operations) to protect access to Python objects. It prevents race conditions (or multiple threads trying to use the same resource at once) and keeps memory states consistent (a file is not changed at the same time by two threads). In practice, the GIL causes only one thread to execute at a time for Python code. The theory of the GIL is used all the time in C/C++ applications. The main difference is that the user controls all the settings in those languages. In Python, one has nearly no control over the GIL. Therefore, one can never achieve true parallelism (where multiple threads run at the same time) but only concurrency (multiple threads make progress over time). 

For our little corner detector, we can implement parallelism (with some minor overhead additions) to improve the speed of our detections. Below is our harris corner response method where we can make just one small change for this boost in performance:

```python
from concurrent.futures import ThreadPoolExecutor

def parallelized_detect_corners(self):
    """
    Parallelized method to detect corners using vectorized operations.
    """
    Ix2 = self.Ix ** 2
    Iy2 = self.Iy ** 2
    IxIy = self.Ix * self.Iy

    # Define a window for convolution
    window = np.ones((self.window_size, self.window_size))

    # Function to perform convolution in parallel
    def convolve_parallel(image_section):
        return convolve(image_section, window, mode='constant', cval=0)

    # Splitting the image into sections for parallel processing
    sections = [Ix2, Iy2, IxIy]

    # Using ThreadPoolExecutor to parallelize convolution
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(convolve_parallel, sections)

    # Unpacking the results
    Sx2, Sy2, Sxy = results

    # Calculate the determinant and trace of M for each pixel
    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2

    # Calculate R for all pixels simultaneously
    self.R = detM - self.k * (traceM ** 2)
```

The optimized operation for finding the second gradient of the image has to performed three separate times. With three threads, those operations can be performed concurrently to save another few seconds in execution. For smaller images, this is a negligible optimization (it may be a little slower because of the overhead of the threads being initialized). But for larger images ($5000x3000$ pixels) it is a game changer.

As with vectorization, parallelism creates cleaner code at the cost of inherent understanding by looking at the code. My computer engineering background also compels me to note that parallelism is often a hardware consideration more than a software one. If you can deploy to hardware that can use parallelization, then look into using it. But also understand the limitations of that hardware. Our corner detector runs amazing on a desktop with a 12-core gaming CPU. If you throw the same detector onto an autonomous robot running a Raspberry Pi that has a lot of other overhead, it may not perform as well and need to be single threaded. That is enough theory now, let's see how fast we made our corner detector. 

### Experiments

Now we start the fun part. The experiment is simple: take in an images of various sizes and time how long it takes to perform corner detection for a determine number of keypoints. Our dataset consists of four images ranging from a small to large size shown below.

The number of keypoints detected are scaled in relation to the image size. The smaller the image, the less keypoints detected and vice versa. The three corner detection methods will be the original implementation that is un-optimized, the vectorized optimization, and the parallelized + vectorized optimization. The purely parallelized un-optimized method is excluded purely for my own sanity as the unoptimized method itself is too long for my short attention span.

![](https://github.com/LandonSwartz/landonswartz.github.io/blob/master/images/P2_optharris/small-image.jpg)

![](https://github.com/LandonSwartz/landonswartz.github.io/blob/master/images/P2_optharris/small_med.jpg)

![](https://github.com/LandonSwartz/landonswartz.github.io/blob/master/images/P2_optharris/medium-image.jpg)

![](https://github.com/LandonSwartz/landonswartz.github.io/blob/master/images/P2_optharris/large-image.jpg)

Here are the results and small discussions for each size image:

**Small Image (160x160)**

| Method        | Number of Corners Detected | Time (s)  |
|---------------|:---------------------------:|:-----------:|
| **Un-Optimized**  |                          |            |
|               | 32                         | 0.59       |
|               | 64                         | 0.44       |
|               | 128                        | 0.44       |
|               | 256                        | 0.43       |
|               | 512                        | 0.43       |
| **Vectorized**     |                          |            |
|               | 32                         | 0.01       |
|               | 64                         | 0.01       |
|               | 128                        | 0.01       |
|               | 256                        | 0.01       |
|               | 512                        | 0.01       |
| **Parallelized**   |                          |            |
|               | 32                         | 0.01       |
|               | 64                         | 0.01       |
|               | 128                        | 0.01       |
|               | 256                        | 0.01       |
|               | 512                        | 0.01       |

For an image unrealistically small, the vectorized and parallel version performs basically the same compared to the un-optimized version by orders of magnitude. These optimizations will only scale, as we will see. But the important thing to note is that optimization still matters even at a small scale like this. Even if you process only images this small, you can do 1000 images in 10 seconds with optimizations versus 500 seconds un-optimized.

**Small-Medium Image (640x470)**

| Method        | Number of Corners Detected | Time (s)  |
|---------------|:---------------------------:|:-----------:|
| **Un-Optimized**  |                          |            |
|               | 64                         | 5.06       |
|               | 128                        | 4.90       |
|               | 256                        | 5.12       |
|               | 512                        | 4.77       |
|               | 1024                       | 4.78       |
| **Vectorized**     |                          |            |
|               | 64                         | 0.08       |
|               | 128                        | 0.06       |
|               | 256                        | 0.06       |
|               | 512                        | 0.06       |
|               | 1024                       | 0.06       |
| **Parallelized**   |                          |            |
|               | 64                         | 0.06       |
|               | 128                        | 0.04       |
|               | 256                        | 0.03       |
|               | 512                        | 0.04       |
|               | 1024                       | 0.04       |

This image, taken from the hpatches dataset, is small-medium because it is a realistic image with a small size. It was a late addition to the party because I felt the small image was really small and the medium image was really big in comparison. Here we see our optimizations scaling beautifully in comparison to the un-optimized version. Furthermore, the parallelized version starts to creep faster than the vectorized. But how would it do with an image that's closer to the quality that a modern IPhone could capture?

**Medium Image (3840x2160)**

| Method        | Number of Corners Detected | Time (s)  |
|---------------|:---------------------------:|:-----------:|
| **Un-Optimized**  |                          |            |
|               | 256                        | 130.78     |
|               | 512                        | 132.04     |
|               | 1024                       | 135.32     |
|               | 2048                       | 137.15     |
|               | 4096                       | 130.32     |
| **Vectorized**     |                          |            |
|               | 256                        | 1.66       |
|               | 512                        | 1.55       |
|               | 1024                       | 1.64       |
|               | 2048                       | 1.68       |
|               | 4096                       | 1.55       |
| **Parallelized**   |                          |            |
|               | 256                        | 1.11       |
|               | 512                        | 1.11       |
|               | 1024                       | 1.11       |
|               | 2048                       | 1.12       |
|               | 4096                       | 1.11       |

Now I hope my point of scale is clear. The un-optimized version here takes over two minutes to process a single image! Just vectorizing the function causes the algorithm to speed up 130x. But the parallelization gives saves almost 0.5 seconds on top of that. While that small optimization seems like not a huge deal, image a dataset of 1000 images of this size capturing 2048 keypoints (a standard amount for quality keypoints). The vectorized would take 1680 seconds (28 minutes) while the parallelized would take 1120 seconds (18.67 minutes). Scaling an algorithm and its optimizations requires every ounce of performance to be squeezed out. 

**Large Image (5143x3209)**

| Method        | Number of Corners Detected | Time (s)  |
|---------------|:---------------------------:|:-----------:|
| **Un-Optimized**  |                          |            |
|               | 512                        | 258.14     |
|               | 1024                       | 269.77     |
|               | 2048                       | 261.14     |
|               | 4096                       | 259.72     |
|               | 8192                       | 288.20     |
| **Vectorized**     |                          |            |
|               | 512                        | 2.64       |
|               | 1024                       | 2.62       |
|               | 2048                       | 2.60       |
|               | 4096                       | 2.60       |
|               | 8192                       | 2.58       |
| **Parallelized**   |                          |            |
|               | 512                        | 1.87       |
|               | 1024                       | 1.87       |
|               | 2048                       | 1.87       |
|               | 4096                       | 1.86       |
|               | 8192                       | 1.96       |

This image is typical of something capture from a high-definition camera - like from a high-grade drone or film camera. Our un-optimized function would still be processing by the time you finished reading this section at this point. The parallelized version really starts to take over as the dominant method with being almost a second (~77x) faster. Now that we have optimized and saved a ton of time in execution, let's add a little bit of that time back for quality.

## Conclusions

Now that we can actually run our experiments without my computer having a heart attack with generating more than 256 keypoints, next blog post we will play with a simple feature descriptor (a patch of pixels) and learn about evaluating reprojection and homographies. There are a few more improvements we could make now that we have optimizations (multi-scale, thresholding the response matrix, etc), but we can save those for another blog post. Thanks for reading!
