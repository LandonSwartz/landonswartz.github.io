---
title: "Optimizing the Harris Corner Detector"
date: 2024-1-1
permalink: /posts/2024/1/OptHarrisCorner/
tags:
  - ComputerVision
  - Corners
  - Harris Corner Detector
  - Optimization
---

# Optimizing the Harris Corner Detector

## Brief Introduction

One of the reasons that I started this blog was to start doing project-based learning on computer vision concepts outside of my graduate classes. That was why I chose the Harris Corner Detector as one of my first posts - it was one of the first projects in my Computer Vision graduate class. But that lead me into one of the greatest problems of project-based learning outside of the classroom. After a class project, one moves on to new work with little regard for state of the project they just did. But, as I found with this project, that mentality doesn't always translate to the real world. 

The corner detector we implemented in the last posts was, at the time, not too far off from being a finished product. It was quite similar to my original implementation I submitted. But then I started working on implement feature descriptors and matching them. It was a long time, like a REALLY long time. That is what led to this blog post. This post is all about optimizing through vectorizing, parallelization, and generally smarter ways to implement things. It was a great learning experience for learning that a project is never truly finished but just reaches steps of maturity.

## How to Optimize Python Corner Detection

Python is notorious for being slow. It is for a variety of reasons (such as Global Interpreter Lock) but at the end of the day, it boils down to the fact that it is a high-level language. Python was designed to abstract away the complexities and annoyances of memory management, types, and all of the nuances that can drive one crazy in low level languages like C/C++. The trade off is less development time for slower speeds. It is perfect for prototyping but not always the best for implementation onto a hardware system. But there are several different methods we can implement to shave off a lot of time by using libraries implemented in C with python wrappers. 

Another thing to discuss before we begin is the idea of scale. When I first implemented the corner detector, I chose images from the hpatches dataset for ease of accessibility. But like the MNIST dataset of machine learning, the dataset is saturated. While revolutionary at the time of the release, hpatches has been eclipsed by the needs of modern computer vision. For example, our corner detector took around $25-30$ seconds for images in hpatches. That is a great start for just implementing the detector. But then when one starts to implement experiments where you do an entire scene of six images and find matches between them, that is now 3 minutes per scene. If we use an image that is $5000x3000$ pixels, we now have to wait several minutes just for one image! In a world where paradigms change almost daily for computer scientists, knowing how to scale your project for the future is vital. So now let us scale first with vectorization.

A quick note: I changed our original implementation to detect a given number of keypoints instead of the top 10% of responses to make later matching evalation more reproducible and comparable. 

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

### Experiments

## Adaptive Non-Maximal Suppression

### Theory

### Experiments

## Conclusions
