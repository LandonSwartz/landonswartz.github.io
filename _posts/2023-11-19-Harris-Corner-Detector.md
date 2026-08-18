---
layout: single
title: "Implementing the Harris Corner Detector"
date: 2023-11-19
categories: blog
tags: computer-vision corners harris-corner-detector
excerpt: Implementing to understand and visualize the Harris Corner Detector
---

<!-- # Implementing the Harris Corner Detector -->

NOTE: This post was written when I was younger, dumber, and more focused on showing what I was learning. Forgive mistakes and hastily written code.

## The Harris Corner Detector

The Harris Corner Detector is one of the oldest interest point detectors in the toolkit of computer vision. First introduced in the 1988 paper "A Combined Corner and Edge Detector" by Chris Harris and Mike Stephens as an improvement on the Moravec corner algorithm, the algorithm stands as one of the easiest interest point detectors to implement for the aspiring computer vision scientist. In this blog post, we will implement the algorithm piece by piece to see how it works with parameters.

![dime_building](/assets\images\harris_corner_detector\harris_corner_detector_1.png)

## Overview

The algorithm is a simple:

1. Convert the image to grayscale
2. Find the gradients (spatial derivatives) with respect to x and y
3. Set up the structure tensor based on the gradients
4. Calculate the Harris response
5. Perform non-maximal suppression for optimal values

We'll be implementing the algorithm in python for ease of use and simplicity. The focus here is learning the algorithm by building the parts of the algorithm from the ground up, not the building blocks themselves. We'll be utilizing the latest and greatest of built-in functions in OpenCV and numpy. 

### Convert the image to grayscale

Finding the gradients in the x and y direction is can be an expensive processing step for each channel of an RGB image. Therefore, the traditional harris corner detector usually starts with converting the RGB image into a grayscale image to only deal with one channel for calculations.

```python
def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
```

### Find the gradients

Computing the gradients of an image requires a special kind of process that some may be familar with if you have any experience with neural networks and convolutions where you convolve two 3x3 kernels with an image to calculate the approximations of the deriviates in both the x and y direction. The best way to understand it is to see the results in action.

![image_gradients](/assets\images\harris_corner_detector\harris_corner_detector_2.png)

As you can see from the image above, Ix (the gradient in the x / horizontal direction) finds the edges in the horizontal direction of the image. Iy finds the edges in the vertical direction of the image. That is where the _"Combined Corner and Edge Detection"_ part of the method comes into play. Traditionally, we apply a gaussian filter before finding the gradients to remove any noise from the image (especially back in the early 1990's and before Iphones when everyone had a 4K camera in their pocket).

```python
def apply_gaussian_blur(self):
        self.blurred_image = cv2.GaussianBlur(self.gray_image, (self.window_size, self.window_size), 0)

    def compute_gradients(self):
        self.Ix = cv2.Sobel(self.blurred_image, cv2.CV_64F, 1, 0, ksize=self.window_size)
        self.Iy = cv2.Sobel(self.blurred_image, cv2.CV_64F, 0, 1, ksize=self.window_size)
```

Because gradients play such a huge role in computer vision, it is important we dive a little deeper into the inner workings of them. We use an operator called the Sobel operator to find gradients traditionally (remember gradients is just spatial derivatives, so rate of change of pixels in an area). Take $G_x$ and $G_y$ as the gradients and $A$ as the original image.

$$G_x = \begin{bmatrix} +1 & 0 & -1 \\\ +2 & 0 & -2 \\\ +1 & 0 & -1 \end{bmatrix} * A$$

$$G_y = \begin{bmatrix} +1 & +2 & +1 \\\ 0 & 0 & 0 \\\ +1 & +2 & +1 \end{bmatrix} * A$$

### Structure Tensor Construction

The Structure Tensor, or the second-moment matrix, is a matrix consisting of the gradients of a function. For our purposes, it describes the distribution of the gradients of an image in the a specific neighborhood around a point. We describe it as below:

$$ M = \Sigma_{(x, y) \in W} \begin{bmatrix} I^2_x & I_xI_y \\\ I_xI_y & I^2_y \end{bmatrix} $$

where Ix and Iy are the previous found image gradients and W is the neighborhood of the pixel we are looking. The Structure Tensor is powerful because we now have the gradients at every position in x and y as a lookup table. We will be using this look up table in the next step to calculate the Harris response 

### Calculating the Harris Response

Mathematically, it is important to remember that a corner is a point whose local neighborhood is characterized by large intensity variation in all directions. Which is a fancy way to say, a corner is a point in a patch of pixels that has the largest changes in x and y. If it was just a change in one direction, it would just be an edge. We can see this when we look at the Harris Response calculated across the entire image.

![corner_response](/assets\images\harris_corner_detector\harris_corner_detector_3.png)

If we want to see even better, we can zoom in on a section of the image.

![zoomed_response](/assets\images\harris_corner_detector\harris_corner_detector_4.png))

Now that looks pretty good! We can see edges being defined and the corners being defined clearly.

To understand what is happening on a math level, we are observing the eigenvalues of the structure tensor to find the corners as seen below:

$$ \lambda_{min} \approx \frac{\lambda_{1}\lambda_{2}}{(\lambda_{1} + \lambda_{2})} $$

To put it into values that we can understand:

$$ R = det(M) - k * tr(M)^2 $$

where the R is the corner response in a patch of pixels. The $det(M)$ is the determinant of the structure tensor and $tr(M)$ is the trace of the structure tensor. The $k$ value is empirically determined in the original implementation to be between $[0.04, 0.06]$. But in the original paper, this empirically determined value is based on a balance of precision and recall. A higher $k$ value will produce precise corners, where false corner are filtered out better but also some true corners. A lower $k$ will raise the recall that produces more corners overall, including false corners.

Why do we care about the eigenvalues then? Well we can see from the image below what we mean visually about changing all directions.

![harris_region](/assets\images\harris_corner_detector\harris_corner_detector_5.jpg)

```python
def detect_corners(self):
    Ix2 = self.Ix ** 2
    Iy2 = self.Iy ** 2
    IxIy = self.Ix * self.Iy

    offset = self.window_size // 2
    height, width = self.gray_image.shape
    self.R = np.zeros((height, width), dtype=np.float64)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sx2 = np.sum(Ix2[y - offset:y + offset + 1, x - offset:x + offset + 1])
            Sy2 = np.sum(Iy2[y - offset:y + offset + 1, x - offset:x + offset + 1])
            Sxy = np.sum(IxIy[y - offset:y + offset + 1, x - offset:x + offset + 1])

            detM = (Sx2 * Sy2) - (Sxy ** 2)
            traceM = Sx2 + Sy2

            self.R[y, x] = detM - self.k * (traceM ** 2)
```

### Non-Maximal Suppression

One of the last things to consider for our little corner detector is Non-Maximal Suppression. If we go back to our zoomed in photo of Harris Responses from above, we would see that there are several pixels in an area that have corner response. This happens because corners can occupy several pixels in an image. It would be redundant to compute descriptors for each and every pixel of the corner. That's where Non-Maximal Suppression comes to the rescue.

The basic process of NMS is to take a sliding window across the corner responses to find the max corner response in a local patch. This can be done quite easily with below:

```python
def apply_non_maximal_suppression(self, neighborhood_size=3):
    height, width = self.R.shape
    offset = neighborhood_size // 2
    suppressed_R = np.zeros((height, width), dtype=np.float64)
    
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            local_max = np.max(self.R[y - offset:y + offset + 1, x - offset:x + offset + 1])
            if self.R[y, x] == local_max:
                suppressed_R[y, x] = self.R[y, x]
                
                
    self.R = suppressed_R
```

As we'll see later, the neighborhood size here is an important parameter that is image and application specific. To see the results it can produced, here's the same Harris Response images from above after NMS.

![nms_response](/assets\images\harris_corner_detector\harris_corner_detector_6.png)

![nms_zoom_response](/assets\images\harris_corner_detector\harris_corner_detector_7.png)

As we see below, the number of corners is greatly reduced and cleaner.

![nms_corners](/assets\images\harris_corner_detector\harris_corner_detector_8.png)

### Conclusion

The Harris Corner Detector works great for a lot of situations in computer vision. If you want to a guess at what situations/images it would be best in, think deeply about the name. It's important to recognize that corners are not universal especially in cases of biological data or organic architecture. Occlusions and different lightnings can affect the harris corner response as well. As anyone who's worked in computer vision will learn, your data is never as pretty as you want. Speed is something to consider to in implementation. Do you need to capture EVERY corner or just enough to do your task as hand on a raspberry pi? Quality and speed are often trade-off you have to consider and tune your algorithm to. 

![castle_1](/assets\images\harris_corner_detector\harris_corner_detector_9.png)

![resort_3](/assets\images\harris_corner_detector\harris_corner_detector_10.png)

![wall_1](/assets\images\harris_corner_detector\harris_corner_detector_11.png)

In the next blog post, we will implement some simple feature descriptors to see how our little corner detector performs in producing homographies.

### References and Code

Harris, Christopher G. and M. J. Stephens. “A Combined Corner and Edge Detector.” Alvey Vision Conference (1988).

ChatGPT-4 for automation of code construction

Corner Detector is implemented at this [repo](https://github.com/LandonSwartz/HarrisCornerDetector)