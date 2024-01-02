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

Another thing to discuss before we begin is the idea of scale. When I first implemented the corner detector, I chose images from the hpatches dataset for ease of accessibility. But like the MNIST dataset of machine learning, the dataset is saturated. While revolutionary at the time of the release, hpatches has been eclipsed by the needs of modern computer vision. For example, our corner detector took around $25-30$ seconds for images in hpatches. That is a great start for just implementing the detector. But then when one starts to implement experiemnts where you do an entire scene of six images and find matches between them, that is now 3 minutes per scene. If we use an image that is $5000x3000$ pixels, we now have to wait several minutes just for one image! In a world where paradigms change almost daily for computer scientists, knowing how to scale your project for the future is vital. So now let us scale first with vectorization.

### Vectorization

Vectorization is the process of applying operations across an entire array at once rather than going one array element at a time and performing the operation.  

### Parallelization 

### Experiments

## Adaptive Non-Maximal Suppression
