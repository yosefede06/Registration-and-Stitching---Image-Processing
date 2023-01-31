# Registration-and-Stitching---Image-Processing


A software that performs automatic Stereo Mosaicing, which involved the combination of a sequence of images into a single panoramic image. The input of the algorithm was a sequence of images taken from left to right, which captured a scene and had significant overlap in the field of view of consecutive frames.

To accomplish this task, I followed two key steps, which were Registration and Stitching. The Registration step involved finding the geometric transformation between each consecutive image pair by detecting Harris feature points, extracting their MOPS-like descriptors, matching these descriptors, and then fitting a rigid transformation using the RANSAC algorithm.

In the Stitching step, I combined strips from aligned images into a sequence of panoramas. To do this, I compensated for global motion and made the residual parallax, as well as other motions, visible. Through this process, I was able to successfully produce a panoramic image from a sequence of individual images.
