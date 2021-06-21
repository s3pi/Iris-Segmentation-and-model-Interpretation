### Segmentation of Iris and model interpretation
This problem discusses pixel-wise image segmentation using Encoder-Decoder architecture with skip connections resembling UNet and the model's interpretation using LRP (Layerwise Relevance Propogation: http://heatmapping.org/) algorithm. The image segmentation problem is a core vision problem with a longstanding history of research. Historically, this problem has been studied in the unsupervised setting as a clustering problem: given an image, produce a pixel wise prediction that segments the image into coherent clusters corresponding to objects in the image. We segment and explain how much is the relevance of each pixel in the decision of segmention.

##### Explaining LRP paper (Explaining non linear classification decisions using Deep Taylor Decomposition): https://youtu.be/BQgfZeSrVc8. 
Interpreting multi layer neural networks by decomposing the network's classification decisions into contributions of the input.
