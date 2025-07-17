
### <h3 id="toc_3">Image enhancement & implementation of CLAHE algorithm </h3>

The project contains a series of image enhancment algorithms. Noise removal, contrast enhancment, binarization etc. <br />
It also contains my own implementation of the CLAHE (contrast-limited adaptive histogram equalization) algorithm.

**Getting started:**

* Download the project and run a section and you will see a bad input image being enhanced step by step
* Enhanced images are **automatically saved to example_output/** directory 
* Play with the parameters for the dos_clahe function and watch the output change quality

**Enhancements Examples:**

Original noisy image (impulse/salt-and-pepper noise)           |  Enhanced image
:-------------------------:|:-------------------------:
![](https://github.com/gordicaleksa/digital-image-processing/blob/master/project3/example_input/enigma.png)  |  ![](https://github.com/gordicaleksa/digital-image-processing/blob/master/project3/example_output/enigma_out.jpg)

Original image           |  Enhanced image
:-------------------------:|:-------------------------:
![](https://github.com/gordicaleksa/digital-image-processing/blob/master/project3/example_input/text_stripes.png)  |  ![](https://github.com/gordicaleksa/digital-image-processing/blob/master/project3/example_output/binarization.png)

**Histogram Equalization Examples:**

Original image <br /> (high contrast)           |  Regular HE algorithm  |       CLAHE algorithm
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/gordicaleksa/digital-image-processing/blob/master/project3/example_input/mars_moon.png)  |  ![](https://github.com/gordicaleksa/digital-image-processing/blob/master/project3/example_output/mars_clahe_he.jpg) | ![](https://github.com/gordicaleksa/digital-image-processing/blob/master/project3/example_output/mars_clahe_best.jpg) 
