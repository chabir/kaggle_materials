Carvana competition: 78th/735 - bronze medal -september 2017
 


Main idea for Carvana competition: 


A. the pictures are grouped by car to avoid leakage in the validation set

B. background is separated using USV and reused as 4th channel along with the 3 channels of the picture as an input to the Unet model
 
C. any RESIZE of pictures at input or output induces some loss in this high resolution picture format, so no resize function is used:
   the pictures are cut on top and bottom and extended with a black screen on left and right sides.


Note: a proper CV shall have help to gain about 25 ranks but I got to busy in the last two weeks of the competition to concentrate on it.
