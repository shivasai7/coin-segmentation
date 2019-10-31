# coin-segmentation
    Image segmentation using coins. Coin image enhances and will be able to classify different coins easily. 
    In this we initially use canny’s algorithm to find edges for each coins. 
    Later we use distance transform to find edges of each coins and labelling them. 
    And later we find center of each coin that is represented using lock peak maximum. 
    For each center we apply watershed algorithm. That is this algorithm colors each coin with a different color. 
    Using this colors the system will be able to identify various coins.
    It is used in image segmentation and autonomous driving and for object classification.
        Comparing edge-based and region-based segmentation¶. In this example, we will see how to segment objects from a background.
        We use the coins image from skimage.data, which shows several coins outlined against a darker background.
        A demo of structured Ward hierarchical clustering on an image of coins¶ Compute the segmentation of a 2D image with Ward 
        hierarchical clustering. The clustering is spatially constrained in order for each segmented region to be in one piece.
#segmentation fault
    I have a segmentation Fault above. And i cant locate where it is. 
    Execute the code with gdb. gdb ./exec then type 'run' and gdb will do his magic and tell you exactly where the 
    segmentation fault is and you will be able to understand why. From reading the code it looks like we are accessing vert.
    contents pointer without allocate memory for it...
