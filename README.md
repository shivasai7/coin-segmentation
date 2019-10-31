# coin-segmentation
    Image segmentation using coins. Coin image enhances and will be able to classify different coins easily. 
    In this we initially use canny’s algorithm to find edges for each coins. 
    Later we use distance transform to find edges of each coins and labelling them. 
    And later we find center of each coin that is represented using lock peak maximum. 
    For each center we apply watershed algorithm. That is this algorithm colors each coin with a different color. 
    Using this colors the system will be able to identify various coins.
    It is used in image segmentation and autonomous driving and for object classification.
