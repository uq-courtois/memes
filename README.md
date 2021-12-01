Please download and add https://github.com/oyyd/frozen_east_text_detection.pb to your project folder. Also extract .zip files with distractor and meme templates (used in the performance test in the paper) to source_images folder.

###

Order of running scripts:

1. cleanimages.py to blur text (necessary pre-processing)
2. featurematching.py to assess feature similarity between all pairs of clean images (presets d = 25, m = 10 - as evaluated the best in the paper)
3. evaluation.py to generate component folders, generate statistics, and potentially change local parameters in components (d - distance and m - minumum number of matches)

###

imagecollage.py is a helper script that generates the image collages per component.
