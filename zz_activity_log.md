# noted that the clam model does not learn properly. to investigate the reason generated synthetic data where the task difficulty can be adjusted. the model was able to learn based on the synthetic data. suspected that the issue lay somewhere in the data-processing pipeline. found that the patching as represented by masks/stitches were not optimal representations of the tissue regions of the whole-slide-images. decided to try to improve segmentation for better patches.

## ----
# was planning to implement dl based segmentation, but now it seems throught testing that otsu thresholding can do well enough.

# i made an early definition of what i wanted to achieve before the time of the meeting: to have a model that has learned to use the data to some extent for prediction and to reproduce some of the results (be specific).

task: see if otsu can give the same quality results in the pipeline, and if they extend through contouring. may need to use different segmentation level.
# otsu works but significant chunks of tissue are often lost downstream. can i change contouring to be less discriminating, or can i design my own patching algorithm? the advantage of contouring is that it can be easier to filter out holes.

task: figure out why the tissue sections are excluded.
# answer: the filtering parameters were not tuned to my slides and so filtered too many contours. by adjusting, more separate tissue patches can be included.

## RECAP: issue was the model did not learn. investigated the models ability by designing synthetic data and applying the model to it.this worked fine. i then suspected the issue was that the patches did not well represent the tissue on the slides. turned out patching was only representative of a fraction of the tissue on a given slide. attempted to adjust preprocessing parameters such as thresholds and otsu. based on stitch visualizations, this had limited success. isolated testing and found that otsu thresholding works just fine. realized that the wsi sections included are based on the contouring. after testing concluded that filtering params are not tuned to my slides and that this has a significant impact on patchset representation of slide.

task: find a better set of filtering parameters and see how well the performance of those parameters generalize across slides.
# through testing found parameters that cover a much greater amount of tissue. on a few sample wsi, seems to still leave out significant tissue chuncks. as i dont yet know exactly what is optimal to include, i shall leave it like that for now. when i have the knowledge of how to identify tissue that is useful, i will return to this point. ive also simplified the patch extraction code. i removed the need of process_list.csv, and revised what parameters are useful to adjust manually. hopefully i did not oversimplify

task: verify the quality of patches generated and that they represent tissue segments of the whole-slide-images in the way that the stitches suggest.
# results appear ok, but need expert opinion to confirm and inform next steps.
# update: given the task of detecting TINT and with it the lack on certainty in exactly what tissue types to filter out the path forward is not obvious since im not able to do heavy computations right now.

task: create a filter for blur and background to apply post patch extraction