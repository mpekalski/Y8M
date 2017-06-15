This repository contains all code used by team Y8M ([Xpeuler](https://www.kaggle.com/xpeuler), [MihaSkalic](https://www.kaggle.com/mihaskalic) and [MPekalski](https://www.kaggle.com/mpekalski)) in the Kaggle's competition Google Cloud & YouTube-8M Video Understanding Challenge. For more details about the competition please go to https://www.kaggle.com/c/youtube8m . For more details about the YouTube 8M dataset please refer to Google Research's website dedicated to that dataset https://research.google.com/youtube8m/ . The starter code we based our solution on was provided by the organizers under https://github.com/google/youtube-8m/ 


This repository contains three additional subfolders:

`video_level_code` contains all the code needed to reproduce our video level models that we have incorporated in the final ensemble.

`frame_level_code` contains all the code needed to reproduce our frame level models that we have incorporated in the final ensemble.

`C` that contains a code written in C++ that we used for ensemble, calculating GAP score, calculating performance by label. For more details please see REAMDE file in the folder.

`bstnet`, which contains a bit older version of the code that also includes boosting network. So, not all changes we have introduced that are available in this folder are also available in the bstnet. For a sample usage script please refer to the bash script in the bstnet.

To reproduce the final results follow readmes in each of the two *_code folders. Finally, you can ensemble the predictions using addPred from the C folder. Execute: 

```
./c/addPred ./mypredictionfolder/MoNNs.csv ./mypredictionfolder/LSTMs.csv ./mypredictionfolder/GRUs.csv 0.4 0.36 0.24 /mypredictionfolder/final_prediction.csv
```

As in the example we used weights of 0.4, 0.36, 0.24 for MoNNs, LSTMs and GRUs, respectively.
