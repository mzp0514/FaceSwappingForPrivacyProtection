# FaceSwappingForPrivacyProtection
Substitute random faces with the same soft biometrics for the real face in the original videos to enable data analysis on soft biometrics while protecting facial data privacy.

## Motivation
In recent years the misuse of people’s face data has raised concern on privacy issue.  In many data analysis tasks, we don’t need to identify the person in the videos with all the facial information, and we just want to know, for example, what kind of people like to shop in certain store, like female, elder people and so on. What we need is people's soft biometrics (such as age, gender, height, weight, etc.). These features can be useful to data analysis, but they are not decisive features to identify a person. In other words, we can’t identify a person simply by his soft biometrics.

So, how can we enable data analysis on people’s soft biometrics and in the meanwhile prevent the abuse of people’s whole face information? One solution may be to replace the real faces in the original videos with random faces with the same soft biometrics retained, so that data analyzers can gain enough needed information from the adapted videos for analysis, and they won’t be able to identify the people in the video. 

The project is to preliminarily demonstrate the concept and feasibility of the proposal. For easy implementation, we will only involve age and gender as the soft biometrics.  The whole process include *face extraction*, *age & gender estimation*, *target face generation* and finally *face swapping*.


## Usage

see *demo.ipynb*

## Results
https://drive.google.com/file/d/1jdw7L8QSsQ0HN0y-1rlFjzuAI1iqeGPo/view?usp=sharing

![result](https://user-images.githubusercontent.com/34649843/148908011-22956581-3ae2-47c3-bea4-45b96b856ffb.png)

## More information
https://docs.google.com/presentation/d/1bN9mc9iMqKe-eH2HG9eyji0mkkokfc6V/edit?usp=sharing&ouid=114134881174581320604&rtpof=true&sd=true


## Credit
https://github.com/neuralchen/SimSwap

https://github.com/serengil/deepface

https://github.com/Linzaer/Face-Track-Detect-Extract


## References
[1] A. Dantcheva, C. Velardo, A. D’Angelo and J.-L. Dugelay, "Bag of soft biometrics for person identification," Multimedia Tools and Applications, vol. 51, no. 2, pp. 739-777, 2011.

[2] J. Deng, J. Guo, S. Zafeiriou and X. Niannan, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in CVPR, 2019. 

[3] S. I. Serengil and A. Ozpinar, "LightFace: A Hybrid Deep Face Recognition Framework," 2020 Innovations in Intelligent Systems and Applications Conference (ASYU), 2020.

[4] S. I. Serengil and A. Ozpinar, "HyperExtended LightFace: A Facial Attribute Analysis Framework," in 2021 International Conference on Engineering and Emerging Technologies (ICEET), 2021. 

[5] Z. Zhang, Y. Song and H. Qi, "Age Progression/Regression by Conditional Adversarial Autoencoder," in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 

[6] R. Chen, X. Chen, B. Ni and Y. Ge, "SimSwap: An Efficient Framework For High Fidelity Face Swapping," in Proceedings of the 28th ACM International Conference on Multimedia, Seattle, WA, USA, 2020. 
