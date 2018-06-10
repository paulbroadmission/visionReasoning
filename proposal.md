# VisionReasoning

# June 8  2018 #

# Human Skeleton

## 1. Single Picture (Feature Detection)

### Process
  * Input Image --> Two "Maps" for components and joint --> Bipartite Matching to connect --> Result
  
  ![](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/8097368/8099483/8099626/8099626-fig-2-small.gif)
  * Map 1: Part Confidence Maps
      
      $S=(S_1, S_2, ..., S_J)$, body part locations
      
  * Map 2: Part Affinity Fields
  
      $L=(L_1, L_2, ..., L_C)$set of 2D vector fields L of part affinities, which encode the degree of association between parts. Each "limb"(part pairs besides face) has C vector fields.

### Recogition
  * Produce Feature Map
    analyzed by a convolutional network (initialized by the first 10 layers of VGG-19 and fine-tuned), generating a set of feature maps F that is input to the first stage of each branch.
    
  * Two-branch Recognition :
    * Branch 1 for Part Confidence Maps
    * Branch 2 for Part Affinity Fields
 
  ![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_3.png)

  First stage, network produces a set of detection confidence maps $S^1=ρ^1(F)$ and a set of part affinity fields $L^1=ϕ^1(F)$, $ρ^1$ and $ϕ^1$ are the CNNs for inference at Stage 1. 
  
  In each subsequent stage,  concatenated previous predictions and used to produce refined predictions.
  
  \begin{align*} &\mathbf{S}^{t} = \rho^{t}(\mathbf{F}, \mathbf{S}^{t-1}, \mathbf{L}^{t-1}), \forall t\geq 2, \tag{1}\\ &\mathbf{L}^{t} = \phi^{t}(\mathbf{F}, \mathbf{S}^{t-1}, \mathbf{L}^{t-1}), \forall t\geq 2, \tag{2} \end{align*}
  
  $ρ^t$  and $ϕ^t$ are the CNNs for inference at Stage t
  
### Part Association

* Overcome Problem of Part association

  Parts from different people may be joint together, especially occlusion or too near
![](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/8097368/8099483/8099626/8099626-fig-5-small.gif)

* Joint 

  * Vector Algorithm for Affinity
  
     Let $x_{j_1, k}, x_{j_2, k}$ be be groundtruth positions of body parts $j_1$ and $j_2$ from the limb c for person k in the image.
     
     \begin{equation*} \mathbf{L}_{c, k}^{\ast}(\mathbf{p})=\begin{cases} \mathbf{v} &\text{if}\ \mathbf{p}\ \text{on limb}\ c, k\\ 0\ &\text{otherwise}. \end{cases} \tag{8} \end{equation*}
      ![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_chatu.png)

         (即在 Affinity 2D vectors 上就連，不在就不連，用軀幹趨勢將可能錯誤歸零，用以判定是否是同一人的肢幹)
     
 * Bipartite Graph to Connect
![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_6.png)
    Use bipartite graph to reduce error joint between people
    
## 2. Continous Picture Tracking

### Pipeline [[2](https://ieeexplore.ieee.org/document/8237623/)]

* At time t, given a series of past poses $P_{1...t}$ and and their corresponding velocities $Y_{1…t}$ in input video $X_t$

* Encoder: Use LSTM to encode the past information sequence

  Past Encoder takes  past information $X_t,P_{1..t}$, and $Y_{1..t}$ and encodes it in a hidden representation Ht
  
* Decoder

  We also have Past Decoder module to reconstruct the past information from the hidden state. 
  
    ![](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/8234942/8237262/8237623/8237623-fig-3-large.gif)
    
### Encoder/Decorder Models With LSTM [[3](https://dl.acm.org/citation.cfm?id=3045209)][PDF](https://arxiv.org/pdf/1502.04681.pdf)]
![Imgur](https://i.imgur.com/cOXj5bl.png)

* The model consists of two RNNs – the encoder LSTM and the decoder LSTM. 
*  Input to the model is a sequence of vectors (image
patches or features).  After the last input has been read, decoder
LSTM takes over and outputs a prediction for the target sequence. The target sequence is same as the input sequence, but in reverse order. 
*  Reversing the target sequence makes optimization easier because the model can get off the ground by looking at low range correlations.

## 3. Motion Recogition and Cognition [3]

* Process 

  Use results of "Past Decorder" from last step, which are reduced from video based on a time-series sequence, contains pose and velocity. Then we do supervised learning by CNN/LSTM mixed network and ground-truth label (manually labeled by man for each video clip per action) 
  
 ![](https://i.stack.imgur.com/v77Pv.png)
 
  
 * method 1: easy and efficient [4]
 
    This CNN (or more) architecture consists of one hardwired layer, three convolution layers, two subsampling layers, and one full connection layer.

 ![](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/34/6353858/6165309/6165309-fig-3-hires.gif)
 
   So that we can have an action recogition 
 
 * method 2: Feature Optimization and Action Classification [5]
 
    If fact, so far from step 1, 2 until now, we have built a "skeleton trending" feature encoding/decoding representations. Now we can go for feature recogition for actions. 
    
 ![](https://ars.els-cdn.com/content/image/1-s2.0-S0925231216315570-gr1.jpg)
   
   We don't need "skeleton descriptor" in this paper[5], never mind. We need only last two blocks of above image, namely, feature optimization and action classification. 
 
 * method 3: Fushion Classification [6] (suggested)
 
  ![Imgur](https://i.imgur.com/2FF0NcV.png)
  
  We can replace the second source (optical flow) as our velocity results while  first source as skeleton pose description, so that fusion for the final classification is performed with a multi-class linear Support Vector Machine (SVM)[7]. The output will be an action class, comparing the ground-truth labeling.

## 4.  Analysis and Suggestions

So far we have posture recognition data along with time stamps. Suggest that we need have multiple camera here (for example, camera from roof) to decide the location information of people.

* Person Description 

  We can have a Person description $P_{i,t}=(p_{i,t}, loc_{i,t}, t)$, where $i$ is person ID, $t$ is timestamp, $p$ is posture description, $loc$ is location information for the person $i$ in time $t$ calculated from information of multple cameras (roof, for example).

* Team Description 

   A Team contains one or more people description, $T_t = (P_1, P_2, ..., P_N)$, for N people in a team at time t.
   
* Analysis
  
  We can define a strategy $S = (T_0, T_1, T_2, ... , T_t )$ during time 0 to time t.
  
  Let domain knowlege experts (like couch, trainer, physian) to label strategy by some videos. 
  
  We can do analysis by:
  
    * rule-based
    
      calculated by knowlege bases, for example,  distance calculation to identify if a person is in right position (a range of distance) for a strategy.
      
    * data-driven
     
       Build up dataset from experts, use deep learning models to do confidence review (percentage for a team or some person execute a strategy perfectly)
    
  The we can do suggestions by the result of analysis, which can be based on the rule-based warning logs or data-driven percentage below thresholds.
       

## Reference

[1] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh, "Realtime Multi-person 2D Pose Estimation Using Part Affinity Fields", in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, pp. 1302-1310, Jul. 21-26, 2017

[2] Jacob Walker, Kenneth Marino, Abhinav Gupta, and Martial Hebert, "The Pose Knows: Video Forecasting by Generating Pose Futures", in Proceedings of IEEE International Conference on Computer Vision (ICCV), Venice, Italy, pp. 3352-3361, Oct. 22-29, 2017

[3] Nitish Srivastava, Elman Mansimov, and Ruslan Salakhutdinov, "Unsupervised learning of video representations using LSTMs", in Proceedings of the 32nd International Conference on International Conference on Machine Learning, Lille, France, Vol. 37, pp. 843-852, Jul. 06-11, 2015 

[4] Shuiwang Ji, Wei Xu, Ming Yang, and Kai Yu, "3D Convolutional Neural Networks for Human Action Recognition", IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 35, No. 1, pp. 221-231, Jan. 2013

[5] Hanling Zhang, Ping Zhong, Jiale He, and Chenxing Xia, "Combining depth-skeleton feature with sparse coding for action recognition", Neurocomputing, Vol. 230, , pp. 417-426, Mar. 2017

[6] Irina Mocanu, Bogdan Cramariuc, Oana Balan, and Alin Moldoveanu, "A Framework for Activity Recognition Through Deep Learning and Abnormality Detection in Daily Activities", in Proceedings of Image Analysis and Processing (ICIAP), Catania, Italy, pp. 730-740, Sep. 11-15, 2017

[7] Koby Crammer and Yoram Singer, "On the algorithmic implementation of multiclass kernelbased vector machines.", Journal of Machine Learning Research, Journal of Machine Learning Research, Vol. 2, pp. 265-292, 2001




