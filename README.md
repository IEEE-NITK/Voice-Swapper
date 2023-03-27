# Voice-Swapper
A voice-changing dictaphone


Voice-Swapper is a dictaphone that will be used to convert the user’s voice(source) to a target voice without any loss of linguistic information. VC is useful in many applications, such as customizing audio book and avatar voices, dubbing, voice modification, voice restoration after surgery, and cloning of voices of historical persons. VC models are primarily implemented with Generative Adversarial Networks(GANs) which provide promising results by generating the user fed-in statements in the target’s voice. We aim to build these models from scratch and implement them using a web application(with Streamlit). This project would be an inter-sig project between Diode and CompSoc.  

## Objectives
To be the first to implement [CycleGAN](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/) in Tensorflow 2.0 (*NO* existing implementation of the same)   
To train the CycleGAN model on the "Trump" and "Peter Griffin" datasets.  
To implement these models on a web application.  
To perform voice swapping(conversion) in real-time.  

## Scope
If time permits, we aim to propose a novel model based on the survey/summary of model performances in VCC2016 and write a research 
paper based on its performance compared to the existing models.  

Click [here](https://docs.google.com/document/d/1yTXpZWsHaKSjoWy35lg43YfADTnA-_4a/edit) for the complete proposal.

