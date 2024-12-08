# [NeurIPS 2024] Enhancing Consistency-Based Image Generation via Adversarialy-Trained Classification and Energy-Based Discrimination

This repository contains the official code and checkpoints used in the paper "[Enhancing Consistency-Based Image Generation via Adversarialy-Trained Classification and Energy-Based Discrimination] ([https://arxiv.org/abs/2404.04465](https://www.arxiv.org/abs/2405.16260))"


## Abstract
The recently introduced Consistency models pose an efficient alternative to diffusion algorithms, enabling rapid and good quality image synthesis. These methods overcome the slowness of diffusion models by directly mapping noise to data, while maintaining a (relatively) simpler training. Consistency models enable a fast one- or few-step generation, but they typically fall somewhat short in sample quality when compared to their diffusion origins. In this work we propose a novel and highly effective technique for post-processing Consistency-based generated images, enhancing their perceptual quality. Our approach utilizes a joint classifier-discriminator model, in which both portions are trained adversarially. While the classifier aims to grade an image based on its assignment to a designated class, the discriminator portion of the very same network leverages the softmax values to assess the proximity of the input image to the targeted data manifold, thereby serving as an Energy-based Model. By employing example-specific projected gradient iterations under the guidance of this joint machine, we refine synthesized images and achieve an improved FID scores on the ImageNet 64x64 dataset for both Consistency-Training and Consistency-Distillation techniques.

## Checkpoint

Checkpoint ResNet50 model [https://drive.google.com/file/d/1KTP2vL8cjK0OAL2CXIC53fREVg8pRisz/view?usp=sharing
](https://drive.google.com/file/d/1L6Nd3ldlu_rL6OIPvHioOI5iUewn0cfZ/view?usp=sharing)
Checkpoint Wide-RN 50-2 model [https://drive.google.com/file/d/1KTP2vL8cjK0OAL2CXIC53fREVg8pRisz/view?usp=sharing](https://drive.google.com/file/d/1KTP2vL8cjK0OAL2CXIC53fREVg8pRisz/view?usp=sharing)

## Citation
```
@misc{golan2024enhancingconsistencybasedimagegeneration,
      title={Enhancing Consistency-Based Image Generation via Adversarialy-Trained Classification and Energy-Based Discrimination}, 
      author={Shelly Golan and Roy Ganz and Michael Elad},
      year={2024},
      eprint={2405.16260},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
