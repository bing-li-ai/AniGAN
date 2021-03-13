# AniGAN: Style-Guided Generative Adversarial Networks for Unsupervised Anime Face Generation

<p align="center"> 
  <img src="./imgs/top.jpg" height="290">
</p>

[**AniGAN: Style-Guided Generative Adversarial Networks for Unsupervised Anime Face Generation**](https://arxiv.org/abs/2102.12593)<br/>

Bing Li<sup>1</sup>,
Yuanlue Zhu<sup>2</sup>,
Yitong Wang<sup>2</sup>,
Chia-Wen Lin<sup>3</sup>,
Bernard Ghanem<sup>1</sup>,
Linlin Shen<sup>4</sup><br/>

<sup>1</sup>*Visual Computing Center, KAUST, Thuwal, Saudi Arabia*<br/>
<sup>2</sup>*ByteDance, Shenzhen, China*<br/>
<sup>3</sup>*Department of Electrical Engineering, National Tsing Hua University, Hsinchu, Taiwan*<br/>
<sup>4</sup>*Computer Science and Software Engineering, Shenzhen University, Shenzhen, China*<br/>

### Datasets

We build a new dataset called **face2anime**, which is larger and contains more diverse anime styles (e.g., face poses, drawing styles, colors, hairstyles, eye shapes, strokes, facial contours) than selfie2anime. The **face2anime** dataset contains 17,796 images in total, where the number of both anime-faces and natural photo-faces is 8,898. The anime-faces are collected from the Danbooru2019 dataset, which contains many anime characters with various anime styles. We employ a pretrained cartoon face detector to select images containing anime-faces. For natural-faces, we randomly select 8,898 female faces from the CelebA-HQ dataset. All images are aligned with facial landmarks and are cropped to size 128 × 128. We separate images from each domain into a training set with 8,000 images and a test set with 898 images.

You can download the **face2anime** dataset from [Google Drive](https://drive.google.com/file/d/1Exc6QumR2r0aFUtfHOdAgle4F4I9zwF3/view?usp=sharing).

If you find this work useful or use the **face2anime** dataset, please cite our [paper](https://arxiv.org/abs/2102.12593):
```bibtex
@misc{li2021anigan,
      title={AniGAN: Style-Guided Generative Adversarial Networks for Unsupervised Anime Face Generation}, 
      author={Bing Li and Yuanlue Zhu and Yitong Wang and Chia-Wen Lin and Bernard Ghanem and Linlin Shen},
      year={2021},
      eprint={2102.12593},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

![](./imgs/2.jpg)

