# arXiv:2310.11441v2 [cs.CV] 6 Nov 2023

## 논문 정보

**파일명**: `Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V.pdf`

**위치**: `vision_language/vlm/Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V.pdf`

---

## Abstract

We present Set-of-Mark (SoM), a new visual prompting method, to unleash the
visual grounding abilities of large multimodal models (LMMs), such as GPT-4V.
As illustrated in Fig. 1 (right), we employ off-the-shelf interactive segmentation
models, such as SEEM/SAM, to partition an image into regions at different levels
of granularity, and overlay these regions with a set of marks e.g., alphanumerics,
masks, boxes. Using the marked image as input, GPT-4V can answer the questions
that require visual grounding. We perform a comprehensive empirical study to
validate the effectiveness of SoM on a wide range of fine-grained vision and
multimodal tasks. For example, our experiments show that GPT-4V with SoM
outperforms the state-of-the-art fully-finetuned referring expression comprehension
and segmentation model on RefCOCOg in a zero-shot setting. Code for SoM
prompting is made public here: https://github.com/microsoft/SoM.

---

## 1. Introduction

In the past few years, we have witnessed a significant advancement in large language models
(LLMs) [2, 3, 10, 43, 60, 35]. In particular, Generative Pre-trained Transformers or GPTs [2, 35]
have led a number of breakthroughs in the industry and academia. Since the release of GPT-4, large
multimodal models (LMMs) have gained a growing interest in the research community. A number of
works strive to build multimodal GPT-4 based on open-sourced models [28, 63, 57, 12, 47, 50, 20].
Recently, GPT-4V(ision) is released and attracts immediate attention from the community for its
outstanding multimodal perception and reasoning capability. Its superiority and generality are
showcased in [49]. Despite the unprecedented, strong, vision-language understanding capabilities,
GPT-4V’s fine-grained visual grounding ability is relatively weak, or yet to be unleashed. For
example, GPT-4V can hardly produce a sequence of accurate coordinates for a dog in the image1 , or
a few traffic lights [54].
Our study is motivated by addressing the failed cases of GPT-4V on fine-grained vision tasks, such
as object referring. In principle, these tasks require the model to have rich semantic understanding
and accurate spatial understanding of visual contents. However, prompting GPT-4V to produce
a sequence of tokens that contains textual description and numeric coordinates not only hurts the
natural fluency in LLMs but also dismisses the spatial understanding ability in vision models used
by LMMs, as pointed out by [7]. Therefore, in this study we focus on improving visual inputs
by proposing a new visual prompting method to unleash the visual grounding capability of LMMs.
Visual prompting has been explored for various vision and multimodal tasks [9, 19, 65, 45, 41]. These
works can be categorized into two categories. One is to encode visual prompts such as point, box,
and stroke into latent features, which are then used to prompt the vision models [65, 19]. The other is
to overlay visual marks directly onto the input images. The overlaid marks could be a red circle [41],
a highlighted region [48], or a few circles with arrows [49]. Although these studies demonstrate
the promise of pixel-level visual prompting, they are limited to visually referring to one or a few
objects. Moreover, all these marks are not easily “speakable” by LLMs, thus impeding the grounding
capability for GPT-4V.
In this paper we present a new prompting mechanism called Set-of-Mark (SoM) prompting , i.e.
simply adding a set of visual marks on top of image regions. We first partition an input image into a
set of semantically meaningful regions. Then, we overlay each region with a mark of various formats
such as numbers, alphabets, masks, or boxes. We perform an empirical study to validate whether
GPT-4V could better ground the regions given these interpretable and “spreadable” visual marks. Our
experiments show that SoM can drastically unleash the grounding capability of GPT-4V. As shown
in Fig. 1, compared 

...(계속)

---

## 2. Related Work

Mathematically, Eq. (1) becomes
T o = F(SoM(I), T i ).
(2)
| {z }
Im

Although it is straightforward to apply SoM to all LMMs, we find that not all LMMs have the ability
to “speak out” about the marks. Actually, we find that only GPT-4V, when equiped with SoM, shows
the emergent grounding ability and significantly outperforms the other LMMs. In what follows, we
will explain how we partition an image into regions and mark the image regions in SoM.
2.2

Image Partition

Given an input image, we need to extract meaningful and semantically aligned regions. Ideally, the
extraction should be automatic or semi-automatic to avoid extra burden on users. To achieve this, we
employ a suite of image segmentation tools. To support different use cases, the segmentation tools
need to possess the following properties:

---

## 3. Method / Approach

[4] Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali
Wang, Yu Qiao, Tong Lu, and Limin Wang. Videollm: Modeling video sequence with large
language models, 2023.
[5] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechu Liu, Pengchuan Zhang, Raghuraman
Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: Large
language model as a unified interface for vision-language multi-task learning. github, 2023.
[6] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra:
Unleashing multimodal llm’s referential dialogue magic, 2023.
[7] Ting Chen, Saurabh Saxena, Lala Li, David J. Fleet, and Geoffrey Hinton. Pix2seq: A language
modeling framework for object detection, 2022.
[8] Xi Chen, Zhiyan Zhao, Feiwu Yu, Yilei Zhang, and Manni Duan. Conditional diffusion
for interactive segmentation. In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 7345–7354, 2021.
[9] Xi Chen, Zhiyan Zhao, Yilei Zhang, Manni Duan, Donglian Qi, and Hengshuang Zhao. Focalclick: Towards practical interactive image segmentation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 1300–1309, 2022.
[10] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam
Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm:
Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
[11] Haixing Dai, Chong Ma, Zhengliang Liu, Yiwei Li, Peng Shu, Xiaozheng Wei, Lin Zhao, Zihao
Wu, Dajiang Zhu, Wei Liu, et al. Samaug: Point prompt augmentation for segment anything
model. arXiv preprint arXiv:2307.01187, 2023.
[12] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng
Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose
vision-language models with instruction tuning, 2023.
[13] Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, and Xiaojuan Qi. Pla:
Language-driven open-vocabulary 3d scene understanding. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 7010–7019, 2023.
[14] Zheng Ding, Jieke Wang, and Zhuowen Tu. Open-vocabulary panoptic segmentation with
maskclip. arXiv preprint arXiv:2208.08984, 2022.
[15] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing
Xu, and Zhifang Sui. A survey for in-context learning. arXiv preprint arXiv:2301.00234, 2022.
[16] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In European Conference on Computer Vision, pages 540–557.
Springer, 2022.
[17] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via
vision and language knowledge distillation. arXiv preprint arXiv:2104.13921, 2021.
[18] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring
to object

...(계속)

---

## 4. Experiments

Surfboard 5. Handbag

User: I have labeled a bright numeric ID at
the center for each visual object in the image.
Please tell me the IDs for: The laptop behind
the beer bottle; Laptop turned on.
GPT-4V The IDs for the items you’ve
mentioned: The laptop behind the beer bottle:
6 Laptop turned on: 2

Video Object Segmentation

Grounded Segmentation

User: The 2 images are from the same video,
where the first image is the first frame and
the second image is a later frame. In the first
image, there are 2 objects labeled with 1,2.
Can you track these 2 objects in the second
image?
GPT-4V 1. The object labeled with 1 (a
boxer in red headgear) is most similar to the
object labeled with 2 (another boxer in red
headgear). 2. The object labeled with 2 (a
boxer in blue) is most similar to the object
labeled with 1 (another boxer in blue).

User: I have labeled a bright numeric ID at
the center for each visual object in the image.
Given the image showing a man in glasses
holding a piece of paper, find the corresponding regions for a man in glasses, a piece of
paper.
GPT-4V The "man in glasses" corresponds
to the region labeled with "2". The "piece of
paper" corresponds to the region labeled "5".

Figure 7: Examples of some vision tasks we benchmark on GPT-4V.
• Phrase Grounding: Slightly different from referring segmentation, phrase grounding uses a complete sentence consisting of multiple noun phrases. We ask GPT-4V to allocate the corresponding
regions for all labeled phrases.
• Video Object Segmentation: It takes two images as input. The first image is the query image
which contains a few objects of interest to identify in the second image. Given that GPT-4V
supports multiple images as input, our prompting method can also be applied to ground visual
objects across frames in a video.
We show how we prompt GPT-4V using SoM for the above vision tasks in Fig. 7. It is worth noting
that SoM can be applied to broader tasks, such as region captioning, keypoint localization, and part
segmentation, etc. Due to the limited access (quota) to GPT-4V, we focus on the aforementioned
7

Vision Task
Generic Segmentation
OV Segmentation
Phrase Grounding
Referring Expression Segmentation
Referring Expression Comprehension
Video Object Segmentation

Dataset

#Images

#Instances

Marks

Metric

COCO [26]
ADE20K [62]
Flickr30K [38]
RefCOCO [18]
RefCOCO [18]
DAVIS [37]

100
100
100
100
100
71

567
488
274
177
177
157

Number & Mask
Number & Mask
Number & Box
Number & Mask
Number & Mask
Number & Mask

Precision
Precision
Recall@1
mIoU
ACC@0.5
J&F

Table 1: Benchmarks used to evaluate the visual grounding capability of GPT-4V. Note that we select
a small subset of images or videos from the corresponding datasets.
vision tasks and leave the explorations to other tasks for future work. We present the detailed
empirical study in the next section.

4

Experiments

4.1

Experimental Setup

Implementation. We do not need to train any models for our method. However, due to the limited

...(계속)

---

## 5. Results

We present the experimental results in Table 2.
Image Segmentation On image segmentation tasks, we evaluate the classification precision of
GPT-4V + SoM and compare it with the strong segmentation model MaskDINO on COCO Panoptic
segmentation dataset and OpenSeeD on ADE20K Panoptic segmentation dataset. For MaskDINO
8

OV Segmentation
COCO ADE20K

RefCOCOg
REC
RES

✗
✓
✗
✗
✗
✗
✓
✗

80.7
n/a
n/a
n/a
-

23.4
n/a
n/a
n/a
-

n/a
n/a
86.1∗
85.8∗
n/a
-

✓
✓

✓
✓

n/a
n/a

n/a
n/a

Shikra [6]
LLaVA-1.5 [27]
MiniGPT-v2 [5]
Ferret [54]

✓
✓
✓
✓

✗
✗
✗
✗

n/a
n/a
n/a
n/a

GPT-4V [36]
GPT-4V [36] + SoM (Ours)

✓
✓

✓
✓

n/a
75.7

Model

LMM

Zero-shot

MaskDINO [23]
OpenSeeD [55]
GLIPv2 [56]
GDINO [31]
X-Decoder [64]
PolyFormer [29]
SegGPT [45]
SEEM [65]

✗
✗
✗
✗
✗
✗
✗
✗

RedCircle [41]
FGVP [48]

Phrase Grounding
Flickr30K

VOS
DAVIS2017

n/a
n/a
n/a
64.6∗
67.2
n/a
65.7∗

n/a
n/a
87.7∗
90.5
n/a
n/a
n/a
-

n/a
n/a
n/a
n/a
62.8
n/a
75.6
62.8

59.4∗
63.3∗

n/a
n/a

n/a
n/a

n/a
n/a

n/a
n/a
n/a
n/a

82.6∗
63.3
84.4∗
85.8*

n/a
n/a
n/a
n/a

77.4
n/a
n/a
81.1

n/a
n/a
n/a
n/a

n/a
63.4

25.7
86.4

n/a
75.6

n/a
89.2

n/a
78.8

Table 2: Main quantitative results. ∗ denotes that the number is reported in the original papers
which are evaluated in the full validation datasets. Other numbers are evaluated in our sub-sampled
validation datasets constraint by GPT-4V interface.
Mark Type
Number & Mask
Number & Mask & Box

Flickr30K (R@1)

Mask type

Refcocog (mIoU)

84.4
89.2

MaskDINO
GT mask

75.6
90.1

Table 3: The performance on Flickr30K with
different mark types on a subset of our dataset.

Table 4: The performance on Flickr30K with
different mask types.

and OpenSeeD, we give them GT boxes and evaluate the output class. For GPT-4V + SoM, we
overlay GT masks with alpha=0.4 and 0.2 for COCO and OpenSeeD, respectively, and add the ID
number on each mask. We provide GPT-4V with the vocabulary of the datasets and ask it select a
class label for each region. The results show that zero-shot performance of GPT-4V + SoM is close
to the performance of fine-tuned MaskDINO and is much higher than the zero-shot performance of
OpenSeeD on ADE20K. The similar performance on COCO and ADE20K for GPT-4V indicates its
strong generalization ability to a wide of visual and semantic domains.
Referring For referring tasks, we evaluate RES and REC on RefCOCOg. We use MaskDINO to
propose masks and overlay the masks and numbers on the images. We use mIoU as the evaluation
metric and compare it with state-of-the-art specialist PolyFormer [29] and SEEM [65]. Accordingly,
GPT-4V + SoM outperforms PolyFormer by a large margin. Note that the performance of PolyFormer
is evaluated on our dataset for apple-to-apple comparison. For REC, we convert the masks into boxes
for our method. When prompting GPT-4V to directly output the coordinates, we attain significantly
poor performance (25.7), which verifies our earlier hypothesis. Once augmented with SoM, GPT4V beats both specialists such as Ground

...(계속)

---

## 6. Conclusion

We have presented Set-of-Mark Prompting, a simple yet effective visual prompting mechanism for
LMMs, particularly GPT-4V. We show that simply overlaying a number of symbolic marks on a
set of regions of an input image can unleash the visual grounding ability of GPT-4V. We present
a comprehensive empirical study on a wide range of fine-grained vision tasks to demonstrate that
SoM-prompted GPT-4V is superior to fully-finetuned specialist models and other open-sourced
LMMs. Moreover, our qualitative results show that GPT-4V with SoM possesses extraordinary
fine-grained multimodal perception, cognition, and reasoning capabilities across the board. We hope
that SoM would inspire future works on multimodal prompting for LMMs and pave the road towards
multimodal AGI.

Acknowledgement
We thank Fangrui Zhu for the thoughtful discussions. We thank Xuan Li and Biyi Fang from
Microsoft Office Team for early brainstorming.

---

## Notes

- 📝 이 문서는 PDF에서 자동 추출되었습니다
- 📄 전체 내용은 원본 PDF를 참조하세요
- ⚠️ 수식, 그림, 표는 추출되지 않았습니다

---

## References

📎 원본 논문: [Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V.pdf](./Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V.pdf)
