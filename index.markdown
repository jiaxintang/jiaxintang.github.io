---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
## paper reading

### Survey

#### IEEE communication magazine (16年-)
A Survey of Coflow Scheduling Schemes for Data Center Networks
+ 数据中心调度问题，许多同步流中最慢的那条流决定了该任务的完成时间，降低整体传输时间。介绍一个“新概念”的survey
+ 论文架构：
    + Introduction：先介绍Cluster computing applications，引出coflows调度问题，该问题的challenge。
    + cluster computing applications：分为三类
    + coflow model：characteristics/structure/objective
    + Information-aware coflow scheduling: centralized/distributed
    + Information-Agnostic coflow scheduling
    + optimize other objectives
    + challenges and broader perspective
    + conclusion
    
A Survey on Access Control in Fog Computing
+ 论文架构：
    + Introduction：雾计算的应用背景、雾计算的安全问题、访问控制的重要性
    + Fog computing overview
    + Access control In Fog computing：requirements/models
    + state-of-the-art Access control In Fog computing
    + challenges And Future research directions
    + conclusion
    
Survey of Bio-Inspired Resource Allocation Algorithms and MAC Protocol Design
+ 不是标准的survey，作者提出了一个新的协议设计，并且通过simulation做了performance evaluation



### Application of ML

Srijan Kumar, Francesca Spezzano, and V.S. Subrahmanian. 2015. VEWS: A Wikipedia Vandal Early Warning System. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’15). Association for Computing Machinery, New York, NY, USA, 607–616. DOI:https://doi.org/10.1145/2783258.2783367

+ 发现Wikipedia的恶意修改用户，提取修改特征，使用auto-encoder降低特征维度，使用简单机器学习分类器，baseline为ClueBot NG和STiki

Fei Wu, Xiao-Yuan Jing, Jun Zhou, Yimu Ji, Chao Lan, Qinghua Huang, and Ruchuan Wang. 2019. Semi-supervised Multi-view Individual and Sharable Feature Learning for Webpage Classification. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 3349–3355. DOI:https://doi.org/10.1145/3308558.3313492

+ 网页分类，网页上有各种图片、超链接、url等，因此成为multi-view，基于相似性来做分类，定义了了multi-view的相似性概念，从维持同类别中不同view、view共同特征的相似性角度来做分类，baseline是过去很多网页分类工作的方法和multi-view半监督特征学习方法

Jing X Y, Wu F, Dong X, et al. Semi-supervised multi-view correlation feature learning with application to webpage classification[C]//Thirty-First AAAI Conference on Artificial Intelligence. 2017.
+ 上文的一个较新的baseline，和上文工作非常相似，也是通过半监督的方法做网页分类，同类之间的相关性尽可能大，不同类尽可能小，论文本身的方法比较简单，难的是求解的过程，要讲非凸优化问题转化为凸优化问题。这两篇论文的共同特点是强大的数学基础。




### Reinforcement Learning & Network
