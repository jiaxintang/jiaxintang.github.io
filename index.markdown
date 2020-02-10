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

A Survey on Behavior Recognition Using WiFi Channel State Information
+ 通过WiFi进行室内行为识别
+ 论文架构：
    + background on traditional activity recognition systems
    + WiFi channel state information：介绍WiFi，人的活动对WiFi的影响
    + Wi-Fi csi-based behavior recognition：除了介绍别人的方法，自己提出了使用LSTM的方法，但是只写了一小段
    + evaluation of different methods
    + discussions
    + conclusion and future work

A Survey on Rapidly Deployable Solutions for Post-Disaster Networks
+ 灾后部署
+ 论文架构：
    + Introduction
    + Requirements
    + Wireless technologies
    + rapidly deployable networks
    + metropolitan Area Approach
    + local Area Approach
    + Disscussion

Approaches to End-User Applications Portability in the Cloud: A Survey
+ 论文架构：
    + Introduction：简单介绍三种云计算服务提供方式：SaaS、PaaS、IaaS，便携性要求应用可以在不同的PaaS之间迁移，与之前这类的survey的区别。
    + requirements for cloud end-user applications portability：云应用的life cycle，development、deployment、management的要求
    + work done within the standardization bodies：已有的一些云应用的标准
    + work done within research projects And academia：一些研究项目概述、评价
    + research directions：一些研究方法，已有的项目涉及了什么方向
    + conclusion

Network Slicing in 5G: Survey and Challenges
+ 第一篇关于5G的survey
+ 论文架构
    + Introduction：因为5G还没有明确的定义，描述了5G未来可能的样子，告知这是第一篇5G的survey
    + network slicing in the 5g architecture：一些应用场景和要求、目前已有的架构
    + state of the art in 5g network slicing：每一层都讲了scope、existing work、一些需要说明的点
        + Infrastructure layer
    	+ network function layer
    	+ service layer and MANO
    + challenges
    + summary

Information-Centric Networking for Connected Vehicles:
A Survey and Future Perspectives
+ 论文架构
	+ Introduction：车联网引入information-centric networking (ICN)
	+ Information-centric networking
	+ ICN for connected vehicles: motivations
	+ Conclusion

Hybrid Beamforming for Massive MIMO: A Survey
+ 这一篇有比较多的图表，大部分来自引用，但自己重新绘制了

Recent Research on Massive MIMO Propagation Channels: A Survey
+ 论文架构
	+ Introduction
	+ Key Issues In massive MIMO Propagation channels：特征、与传统的不同
	+ recent massive MIMO channel Measurements：设备、结果、挑战
	+ recent massive MIMO Propagation Modeling approaches
	+ future directions for massive MIMO channel Modeling
	+ conclusion

A Survey of Point-of-Interest Recommendation in Location-Based Social Networks
+ AAAI,15 workshop，按照主要影响的特征对方法进行分类

### Application of ML

Srijan Kumar, Francesca Spezzano, and V.S. Subrahmanian. 2015. VEWS: A Wikipedia Vandal Early Warning System. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’15). Association for Computing Machinery, New York, NY, USA, 607–616. DOI:https://doi.org/10.1145/2783258.2783367

+ 发现Wikipedia的恶意修改用户，提取修改特征，使用auto-encoder降低特征维度，使用简单机器学习分类器，baseline为ClueBot NG和STiki

Fei Wu, Xiao-Yuan Jing, Jun Zhou, Yimu Ji, Chao Lan, Qinghua Huang, and Ruchuan Wang. 2019. Semi-supervised Multi-view Individual and Sharable Feature Learning for Webpage Classification. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 3349–3355. DOI:https://doi.org/10.1145/3308558.3313492

+ 网页分类，网页上有各种图片、超链接、url等，因此成为multi-view，基于相似性来做分类，定义了了multi-view的相似性概念，从维持同类别中不同view、view共同特征的相似性角度来做分类，baseline是过去很多网页分类工作的方法和multi-view半监督特征学习方法

Jing X Y, Wu F, Dong X, et al. Semi-supervised multi-view correlation feature learning with application to webpage classification[C]//Thirty-First AAAI Conference on Artificial Intelligence. 2017.
+ 上文的一个较新的baseline，和上文工作非常相似，也是通过半监督的方法做网页分类，同类之间的相关性尽可能大，不同类尽可能小，论文本身的方法比较简单，难的是求解的过程，要讲非凸优化问题转化为凸优化问题。这两篇论文的共同特点是强大的数学基础。

Srijan Kumar, Xikun Zhang, and Jure Leskovec. 2019. Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 1269–1278. DOI:https://doi.org/10.1145/3292500.3330895
+ 计算user和item的embedding，这里的embedding包括动态和静态（one-hot）的embedding。两个操作：update：利用两个RNN来更新user和item的embedding，互相影响，互相输入；project：利用一个Attention层来对未来的行为进行预测。训练中突出了t-batch的方法来提高效率。
+ 论文优点：非常突出模型优势，文中不断将自己的方法与baseline进行对比，强调自己的创新点；实验丰富，对两个任务证明了模型的正确性，同时有runtime和embedding size的结果

Chenyang Wang, Min Zhang, Weizhi Ma, Yiqun Liu, and Shaoping Ma. 2019. Modeling Item-Specific Temporal Dynamics of Repeat Consumption for Recommender Systems. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 1977–1987. DOI:https://doi.org/10.1145/3308558.3313594
+ 研究了重复购买行为的特征，并且利用重复购买构建了一种推荐系统。传统推荐系统也倾向于用户会购买重复或相似的商品。重复购买行为有两个特征，一种是短期影响（因为好而再买），作者用指数分布刻画这种特征；另一种是生命周期特征（因为用完了而再买），用高斯分布刻画该特征（作者说单峰即可），重复购买行为是这两种特征的结合。作者提出的SLRC模型就是CF和重复购买结合，使用了9种Baseline，既有和别人的模型的对比，也有和自己提出的模型的一部分的对比
+ 论文值得借鉴的地方：Baseline很多，先分析再根据分析结果提出自己的模型，文中的一些部分实际也是作者根据现象直接给出的（比如为什么使用单峰高斯分布、指数分布等），并没有理论说明，但是通过写作圆过去了。

Enrico Mariconti, Jeremiah Onaolapo, Syed Sharique Ahmad, Nicolas Nikiforou, Manuel Egele, Nick Nikiforakis, and Gianluca Stringhini. 2017. What’s in a Name? Understanding Profile Name Reuse on Twitter. In Proceedings of the 26th International Conference on World Wide Web (WWW ’17). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE, 1161–1170. DOI:https://doi.org/10.1145/3038912.3052589
+ 这篇论文是一篇纯分析的论文，主要分析了重用名的影响，论文讲用户分为了高影响力和普通用户，又将普通用户分为了3类，作者的分析非常详细，包括数据集的局限性对分析可能造成的影响，以及为了减轻这些影响做了哪些操作。最后作者还调研了其他社交网络是否会存在类似的现象，使讨论的问题更具有普遍性。

Camille Cobb and Tadayoshi Kohno. 2017. How Public Is My Private Life? Privacy in Online Dating. In Proceedings of the 26th International Conference on World Wide Web (WWW ’17). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE, 1231–1240. DOI:https://doi.org/10.1145/3038912.3052592
+ 纯分析的论文，作者使用了survey和interview两种方法来研究网络约会的问题，针对约会网站中的隐私泄露相关问题进行了分析并提出了一些建议，作者的分析过程中不断使用采访的原文进行论文，分析的条理很清晰。

Dingqi Yang, Bingqing Qu, Jie Yang, and Philippe Cudre-Mauroux. 2019. Revisiting User Mobility and Social Relationships in LBSNs: A Hypergraph Embedding Approach. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 2147–2157. DOI:https://doi.org/10.1145/3308558.3313635
+ 数据集：从Foursquares上获取check-in，从Twitter上获取社交关系，作者先对影响因素进行了分析，在提出了LBSN2Vec这种Embedding方法，LBSN2Vec基于随机游走，除了在基于社交关系的用户图上游走，还可以停留在一个用户点上对时空语义特征进行游走，同时作者提出了基于余弦相似性的最优拟合方法。在实验部分，分朋友预测和地址预测两个任务和多种Baseline进行了详细对比。这篇文章值得我借鉴的点是对特征的分析，以及作者每提出一个观点都有充分的理由进行论证，如为什么使用余弦相似性而不适用点乘。

Yao H, Wu F, Ke J, et al. Deep multi-view spatial-temporal network for taxi demand prediction[C]//Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
+ 对出租车的需求量进行预测，使用local CNN处理空间特征，LSTM处理时间特征，构建图来描述区域之间功能相似性，并使用了LINE来计算每个点的Embedding，参与训练。这篇论文的模型和我们的模型有一定的相似性，可以参考学习它对模型的介绍。论文选择的baseline是一些简单的机器学习方法，这一点可能不太好，但是对与baseline比较的分析比较细致。

A semantic based Web page classification strategy using multi-layered domain ontology

Wang B, Zhang L, Gong N Z. SybilSCAR: Sybil detection in online social networks via local rule based propagation[C]//IEEE INFOCOM 2017-IEEE Conference on Computer Communications. IEEE, 2017: 1-9.
+ 论文提出了一种新的Sybil detection的方法，作者认为之前的方法分为两类，一类是基于随机游走的（如SybilRank，不能同时利用确定的正常用户和确定的恶意用户，对标注的噪音不具有鲁棒性）；一类是基于循环信念传播的（如SybilBelief，不scalable，因为要保存每条边的影响，不一定能收敛），作者提出了新方法综合了这两种方法，作者提出了local rule的概念，即后验概率由先验概率（theta，0.5，1-theta；实验中theta=0.9）和邻居影响共同决定，作者对于方法的收敛性做出了严格的理论证明，关于算法的效率，渐进复杂度是O(E)，但是在实验部分证明实际使用的空间、时间比SybilBelief少。
+ 论文优点：严格的理论证明，对之前工作的整理分析
+ 疑问：实验机器内存16G，最大的一个数据集，21,297,772 nodes，265,025,545 edges，如果算法使用文中所示伪代码，使用邻接矩阵进行存储，至少需要422443G；如果使用边际数组进行存储，大约需要2G，但是实现会非常复杂，且根据作者所言，是用C++实现的。在Github找到了代码。

Zongtao Liu, Yang Yang, Wei Huang, Zhongyi Tang, Ning Li, and Fei Wu. 2019. How Do Your Neighbors Disclose Your Information: Social-Aware Time Series Imputation. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 1164–1174. DOI:https://doi.org/10.1145/3308558.3313714
+ 对缺失数据的预测，作者使用社交影响和时间序列预测共同推测多维时间序列中缺失的值，主要使用了TLSTM（LSTM的变体，利用衰减，考虑了时间序列之间的时间间隔）和注意力机制，但这两者都不是作者提出的，baseline较多，也有自身模型各部分的分析。
+ 值得借鉴：也没有自己提出的模块，但是论文对模型的分析非常具体，也说明了自己的创新点，即应用和模型的第一次结合。

Chuxu Zhang, Dongjin Song, Chao Huang, Ananthram Swami, and Nitesh V. Chawla. 2019. Heterogeneous Graph Neural Network. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 793–803. DOI:https://doi.org/10.1145/3292500.3330961
+ 异构图神经网络，已有的图神经网络大多是基于同质的图，即每个点的类型是一样的，比如都是作者；作者针对的是每个点的类型不同的图，同时每个点本身的特征也有多种类型。方法如下：先使用random walk来选出和一个点有关的点，并根据类型进行分类；使用两个神经网络分别来结合不同类型的特征和不同类型的点，用了bilstm来结合不同特征是论文的一个亮点。实验部分，选择了link prediction、Recommendation、Classification and Clustering和Inductive Classification and Clustering任务，5种graph Embedding的baseline，并对方法本身进行了超参数分析和各部分的分析。

Liu Q, Wu S, Wang L, et al. Predicting the next location: A recurrent model with spatial and temporal contexts[C]//Thirtieth AAAI conference on artificial intelligence. 2016.
+ 位置预测，使用了改进的RNN模型，引入了时间间隔和地理位置距离的因素，pairwize ranking作为目标优化，处理冷启动问题，作者只考虑了将时间间隔和地理距离在一定范围内的数据，且使用线性调整到一定范围。实验中与多种baseline进行对比，并对参数选择和收敛性进行了分析。

Wenyi Xiao, Huan Zhao, Haojie Pan, Yangqiu Song, Vincent W. Zheng, and Qiang Yang. 2019. Beyond Personalization: Social Content Recommendation for Creator Equality and Consumer Satisfaction. In Proc. KDD 19. Association for Computing Machinery, New York, NY, USA, 235–245. DOI:https://doi.org/10.1145/3292500.3330965
+ 推荐算法，向用户推荐文章，评价时不仅考虑到了消费者的满意度，还使用了Gini系数考虑到了创作者的均衡性，尽可能向用户推荐丰富的文章，模型中使用CNN处理词级别，GRU处理句级别，使用attention结合社交信息，使用蒙特卡洛树选择高影响力的朋友（使用了多种标准并进行了比较）。评价时使用多种baseline，也做了参数分析和切割分析。

### Reinforcement Learning & Network
Lynn, T., Hanford, N., & Ghosal, D. Impact of Buffer Size on a Congestion Control Algorithm Based on Model Predictive Control.
+ 关于Buffer size的一篇论文，作者设计并实现了一种拥塞控制协议MPC，亮点是对未来的瓶颈速率进行了预测。作者对buffer size分析的结论如下：（1）Buffer size很大（BDP的很多倍）时，丢包会很高；（2）Buffer size很低时，吞吐量和RTT会很稳定；（3）Buffer size很大或很小时都会使得吞吐量和瓶颈链路不相符；（4）适当增加buffer size会使得更容易达到瓶颈链路。总之，作者认为buffer size应该设为1/4 BDP比较合理，但是当有很多短流时，可以更大一点。

Peng B, Li X, Gao J, et al. Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning[C]//Proceedings of the 56th ACL. 2018: 2182-2192.
+ 这篇论文将Dyna-Q应用在了对话的场景中，Dyna是一种model-based RL框架，包括一个类似DQN的模型来反应真实环境中针对每个action的reward和next state以及一个world model来模拟环境，通过nn给出action的reward和next，最终的模型训练都基于这两种环境一起对策略网络进行训练。

Racanière S, Weber T, Reichert D, et al. Imagination-augmented agents for deep reinforcement learning[C]//NIPS. 2017: 5690-5701.
+ 这篇论文提出一种model-based的模型I2A，环境模型基于现在的环境和现在的action对未来的环境和可能的奖励进行预测，策略网络会同时基于环境模型的输出和真实观察做出决策，作者没有像上一篇论文一样给出清晰的伪代码，因为细节理解不太清楚，需要进一步看源代码。

Feinberg V, Wan A, Stoica I, et al. Model-based value estimation for efficient model-free reinforcement learning[J]. ICML, 2018.
+ 一种新的把环境模型引入model-free的方法，作者把环境模型先展开一定步数之后再进行Q值预估。也就是说，在传统的更新方式中，target Q值是下一步Q值的预估，而在这里，target Q值是先通过环境模型进行模拟一段路径之后，再进行Q值预估。这样Q值的预估就融合了基于环境模型的短期预估以及基于target_Q网络的长期预估。文中对过去Model-based RL的总结很好，分为三类：（1）直接将动态系统融合进值梯度（在进行Q值或者V值预估时，environment model 和agent做交互，交互过程中的信息作为context提供给agent来帮助其决策）；（2）将想象作为新的额外的训练数据；（3）将想象的context用来做值估计。
