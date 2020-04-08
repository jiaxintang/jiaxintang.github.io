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

Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. 2019. Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks. In Proc. KDD 19. Association for Computing Machinery, New York, NY, USA, 257–266. DOI:https://doi.org/10.1145/3292500.3330925
+ 对GCN的效率进行优化，传统的多层GCN无论是训练效率还是内存占用都很大，因此有使用mini-batch的方法，但是复杂度依然和层数成指数关系，文章定义了Embedding利用率的概念，对图进行聚类，减少batch之间的边数，提高Embedding利用率，但由于聚类会使得图中一些类与类之间的边被忽略，同时改变分布，因此每一个batch选几类进行训练。文中作者强调了之前此类工作的缺点，同时也说明了提出的方法可能存在的问题，以及可做的优化，实验部分对多个数据集、多层的网络进行了对比，非常详细。

Haoji Hu and Xiangnan He. 2019. Sets2Sets: Learning from Sequential Sets with Neural Networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 1491–1499. DOI:https://doi.org/10.1145/3292500.3330979
+ 时间序列集到时间序列集的预测，用一些用户过去的时间序列数据来预测未来一段时间的数据，作者定义了这一类问题，并提出了基于encoder-decoder框架的模型，使用了特殊的Embedding，并利用了attention机制。作者考虑到了过去出现过的商品更有可能未来被使用，因此对最后一层网络进行了调整。同时loss function考虑到了label数量不平衡的问题。在实验部分，作者列出了要分析的问题，使文章逻辑很清晰；作者在性能和其他方法的对比的分析上，写得非常详细，并说明了可能的理由。作者也对自身模型的一些变体做了研究。

Kai Shu, Limeng Cui, Suhang Wang, Dongwon Lee, and Huan Liu. 2019. DEFEND: Explainable Fake News Detection. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 395–405. DOI:https://doi.org/10.1145/3292500.3330935
+ 假新闻探测，通过contents和comments进行分类，考虑了词和句层次的特征，使用了双向GRU分别建模，并使用attention机制连接不同特征，论文重点强调了可解释性，我觉得这部分是整个论文的亮点，作者通过attention的系数决定重要性排序，评论部分明确提出了三个问题，并分别与一些Baseline进行对比，并使用case study直观展示。

Namyong Park, Andrey Kan, Xin Luna Dong, Tong Zhao, and Christos Faloutsos. 2019. Estimating Node Importance in Knowledge Graphs Using Graph Neural Networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 596–606. DOI:https://doi.org/10.1145/3292500.3330855
+ 知识图谱中的点重要性估计，使用了监督学习的方法对重要性进行拟合，与传统PageRank一类的方法不同，文中使用的数据集都包含可以代表重要性的特征，结果现在在同一领域使用监督学习的方法比不用训练的方法好，文中使用的方法为基于Attention的神经网络，使用attention对邻居节点的分数进行汇总，对于谓语（两点关系）也使用attention，中心性使用度来衡量，但由于中心性的影响不大，使用了一个参数来调整该属性的大小。论文介绍模型时，根据功能介绍了每个部分使用的模型，再汇总介绍。实验部分比较了计算重要性的一些传统方法和监督学习的一些常用方法，个人觉得实验部分不如之前读的几篇论文写得好。

Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and Tat-Seng Chua. 2019. KGAT: Knowledge Graph Attention Network for Recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 950–958. DOI:https://doi.org/10.1145/3292500.3330989
+ 利用知识图谱辅助推荐。知识图谱中的高阶特征对推荐有很大帮助，比如两个item和同一个entity有不同性质的关联。作者提出的模型首先使用Embedding层来计算Embedding，该层使用了知识图谱中常用的TransR方法；再使用attention层，这一层是自网络和邻居影响的结合，作者使用了3种结合的方法并在实验部分对比；最终的loss是这两层的loss之和。在实验部分，提出并回答了3个问题：与baseline对比，自身各部分的性能（对各部分替换不同的方法，对性能进行对比），可解释性（case study），使文章更有条理。

Xiaoli Tang, Tengyun Wang, Haizhi Yang, and Hengjie Song. 2019. AKUPM: Attention-Enhanced Knowledge-Aware User Preference Model for Recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 1891–1899. DOI:https://doi.org/10.1145/3292500.3330705
+ 同样是利用知识图谱来辅助推荐，作者以click-through rate (CTR)推荐为例，模型主要分为intra-entity和inter-entity两部分，其中intra-entity部分主要刻画entity自身属性，也使用了TransR方法，inter-entity部分刻画entity之间的关系，使用了self-attention，最终user和item的时间使用了矩阵内积，这篇论文更理论化，使用了很多公式来描述过程，包括loss和train的过程，实验部分在两个数据集上做了测试，与多种推荐方法对比，也有自身的变体比较，说明两部分的必要性。

Xiao Zhou, Cecilia Mascolo, and Zhongxiang Zhao. 2019. Topic-Enhanced Memory Networks for Personalised Point-of-Interest Recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 3018–3028. DOI:https://doi.org/10.1145/3292500.3330781
+ POI推荐，考虑了时序特征和地理特征，使用memory network来建模check-in数据（使用了attention），TLDA模型来刻画check-in序列的时序特征；为了描述地理特征，作者使用了用户位置影响、POI位置影响和距离影响三个变量。实验部分使用了attention的结果来对比不同模式。

Songgaojun Deng, Huzefa Rangwala, and Yue Ning. 2019. Learning Dynamic Context Graphs for Predicting Social Events. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 1007–1016. DOI:https://doi.org/10.1145/3292500.3330919
+ 预测新闻事件，实际是一篇NLP的论文，通过分析新闻的文本来预测事件，提出了动态的GCN模型，作者筛选一些重要单词作为点，出现的文章频率作为边，构建图，因为每天的新闻是不同的，所以图每天都是变化的，利用动态GCN模型和时间特征来预测未来事件，baseline除了LR只有一种之前的事件预测模型，此外还使用了一些提出的模型的变体，除了对预测性能做对比，还给出了一些case study和图的可视化。

Drakonakis K, Ilia P, Ioannidis S, et al. Please Forget Where I Was Last Summer: The Privacy Risks of Public Location (Meta) Data[J].NDSS 2019
+ The paper proposed LPAuditor which conducts a comprehensive evaluation of the privacy loss caused by public location metadata. 
+ Data Collection and Pre-processing
	+ Data Collection: Twitter's stream API for collecting tweets
	+ Dataset: users (the mainland area of USA) with at least one tweet containing GPS coordinates, 87114 users, 15263317 tweets
	+ Ground Truth: manual process, 2047 users
	+ Data Labeling: GPS coordinates to Postal address (reverse geocoding API by ArcGIS, Google Maps Geocoding API)
		Initial clustering: Tweets with the same postal address are grouped into a single clustering
		Second-level clustering: Using DBSCAN to group neighboring clusters into a larger one
		Clustering results: power law distribution
+ Identifying Key User Locations
	+ heuristics
	+ Home
		the cluster with the broadest time frame in the five most active clusters
		Evaluation: Outperform previous approaches
	+ Work
		the cluster with the largest number of active tweets in the dominant time frame
		Evaluation: Not very effective (Possible reason: The user can exhibit similar characteristics in other locations.)
	+ Selection bias(Ground truth):
		They selected 100 users randomly from the dataset and manually investigate their tweets.
+ Identifying Highly-Sensitive Places	(health/religion/sex)
	+ Foursquare API returns the name of each venue as well as its type.
	+ Content-based corroboration
		manually-curated wordlist
	+ Duration-based corroboration
		consecutive tweets in the span of a few hours
+ Impact of Historical Data
	+ A bug of Twitter before April 2015: The tweets with coarse-grained labels have coordinates in their metadata.
	+ Historical data lead to unavoidable privacy leakage.

Guolin Ke, Zhenhui Xu, Jia Zhang, Jiang Bian, and Tie-Yan Liu. 2019. DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 384–394. DOI:https://doi.org/10.1145/3292500.3330858
+ online预测，方法是将GBDT与NN结合，也就是用NN来模拟GBDT的树结构。GBDT的优势在于处理高密度的数值特征，但是对于新增的数据需要重新训练模型树，效率很低；NN的优势是处理比较稀疏的类别特征，同时可以做增量的训练。作者使用NN来模拟树的操作，即模拟决策树的特征，对于处理多颗树的问题，使用了分组和leaf embedding的做法提高效率。训练时，leaf embedding是利用离线模型先训练的，之后就是end-to-end的模型，增量训练时只使用了end-to-end的loss。模型评价使用了多个常用数据库来测试分类和回归，并与一些现有的预测方法做对比，分为offline和online部分，实验证明提出的模型收敛速度快，准确性高。

Hulsebos M, Hu K, Bakker M, et al. Sherlock: A deep learning approach to semantic data type detection[C]//Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019: 1500-1508.
+ 对列数据进行语义分类，这篇文章看起来中规中矩，模型很简单，就是简单的全连接层，特征提取也很简单，字符、单词、句子、全局四个层面，对比的baseline就是基本的ml模型、正则表达式和字典方法，文章结构清晰，数据获取、特征提取、模型介绍、评价（与baseline对比，特征重要性分析等），看起来没什么亮点，但是每一部分表达都比较清晰，之前也的确没有人用deep learning来解决该问题。

Xin Wang, Wenwu Zhu, and Chenghao Liu. 2019. Social Recommendation with Optimal Limited Attention. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 1518–1527. DOI:https://doi.org/10.1145/3292500.3330939
+ 社交推荐，作者引入了社会科学中有限注意的问题，认为只有一部分的朋友会影响用户行为，且这一部分的朋友对用户的影响也是不同的，但是作者没有直接使用神经网络中的attention模型，而是引入了权重后，结合MF对整个优化问题进行了严格的数学推导和证明，并使用EM的思想来做优化，与之前KDD上见到的文章不同，这篇论文非常理论，文章的亮点是严格的数学功底，评论部分使用了多种评论指标证明优越性，同时考虑到了冷启动问题。

Hanwen Zha, Wenhu Chen, Keqian Li, and Xifeng Yan. 2019. Mining Algorithm Roadmap in Scientific Publications. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 1083–1092. DOI:https://doi.org/10.1145/3292500.3330913
+ 使用NLP的方法来整理论文，绘制算法的演化图，作者使用正则表达式的方法从论文中提取简称（如CNN、GAN）来代表算法，使用PCNN、Transformer和BERT来提取论文中不同简称之间的关系，这里的关系主要是比较关系，为了获取训练数据，作者认为在表格中的一列或一行算法可以认为是具有比较关系的，最终确定演化关系时，作者认为发表年份和出现频率可以用来确定演化边的方向。因为不同领域的简称可能有重复，作者还同时训练了实体类别。在评价时，数据集爬取了NeurIPS、ACL和VLDB的论文数据，数量既都为几k篇论文，因为问题比较新颖，比较时使用了一些简单的方法，如Word similarity等，同时也与自身模型变体做了比较，还分析了case study。

Quanyu Dai, Xiao Shen, Liang Zhang, Qiang Li, and Dan Wang. 2019. Adversarial Training Methods for Network Embedding. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 329–339. DOI:https://doi.org/10.1145/3308558.3313445
+ 使用GAN来优化node embedding，论文以DeepWalk为例，使用GAN来训练正则化项，避免过拟合，根据作者所言固定的正则化项是不合理的，使用GAN做这种优化已经被证明很有效，他将其运用在了DeepWalk上，其中基于random walk的sampling和negative sampling都不变，只是增加了使用adv正则化项，同时两点的similarity越大，该项越小。在这种方法的基础上，作者还提出了可解释性更强的优化模型，即adv正则化只当做稀疏，同时使用另一个点作为方向，使得该项具有可解释性。实验部分与现有的node embedding方法做了比较，在训练过程中效果一直非常好，且对超参数不敏感。该优化还可以运用于除了Deepwalk以外的其他方法上，如LINE。

Min-hwan Oh and Garud Iyengar. 2019. Sequential Anomaly Detection using Inverse Reinforcement Learning. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 1480–1490. DOI:https://doi.org/10.1145/3292500.3330932
+ 根据序列数据进行异常检测，使用了逆向强化学习（IRL），即没有给定的Reward方程，而是试图找出符合最优策略demonstration行为的Reward方程（一个神经网络）。最大熵IRL：demonstrations服从Boltzmann分布，demonstrator的偏好呈指数偏向高reward的轨迹，最终的目标是最大化给定轨迹的可能性。Bayesian框架，Reward有一个先验分布，根据demonstration计算后验分布。这篇论文模型非常复杂，但是基本是使用了已有的东西，但是第一次运用在异常检测上，且效果很好。实验部分进行了可视化，对比的baseline是很简单的机器学习。

Zheyi Pan, Yuxuan Liang, Weifeng Wang, Yong Yu, Yu Zheng, and Junbo Zhang. 2019. Urban Traffic Prediction from Spatio-Temporal Data Using Deep Meta Learning. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19). Association for Computing Machinery, New York, NY, USA, 1720–1730. DOI:https://doi.org/10.1145/3292500.3330884
+ 交通预测，作者认为目前交通预测的挑战在于时间和空间的复杂联系，作者首先使用RNN（GRU）来提取各个点各个时间的时序特征（encoder），接着FCN来学习点和边的特征（NMK和EMK），输入进GAT获取不同边的weight同时获得点的embedding，最终使用RNN来结合NMK和点embedding来获得最终的预测结果（decoder）。实验部分预测了taxi的流和速度，在baseline中既有state-of-the-art的DCRNN，也有算法变体GAT-Seq2Seq，同时做了参数敏感性测试。这篇论文的内容比较简单，设计的模型也是多种已有模型的拼接，但是每一部分的用途、计算就写得很详细，实验部分比较也很清晰。

Alessandro Epasto and Bryan Perozzi. 2019. Is a Single Embedding Enough? Learning Node Representations that Capture Multiple Social Contexts. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 394–404. DOI:https://doi.org/10.1145/3308558.3313660
+ node embedding，作者认为如果把一个点对应到一个embedding不能反映点的全部信息，他认为基于不同角色，一个点应当对应于多个embedding；因为他们引入了重叠聚类，即一个点属于多个集合，在每个集合中都有一个embedding，作者对已有的Persona Decomposition重叠聚类方法提出了改进，但是没有对具体的embedding方法改进。在应用方面，分别作了link prediction和可视化，其中可视化这一应用给了读者更直观的感受，也是之前node embedding中不常见的。这篇论文的亮点就在于idea很新，对重叠分类和node embedding进行结合，其方法和实验并不难。

Chao Huang, Chuxu Zhang, Jiashu Zhao, Xian Wu, Dawei Yin, and Nitesh Chawla. 2019. MiST: A Multiview and Multimodal Spatial-Temporal Learning Framework for Citywide Abnormal Event Forecasting. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 717–728. DOI:https://doi.org/10.1145/3308558.3313730
+ 预测城市异常事件，作者首先分析了挑战，包括区域内部的时间特征、区域之间的空间特征以及其他影响因素，模型分为三部分，RNN提取区域内部特征，attention结合不同的区域特征，最后RNN来总结最终结果。这篇文章很值得我借鉴，因为就模型本身来说，并不新颖，但是作者的介绍非常详细，而且在介绍模型时由于融入了一些与应用场景相结合的概念，因此不显得是完全套用已有模型。在评价部分，根据提出的问题，分析与baselines的性能对比，已经component-wise的性能，对超参数也进行了分析，简单的介绍了一个case study。baseline挑选的比较多，但很多也原本也不是用来解决该问题的，只是也是处理时空特征。

Jie Feng, Mingyang Zhang, Huandong Wang, Zeyu Yang, Chao Zhang, Yong Li, and Depeng Jin. 2019. DPLink: User Identity Linkage via Deep Neural Network From Heterogeneous Mobility Data. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 459–469. DOI:https://doi.org/10.1145/3308558.3313424
+ 不同数据集中的用户对齐， 针对的是基于位置的应用，数据集是ISP和Foursquare-twitter，两者的质量不同，前者较高，作者先通过RNN提取轨迹特征和位置特征，通过attention来选择两个轨迹之间重要的部分，最终通过FC分类，模型本身比较简单，但是效果并不好，作者认为是由于数据质量的问题，即两个数据集差距较大，因此作者引入了transfer learning的概念，先预训练一个任务，即同一个数据集中，不同时间的两个轨迹是否属于同一个用户，这样数据质量比较一致，效果也较好，再使用部分训练好的参数来完成最终的任务。这篇文章的另一个优点是图片很清晰漂亮。

Kang-Min Kim, Yeachan Kim, Jungho Lee, Ji-Min Lee, and SangKeun Lee. 2019. From Small-scale to Large-scale Text Classification. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 853–862. DOI:https://doi.org/10.1145/3308558.3313563
+ NLP，文本分类，由于做大规模分类时，类别较多，每一类的数据量不足也不均匀，因此作者提出了多任务的模型，即同时进行小规模和大规模的分类，作者使用了Text-CNN模型，CNN输出两种层，一种是私有特征，一种是公有特征，其中小规模任务由其私有特征和公有特征决定，大规模任务由其私有特征和公有特征、以及经过了一个gate的小规模任务的私有特征决定。一些可以参考的细节：pre-training的四种方法、zero-padding、over-sampling（对不均匀数据重复取样）

Desheng Hu, Shan Jiang, Ronald E. Robertson, and Christo Wilson. 2019. Auditing the Partisanship of Google Search Snippets. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 693–704. DOI:https://doi.org/10.1145/3308558.3313654
+ 分析google搜索索引简介的政治倾向，发现有扩大政治倾向的现象，作者认为这样的现象不是故意的。文章使用了一些简单的NLP方法，首先从一些演讲中提取政治词汇，结果计算索引和原文的政治倾向分数。文章以分析为主，并没有提出算法，分析分类非常清晰，如位置、派系、主题等。

Miriam Redi, Besnik Fetahu, Jonathan Morgan, and Dario Taraborelli. 2019. Citation Needed: A Taxonomy and Algorithmic Assessment of Wikipedia’s Verifiability. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 1567–1578. DOI:https://doi.org/10.1145/3308558.3313618
+ 这篇文章很值得借鉴，它提出了wikipedia上的引用问题，以及引用原因，文章花了很大笔墨介绍数据标注过程，分为expert和non-expert两部分，作者详细介绍了为什么要分两类人标注，这样做的可行性，两类人标注结果是否有相似性，另一方面，作者也提出了NN模型来预测是否需要引用，模型很简单，对比的baseline是手动提取特征的模型。文章有很多的可视化。

Fabrizio Lillo and Salvatore Ruggieri. 2019. Estimating the Total Volume of Queries to Google. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 1051–1060. DOI:https://doi.org/10.1145/3308558.3313535
+ 估计Google的查询量，纯数学的论文，作者认为不同查询的查询量符合zipf分布，论文将这个问题分解为两个子问题，第一个是怎样获取数据，使用了Google Trend获得相对查询数据，结合SEO工具（比较贵，不能做大量查询）的部分绝对数据量找到相对和绝对之间的关系，得到其他查询的绝对量；第二个子问题是怎样通过一些子查询来获取某领域内所有查询的总量，这部分主要介绍了怎样使用采样方法估计该分布的各个参数。最后作者以一类查询（烹饪）为例概括了计算方法。这篇论文数学功底非常强，每个部分都比较了多种方法，如怎么样采样、怎样求参数等，并给出了严格的数学推导。

Yang Z, Yang D, Dyer C, et al. Hierarchical attention networks for document classification[C]//Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016: 1480-1489.
+ 文本分类，使用了两级Encoder+Attention结构，其中Encoder使用了双向的GRU，Attention选择了self-attention，选择的baseline是CNN，LSTM，Conv-GRNN等，对多个数据集进行了测试，本文值得借鉴的一点是其对attention部分的测量和可视化，通过good和bad的attention weight展示证明了模型的有效性。

Shangsong Liang, Xiangliang Zhang, Zhaochun Ren, and Evangelos Kanoulas. 2018. Dynamic Embeddings for User Profiling in Twitter. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’18). Association for Computing Machinery, New York, NY, USA, 1764–1773. DOI:https://doi.org/10.1145/3219819.3220043
+ 改进了SkipGram，获取Twitter用户和word的动态embedding，因为方法部分太复杂，主要看了评价部分，评价部分主要讨论了该方法获取的Embedding与baseline相比的优势，一些变量（时间片、Embedding维度）对模型的影响、可解释性，数据集的Ground Truth使用了人工标注的结果，分别为Relevance-oriented (RGT) performance 和 Diversity-oriented (DGT) performance.这篇论文的数学性很强，但是可读性比较差，即使是评价部分也以公式为主，描述性的语言很少，可借鉴性不强。

Yujing Hu, Qing Da, Anxiang Zeng, Yang Yu, and Yinghui Xu. 2018. Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’18). Association for Computing Machinery, New York, NY, USA, 368–377. DOI:https://doi.org/10.1145/3219819.3219846
+ 将强化学习运用在搜索排序算法，淘宝的工作，作者认为用户从开始搜索到买某个商品或者退出淘宝是一个完整的session，其中每一次的搜索结果应该是相关的，因此可以使用强化学习来训练这个过程，论文先证明了这个过程是马尔科夫过程，为强化学习的使用提供给了理论依据，使用的reward function类似游戏，只有买的那一次reward是交易价格，其他的reward都是0，在算法上改进了dpg算法，对Q值的loss计算进行了改进。实验部分既有使用用户行为模拟器来进行模拟实验，也有真实A/B test，证明了算法的有效性。

Rahul Bhagat, Srevatsan Muralidharan, Alex Lobzhanidze, and Shankar Vishwanath. 2018. Buy It Again: Modeling Repeat Purchase Recommendations. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’18). Association for Computing Machinery, New York, NY, USA, 62–70. DOI:https://doi.org/10.1145/3219819.3219891
+ 预测重复购买的行为，亚马逊的工作，作者认为之前没有针对该问题的具体方法，文中提出了4种模型来解决这个问题，在评论部分也是就这4中方法进行了对比，其中前三种RCP（重复购买历史排序）、ATD（正态分布）、PG（Possion-Gamma分布）都是已有方法的应用，MPG针对PG方法进行了一些更贴合场景的改进，作者使用了一些具体的例子来比较这些方法。评价部分分为两部分，offline和online，online部分也是使用了A/B test，用不同的方法分别运行14天，每天对模型进行一次update。

Supreeth P. Shashikumar, Amit J. Shah, Gari D. Clifford, and Shamim Nemati. 2018. Detection of Paroxysmal Atrial Fibrillation using Attention-based Bidirectional Recurrent Neural Networks. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’18). Association for Computing Machinery, New York, NY, USA, 715–723. DOI:https://doi.org/10.1145/3219819.3219912
+ 心颤检测，生物信息学，神经网络结构比较简单，CNN提取特征、RNN提取序列特征、最后通过Attention，并加入了额外的covariant特征，在模型介绍时结合问题本身的特点介绍模型每个模块，评价部分只有一种baseline，比较少，通过t-SNE聚类做了特征可视化，亮点是使用了迁移学习证明了模型可以被迁移到别的数据集上。本文在技术上和写作上亮点都不太突出，但是结构比较完整，可能在18年attention模型的应用也比较少，再加上迁移学习的应用使它能中KDD。


### Reinforcement Learning & Network
Lynn, T., Hanford, N., & Ghosal, D. Impact of Buffer Size on a Congestion Control Algorithm Based on Model Predictive Control.
+ 关于Buffer size的一篇论文，作者设计并实现了一种拥塞控制协议MPC，亮点是对未来的瓶颈速率进行了预测。作者对buffer size分析的结论如下：（1）Buffer size很大（BDP的很多倍）时，丢包会很高；（2）Buffer size很低时，吞吐量和RTT会很稳定；（3）Buffer size很大或很小时都会使得吞吐量和瓶颈链路不相符；（4）适当增加buffer size会使得更容易达到瓶颈链路。总之，作者认为buffer size应该设为1/4 BDP比较合理，但是当有很多短流时，可以更大一点。

Peng B, Li X, Gao J, et al. Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning[C]//Proceedings of the 56th ACL. 2018: 2182-2192.
+ 这篇论文将Dyna-Q应用在了对话的场景中，Dyna是一种model-based RL框架，包括一个类似DQN的模型来反应真实环境中针对每个action的reward和next state以及一个world model来模拟环境，通过nn给出action的reward和next，最终的模型训练都基于这两种环境一起对策略网络进行训练。

Racanière S, Weber T, Reichert D, et al. Imagination-augmented agents for deep reinforcement learning[C]//NIPS. 2017: 5690-5701.
+ 这篇论文提出一种model-based的模型I2A，环境模型基于现在的环境和现在的action对未来的环境和可能的奖励进行预测，策略网络会同时基于环境模型的输出和真实观察做出决策，作者没有像上一篇论文一样给出清晰的伪代码，因为细节理解不太清楚，需要进一步看源代码。

Feinberg V, Wan A, Stoica I, et al. Model-based value estimation for efficient model-free reinforcement learning[J]. ICML, 2018.
+ 一种新的把环境模型引入model-free的方法，作者把环境模型先展开一定步数之后再进行Q值预估。也就是说，在传统的更新方式中，target Q值是下一步Q值的预估，而在这里，target Q值是先通过环境模型进行模拟一段路径之后，再进行Q值预估。这样Q值的预估就融合了基于环境模型的短期预估以及基于target_Q网络的长期预估。文中对过去Model-based RL的总结很好，分为三类：（1）直接将动态系统融合进值梯度（在进行Q值或者V值预估时，environment model 和agent做交互，交互过程中的信息作为context提供给agent来帮助其决策）；（2）将想象作为新的额外的训练数据；（3）将想象的context用来做值估计。


A modern AQM is just one piece of the solution to bufferbloat.
+ CoDel，论文首先解释了Bufferbloat，描述了当带宽发生变化时的包的情况，以及在没有突发流量的情况下buffer中队列为什么不为0；同时论文对之前的AQM的缺点进行了总结：参数难配置、不能区分好排队和坏排队、对链路带宽、流量、rtt敏感、不能随链路带宽动态变化、不够简单高效，这些同时就是CoDel的优点。CoDel通过记录一段interval（通过瓶颈的最坏情况RTT的时间）中的最低排队时延来控制丢包，当该值小于target时不丢包，当大于时丢包直到buffer中的包数量小于MTU对应的应有的字节数。这种方法实现简单，不需要锁，作者在ns2中做了模拟，分为静态网络和动态网络特征，分别与RED和Drop-tail做了对比，结果显示时延较低，利用率较高，公平性较高。

Machine Learning for Networking: Workflow, Advances and Opportunities
+ 机器学习与网络结合的总结，工作流：问题阐述、数据收集、数据分析、模型构建、模型验证、部署推论；作者按功能对MLN进行了划分：
	+ 信息认知：监督学习，预测很难直接测量的数据
	+ 流量预测：监督学习，HMM模型
	+ 流量分类：监督学习、无监督学习，不同类型的流量适合不同类型的应用
	+ 资源配置：强化学习
	+ 网络适配：路由策略（监督学习、深度信念网络）、TCP拥塞控制（强化学习）
	+ 性能预测：监督学习，HMM
	+ 配置推断：监督学习
+ 可行性问题：对延时敏感，数据量、标注数据

Dong M, Li Q, Zarchy D, et al. {PCC}: Re-architecting congestion control for consistent high performance[C]//12th {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 15). 2015: 395-408.
+ TCP会根据丢包来降低发送速度，在很多情况下丢包并不是因为拥塞，作者提出在发送端尝试增加和降低发送速率，一个RTT后根据utility function来决定是否按此高速率还是低速率发送，严格来说并不是RL，因为只关注了现在的utility，没有考虑影响，但是好处是不需要训练，可以直接使用，作者证明了其公平性（多发送端速率最终会趋于一致）和收敛性，同时作者通过实验证明了该方法可以是TCP友好的，可以在同一网络中一起使用。

Hongzi Mao, Malte Schwarzkopf, Shaileshh Bojja Venkatakrishnan, Zili Meng, and Mohammad Alizadeh. 2019. Learning scheduling algorithms for data processing clusters. In Proceedings of the ACM Special Interest Group on Data Communication (SIGCOMM ’19). Association for Computing Machinery, New York, NY, USA, 270–288. DOI:https://doi.org/10.1145/3341302.3342080
+ 使用GNN和PG来实现任务调度，使用GNN来获取DAG的各个点和每个DAG的Embedding，文中的两个技巧是：初期策略效果很差，所以前一些epoch可以提早结束；考虑到job到来的随机性，作者对同一组实验进行N次，取N次的平均值作为baseline（疑问：因为使用了PG，所以随机性较大，但是如果随机性不够，这N次的结果应该是一样的？）

Hongzi Mao, Mohammad Alizadeh, Ishai Menache, and Srikanth Kandula. 2016. Resource Management with Deep Reinforcement Learning. In Proceedings of the 15th ACM Workshop on Hot Topics in Networks (HotNets ’16). Association for Computing Machinery, New York, NY, USA, 50–56. DOI:https://doi.org/10.1145/3005745.3005750
+ 使用PG来实现任务调度，是上一篇的一个简化版，只考虑简单的资源分配，即每个任务到达时可知其需要的各项资源的多少，然后根据现有资源对安排执行那些任务，state为现有资源分配情况，已到达等待执行的每个任务需要的资源，action为之后一段timestep运行那些任务，reward为平均等待时间。baseline和上一篇论文一样，都是N次实验，每个时间t取平均作为baseline。
作者总结的挑战：1、系统复杂，很难建模；2、实际环境下的噪音；3、评价指标难以优化

Hongzi Mao, Ravi Netravali, and Mohammad Alizadeh. 2017. Neural Adaptive Video Streaming with Pensieve. In Proceedings of the Conference of the ACM Special Interest Group on Data Communication (SIGCOMM ’17). Association for Computing Machinery, New York, NY, USA, 197–210. DOI:https://doi.org/10.1145/3098822.3098843
+ 视频的比特率确定，使用了A3C的方法，作者说明了传统方法的局限性：固定策略不能考虑网络吞吐量的变化（预测不准）、视频QoE的要求矛盾（高比特率，最小化再缓冲）、比特率决策的级联效应、ABR决策的粗粒度，并通过具体的例子证明了这些缺陷。作者设计了模拟器来使得不用使用真实环境训练，节省了时间，并且实际训练时
使用了多个模拟器并行，A3C中使用critic网络来预测q值（qoe方程具体实验时尝试了多种不同的设计），advantage来衡量与平均reward的差，同时使用entropy来使得在训练初期尽可能尝试不同的action。评价比较了不同的baseline，不同的rl算法，以及在实际环境下的效果来评价模型是否具有generalization。

Daniel S. Berger. 2018. Towards Lightweight and Robust Machine Learning for CDN Caching. In Proceedings of the 17th ACM Workshop on Hot Topics in Networks (HotNets ’18). Association for Computing Machinery, New York, NY, USA, 134–140. DOI:https://doi.org/10.1145/3286062.3286082
+ 利用监督学习来做CDN caching，由于CDN中的状态信息非常复杂，作者认为无论是model-free还是model-based RL训练都需要非常大量的数据，且没有必要使用RL。作者提出了Learning from OPT，首先利用OPT(Optimal Caching Decision)的min-cost flow算法获取过去一段时间的特征，同时利用一些Online的特征，如空余cache大小等，将这些特征输入GBDT分类器，根据得分排名觉得cache的object。
