**Resource List w/ Abstract**

- [Paper](#paper)
    - [Distant Supervision for Relation Extraction without Labeled Data](#distant-supervision-for-relation-extraction-without-labeled-data)
    - [Distance Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](#distance-supervision-for-relation-extraction-via-piecewise-convolutional-neural-networks)
    - [Building a Large-scale Multimodal Knowledge Base System for Answering Visual Queries](#building-a-large-scale-multimodal-knowledge-base-system-for-answering-visual-queries)
    - [Neural Relation Extraction with Selective Attention over Instances](#neural-relation-extraction-with-selective-attention-over-instances)
    - [Towards Building Large Scale Multimodal Domain-Aware Conversation Systems](#towards-building-large-scale-multimodal-domain-aware-conversation-systems)
    - [A Multimodal Translation-Based Approach for Knowledge Graph Representation Learning](#a-multimodal-translation-based-approach-for-knowledge-graph-representation-learning)
    - [MMKG: Multi-Modal Knowledge Graphs](#mmkg-multi-modal-knowledge-graphs)
    - [Multimodal Data Enhanced Representation Learning for Knowledge Graphs](#multimodal-data-enhanced-representation-learning-for-knowledge-graphs)
    - [Multi-modal Knowledge-aware Hierarchical Attention Network for Explainable Medical Question Answering](#multi-modal-knowledge-aware-hierarchical-attention-network-for-explainable-medical-question-answering)
- [Tutorials](#tutorials)
    - [Multimodal Knowledge Graphs: Automatic Extraction & Applications](#multimodal-knowledge-graphs-automatic-extraction--applications)
    - [Towards Building Large-Scale Multimodal Knowledge Bases](#towards-building-large-scale-multimodal-knowledge-bases)
    - [How To Build a Knowledge Graph](#how-to-build-a-knowledge-graph)
- [Datasets](#datasets)

## Paper

**2009** {#1}

#### Distant Supervision for Relation Extraction without Labeled Data
  * [[pdf](https://www.aclweb.org/anthology/P09-1113.pdf)] [[repo](paper/mintz2009distant.pdf)]
  * Mintz et al. (2009.08)
  * ACL'09
  * Modern models of relation extraction for tasks like ACE are based on supervised learning of relations from small hand-labeled corpora. We investigate an alternative paradigm that does not require labeled corpora, avoiding the domain dependence of ACE-style algorithms, and allowing the use of corpora of any size. Our experiments use Freebase, a large semantic database of several thousand relations, to provide distant supervision. For each pair of entities that appears in some Freebase relation, we find all sentences containing those entities in a large unlabeled corpus and extract textual features to train a relation classifier. Our algorithm combines the advantages of supervised IE (combining 400,000 noisy pattern features in a probabilistic classifier) and unsupervised IE (extracting large numbers of relations from large corpora of any domain). Our model is able to extract 10,000 instances of 102 relations at a precision of 67.6%. We also analyze feature performance, showing that syntactic parse features are particularly helpful for relations that are ambiguous or lexically distance in their expression.

**2015** {#2}

#### Distance Supervision for Relation Extraction via Piecewise Convolutional Neural Networks
  * [[pdf](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)] [[repo](paper/zeng2015distant.pdf)]
  * Zeng et al. (2015.08)
  * EMNLP'15
  * Two problems arise when using distant supervision for relation extraction. First, in this method, an already existing knowledge base is heuristically aligned to texts, and the alignment results are treated as labeled data. However, the heuristic alignment can fail, resulting in wrong label problem. In addition, in previous approaches, statistical models have typically been applied to ad hoc features. The noise that originates from feature extraction process can cause poor performance. In this paper, we propose a novel model dubbed the Piecewise Convolutional Neural Networks (PCNNs) with multi-instance learning to address these two problems. To solve the first problem, distant supervised relation extraction is treated as a multi-instance problem in which the uncertainty of instance labels is taken into account. To address the latter problem, we avoid feature engineering and instead adopt convolutional architecture with piecewise max pooling to automatically learn relevant features. Experiments show that our method is effective and outperforms several competitive baseline methods.

#### Building a Large-scale Multimodal Knowledge Base System for Answering Visual Queries 
  * [[pdf](https://arxiv.org/pdf/1507.05670.pdf)] [[repo](paper/zhu2015building.pdf)]
  * Zhu et al. (2015.11)
  * arXiv
  * The complexity of the visual world creates significant challenges for comprehensive visual understanding. In spite of recent successes in visual recognition, today's vision systems would still struggle to deal with visual queries that require a deeper reasoning. We propose a knowledge base (KB) framework to handle an assortment of visual queries, without the need to train new classifiers for new tasks. Building such a large-scale multimodal KB presents a major challenge of scalability. We cast a large-scale MRF into a KB representation, incorporating visual, textual and structured data, as well as their diverse relations. We introduce a scalable knowledge base construction system that is capable of building a KB with half billion variables and millions of parameters in a few hours. Our system achieves competitive results compared to purpose-built models on standard recognition and retrieval tasks, while exhibiting greater flexibility in answering richer visual queries.

**2016** {#1}

#### Neural Relation Extraction with Selective Attention over Instances
  * [[pdf](https://www.aclweb.org/anthology/P16-1200v2.pdf)] [[repo](paper/lin2016neural.pdf)]
  * Lin et al. (2016.08)
  * ACL'16
  * Distant supervised relation extraction has been widely used to find novel relational facts from text. However, distant supervision inevitably accompanies with the wrong labelling problem, and these noisy data will substantially hurt the performance of relation extraction. To alleviate this issue, we propose a sentence-level attention-based model for relation extraction. In this model, we employ convolutional neural networks to embed the semantics of sentences. Afterwards, we build sentence-level attention over multiple instances, which is expected to dynamically reduce the weights of those noisy instances. Experimental results on real-world datasets show that, our model can make full use of all informative sentences and effectively reduce the influence of wrong labelled instances. Our model achieves significant and consistent improvements on relation extraction as compared with baselines.

**2018** {#2}

#### Towards Building Large Scale Multimodal Domain-Aware Conversation Systems 
  * [[pdf](https://arxiv.org/pdf/1704.00200.pdf)] [[repo](paper/saha2018towards.pdf)]
  * Saha et al. (2018.01)
  * AAAI'18
  * While multimodal conversation agents are gaining importance in several domains such as retail, travel etc., deep learning research in this area has been limited primarily due to the lack of availability of large-scale, open chatlogs. To overcome this bottleneck, in this paper we introduce the task of multimodal, domain-aware conversation, and propose the MMD benchmark dataset. This dataset was gathered by working in close coordination with large number of domain experts in the reail domain. These experts suggested various conversations flows and dialog states which are typically seen in multimodal conversations in the fashion domain. Keeping these flows and states in mind, we created a dataset consisting of over 150K conversation sessions between shoppers and sale agents, with the help of in-house annotators using a semi-automated manually intense iterative process. With this dataset, we propose 5 new sub-tasks for multimodal conversations along with their evaluation methodology. We also propose two multimodal neural models in the encode-attend-decode paradigm and demonstrate their performance on two of the sub-tasks, namely text response generation and best image response selection. These experiments serve to establish baseline performance and open new research directions for each of these sub-tasks. Further, for each of the sub-tasks, we present a 'per-state evaluation' of 9 most significant dialog states, which would enable more focused research into understanding the challenges and complexities involved in each of these states.

#### A Multimodal Translation-Based Approach for Knowledge Graph Representation Learning 
  * [[pdf](https://www.aclweb.org/anthology/S18-2027.pdf)] [[repo](paper/mousselly2018multimodal.pdf)]
  * Mousselly-Sergieh et al. (2018.06)
  * SEM'18
  * Current methods for knowledge graph (KG) representation learning focus solely on the structure of the KG and do not exploit any kind of external information, such as visual and linguistic information corresponding to the KG entities. In this paper, we propose a multimodal translation-based approach that defines the energy og a KG triple as the sum of sub-energy functions that leverage both multimodal (visual and linguistic) and structural KG representations. Next, a ranking-based loss is minimized using a simple neural network architecture. Moreover, we introduce a new large-scale dataset for multimodal KG representation learning. We compared the performance of our approach to other baselines on two standard tasks, namely knowledge graph completion and triple classification, using our as well as the WN9-IMG dataset. The results demonstrate that our approach outperforms all baselines on both tasks and datasets.

**2019** {#3}

#### MMKG: Multi-Modal Knowledge Graphs 
  * [[pdf](https://arxiv.org/pdf/1903.05485.pdf)] [[repo](paper/liu2019mmkg.pdf)]
  * Liu et al. (2019.03)
  * arXiv
  * We present MMKG, a collection of three knowledge graphs that contain both numerical features and (links to) images for all entities as well as entity alignments between pairs of KGs. Therefore, multi-relational link prediction and entity matching communities can benefit from this resource. We believe this data set has the potential to facilitate the development of novel multi-modal learning approaches for knowledge graphs. We validate the utility of MMKG in the sameAs link prediction task with an extensive set of experiments. These experiments show that the task at hand benefits from learning of multiple feature types.

#### Multimodal Data Enhanced Representation Learning for Knowledge Graphs 
  * [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8852079)] [[repo](paper/wang2019multimodal.pdf)]
  * Wang et al. (2019.06)
  * IJCNN'19
  * Knowledge graph, or knowledge base, plays an important role in a variety of applications in the field of artificial intelligence. In both research and application of knowledge graph, knowledge representation learning is one of the fundamental tasks. Existing representation learning approaches are mainly based on structural knowledge between entities and relations, while knowledge among entities per se is largely ignored. Though a few approaches integrated entity knowledge while learning representations, these methods lack the flexibility to apply to multimodalities. To tackle this problem, in this paper, we propose a new representation learning method, TransAE, by combining multimodal autoencoder with TransE model, where TransE is a simple and effective representation learning method for knowledge graphs. In TransAE, the hidden layer of autoencoder is used as the representation of entities in the TransE model, thus it encodes not only the structural knowledge, but also the multimodal knowledge, such as visual and textual knowledge, into the final representation. Compared with traditional methods based on only structured knowledge, TransAE can significantly improve the performance in the sense of link prediction and triplet classification. Also, TransAE has the ability to learn representations for entities out of knowledge base in zero-shot. Experiments on various tasks demonstrate the effectiveness of our proposed TransAE method.

#### Multi-modal Knowledge-aware Hierarchical Attention Network for Explainable Medical Question Answering 
  * [[pdf](https://dl.acm.org/doi/10.1145/3343031.3351033)] [[repo](paper/zhang2019multimodal.pdf)]
  * Zhang et al. (2019.10)
  * ACM-MM'19
  * Online healthcare services can offer public ubiquitous access to the medical knowledge, especially with the emergence of medical question answering websites, where patients can get in touch with doctors without going to hospital. Explainability and accuracy are two main concerns for medical question answering. However, existing methods mainly focus on accuracy and cannot provide good explanation for retrieved medical answers. This paper proposes a novel Multi-Modal Knowledge-aware Hierarchical Attention Network (MKHAN) to effectively exploit multi-modal knowledge graph (MKG) for explainable medical question answering. MKHAN can generate path representation by composing the structural, linguistics, and visual information of entities, and infer the underlying rationale of question-answer interactions by leveraging the sequential dependencies within a path from MKG. Furthermore, a novel hierarchical attention network is proposed to discriminate the salience of paths endowing our model with explainability. We build a large-scale multi-modal medical knowledge graph and two real-world medical question answering datasets, the experimental results demonstrate the superior performance on our approach compared with the state-of-the-art methods. 

## Tutorials

#### Multimodal Knowledge Graphs: Automatic Extraction & Applications 
  * [[pdf](http://www.ee.columbia.edu/~sfchang/papers/CVPR2019_MM_Knowledge_Graph_SF_Chang.pdf)] [[repo](tutorials/MultimodalKnowledgeGraphs@Columbia.pdf)]
  * Shih-Fu Chang
  * CVPR'19 & Uni. Columbia

#### Towards Building Large-Scale Multimodal Knowledge Bases 
  * [[pdf](https://www.cise.ufl.edu/~dihong/assets/MKBC.pdf)] [[repo](tutorials/TowardsBuildingLargeScaleMultimodalKnowledgeBases@Florida.pdf)]
  * Dihong Gong
  * Uni. Florida

#### How To Build a Knowledge Graph
  * [[pdf](https://2019.semantics.cc/sites/2019.semantics.cc/files/how-to-build-a-knowledge-graph.pdf)] [[repo](tutorials/HowToBuildKnowledgeGraph@Innsbruck.pdf)]
  * Elias Karle & Umutcan Simsek
  * Uni Innsbruck

## Datasets

