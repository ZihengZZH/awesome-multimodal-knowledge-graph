**Resource List w/ Abstract**

- [Paper](#paper)
    - [Distant Supervision for Relation Extraction without Labeled Data](#distant-supervision-for-relation-extraction-without-labeled-data)
    - [Distance Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](#distance-supervision-for-relation-extraction-via-piecewise-convolutional-neural-networks)
    - [Building a Large-scale Multimodal Knowledge Base System for Answering Visual Queries](#building-a-large-scale-multimodal-knowledge-base-system-for-answering-visual-queries)
    - [Order-Embeddings of Image and Language](#order-embeddings-of-image-and-language)
    - [Neural Relation Extraction with Selective Attention over Instances](#neural-relation-extraction-with-selective-attention-over-instances)
    - [Towards Building Large Scale Multimodal Domain-Aware Conversation Systems](#towards-building-large-scale-multimodal-domain-aware-conversation-systems)
    - [A Multimodal Translation-Based Approach for Knowledge Graph Representation Learning](#a-multimodal-translation-based-approach-for-knowledge-graph-representation-learning)
    - [Embedding Multimodal Relational Data for Knowledge Base Completion](#embedding-multimodal-relational-data-for-knowledge-base-completion)
    - [MMKG: Multi-Modal Knowledge Graphs](#mmkg-multi-modal-knowledge-graphs)
    - [Answering Visual-Relational Queries in Web-Extracted Knowledge Graphs](#answering-visual-relational-queries-in-web-extracted-knowledge-graphs)
    - [Multimodal Data Enhanced Representation Learning for Knowledge Graphs](#multimodal-data-enhanced-representation-learning-for-knowledge-graphs)
    - [Multi-modal Knowledge-aware Hierarchical Attention Network for Explainable Medical Question Answering](#multi-modal-knowledge-aware-hierarchical-attention-network-for-explainable-medical-question-answering)
    - [MHSAN: Multi-Head Self-Attention Network for Visual Semantic Embedding](#mhsan-multi-head-self-attention-network-for-visual-semantic-embedding)
- [Tutorials](#tutorials)
    - [Multimodal Knowledge Graphs: Automatic Extraction & Applications](#multimodal-knowledge-graphs-automatic-extraction--applications)
    - [Towards Building Large-Scale Multimodal Knowledge Bases](#towards-building-large-scale-multimodal-knowledge-bases)
    - [How To Build a Knowledge Graph](#how-to-build-a-knowledge-graph)
    - [Mining Knowledge Graphs from Text](#mining-knowledge-graphs-from-text)
    - [Injecting Prior Information and Multiple Modalities into Knowledge Base Embeddings](#injecting-prior-information-and-multiple-modalities-into-knowledge-base-embeddings)
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

**2016** {#2}

#### Order-Embeddings of Image and Language 
  * [[pdf](https://arxiv.org/pdf/1511.06361.pdf)] [[repo](paper/vendrov2016order.pdf)] [[github](https://github.com/ivendrov/order-embedding)]
  * Vendrov et al. (2016.03)
  * ICLR'16
  * Hypernymy, textual entailment, and image captioning can be seen as special cases of a single visual-semantic hierarchy over words, sentences, and images. In this paper we advocate for explicitly modelling the partial order structure of this hierarchy. Towards this goal, we introduce a general method for learning ordered representations, and show how it can be applied to a variety of tasks involving images and language. We show that the resulting representations improve performance over current approaches for hypernym prediction and image-caption retrieval.

#### Neural Relation Extraction with Selective Attention over Instances
  * [[pdf](https://www.aclweb.org/anthology/P16-1200v2.pdf)] [[repo](paper/lin2016neural.pdf)]
  * Lin et al. (2016.08)
  * ACL'16
  * Distant supervised relation extraction has been widely used to find novel relational facts from text. However, distant supervision inevitably accompanies with the wrong labelling problem, and these noisy data will substantially hurt the performance of relation extraction. To alleviate this issue, we propose a sentence-level attention-based model for relation extraction. In this model, we employ convolutional neural networks to embed the semantics of sentences. Afterwards, we build sentence-level attention over multiple instances, which is expected to dynamically reduce the weights of those noisy instances. Experimental results on real-world datasets show that, our model can make full use of all informative sentences and effectively reduce the influence of wrong labelled instances. Our model achieves significant and consistent improvements on relation extraction as compared with baselines.

**2018** {#3}

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

#### Embedding Multimodal Relational Data for Knowledge Base Completion
  * [[pdf](https://www.aclweb.org/anthology/D18-1359.pdf)] [[repo](paper/pezeshkpour2018embedding.pdf)]
  * Pezeshkpour et al. (2018.11)
  * EMNLP'18
  * Representing entities and relations in an embedding space is well-studied approach for machine learning on relational data. Existing approaches, however, primarily focus on simple link structure between a finite set of entities, ignoring the variety of data types that are often used in knowledge bases, such as text, images, and numerical values. In this paper, we propose multimodal knowledge base embeddings (MKBE) that use different neural encoders for this variety of observed data, and combine them with existing relational models to learn embeddings of the entities and multimodal data. Further, using these learned embeddings and different neural decoders, we introduce a novel multimodal imputation model to generate missing multimodal values, like text and images, from information in the knowledge base. We enrich existing relational datasets to create two novel benchmarks that contain additional information such as textual descriptions and images of the original entities. We demonstrate that our models utilize this additional information effectively to provide more accurate link prediction, achieving state-of-the-art results with a considerable gep of 5-7% over existing methods. Further, we evaluate the quality of our generated multimodal values via a user study.

**2019** {#4}

#### MMKG: Multi-Modal Knowledge Graphs 
  * [[pdf](https://arxiv.org/pdf/1903.05485.pdf)] [[repo](paper/liu2019mmkg.pdf)]
  * Liu et al. (2019.03)
  * ESWC'19
  * We present MMKG, a collection of three knowledge graphs that contain both numerical features and (links to) images for all entities as well as entity alignments between pairs of KGs. Therefore, multi-relational link prediction and entity matching communities can benefit from this resource. We believe this data set has the potential to facilitate the development of novel multi-modal learning approaches for knowledge graphs. We validate the utility of MMKG in the sameAs link prediction task with an extensive set of experiments. These experiments show that the task at hand benefits from learning of multiple feature types.

#### Answering Visual-Relational Queries in Web-Extracted Knowledge Graphs
  * [[pdf](https://arxiv.org/pdf/1709.02314.pdf)] [[repo](paper/rubio2019answering.pdf)]
  * Rubio et al. (2019.05)
  * AKBC'19
  * A visual-relational knowledge graph (KG) is a multi-relational graph whose entities are associated with images. We explore novel machine learning approaches for answering visual-relational queries in web-extracted knowledge graphs. To this end, we have created ImageGraph, a KG with 1330 relation types, 14870 entities, and 829,931 images crawled from the web. With visual-relational KGs such as ImageGraph one can introduce novel probabilistic query types in which images are treated as first-class citizens. Both the prediction of relations between unseen images as well as multi-relational image retrieval can be expressed with specific families of visual-relational queries. We introduce novel combinations of convolutional networks and knowledge graph embedding methods to answer such queries. We also explore a zero-shot learning scenario where an image of an entirely new entity is linked with multiple relations to entities of an existing KG. The resulting multi-relational grounding of unseen entity images into a knowledge graph serves as a semantic entity representation. We conduct experiments to demonstrate that the proposed methods can answer these visual-relational queries efficiently and accurately. 

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

**2020** {#1}

#### MHSAN: Multi-Head Self-Attention Network for Visual Semantic Embedding
  * [[pdf](https://arxiv.org/pdf/2001.03712.pdf)] [[repo](paper/park2020mhsan.pdf)]
  * Park et al. (2020.01)
  * WACV'20
  * Visual-semantic embedding enables various tasks such as image-text retrieval, image captioning, and visual question answering. The key to successful visual-semantic embedding is to express visual and textual data properly by accounting for their intricate relationship. While previous studies have achieved much advance by encoding the visual and textual data into a joint space where similar concepts are closely located, they often represent data by a single vector ignoring the presence of multiple important components in an image or text. Thus, in addition to the joint embedding space, we propose a novel multi-head self-attention network to capture various components of visual and textual data by attending to important parts in data. Our approach achieves the new state-of-the-art results in image-text retrieval tasks on MS-COCO and Flickr30K datasets. Through the visualization of the attention maps that capture distinct semantic components at multiple positions in the image and the text, we demonstrate that our method achieves an effective and interpretable visual-semantic joint space.

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

#### Mining Knowledge Graphs from Text
  * [[pdf](https://kgtutorial.github.io/)]
  * [[repo-Tutorial1-Introduction](tutorials/KGTutorial1_Introduction.pdf)]
  * [[repo-Tutorial2-A-InformationExtraction](tutorials/KGTutorial2_A_InformationExtraction.pdf)]
  * [[repo-Tutorial2-B-InformationExtraction](tutorials/KGTutorial2_B_InformationExtraction.pdf)]
  * [[repo-Tutorial3-Construction](tutorials/KGTutorial3_Construction.pdf)]
  * [[repo-Tutorial4-Embedding](tutorials/KGTutorial4_Embedding.pdf)]
  * [[repo-Tutorial5-Conclusion](tutorials/KGTutorial5_Conclusion.pdf)]
  * Jay Pujara & Sameer Singh
  * WSDM‘18

#### Injecting Prior Information and Multiple Modalities into Knowledge Base Embeddings
  * [[pdf](http://exobrain.kr/images/(1-1)Injecting%20Prior%20Information%20and%20Multiple%20Modalities%20into%20KB%20Embeddings(Sameer%20Singh).pdf)]
  * Sameer Singh
  * Uni California, Irvine

## Datasets

