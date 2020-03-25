**Resource List w/ Abstract**

> [pdf]: paper PDF online link
> [repo]: paper PDF repo link
> [github]: github link
> [web]: website link

- [Paper](#paper)
    - [Distant Supervision for Relation Extraction without Labeled Data](#distant-supervision-for-relation-extraction-without-labeled-data)
    - [Translating Embeddings for Modeling Multi-relational Data](#translating-embeddings-for-modeling-multi-relational-data)
    - [Relation Classification via Convolutional Deep Neural Network](#relation-classification-via-convolutional-deep-neural-network)
    - [Distance Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](#distance-supervision-for-relation-extraction-via-piecewise-convolutional-neural-networks)
    - [Building a Large-scale Multimodal Knowledge Base System for Answering Visual Queries](#building-a-large-scale-multimodal-knowledge-base-system-for-answering-visual-queries)
    - [Order-Embeddings of Image and Language](#order-embeddings-of-image-and-language)
    - [Neural Relation Extraction with Selective Attention over Instances](#neural-relation-extraction-with-selective-attention-over-instances)
    - [Image-embodied Knowledge Representation Learning](#image-embodied-knowledge-representation-learning)
    - [Towards Building Large Scale Multimodal Domain-Aware Conversation Systems](#towards-building-large-scale-multimodal-domain-aware-conversation-systems)
    - [A Multimodal Translation-Based Approach for Knowledge Graph Representation Learning](#a-multimodal-translation-based-approach-for-knowledge-graph-representation-learning)
    - [Multimodal Unsupervised Image-to-Image Translation](#multimodal-unsupervised-image-to-image-translation)
    - [Embedding Multimodal Relational Data for Knowledge Base Completion](#embedding-multimodal-relational-data-for-knowledge-base-completion)
    - [MMKG: Multi-Modal Knowledge Graphs](#mmkg-multi-modal-knowledge-graphs)
    - [Shared Predictive Cross-Modal Deep Quantization](#shared-predictive-cross-modal-deep-quantization)
    - [Answering Visual-Relational Queries in Web-Extracted Knowledge Graphs](#answering-visual-relational-queries-in-web-extracted-knowledge-graphs)
    - [Multimodal Data Enhanced Representation Learning for Knowledge Graphs](#multimodal-data-enhanced-representation-learning-for-knowledge-graphs)
    - [Aligning Linguistic Words and Visual Semantic Units for Image Captioning](#aligning-linguistic-words-and-visual-semantic-units-for-image-captioning)
    - [A New Benchmark and Approach for Fine-grained Cross-media Retrieval](#a-new-benchmark-and-approach-for-fine-grained-cross-media-retrieval)
    - [Annotation Efficient Cross-Modal Retrieval with Adversarial Attentive Alignment](#annotation-efficient-cross-modal-retrieval-with-adversarial-attentive-alignment)
    - [Knowledge-guided Pairwise Reconstruction Network for Weakly Supervised Referring Expression Grounding](#knowledge-guided-pairwise-reconstruction-network-for-weakly-supervised-referring-expression-grounding)
    - [Multimodal Dialog System: Generating Responses via Adaptive Decoders](#multimodal-dialog-system-generating-responses-via-adaptive-decoders)
    - [Multi-modal Knowledge-aware Hierarchical Attention Network for Explainable Medical Question Answering](#multi-modal-knowledge-aware-hierarchical-attention-network-for-explainable-medical-question-answering)
    - [MHSAN: Multi-Head Self-Attention Network for Visual Semantic Embedding](#mhsan-multi-head-self-attention-network-for-visual-semantic-embedding)
    - [MULE: Multimodal Universal Language Embedding](#mule-multimodal-universal-language-embedding)
    - [Modality to Modality Translation: An Adversarial Representation Learning and Graph Fusion Network for Multimodal Fusion](#modality-to-modality-translation-an-adversarial-representation-learning-and-graph-fusion-network-for-multimodal-fusion)
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

**2013** {#1}

#### Translating Embeddings for Modeling Multi-relational Data
  * [[pdf](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)] [[repo](paper/bordes2013translating.pdf)]
  * Bordes et al. (2013.12)
  * NIPS'13
  * We consider the problem of embedding entities and relationships of multi-relational data in the low-dimensional vector spaces. Our objective is to propose a canonical model which is easy to train, contains a reduced number of parameters and can scale up to very large databases. Hence, we propose TransE, a method which models relationships by interpreting them as translations operating on the low-dimensional embeddings of the entities. Despite its simplicity, this assumption proves to be powerful since extensive experiments show that TransE significantly outperforms state-of-the-art methods in link prediction on two knowledge bases. Besides, it can be successfully trained on a large scale data set with 1M entities, 25k relationships and more than 17M training samples.

**2014** {#1}

#### Relation Classification via Convolutional Deep Neural Network
  * [[pdf](https://www.aclweb.org/anthology/C14-1220.pdf)] [[repo](paper/zeng2014relation.pdf)]
  * Zeng et al. (2014.08)
  * COLING'14
  * The state-of-the-art methods used for relation classification are primarily based on statistical machine learning, and their performance strongly depends on the quality of the extracted features. The extracted features are often derived from the output of pre-existing natural language processing (NLP) systems, which leads to the propagation of the errors in the existing tools and hinders the performance of these systems. In this paper, we exploit a convolutional deep neural network (DNN) to extract lexical and sentence level features. Our method takes all of the word tokens as input without complicated pre-processing. First, the work tokens are transformed to vectors by looking up word embeddings. Then, lexical level features are extracted according to the given nouns. Meanwhile, sentence level features are learned using a convolutional approach. These two level features are concatenated to form the final extracted feature vector. Finally, the features are fed into a softmax classifier to predict the relationship between two marked nouns. The experimental results demonstrate that our approach significantly outperforms the state-of-the-art methods.

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

**2017** {#1}

#### Image-embodied Knowledge Representation Learning
  * [[pdf](https://www.ijcai.org/Proceedings/2017/0438.pdf)] [[repo](paper/xie2017image.pdf)]
  * Xie et al. (2017.08)
  * IJCAI'17
  * Entity images could provide significant visual information for knowledge representation learning. Most conventional methods learn knowledge representations merely from structured triples, ignoring rich visual information extracted from entity images. In this paper, we propose a novel Image-embodied Knowledge Representation Learning model (IKRL), where knowledge representations are learned with both triple facts and images. More specially, we first construct representations for all images of an entity with a neural image encoder. These image representations are then integrated into an aggregated image-based representation via an attention-based method. We evaluate out IKRL models on knowledge graph completion and triple classification. Experimental results demonstrate that our models outperform all baselines on both tasks, which indicates the significance of visual information for knowledge representations and the capability of our models in learning knowledge representations with images.

**2018** {#4}

#### Towards Building Large Scale Multimodal Domain-Aware Conversation Systems
  * [[pdf](https://arxiv.org/pdf/1704.00200.pdf)] [[repo](paper/saha2018towards.pdf)]
  * Saha et al. (2018.01)
  * AAAI'18
  * While multimodal conversation agents are gaining importance in several domains such as retail, travel etc., deep learning research in this area has been limited primarily due to the lack of availability of large-scale, open chatlogs. To overcome this bottleneck, in this paper we introduce the task of multimodal, domain-aware conversation, and propose the MMD benchmark dataset. This dataset was gathered by working in close coordination with large number of domain experts in the reail domain. These experts suggested various conversations flows and dialog states which are typically seen in multimodal conversations in the fashion domain. Keeping these flows and states in mind, we created a dataset consisting of over 150K conversation sessions between shoppers and sale agents, with the help of in-house annotators using a semi-automated manually intense iterative process. With this dataset, we propose 5 new sub-tasks for multimodal conversations along with their evaluation methodology. We also propose two multimodal neural models in the encode-attend-decode paradigm and demonstrate their performance on two of the sub-tasks, namely text response generation and best image response selection. These experiments serve to establish baseline performance and open new research directions for each of these sub-tasks. Further, for each of the sub-tasks, we present a 'per-state evaluation' of 9 most significant dialog states, which would enable more focused research into understanding the challenges and complexities involved in each of these states.

#### A Multimodal Translation-Based Approach for Knowledge Graph Representation Learning
  * [[pdf](https://www.aclweb.org/anthology/S18-2027.pdf)] [[repo](paper/mousselly2018multimodal.pdf)]
  * Mousselly-Sergieh et al. (2018.06)
  * SEM'18
  * Current methods for knowledge graph (KG) representation learning focus solely on the structure of the KG and do not exploit any kind of external information, such as visual and linguistic information corresponding to the KG entities. In this paper, we propose a multimodal translation-based approach that defines the energy of a KG triple as the sum of sub-energy functions that leverage both multimodal (visual and linguistic) and structural KG representations. Next, a ranking-based loss is minimized using a simple neural network architecture. Moreover, we introduce a new large-scale dataset for multimodal KG representation learning. We compared the performance of our approach to other baselines on two standard tasks, namely knowledge graph completion and triple classification, using our as well as the WN9-IMG dataset. The results demonstrate that our approach outperforms all baselines on both tasks and datasets.

#### Multimodal Unsupervised Image-to-Image Translation
  * [[pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf)] [[repo](paper/huang2018multimodal.pdf)]
  * Huang et al. (2018.09)
  * ECCV'18
  * Unsupervised image-to-image translation is an important and challenging problem in computer vision. Given an image in the source domain, the goal is to learn the conditional distribution of corresponding images in the target domain, without seeing any examples of corresponding image pairs. While this conditional distribution is inherently multimodal, existing approaches make an overly simplified assumption, modeling it as a deterministic one-to-one mapping. As a result, they fail to generate diverse outputs from a given source domain image. To address this limitation, we propose a Multimodal Unsupervised Image-to-image Translation (MUNIT) framework. We assume that the image representation can be decomposed into a content code that is domain-invariant, and a style code that captures domain-specific properties. To translate an image to another domain, we recombine its content code with a random style code sampled from the style space of the target domain. We analyze the proposed framework and establish several theoretical results. Extensive experiments with comparisons to the state-of-the-art approaches further demonstrate the advantage of the proposed framework. Moreover, our framework allows users to control the style of translation outputs by providing an example style image. Code and pretrained models are available at https://github.com/nvlabs/MUNIT.

#### Embedding Multimodal Relational Data for Knowledge Base Completion
  * [[pdf](https://www.aclweb.org/anthology/D18-1359.pdf)] [[repo](paper/pezeshkpour2018embedding.pdf)]
  * Pezeshkpour et al. (2018.11)
  * EMNLP'18
  * Representing entities and relations in an embedding space is well-studied approach for machine learning on relational data. Existing approaches, however, primarily focus on simple link structure between a finite set of entities, ignoring the variety of data types that are often used in knowledge bases, such as text, images, and numerical values. In this paper, we propose multimodal knowledge base embeddings (MKBE) that use different neural encoders for this variety of observed data, and combine them with existing relational models to learn embeddings of the entities and multimodal data. Further, using these learned embeddings and different neural decoders, we introduce a novel multimodal imputation model to generate missing multimodal values, like text and images, from information in the knowledge base. We enrich existing relational datasets to create two novel benchmarks that contain additional information such as textual descriptions and images of the original entities. We demonstrate that our models utilize this additional information effectively to provide more accurate link prediction, achieving state-of-the-art results with a considerable gep of 5-7% over existing methods. Further, we evaluate the quality of our generated multimodal values via a user study.

**2019** {#10}

#### MMKG: Multi-Modal Knowledge Graphs
  * [[pdf](https://arxiv.org/pdf/1903.05485.pdf)] [[repo](paper/liu2019mmkg.pdf)]
  * Liu et al. (2019.03)
  * ESWC'19
  * We present MMKG, a collection of three knowledge graphs that contain both numerical features and (links to) images for all entities as well as entity alignments between pairs of KGs. Therefore, multi-relational link prediction and entity matching communities can benefit from this resource. We believe this data set has the potential to facilitate the development of novel multi-modal learning approaches for knowledge graphs. We validate the utility of MMKG in the sameAs link prediction task with an extensive set of experiments. These experiments show that the task at hand benefits from learning of multiple feature types.

#### Shared Predictive Cross-Modal Deep Quantization
  * [[pdf](https://arxiv.org/pdf/1904.07488.pdf)] [[repo](paper/yang2019shared.pdf)]
  * Yang et al. (2019.04)
  * arXiv
  * With explosive growth of data volume and ever-increasing diversity of data modalities, cross-modal similarity search, which conducts nearest neighbor search across different modalities, has been attracting increasing interest. This paper presents a deep compact code learning solution for efficient cross-modal similarity search. Many recent studies have proven that quantization based approaches perform generally better than hashing based approaches on single-modal similarity search. In this work, we propose a deep quantization approach, which is among the early attempts of leveraging deep neural networks into quantization based cross-modal similarity search. Our approach, dubbed shared predictive deep quantization (SPDQ), explicitly formulates a shared subspace across different modalities and two private subspaces for individual modalities, representations in the shared subspace and the private subspaces are learned simultaneously by embedding them to a reproducing kernel Hilbert space where the mean embedding of different modality distributions can be explicitly compared. Additionally, in the shared subspace, a quantizer is learned to produce the semantics preserving compact codes with the help of label alignment. Thanks to this novel network architecture in cooperation with supervised quantization training, SPDQ can preserve intra- and inter-modal similarities as much as possible and greatly reduce quantization error. Experiments on two popular benchmarks corroborate that our approach outperforms state-of-the-art methods.

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

#### Aligning Linguistic Words and Visual Semantic Units for Image Captioning
  * [[pdf](https://arxiv.org/pdf/1908.02127.pdf)] [[repo](paper/guo2019aligning.pdf)] [[github](https://github.com/ltguo19/VSUA-Captioning)]
  * Guo et al. (2019.10)
  * ACM-MM'19
  * Image captioning attempts to generate a sentence composed of several linguistic words, which are used to describe objects, attributes, and interactions in an image, denoted as visual semantic units in this paper. Based on this view, we propose to explicitly model the object interactions in semantics and geometry based on Graph Convolutional Networks (GCNs), and fully exploit the alignment between linguistic words and visual semantic units for image captioning. Particularly, we construct a semantic graph and a geometry graph, where each node corresponds to a visual semantic unit, i.e., an object, an attribute, or a semantic (geometrical) interaction between two objects. Accordingly, the semantic (geometrical) context-aware embeddings for each unit are obtained through the corresponding GCN learning processors. At each time step, a context gated attention module takes as inputs the embeddings of the visual semantic units and hierarchically align the current word with these units by first deciding which type of visual semantic unit (object, attribute, or interaction) the current word is about, and then finding the most correlated visual semantic units under this type. Extensive experiments are conducted on the challenging MS-COCO image cpationing dataset, and superior results are reported when comparing to state-of-the-art approaches. The code is publicly available at https://github.com/ltguo19/VSUA-Captioning.

#### A New Benchmark and Approach for Fine-grained Cross-media Retrieval
  * [[pdf](https://arxiv.org/pdf/1907.04476.pdf)] [[repo](paper/he2019new.pdf)] [[github](https://github.com/PKU-ICST-MIPL/FGCrossNet_ACMMM2019)]
  * He et al. (2019.10)
  * ACM-MM'19
  * Cross-media retrieval is to return the results of various media types corresponding to the query of any media type. Existing researches generally focus on coarse-grained cross-media retrieval. When users submit an image of "Slaty-backed Gull" as a query, coarse-grained cross-media retrieval treats it as "Bird", so that users can only get the results of "Bird", which may include other bird species with similar appearance (image and video), descriptions (text) or sounds (audio), such as "Herring Gull". Such coarse-grained cross-media retrieval is not consistent with human lifestyle, where we generally have the fine-grained requirement of returning the exactly relevant results of "Slaty-backed Gull" instead of "Herring Gull". However, few researches focus on fine-grained cross-media retrieval, which is a highly challenging and practical task. Therefore, in this paper, we first construct a new benchmark for fine-grained cross-media retrieval, which consists of 200 fine-grained subcategories of the "Bird", and contains 4 media types, including image, text, video and audio. To the best of our knowledge, it is the first benchmark with 4 media types for fine-grained cross-media retrieval. Then, we propose a uniform deep model, namely FGCrossNet, which simultaneously learns 4 types of media without discriminative treatments. We jointly consider three constraints for better common representation learning: *classification constraint* ensures the learning of discriminative features for fine-grained subcategories, *center constraint* ensures the compactness characteristic of the features of the same subcategory, and *ranking constraint* ensures the sparsity characteristic of the features of different subcategories. Extensive experiments verify the usefulness of the new benchmark and the effectiveness of our FGCrossNet. The new benchmark and the source code of FGCrossNet will be made available at https://github.com/PKU-ICST-MIPL/FGCrossNet_ACMMM2019.

#### Annotation Efficient Cross-Modal Retrieval with Adversarial Attentive Alignment
  * [[pdf](http://www.cs.cmu.edu/~poyaoh/data/ann.pdf)] [[repo](paper/huang2019annotation.pdf)]
  * Huang et al. (2019.10)
  * ACM-MM'19
  * Visual-semantic embeddings are central to many multimedia applications such as cross-modal retrieval between visual data and natural language descriptions. Conventionally, learning a joint embedding space relies on large parallel multimodal corpora. Since massive human annotation is expensive to obtain, there is a strong motivation in developing versatile algorithms to learn from large corpora with fewer annotations. In this paper, we propose a novel framework to leverage automatically extracted regional semantics from un-annotated images as additional weak supervision to learn visual-semantic embeddings. The proposed model employs adversarial attentive alignments to close the inherent heterogeneous gaps between annotated and un-annotated portions of visual and textual domains. To demonstrate its superiority, we conduct extensive experiments on sparsely annotated multimodal corpora. The experimental results show that the proposed model outperforms state-of-the-art visual-semantic embedding models by a significant margin for cross-modal retrieval tasks on the sparse Flickr30K and MS-COCO datasets. It is also worth noting that, despite using only 20% of the annotations, the proposed model can achieve competitive performance (Recall at 10 > 80.0% for 1K and > 70% for 5K text-to-image retrieval) compared to the benchmarks trained with the complete annotations.

#### Knowledge-guided Pairwise Reconstruction Network for Weakly Supervised Referring Expression Grounding
  * [[pdf](https://arxiv.org/pdf/1909.02860.pdf)] [[repo](paper/liu2019knowledge.pdf)]
  * Liu et al. (2019.10)
  * ACM-MM'19
  * Weakly supervised referring expression grounding (REG) aims at localizing the referential entity in an image according to linguistic query, where the mapping between the image region (proposal) and the query is unknown in the training stage. In referring expressions, people usually describe a target entity in terms of its relationship with other contextual entities as well as visual attributes. However, previous weakly supervised REG methods rarely pay attention to the relationship between the entities. In this paper, we propose a knowledge-guided pairwise reconstruction network (KPRN), which models the relationship between the target entity (subject) and contextual entity (object) as well as grounds these two entities. Specifically, we first design a knowledge extraction module to guide the proposal selection of subject and object. The prior knowledge is obtained in a specific form of semantic similarities between each proposal and the subject/object. Second, guided by such knowledge, we design the subject and object attention module to construct the subject-object proposal pairs. The subject attention excludes the unrelated proposals from the candidate proposals. The object attention selects the most suitable proposal as the contextual proposal. Third, we introduce a pairwise attention and an adaptive weighting scheme to learn the correspondence between these proposal pairs and the query. Finally, a pairwise reconstruction module is used to measure the grounding for weakly supervised learning. Extensive experiments on four large-scale datasets show our method outperforms existing state-of-the-art methods by a large margin.

#### Multimodal Dialog System: Generating Responses via Adaptive Decoders
  * [[pdf](https://liqiangnie.github.io/paper/fp349-nieAemb.pdf)] [[repo](paper/nie2019multimodal.pdf)]
  * Nie et al. (2019.10)
  * ACM-MM'19
  * On the shoulders of textual dialog systems, the multimodal ones, recently have engaged increasing attention, especially in the retail domain. Despite the commercial value of multimodal dialog systems, they will suffer from the following challenges: 1) automatically generate the right responses in appropriate medium forms; 2) jointly consider the visual cues and the side information while selecting product images; and 3) guide the response generation with multi-faceted and heterogeneous knowledge. To address the aforementioned issues, we present a Multimodal diAloG  system with adaptIve deCoders, MAGIC for short. In particular, MAGIC first judges the response type and the corresponding medium form via understanding the intention of the given multimodal context. Hereafter, it employs adaptive decoders to generate the desired responses: a simple recurrent neural network (RNN) is applied to generating general responses, then a knowledge-aware RNN decoder is designed to encode the multiform domain knowledge to enrich the response, and the multimodal response decoder incorporates an image recommendation model with jointly considers the textual attributes and the visual images via a neural model optimized by the max-margin loss. We comparatively justify MAGIC over a benchmark dataset. Experiment results demonstrate that MAGIC outperforms the existing methods and achieves the state-of-the-art performance.

#### Multi-modal Knowledge-aware Hierarchical Attention Network for Explainable Medical Question Answering
  * [[pdf](https://dl.acm.org/doi/10.1145/3343031.3351033)] [[repo](paper/zhang2019multimodal.pdf)]
  * Zhang et al. (2019.10)
  * ACM-MM'19
  * Online healthcare services can offer public ubiquitous access to the medical knowledge, especially with the emergence of medical question answering websites, where patients can get in touch with doctors without going to hospital. Explainability and accuracy are two main concerns for medical question answering. However, existing methods mainly focus on accuracy and cannot provide good explanation for retrieved medical answers. This paper proposes a novel Multi-Modal Knowledge-aware Hierarchical Attention Network (MKHAN) to effectively exploit multi-modal knowledge graph (MKG) for explainable medical question answering. MKHAN can generate path representation by composing the structural, linguistics, and visual information of entities, and infer the underlying rationale of question-answer interactions by leveraging the sequential dependencies within a path from MKG. Furthermore, a novel hierarchical attention network is proposed to discriminate the salience of paths endowing our model with explainability. We build a large-scale multi-modal medical knowledge graph and two real-world medical question answering datasets, the experimental results demonstrate the superior performance on our approach compared with the state-of-the-art methods.

**2020** {#3}

#### MHSAN: Multi-Head Self-Attention Network for Visual Semantic Embedding
  * [[pdf](https://arxiv.org/pdf/2001.03712.pdf)] [[repo](paper/park2020mhsan.pdf)]
  * Park et al. (2020.01)
  * WACV'20
  * Visual-semantic embedding enables various tasks such as image-text retrieval, image captioning, and visual question answering. The key to successful visual-semantic embedding is to express visual and textual data properly by accounting for their intricate relationship. While previous studies have achieved much advance by encoding the visual and textual data into a joint space where similar concepts are closely located, they often represent data by a single vector ignoring the presence of multiple important components in an image or text. Thus, in addition to the joint embedding space, we propose a novel multi-head self-attention network to capture various components of visual and textual data by attending to important parts in data. Our approach achieves the new state-of-the-art results in image-text retrieval tasks on MS-COCO and Flickr30K datasets. Through the visualization of the attention maps that capture distinct semantic components at multiple positions in the image and the text, we demonstrate that our method achieves an effective and interpretable visual-semantic joint space.

#### MULE: Multimodal Universal Language Embedding
  * [[pdf](https://arxiv.org/pdf/1909.03493.pdf)] [[repo](paper/kim2019MULE.pdf)]
  * Kim et al. (2020.02)
  * AAAI'20
  * Existing vision-language methods typically support two languages at a time at most. In this paper, we present a modular approach which can easily be incorporated into existing vision-language methods in order to support many languages. We accomplish this by learning a single shared Multimodal Universal Language Embedding (MULE) which has been visually-semantically aligned across all languages. Then we learn to relate MULE to visual data as if it were a single language. Our method is not architecture specific, unlike prior work which typically learned separate branches for each language, enabling our approach to easily be adapted to many vision-language methods and tasks. Since MULE learns a single language branch in the multimodal model, we can also scale to support many languages, and languages with fewer annotations can take advantage of the good representation learned from other (more abundant) language data. We demonstrate the effectiveness of MULE on the bidirectional image-sentence retrieval task, supporting up to four languages in a single model. In addition, we show that Machine Translation can be used for data augmentation in multilingual learning, which, combined with MULE, improves mean recall by up to 21.9% on a single-language compared to prior work, with the most significant gains seen on languages with relatively few annotations. Our code is publicly available.

#### Modality to Modality Translation: An Adversarial Representation Learning and Graph Fusion Network for Multimodal Fusion
  * [[pdf](https://arxiv.org/pdf/1911.07848.pdf)] [[repo](paper/mai2019modality.pdf)]
  * Mai et al. (2020.02)
  * AAAI'20
  * Learning joint embedding space for various modalities is of vital importance for multimodal fusion. Mainstream modality fusion approaches fail to achieve this goal, leaving a modality gap which heavily affects cross-modal fusion. In this paper, we propose a novel adversarial encoder-decoder-classifier framework to learn a modality invariant embedding space. Since the distributions of various modalities vary in nature, to reduce the modality gap, we translate the distributions of source modalities into that of target modality via their respective encoders using adversarial training. Furthermore, we exert additional constraints on embedding space by introducing reconstruction loss and classification loss. Then we fuse the encoded representations using hierarchical graph neural network which explicitly explores unimodal, bimodal and trimodal interactions in multi-stage. Our method achieves state-of-the-art performance on multiple datasets. Visualization of the learned embeddings suggests that the joint embedding space learned by our method is discriminative.

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
  * [[pdf](http://exobrain.kr/images/(1-1)Injecting%20Prior%20Information%20and%20Multiple%20Modalities%20into%20KB%20Embeddings(Sameer%20Singh).pdf)] [[repo](tutorials/InjectingPriorInformationAndMultipleModalitiesIntoKBEmbeddings.pdf)]
  * Sameer Singh
  * Uni California, Irvine

## Datasets

