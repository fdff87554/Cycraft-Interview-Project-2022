# Cycraft-Interview-Project-2022
This Repo is for 2022 Cycraft Intern Program Interview Homework.

## Practices
1. Read and understand the paper "Cyber Threat Intelligence Modeling Based on Heterogeneous Graph Convolutional Network"
    > [Paper link](https://www.usenix.org/conference/raid2020/presentation/zhao)
2. Implement the model described in the paper and modified the model to fit the provided datasets.
3. Verify the paper's performance.
4. Give your comment about this paper's pros and cons.

## Datasets
* [CTI-reports-dataset](https://github.com/nlpai-lab/CTI-reports-dataset)

## Summary of Paper
* About CTI and IOC.
    * 網路威脅情資 (Cyber Threat Intelligence, CTI) 
        * 是指關於網路安全威脅的情報，它包括威脅的來源、目標、方法、類型和可能的影響等方面的信息。
        * 是經過 收集、轉換、分析、解釋、整理 等等處理過程之後的資料，主要提供用來分析資安事件跟提供決策的基礎。
    * 妥協指標/危害指標 (Indicator of Compromise, IoC/IOC)
        * 一種用來識別定義電腦遭受威脅的表示方法，可以視為一種被入侵的證據，也可以允許研究人員利用 IoC 來更好的分析事件所發生的行為跟技術。
    * CTI 是關於整體網路安全威脅的情報，IoC 則是關於特定威脅活動的指標，而 CTI 的相關情報是允許被撰寫成 IoC 的形式進行情報分享。
    * 現在的研究中，有期望從非結構化資料中利用 regular expression 的方式提取其中的 IoC 資訊 (如依循 OpenIOC 規範的 *CleanMX*, *PhishTank*, *IOC Finder*, and *Gartner peer insight*)，來建構 CTI 資訊，但會遇到，
        1. IoC 提取的準確率不高。
        2. 孤立的 IoC 資訊基本無法描述威脅事件的樣貌。
        3. 缺乏有效準確的標準來計算不同的 IoC 之間的交互關係。
        > OpenIOC 標準，利用一組正則表示式來提取特定的 IOC 資訊 (例如，malicious IPs, malware, file Hash, etc)
* 論文中提出了三個大區塊解決上面的問題，分別是
    * Multi-granular Attention based IOC Recognition - 基於多粒度注意力的 IOC 識別
    * Heterogeneous Threat Intelligence Modeling -  異構威脅情報建模
    * Threat Intelligence Computing Framework - 威脅情報框架
    並依照上面的三個部分建立了 Threat Intelligence Prototype System (威脅情報模型系統) 來 辨識、分析 資訊。
* Heterogeneous Information Network of Threat Intelligence (HINTI) workflow
    * Extract IOCs - 提取 IOCs: 因為關於 CTI 的敘述中，並不是所有資訊都是非常重要的，因此藉由 B-I-O Tagging Method，可以加速對有意義資訊的提取
        * e.g., *Last week, Lotus exploited CVE-2017-0143 vulnerability to affect a larger number of Vista SP2 and Win7 SP devices in Iran*
            * Extracted IOCs without any relations among them.
                > ![](https://i.imgur.com/EfuDKy1.png)
            * B-I-O Tagging Method.
                * B-X indicates that the element of type X is located at the beginning of the fragment.
                    > B-X 表示 X 類型的元素位於片段的開頭。
                * I-X means that the element belonging to type X is located at the middle of the fragment.
                    > I-X 表示屬於 X 類型的元素位於片段的中間。
                * O represents a non-essential element of other types.
                    > O 代表其他類型的非必要元素。
                > ![](https://i.imgur.com/LsHmZcJ.png)
    * Relationships between IOCs - IOCs 之間的關聯性: 單一的 IOC 資訊往往無法代表任何意義，例如攻擊者的名稱並不會代表任何東西
        * Syntactic dependency parser (e.g., subject-predicate-object, attributive clause, etc.) to extract associated relationships between IOCs. ($IOC_i$, relation, $IOC_j$)
            * e.g., (*Lotus*, *exploit*, *CVE-2017-0143*), (*CVE-2017-0143*, *affect*, *Vista SP2*), etc.
            * Extracted relational triples can be  incrementally pooled into an HIN to model the interactions among IOCs for depicting a more comprehensive threat landscape.
                > ![](https://i.imgur.com/a29Z1nD.png)

### Multi-granular Attention Based IoC Extraction - 基於多粒度注意力的IOC提取
* 由於我們希望從文本之中提取跟 IOCs 有關的 B-I-O Tag，藉由這些 Tag 來辨識文本中的重要資訊，並於後面步驟做到關聯性建立，因此我們首先要識別文本中的這些 B-I-O Tags。
* 而 B-I-O Tags 在這邊也屬於自然語言處理中的一個任務，也就是 Named-entity recognition (NER)，它的目標是將文本中提及的實體識別出來並且標記它們的類型。這些實體可以包括人名、地點、組織、時間等等。例如，在一段文本中，NER 系統可以將 "紐約市" 認定為地點，"約翰·史密斯" 認定為人名，"2022 年 3 月" 認定為時間等。而對於我們來說， malware / IPs 等等則數於我們再億的 NER。
* 而由於 Bidirectional Long Short-Term Memory+Conditional Random Fields (BiLSTM+CRF) model 在 NER 方面有出色的效能，這邊選擇的 BiLSTM + CRF model 作為訓練的模型，但由於 IOC 文本的不穩定性問題，論文中提出了利用 n-gram model 針對訓練資料先做一次前處理之後再給 BiLSTM + CRF model 做訓練的 "Multi-granular" 方法。
* 實作方法可以看 Method Summary 和 `IOC-Detect/model/bilstm_crf.py` 的 code，資料前處理則在 `IOC-Detect/utils/datasets.py` 中的 `CTIDatasetForBiLSTMCRFWithNgram` 還有 `CTIDatasetForBiLSTMCRF`。

### Cyber Threat Intelligence Modeling - 網絡威脅情報建模
* 在論文中說明了，其建立 `IOC and IOC` 之間關係的方式依賴於九種 relation，分別是
    * attacker-exploit-vulnerability
    * attacker-invade-device
    * attacker-cooperate-attacker
    * vulnerability-affect-device
    * vulnerability-belong-attack
    * vulnerability-include-file
    * file-target-device
    * vulnerability-evolve-vulnerability
    * device-belong-platform
* 且利用 syntactic dependency parser 來做訓練提取，這邊有提到使用的 syntactic dependency parser 是 "*A Fast and Accurate Dependency Parser using Neural Networks*" 一論文中的方法。
* 就我的理解，我可以利用 **stanza** 這個工具來標記整個 training data 跟 testing data 的 sentences 的 dependency，然後利用有 B-I tags 的敘述去找前後相依來做到上述的三元提取，但在關於這個部分，我卡在不知道怎麼準備 training dataset 來告訴 model 我希望這樣三組三組的提取資料，因為看起來 Dependency Parser using NN 論文應該也是一個 Dependency Parser，而論文中並沒有特別說明關於上述的九種 Relationships 的提取詳細實作細節，因此沒有實作出來。
* 正在嘗試做的事情是把 Dependency Parser 跟 Relation Link 分開，讓 Dependency Parser 一樣在建立我們常見的 sentences NERs，而後我們再利用一個類似 Dict 的方式直接針對句子中的 IOC-to-IOC 這樣的結構把資料提取出來，在 Method Summary 中有提到我的想法，並搭配寫到一半的 IOC-Pairing 的 Code，但可惜的是並沒有達成。

### Threat Intelligence Computing - 威脅情報計算
* 在有大量的 `IOC-to-IOC` pairs 之後，我們要開始將資料們做關聯，來達到對於整個 CTI 事件的描繪，在論文中定義了計算 pair graph 相似性公式，並且提到一個避免遍歷所有資料群的方式，也就是依照他們提出的 meta-path 去做建模即可，也就是說當資料關係已經滿足 meta-path，就可以停止往下遍歷了，可以有效增加建模效能。
* 而論文中提到了 17 種 meta-path，如下圖，在給定威脅情報圖 $G=(V, E)$ 還有 meta-path $M={P1, P2, ..., Pi}$，
    1. 基於元路徑Pi計算IOC之間的相似度，以生成相應的鄰接矩陣Ai
    2. 通過將IOC的屬性信息嵌入到向量空間中，構造節點Xi的特徵矩陣
    3. 進行圖卷積 $GCN(Ai，Xi)$，通過遵循元路徑Pi量化IOC之間的相互依賴關係，將其嵌入到低維空間中
    > ![](https://i.imgur.com/dZ4T5r0.png)
* 關於上述部分，因為時間上的關係，並沒有實際做到更深入的理解跟實作，但就想像中而言，因為 meta-path 的建立過程中，其實已經將事件的關聯性描繪得蠻仔細的了，而且相近的資料描述，會因為其建構的 meta-path 讓其特徵近似，因此可以想像 IOCs 之間的特徵相比零散的 IOC 會強烈很多，也可以理解這樣的 GCN 效果應該會很不錯。
* 而就我自己的理解，這一部的效果會很不錯是必然的，甚至可以說就算沒有這一步驟，頂多就是不好描繪整個 CTI 事件的全貌，但由於前面對於 IOC 資料的提取已經非常強大（第一個 method 如果可以正確地在"敘述"中精準的提取 IOC Tags，其實就已經能夠幫資安人員釐清很多資訊了，而第二部的 relation pair 的建立更是已經可以描繪很多 IOC 資訊之間的關聯，最後這個步驟更像是必然的因為資訊很明確，因此很好計算資料之間的相似性。
