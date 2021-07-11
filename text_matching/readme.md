### 文本匹配简单流程
目前感觉收获不大，太能滑水了，总结一下
#### 基础思路：
传统模型ESIM 预训练模型 编辑距离等特征
#### 可用的技巧：
数据增强：伪标签，闭包增强

对抗训练

### 例题
[souhu校园文本匹配](https://www.biendata.xyz/competition/sohu_2021/my-submission/)

本题有两种类型的匹配，一种是精确的B文件匹配(两段文字必须是同一个事件)，一种是大致意思的A文件匹配（两端文字是一个话题），样本数据如下：

```
{
    "source": "英国伦敦，20/21赛季英超第20轮，托特纳姆热刺VS利物浦。热刺本赛季18轮联赛是9胜6平3负，目前积33分排名联赛第5位。利物浦本赛季19轮联赛是9胜7平3负，目前积34分排名联赛第4位。从目前的走势来看，本场比赛从热刺的角度来讲，是非常被动的。最终，本场比赛的比分为托特纳姆热刺1-3利",
    "target": " 北京时间1月29日凌晨4时，英超联赛第20轮迎来一场强强对话，热刺坐镇主场迎战利物浦。  热刺vs利物浦，比赛看点如下： 第一：热刺能否成功复仇？双方首回合，热刺客场1-2被利物浦绝杀，赛后穆里尼奥称最好的球队输了，本轮热刺主场迎战利物浦，借着红军5轮不胜的低迷状态，能否成功复仇？ 第二：利物浦近",
    "labelA": "1"
}
```

```
{
    "source": "英国伦敦，20/21赛季英超第20轮，托特纳姆热刺VS利物浦。热刺本赛季18轮联赛是9胜6平3负，目前积33分排名联赛第5位。利物浦本赛季19轮联赛是9胜7平3负，目前积34分排名联赛第4位。从目前的走势来看，本场比赛从热刺的角度来讲，是非常被动的。最终，本场比赛的比分为托特纳姆热刺1-3利",
    "target": " 北京时间1月29日凌晨4时，英超联赛第20轮迎来一场强强对话，热刺坐镇主场迎战利物浦。  热刺vs利物浦，比赛看点如下： 第一：热刺能否成功复仇？双方首回合，热刺客场1-2被利物浦绝杀，赛后穆里尼奥称最好的球队输了，本轮热刺主场迎战利物浦，借着红军5轮不胜的低迷状态，能否成功复仇？ 第二：利物浦近",
    "labelB": "0"
}
```

每种匹配按文本长度又分为了长文本与长文本匹配，长文本短文本匹配，短文本短文本匹配。

### EDA 数据初探

1.长长匹配 B类 

![image-20210519142915906](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519142915906.png)

![image-20210519142936833](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519142936833.png)

可见长长匹配的句子B长度基本分布在2000以内，正负例占比32714：12459

2. 长长匹配A类

   ![image-20210519143405330](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519143405330.png)

![image-20210519143423367](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519143423367.png)

正负例分布为23496：21693

3. 短长匹配B类

   ![image-20210519144745570](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519144745570.png)

   ![image-20210519144212637](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519144212637.png)

   正负例占比9706：35692

   4. 短长匹配A类

      ![image-20210519144630819](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519144630819.png)

      ![image-20210519144640536](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519144640536.png)

      正负例分布：19967：25428

      5. 短短匹配 B类

         ![image-20210519144928617](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519144928617.png)

      ![image-20210519144941318](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519144941318.png)

      正负例分布20230：5015

      6. 短短匹配A类

         ![image-20210519145105962](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519145105962.png)

      ![image-20210519145132753](C:\Users\liwenlong\AppData\Roaming\Typora\typora-user-images\image-20210519145132753.png)

      正负比 10364：14870

      

#### 思路

- [x] 采用统一的一个模型来解决这六类问题，通过将类型变量输入网络，区分是哪种模式的匹配。
- [ ] 建立两个模型，对A文件和B文件单独进行匹配（即训练两个模型），因为之前的单模型实验结果表明，A文件线上的f1值可以在0.79，而B文件的f1值仅仅在0.68。
- [ ] 由于长文本中大多数词对匹配并没有贡献，可以考虑关键词提取如textrank，将长文本转为短文本，则所有的问题都成为了短短匹配。

#### 采用的模型
1. Roberta-wwm-large-ext
2. Ernie
3. Rofomer

每个模型均采用五折交叉，并在验证集中寻找最佳的分类阈值

对这三个模型进行stacking

### 结果

| 模型                 |            线上f1 |      线上f1_a      | 线上f1_b           | 线下f1 |
| -------------------- | ----------------: | :----------------: | ------------------ | ------ |
| Ernie                | 0.738960391717261 | 0.7929843693202471 | 0.6849364141142755 |        |
| Rofomer              | 0.728386311281346 | 0.7864891083869829 | 0.6702835141757088 | 0.749  |
| Roberta-wwm-base-ext | 0.733490495179992 | 0.792749658002736  | 0.6742313323572474 |        |


后续更新
### 例题
[2020 CCF DataFountain 房产行业聊天问答匹配](https://www.datafountain.cn/competitions/474)


### 例题
[小布助手短文本语义匹配](https://tianchi.aliyun.com/competition/entrance/531851/introduction?spm=5176.12281957.1004.2.38b02448ORmlMv)
发现比赛太晚了，没有参加，知识点

1. 脱敏数据（只有词频，没有词）的使用
2. 预训练