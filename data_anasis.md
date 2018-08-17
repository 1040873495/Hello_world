# 数据分析技术在Titanic中的应用

## Overview

可以将分析Titanic乘客信息和幸存结果之间关系的整个流程划分为以下三个基本步骤:

> - 问题定义，数据准备，数据处理，提取最终特征。

> - 模型选择，模型训练，模型测试，模型修正。

> - 撰写工作过程报告，提交最终结果。


总体上，数据分析按照以上过程进行，但是在每个单一过程中可以采用不同的工作流，并且可以反复跳转到某一阶段进行重复处理。



## 第一阶段

### 问题定义

通过深入理解问题所处的背景，我们能够在一定程度上提前把握数据各个属性和目标之间的相关性，有助于识别特征变量，剔除无用变量，并且在特定情况下引入原有数据所没有的属性。最终提升模型的可预测性。

### 数据处理

数据处理阶段的最终目标是筛选出合适的特征变量，在这个过程中需要对数据进行各种变换，可以总结为**7C**原则。

分别是:

> - Classifying: 对样本数据进行分类，并获取不同样本数据和解决方案目标的相关性(**对这一点比较疑惑**)。
>  - Correlating:针对训练数据找出和目标显著相关的属性，也可以找出各个属性之间的相关性，往往相关性强的属性可以选作最终特征。
>  - Converting:对于大多数问题，可能需要将文本数据转换为标量数据。
>  - Completing:获取的数据集中可能存在某些缺失，不完整。需要根据上下文对缺失的数据进行合理的估计，并补全空值。
>  - Correcting:数据集中可能存在和实际情况明显不符合的数据项，需要对这些数据项进行修正，在某些情况下可以将和目标明显无关的属性删除。
>  - Creaing:根据问题的性质和现有特征，可以创造出新的特征，新的特征使得模型的计算更加方便快速，同时表现出和目标显著的相关性。


#### 描述数据集的整体属性

1. 初始数据集包含了哪些**特征**。`train_df.columns.values`

2. 有哪些特征是**数字型**的，比如年龄，费用等。

3. 哪些特征是**类别数据**，比如性别，是否存活等。

4. 哪些特征是**混合型**的，包含数字和文本，比如船票。

5. 哪些特征可能包含**错误数据项**。

6. 哪些特征可能包含**空值数据项**

7. 各个特征的变量类型，比如整形，浮点型或者是一个对象。`train_df.info()`

8. 数字型特征在整个数据集上具有怎样的分布。`train_df.describe()`

9. 类别行特征在整个数据集上如何分布的。`train_df.describe()`

#### 描述数据特征之间的相关性
1. **不能有空数据项，必须是类别特征、序数特征，或者是离散型特征**
2. 比如sex,Pclass,SibSp and Parch
3  常用操为:`traind_df["fiture_A","feature_B"].group_by(["feature_A"],as_index=False).mean().sort_values(by="feature_B",ascending=Fasle)`
这个操作的以某一特征进行分组，这样就可以得到在处于特定特征值时，特征A和特征B的相关性。
4. 通常特征B是目标值，比如这里的`Survived`。


#### 利用图表分析特征之间的相关性以及特征与目标值之间的相关性

1. 对于数字型特征和目标值之间的相关性，可以使用**直方图**，以数字型特征为***x-轴***，以目标值为***y-轴***,能够直观看出在某一范围内目标值的
大小分布，通过的观察目标值随特征变化的波动，来判断它们之间的相关性大小。通常来说波动越大，相关性越强。
2. 可以将**类别型特征**、**数字型特征**、以及**目标值**结合起来，来判断当多个特征组合在一起的时候与目标值之间的相关性。类似的，以数字型特征为x-轴，以目标值为y-轴，以类别型特征为筛选条件，画出直方图。能够直观反映***组合特征***与***目标值***之间的相关性。
3. 组合多个类别型特征观察和目标值之间的相关性。


#### 修正数据集

1. 直接剔除和目标值不相关的的特征。
2.从已有特征中提取新的特征，比如可以从name提取出title这一特征，并判断这一特征和目标值之间的相关性，如果具有明显的相关性，则将新特征保留，并将类别型
特征转换为序数型特征。（主要是对文本性质的类别特征进行转换).
**dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)**

 3. 补全连续型数字特征
 > - 使用所有数据项的均值。
 
 > - 结合该特征和其它特征的相关性确定缺失值。
**for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
 for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]**

 > - 结合以上两种方法。

 4. 将连续型数字特征转化为区间特征，并观察新特征和目标值之间的相关性。如果具有显著相关性，则可以将原有特征表示为序数性特征，并删除区间特征。
**train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

 **for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']**

 5.组合原有特征生成新特征

**for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
  train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)**

 6.补全类别型特征

  在补全类别型特征时，通常选取其中最频繁出现的数据值

**freq_port = train_df.Embarked.dropna().mode()[0]
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)**


## 第二阶段

### 模型选取,训练,测试,修正
根据问题特征选取合适模型(**目前还没有深入研究**）

## 第三阶段
还没有实际经验，下周会进行实战。
