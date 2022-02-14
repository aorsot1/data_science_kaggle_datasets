fraud_detection
================
Akoua Orsot
2/14/2022

-   [Fraud Detection](#fraud-detection)
    -   [1. Environment Set-up](#1-environment-set-up)
    -   [2. Initial Diagnostics](#2-initial-diagnostics)
    -   [3. Data Cleaning](#3-data-cleaning)
    -   [4. Correlation Analysis](#4-correlation-analysis)
    -   [5. Inquiry Exploration](#5-inquiry-exploration)
    -   [6. Class Imbalance](#6-class-imbalance)
    -   [7. Machine Learning set-up](#7-machine-learning-set-up)

# Fraud Detection

This notebook will attempt to build a predictive algorithm to detect a
fraudulent transaction using a training dataset. We will explain the
thinking process at every step using LIME (Local Interpretable
Model-agnostic Explanations) principles making it accessible and
user-friendly.

## 1. Environment Set-up

``` r
## Importing libraries
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(tidyverse)
```

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --

    ## v ggplot2 3.3.5     v purrr   0.3.4
    ## v tibble  3.1.5     v stringr 1.4.0
    ## v tidyr   1.1.4     v forcats 0.5.1
    ## v readr   2.0.2

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(ggplot2)
library(e1071)
library(caret)
```

    ## Loading required package: lattice

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
# install.packages("corrplot")
library(corrplot)
```

    ## Warning: package 'corrplot' was built under R version 4.1.2

    ## corrplot 0.92 loaded

``` r
# install.packages("ROSE")
library(ROSE)
```

    ## Warning: package 'ROSE' was built under R version 4.1.2

    ## Loaded ROSE 0.0-4

``` r
# install.packages("hyperSMURF")
library(hyperSMURF)
```

    ## Warning: package 'hyperSMURF' was built under R version 4.1.2

``` r
## Loading dataset
df <- read_csv(file = 'C:/Users/Akoua Orsot/Desktop/ds_projects_data/creditcard.csv')
```

    ## Rows: 284807 Columns: 31

    ## -- Column specification --------------------------------------------------------
    ## Delimiter: ","
    ## dbl (31): Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14,...

    ## 
    ## i Use `spec()` to retrieve the full column specification for this data.
    ## i Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
head(df)
```

    ## # A tibble: 6 x 31
    ##    Time     V1      V2    V3     V4      V5      V6      V7      V8     V9
    ##   <dbl>  <dbl>   <dbl> <dbl>  <dbl>   <dbl>   <dbl>   <dbl>   <dbl>  <dbl>
    ## 1     0 -1.36  -0.0728 2.54   1.38  -0.338   0.462   0.240   0.0987  0.364
    ## 2     0  1.19   0.266  0.166  0.448  0.0600 -0.0824 -0.0788  0.0851 -0.255
    ## 3     1 -1.36  -1.34   1.77   0.380 -0.503   1.80    0.791   0.248  -1.51 
    ## 4     1 -0.966 -0.185  1.79  -0.863 -0.0103  1.25    0.238   0.377  -1.39 
    ## 5     2 -1.16   0.878  1.55   0.403 -0.407   0.0959  0.593  -0.271   0.818
    ## 6     2 -0.426  0.961  1.14  -0.168  0.421  -0.0297  0.476   0.260  -0.569
    ## # ... with 21 more variables: V10 <dbl>, V11 <dbl>, V12 <dbl>, V13 <dbl>,
    ## #   V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>, V18 <dbl>, V19 <dbl>,
    ## #   V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>, V24 <dbl>, V25 <dbl>,
    ## #   V26 <dbl>, V27 <dbl>, V28 <dbl>, Amount <dbl>, Class <dbl>

## 2. Initial Diagnostics

``` r
## Glimpse of the data
df %>% str()
```

    ## spec_tbl_df [284,807 x 31] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
    ##  $ Time  : num [1:284807] 0 0 1 1 2 2 4 7 7 9 ...
    ##  $ V1    : num [1:284807] -1.36 1.192 -1.358 -0.966 -1.158 ...
    ##  $ V2    : num [1:284807] -0.0728 0.2662 -1.3402 -0.1852 0.8777 ...
    ##  $ V3    : num [1:284807] 2.536 0.166 1.773 1.793 1.549 ...
    ##  $ V4    : num [1:284807] 1.378 0.448 0.38 -0.863 0.403 ...
    ##  $ V5    : num [1:284807] -0.3383 0.06 -0.5032 -0.0103 -0.4072 ...
    ##  $ V6    : num [1:284807] 0.4624 -0.0824 1.8005 1.2472 0.0959 ...
    ##  $ V7    : num [1:284807] 0.2396 -0.0788 0.7915 0.2376 0.5929 ...
    ##  $ V8    : num [1:284807] 0.0987 0.0851 0.2477 0.3774 -0.2705 ...
    ##  $ V9    : num [1:284807] 0.364 -0.255 -1.515 -1.387 0.818 ...
    ##  $ V10   : num [1:284807] 0.0908 -0.167 0.2076 -0.055 0.7531 ...
    ##  $ V11   : num [1:284807] -0.552 1.613 0.625 -0.226 -0.823 ...
    ##  $ V12   : num [1:284807] -0.6178 1.0652 0.0661 0.1782 0.5382 ...
    ##  $ V13   : num [1:284807] -0.991 0.489 0.717 0.508 1.346 ...
    ##  $ V14   : num [1:284807] -0.311 -0.144 -0.166 -0.288 -1.12 ...
    ##  $ V15   : num [1:284807] 1.468 0.636 2.346 -0.631 0.175 ...
    ##  $ V16   : num [1:284807] -0.47 0.464 -2.89 -1.06 -0.451 ...
    ##  $ V17   : num [1:284807] 0.208 -0.115 1.11 -0.684 -0.237 ...
    ##  $ V18   : num [1:284807] 0.0258 -0.1834 -0.1214 1.9658 -0.0382 ...
    ##  $ V19   : num [1:284807] 0.404 -0.146 -2.262 -1.233 0.803 ...
    ##  $ V20   : num [1:284807] 0.2514 -0.0691 0.525 -0.208 0.4085 ...
    ##  $ V21   : num [1:284807] -0.01831 -0.22578 0.248 -0.1083 -0.00943 ...
    ##  $ V22   : num [1:284807] 0.27784 -0.63867 0.77168 0.00527 0.79828 ...
    ##  $ V23   : num [1:284807] -0.11 0.101 0.909 -0.19 -0.137 ...
    ##  $ V24   : num [1:284807] 0.0669 -0.3398 -0.6893 -1.1756 0.1413 ...
    ##  $ V25   : num [1:284807] 0.129 0.167 -0.328 0.647 -0.206 ...
    ##  $ V26   : num [1:284807] -0.189 0.126 -0.139 -0.222 0.502 ...
    ##  $ V27   : num [1:284807] 0.13356 -0.00898 -0.05535 0.06272 0.21942 ...
    ##  $ V28   : num [1:284807] -0.0211 0.0147 -0.0598 0.0615 0.2152 ...
    ##  $ Amount: num [1:284807] 149.62 2.69 378.66 123.5 69.99 ...
    ##  $ Class : num [1:284807] 0 0 0 0 0 0 0 0 0 0 ...
    ##  - attr(*, "spec")=
    ##   .. cols(
    ##   ..   Time = col_double(),
    ##   ..   V1 = col_double(),
    ##   ..   V2 = col_double(),
    ##   ..   V3 = col_double(),
    ##   ..   V4 = col_double(),
    ##   ..   V5 = col_double(),
    ##   ..   V6 = col_double(),
    ##   ..   V7 = col_double(),
    ##   ..   V8 = col_double(),
    ##   ..   V9 = col_double(),
    ##   ..   V10 = col_double(),
    ##   ..   V11 = col_double(),
    ##   ..   V12 = col_double(),
    ##   ..   V13 = col_double(),
    ##   ..   V14 = col_double(),
    ##   ..   V15 = col_double(),
    ##   ..   V16 = col_double(),
    ##   ..   V17 = col_double(),
    ##   ..   V18 = col_double(),
    ##   ..   V19 = col_double(),
    ##   ..   V20 = col_double(),
    ##   ..   V21 = col_double(),
    ##   ..   V22 = col_double(),
    ##   ..   V23 = col_double(),
    ##   ..   V24 = col_double(),
    ##   ..   V25 = col_double(),
    ##   ..   V26 = col_double(),
    ##   ..   V27 = col_double(),
    ##   ..   V28 = col_double(),
    ##   ..   Amount = col_double(),
    ##   ..   Class = col_double()
    ##   .. )
    ##  - attr(*, "problems")=<externalptr>

``` r
## Descriptive Statistics
df %>% summary()
```

    ##       Time              V1                  V2                  V3          
    ##  Min.   :     0   Min.   :-56.40751   Min.   :-72.71573   Min.   :-48.3256  
    ##  1st Qu.: 54202   1st Qu.: -0.92037   1st Qu.: -0.59855   1st Qu.: -0.8904  
    ##  Median : 84692   Median :  0.01811   Median :  0.06549   Median :  0.1799  
    ##  Mean   : 94814   Mean   :  0.00000   Mean   :  0.00000   Mean   :  0.0000  
    ##  3rd Qu.:139321   3rd Qu.:  1.31564   3rd Qu.:  0.80372   3rd Qu.:  1.0272  
    ##  Max.   :172792   Max.   :  2.45493   Max.   : 22.05773   Max.   :  9.3826  
    ##        V4                 V5                   V6                 V7          
    ##  Min.   :-5.68317   Min.   :-113.74331   Min.   :-26.1605   Min.   :-43.5572  
    ##  1st Qu.:-0.84864   1st Qu.:  -0.69160   1st Qu.: -0.7683   1st Qu.: -0.5541  
    ##  Median :-0.01985   Median :  -0.05434   Median : -0.2742   Median :  0.0401  
    ##  Mean   : 0.00000   Mean   :   0.00000   Mean   :  0.0000   Mean   :  0.0000  
    ##  3rd Qu.: 0.74334   3rd Qu.:   0.61193   3rd Qu.:  0.3986   3rd Qu.:  0.5704  
    ##  Max.   :16.87534   Max.   :  34.80167   Max.   : 73.3016   Max.   :120.5895  
    ##        V8                  V9                 V10                 V11          
    ##  Min.   :-73.21672   Min.   :-13.43407   Min.   :-24.58826   Min.   :-4.79747  
    ##  1st Qu.: -0.20863   1st Qu.: -0.64310   1st Qu.: -0.53543   1st Qu.:-0.76249  
    ##  Median :  0.02236   Median : -0.05143   Median : -0.09292   Median :-0.03276  
    ##  Mean   :  0.00000   Mean   :  0.00000   Mean   :  0.00000   Mean   : 0.00000  
    ##  3rd Qu.:  0.32735   3rd Qu.:  0.59714   3rd Qu.:  0.45392   3rd Qu.: 0.73959  
    ##  Max.   : 20.00721   Max.   : 15.59500   Max.   : 23.74514   Max.   :12.01891  
    ##       V12                V13                V14                V15          
    ##  Min.   :-18.6837   Min.   :-5.79188   Min.   :-19.2143   Min.   :-4.49894  
    ##  1st Qu.: -0.4056   1st Qu.:-0.64854   1st Qu.: -0.4256   1st Qu.:-0.58288  
    ##  Median :  0.1400   Median :-0.01357   Median :  0.0506   Median : 0.04807  
    ##  Mean   :  0.0000   Mean   : 0.00000   Mean   :  0.0000   Mean   : 0.00000  
    ##  3rd Qu.:  0.6182   3rd Qu.: 0.66251   3rd Qu.:  0.4931   3rd Qu.: 0.64882  
    ##  Max.   :  7.8484   Max.   : 7.12688   Max.   : 10.5268   Max.   : 8.87774  
    ##       V16                 V17                 V18           
    ##  Min.   :-14.12985   Min.   :-25.16280   Min.   :-9.498746  
    ##  1st Qu.: -0.46804   1st Qu.: -0.48375   1st Qu.:-0.498850  
    ##  Median :  0.06641   Median : -0.06568   Median :-0.003636  
    ##  Mean   :  0.00000   Mean   :  0.00000   Mean   : 0.000000  
    ##  3rd Qu.:  0.52330   3rd Qu.:  0.39968   3rd Qu.: 0.500807  
    ##  Max.   : 17.31511   Max.   :  9.25353   Max.   : 5.041069  
    ##       V19                 V20                 V21           
    ##  Min.   :-7.213527   Min.   :-54.49772   Min.   :-34.83038  
    ##  1st Qu.:-0.456299   1st Qu.: -0.21172   1st Qu.: -0.22839  
    ##  Median : 0.003735   Median : -0.06248   Median : -0.02945  
    ##  Mean   : 0.000000   Mean   :  0.00000   Mean   :  0.00000  
    ##  3rd Qu.: 0.458949   3rd Qu.:  0.13304   3rd Qu.:  0.18638  
    ##  Max.   : 5.591971   Max.   : 39.42090   Max.   : 27.20284  
    ##       V22                  V23                 V24          
    ##  Min.   :-10.933144   Min.   :-44.80774   Min.   :-2.83663  
    ##  1st Qu.: -0.542350   1st Qu.: -0.16185   1st Qu.:-0.35459  
    ##  Median :  0.006782   Median : -0.01119   Median : 0.04098  
    ##  Mean   :  0.000000   Mean   :  0.00000   Mean   : 0.00000  
    ##  3rd Qu.:  0.528554   3rd Qu.:  0.14764   3rd Qu.: 0.43953  
    ##  Max.   : 10.503090   Max.   : 22.52841   Max.   : 4.58455  
    ##       V25                 V26                V27            
    ##  Min.   :-10.29540   Min.   :-2.60455   Min.   :-22.565679  
    ##  1st Qu.: -0.31715   1st Qu.:-0.32698   1st Qu.: -0.070840  
    ##  Median :  0.01659   Median :-0.05214   Median :  0.001342  
    ##  Mean   :  0.00000   Mean   : 0.00000   Mean   :  0.000000  
    ##  3rd Qu.:  0.35072   3rd Qu.: 0.24095   3rd Qu.:  0.091045  
    ##  Max.   :  7.51959   Max.   : 3.51735   Max.   : 31.612198  
    ##       V28                Amount             Class         
    ##  Min.   :-15.43008   Min.   :    0.00   Min.   :0.000000  
    ##  1st Qu.: -0.05296   1st Qu.:    5.60   1st Qu.:0.000000  
    ##  Median :  0.01124   Median :   22.00   Median :0.000000  
    ##  Mean   :  0.00000   Mean   :   88.35   Mean   :0.001728  
    ##  3rd Qu.:  0.07828   3rd Qu.:   77.17   3rd Qu.:0.000000  
    ##  Max.   : 33.84781   Max.   :25691.16   Max.   :1.000000

**Takeaway:** The following percentage breakdown confirms the note in
the project description; indeed, we have a considerable class imbalance
with the target variable. It stays consistent that most fraudulent
activities are much less frequent than non-fraudulent. Before
proceeding, we shall note it to avoid any overfitting issues when
fitting the machine learning models.

``` r
## Target Variable Analysis
df %>% group_by(Class) %>%
  summarise(cnt = n()) %>%
  mutate(freq = round(cnt / sum(cnt), 5)) %>% 
  arrange(desc(freq))
```

    ## # A tibble: 2 x 3
    ##   Class    cnt    freq
    ##   <dbl>  <int>   <dbl>
    ## 1     0 284315 0.998  
    ## 2     1    492 0.00173

**Note:** We did not have any information on the numerical predictors
for privacy, given their transformation and standardization, excluding
Amount & Time. In that regard, Amount presented itself as potentially
most informative for the feature variable analysis. To better understand
the variable’s distribution, we had to transform it using a log scale.

``` r
## Target Variable Analysis
df$Amount %>% summary()
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ##     0.00     5.60    22.00    88.35    77.17 25691.16

``` r
df %>% ggplot(aes(Amount)) +
  geom_histogram(bins=35) +
  scale_x_log10() +
  labs(
  x = "Dollar Amount (Log Scale)",
  y = "Frequency (Count)",
  title= "Distribution of Transaction Amount (log scaled)"
 )
```

    ## Warning: Transformation introduced infinite values in continuous x-axis

    ## Warning: Removed 1825 rows containing non-finite values (stat_bin).

![](credit_fraud_detection_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

## 3. Data Cleaning

``` r
## Missing Values
df %>% is.na() %>% sum()
```

    ## [1] 0

**Takeaway:** As the count shows, we have no missing values given the
pre-processing done prior.

**Note**: With most predictors transformed, there will be little chance
for any outliers in the data points for V1, V2, …, V28. So, we will only
examine Amount as the only meaningful numeric feature.

``` r
df %>% ggplot(aes(x=Amount)) +
  geom_boxplot() +
  labs(
  x = "Amount ($)",
  title= "Distribution of Transaction Amount"
 )
```

![](credit_fraud_detection_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

**Takeaway:** From the boxplot below, we can observe a non-negligible
number of outliers on the upper end of the distribution. It would denote
transactions with high amounts in the order of thousands of dollars. We
would assess the effect of this skewed distribution when building the
predictive models in terms of feature transformation or selecting models
robust to such feature types.

``` r
df %>% duplicated() %>% sum()
```

    ## [1] 1081

**Takeaway:** A quick check reveals 1081 duplicate rows, so we proceed
in removing them from the dataset.

``` r
df <- df[!duplicated(df), ]
```

**Definition:** Feature Engineering

``` r
df$Amount <- scale(df$Amount)
```

## 4. Correlation Analysis

``` r
df_cor <- cor(df)
corrplot(df_cor, method = 'color')
```

![](credit_fraud_detection_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

**Takeaway:** From the correlation matrix plotted, we can observe very
few correlated variables as we would expect after the feature
transformation. The two meaningful features, are Time and Amount, have
some relative correlation with some variables with coefficients
approximating 0.4. With such low values, it would be pretty challenging
to imply a correlation between any of them with any certainty. It also
indicates that there would be a very low incidence of any colinearity
within our data

**Note:** The code below filters those pairs with correlation
coefficients above 0.5 as a threshold. As noted above, those values give
very little to no confidence in any solid correlated relationship
between variables as few crossing the 0.5 mark.

``` r
df_cor <- as.data.frame(df_cor)
df_cor[(abs(df_cor) >= 0.5) & (abs(df_cor) !=1)]
```

    ## [1] -0.533428 -0.533428

## 5. Inquiry Exploration

**Note:** In an attempt to answer the first question, we first split our
dataset by class types; in other words, fraudulent and non-fraudulent
transactions. We then plot the histogram side by side to observe any
unusual behavior. In doing so, the non-fraud transactions were heavily
right-skewed, making it quite challenging to compare the plots. To solve
this issue, we used a logarithmic transformation, making it easier to
see and thus, evaluate any similarities and differences.

``` r
# How does Amount's distribution behaves across classes?

# Splitting data by fraud class
df_no_fraud <- df %>% filter(Class == 0)
df_fraud <- df %>% filter(Class == 1)

# Histogram for Amount Distribution per class
df_no_fraud %>% ggplot(aes(x=Amount)) +
  geom_histogram(color="black", fill="white", bins=100) +
  labs(
  x = "Scaled Amount",
  title= "Distribution of Non-Fraud Transactions"
 )
```

![](credit_fraud_detection_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
df_fraud %>% ggplot(aes(x=Amount)) +
  geom_histogram(color="black", fill="white", bins=50) +
  labs(
  x = "Scaled Amount",
  title= "Distribution of Fraud Transactions"
 )
```

![](credit_fraud_detection_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->

**Takeaway:** Before making a note on the plots, we will first explain
how to interpret logarithmic scales. In short, log scales show relative
values rather than absolute ones. Indeed, 2 minus 1 would be displayed
similarly to 9999 minus 9998, given that we are dealing with percentages
here. In context, the histograms below would depict the order of growth
of transaction value. Both distributions represent a similar trajectory,
with most transactions on the lower end of the graph. It stays
consistent with the mean value found at USD88, even with max values
averaging USD20,000.

**Note:** For the second question, we will check the timing of
transactions to detect anything unusual. We will use only the fraud
dataset and plot a scatterplot accordingly.

``` r
## Are there any noteworthy point in time where fraud occured?
# Scatterplot
df_fraud %>% ggplot(aes(x=Time, y=Amount)) +
  geom_point() +
  labs(
  y = "Amount ($)", 
  x = "Time (s)",
  title= "Fraudulent Transactions Across Time"
 )
```

![](credit_fraud_detection_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

**Takeaway:** The graph above does not appear that there is a clustering
pattern on a time interval. So, we would assume that fraud occurred
across time quite randomly.

## 6. Class Imbalance

**Note:** Our diagnostics observed a stark imbalance between classes of
transactions, with fraud only making up 0.2% of all transaction
statuses. Given the limited pool of examples to train, it poses an issue
in terms of building an effective machine model to predict if there is a
fraud. With the minority class being so small, we would expect poor
performance on the critical task of detecting fraud transactions. In
that vein, we will use different sampling methods (Undersampling &
Oversampling) to tackle this problem.

``` r
# Splitting features & target variable
X <- df %>% subset(select = -c(Class))
y <- df$Class
```

**Definition:** SMOTE (Synthetic Minority Oversampling Technique) is an
oversampling approach to the minority class. In context, it would mean
to randomly increase fraud examples by “artificially” replicating to
have a more balanced class distribution. Further information
[here](https://rikunert.com/smote_explained).

``` r
## now using ROSE for oversampling
ROSE_over <- ovun.sample(Class ~., data=df,
                                  p=0.5, seed=1,
                                  method="over")
```

``` r
data_balanced_over <- ROSE_over$data

## Check class distribution after using SMOTE
data_balanced_over %>% group_by(Class) %>%
  summarise(cnt = n()) %>%
  mutate(freq = round(cnt / sum(cnt), 5)) %>% 
  arrange(desc(freq))
```

    ## # A tibble: 2 x 3
    ##   Class    cnt  freq
    ##   <dbl>  <int> <dbl>
    ## 1     1 283545 0.500
    ## 2     0 283253 0.500

**Definition:** Near-Miss Algorithm is an undersampling approach on the
majority class. In context, we select examples to keep out of the
training set based on the distance of majority class examples to
minority class examples. Further information
[here](https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/).

``` r
## now using ROSE for oversampling
ROSE_under <- ovun.sample(Class ~., data=df,
                                  p=0.5, seed=1,
                                  method="under")
```

``` r
data_balanced_under <- ROSE_under$data

## Check class distribution after using SMOTE
data_balanced_over %>% group_by(Class) %>%
  summarise(cnt = n()) %>%
  mutate(freq = round(cnt / sum(cnt), 5)) %>% 
  arrange(desc(freq))
```

    ## # A tibble: 2 x 3
    ##   Class    cnt  freq
    ##   <dbl>  <int> <dbl>
    ## 1     1 283545 0.500
    ## 2     0 283253 0.500

**Note:** With the risk of overfitting with oversampling and the
possibility to lose valuable information from undersampling, we will
also consider combining both to rebalance the distribution. So, we shall
proceed with the combination to offer curve out the risks we identified.

``` r
## now using ROSE for oversampling
ROSE_both <- ovun.sample(Class ~., data=df,
                                  p=0.5, seed=1,
                                  method="both")
```

``` r
data_balanced_both <- ROSE_under$data

## Check class distribution after using SMOTE
data_balanced_both %>% group_by(Class) %>%
  summarise(cnt = n()) %>%
  mutate(freq = round(cnt / sum(cnt), 5)) %>% 
  arrange(desc(freq))
```

    ## # A tibble: 2 x 3
    ##   Class   cnt  freq
    ##   <dbl> <int> <dbl>
    ## 1     1   473 0.510
    ## 2     0   455 0.490

``` r
df <- data_balanced_both
df
```

    ##       Time            V1            V2            V3           V4            V5
    ## 1    75665 -6.494318e-01  1.644842e+00   0.003989921  0.810399154  -0.066460869
    ## 2    57164 -6.059694e-01 -3.311257e+00  -0.610508242  0.753683902  -1.491984385
    ## 3    48024 -1.242900e+00 -6.187341e-02   1.077574859  3.275117901   1.596855166
    ## 4    59751  1.222724e+00 -7.825671e-01   0.877073698 -0.577728679  -1.404458272
    ## 5   168822  1.971121e+00 -1.819974e-01  -2.419557036 -0.014400195   0.689074237
    ## 6    73389 -3.494827e+00 -3.615634e+00   1.464523615 -1.266595313  -0.708052758
    ## 7     3766  1.391192e+00 -7.006223e-01  -0.920995983 -1.612063384   1.453085428
    ## 8    84660  9.722455e-01 -1.015617e+00   0.366793614  0.424008626  -0.981685581
    ## 9   115858 -1.622427e+00  1.962529e+00  -0.973222636 -0.634073176  -0.528524082
    ## 10   99190 -1.367376e-01  5.647147e-01   0.157283822 -0.719750196   1.175613976
    ## 11  147180  1.968554e+00 -1.495805e-02  -2.363675733  1.186401837   0.932308955
    ## 12  167376  8.825313e-04  1.316590e-01  -0.134153512 -0.557772865  -0.455797832
    ## 13   44873  8.876884e-01 -1.237429e+00   1.171264084 -0.630323007  -1.796794726
    ## 14  151668  1.946192e-01  4.157985e-01   0.847275671 -0.719842189   0.108031665
    ## 15  162203  2.079024e+00 -2.908637e-02  -2.667661638 -0.371559281   1.283622932
    ## 16   60790 -1.545612e+00  5.489596e-01   1.809253580  1.401073531   0.241918002
    ## 17   59865  8.484148e-01 -4.149484e-01   0.810700250  1.139482931   0.042343929
    ## 18   60087  1.177783e+00  3.224111e-01   0.277826210  1.138003519  -0.221832157
    ## 19   37414  1.076787e+00 -1.796288e-01   1.120229599  1.520206844  -0.466351178
    ## 20  165638 -1.211767e+00  7.858829e-01   1.440107625 -0.427937874   0.089547713
    ## 21   34304  1.030420e+00  3.309966e-02   0.326064859  1.065343326  -0.002512249
    ## 22   50074 -1.407463e+00  1.162774e+00   1.476300916 -0.426860692  -0.022218248
    ## 23   67754  1.247658e+00 -1.343582e+00   1.450116028 -0.388536103  -1.820043736
    ## 24  129296  1.704105e+00 -6.960504e-01  -0.570017029  1.328406150  -0.570693898
    ## 25  137112 -9.033212e-01  9.902848e-01  -0.505091025 -0.269046342  -0.005764784
    ## 26  163421  1.390333e-01  9.429695e-01  -0.618631620 -0.780480775   1.197128151
    ## 27  137110  1.785420e+00 -5.903061e-01  -1.421728148  0.347520437  -0.013512909
    ## 28   84702 -6.118960e-01  1.210103e+00   0.218658399 -0.088198267   0.769349099
    ## 29   35966  1.213695e+00  3.363320e-01   0.432760998  0.777539947  -0.533583696
    ## 30  161075  3.664455e-01 -3.729878e+00  -2.101705165 -1.445950083  -1.486706904
    ## 31  143697 -1.320395e+00  1.517447e+00  -0.958338604 -0.512530349   0.569979978
    ## 32   37216  9.459867e-01 -1.006686e+00   0.865973925  0.658456216  -1.512018937
    ## 33  132991  1.972207e+00 -4.798072e-02  -3.671938069 -0.189729856   2.866697326
    ## 34   63456  1.251541e+00  4.784015e-01  -0.772111855  1.130308902   0.457915823
    ## 35  142658  1.959709e+00  4.857429e-02  -1.897056321  0.391321815   0.348388614
    ## 36   75547  1.023225e+00 -1.002348e-01   0.736312291  1.264623290  -0.588951394
    ## 37   43089 -1.638550e-01 -2.719313e-01   1.097969989 -0.996259396  -2.574296335
    ## 38  140876  2.102790e+00 -6.911186e-01  -1.565639934 -0.286486635  -0.399462898
    ## 39   44715  9.895391e-01 -2.697334e-01   1.224888822  1.625076942  -0.654623034
    ## 40  147792 -2.388261e+00  7.905003e-01  -3.042896021 -0.449350071   1.045472812
    ## 41   55405 -1.637908e+00 -8.304664e-01   1.294877599  0.453392000  -0.477077243
    ## 42   41012  1.025131e+00 -7.170971e-01   1.046517341  0.138542764  -1.125144308
    ## 43  146620  2.093510e+00 -6.512087e-01  -1.009194045 -1.776087652  -0.310621614
    ## 44   74510 -1.062351e+00 -6.446857e-02   0.933430280 -4.073485963  -0.841593150
    ## 45  154799  2.088367e+00  2.405427e-01  -1.743555108  0.369063241   0.595459383
    ## 46  127211 -7.684446e-01 -2.789323e-01   1.706255461 -1.422340593  -1.296944911
    ## 47   48881 -5.722353e-01 -2.304633e-01   1.646085976 -2.142374020  -0.156822150
    ## 48   99570 -2.358377e+00  1.175782e+00   1.335738119 -0.402198135  -0.078240486
    ## 49  164273 -1.180363e+00  1.117052e+00   0.015704245 -1.090149578   0.239556011
    ## 50  111549 -8.556157e+00  5.198451e+00  -5.250511688 -1.162437505  -2.652057861
    ## 51   64541 -7.987925e-01  3.182772e-01   1.405072955 -2.454963178   0.068415282
    ## 52   60542 -2.652929e+00 -1.621410e+00   1.568353326 -1.493961838  -2.100139838
    ## 53   18559  1.231842e+00 -7.629263e-01   0.951151267 -0.637086847  -1.042058060
    ## 54   54868 -4.074179e-01  1.051628e+00   1.156057197 -0.183698859   0.467889358
    ## 55  128116  1.899709e+00  1.175602e+00  -2.614994495  3.973631492   1.771638413
    ## 56   89609  1.950240e+00  1.725565e-01  -1.615459105  1.415038234   0.474299906
    ## 57  141675  1.996191e+00 -1.485708e+00   0.692576187  0.012470671  -1.807120380
    ## 58  170284 -3.349289e-01  3.017590e-01  -0.397446814 -3.566535492   0.726077242
    ## 59  165176 -1.212496e+00  1.086628e+00  -0.958941386  0.529942820   1.623695696
    ## 60   74188 -3.049237e+00 -2.636419e+00   1.073319451 -0.755853500   1.033703229
    ## 61   65585 -5.189784e-01  8.315755e-01   1.480315686 -0.153282769   0.263715502
    ## 62   81832 -1.382837e+00  1.329345e+00   0.260763561  1.251569851  -0.784284533
    ## 63   74913  2.273948e-01  1.003535e+00   0.881593800  1.374974329   0.370615591
    ## 64  133410  1.848192e+00 -5.119499e-01  -1.626595275  0.005299777   0.876208186
    ## 65   32403 -3.639705e+00 -1.901409e+00  -0.093032543 -0.890671021  -0.044604205
    ## 66   81577 -7.236870e-01  1.345032e+00   1.223205185  1.380735079  -0.100249846
    ## 67   78386 -4.560964e+00  3.907460e+00  -2.273611915  0.183544436  -1.102063768
    ## 68  150420  2.031012e+00 -1.398974e-01  -1.217156650  0.215844604   0.093016020
    ## 69  153183 -1.315625e-01 -3.762768e-01  -1.581777979 -1.777891839   0.700427481
    ## 70  133533  2.636023e-02  8.137268e-01   0.149616560 -0.805512389   0.699185820
    ## 71  122666 -1.590950e+00  5.849244e-01   1.285604914 -1.304223053   1.434943884
    ## 72   73499 -4.071000e-01  1.015652e+00   1.355589485 -0.141228443   0.208762915
    ## 73  133348  1.697060e+00 -1.162893e+00  -0.707966014 -0.562947164  -0.806092379
    ## 74  126801 -1.323509e+01 -1.148302e+01  -2.748735660 -0.014577507   4.637719918
    ## 75  165072  2.080173e+00  1.830963e-01  -2.037530820  1.152333667   0.946377241
    ## 76   32612 -1.040885e+00  4.317795e-01   1.439919407 -0.467533687   0.787429946
    ## 77  147310 -1.311219e+00  7.395631e-01   1.935335988  0.791780452  -0.389954046
    ## 78  132702  2.923060e-01  5.122586e-01  -1.340612737 -0.312575190   1.670539896
    ## 79  128509 -2.079904e-01  1.302827e+00  -1.187219349  0.781410005   0.892146290
    ## 80  126065  1.993446e+00 -1.900934e-01  -0.257439943  0.392834403  -0.602088406
    ## 81   53810 -1.065254e+00  1.524620e+00  -0.022427024  1.238036988  -0.128125390
    ## 82   11041 -2.566192e+00 -2.189016e+00   1.153200712  1.433477873  -2.093935998
    ## 83  166622  8.301894e-01 -3.132904e+00  -1.345380283  0.014552143  -1.632451474
    ## 84  146589  1.848914e+00 -7.400469e-01   0.138818713  0.312893332  -1.194688985
    ## 85  134461 -6.075836e-01  6.103921e-01   0.235915172 -0.079636854   0.893305912
    ## 86   44216  1.104171e+00  2.374410e-02   0.049770582  0.961536673   0.127051844
    ## 87  124115  1.966076e+00 -3.804893e-01  -1.778758895  0.053588407   0.395440868
    ## 88   63654 -1.434334e+00  2.254643e-01   1.695793082 -0.958573088  -0.040823259
    ## 89   30799 -4.288370e+00 -2.161113e+00   0.778907557 -0.185368904  -1.981949287
    ## 90  128117 -1.179851e-01  1.002102e+00  -1.044349209 -0.217037070   0.369501372
    ## 91  122765  1.890889e+00  3.012609e-01  -0.148633320  4.060890043  -0.073022313
    ## 92   45763  1.387000e+00 -1.224671e+00   0.989313062 -1.369674631  -1.988085848
    ## 93  149044  2.225567e+00 -1.571464e+00  -1.467809386 -1.881785520  -0.587303339
    ## 94  153231  2.184668e+00 -1.806994e+00   0.378459033 -1.391824319  -2.406381859
    ## 95   28535  1.237891e+00  3.387452e-01   0.201965967  0.494319936  -0.101308471
    ## 96  151483  1.960452e+00 -6.038079e-01  -0.424743015  0.301552295  -0.672196997
    ## 97   33469 -7.002204e-01  7.408074e-01   1.796961638 -0.281169214   0.601205807
    ## 98   48792  1.122088e+00  2.363148e-01   0.537681020  1.314211854  -0.124622630
    ## 99   67238 -7.993358e-01  1.051808e+00   1.894249582 -0.586359585   0.144938577
    ## 100 146375 -1.157386e+00  1.280056e+00   2.393901423  2.027639503   0.754739076
    ## 101 131341 -4.691161e+00  3.703992e+00  -3.459048637 -2.599661003  -2.020959235
    ## 102 153683 -6.407069e-01  1.279044e+00  -0.111683583 -0.980309391   0.398272280
    ## 103 144607 -1.208154e+00  4.295481e-01   0.951727709 -0.675106580   0.710642223
    ## 104 125770  2.120467e+00 -1.593830e+00  -1.086153227 -1.754320512  -0.863350439
    ## 105  65200 -1.036817e+00  9.260578e-01   0.907319883  1.514656651  -0.585100083
    ## 106  54882  5.818809e-01 -1.620714e+00  -0.040074070  0.422224662  -1.118175858
    ## 107 151937  1.983655e+00 -3.207716e-01  -0.383010427  0.396313726  -0.414915552
    ## 108  76281  2.427205e-01 -3.001223e+00  -0.388408933 -0.542644894  -2.261294689
    ## 109  48238 -1.485428e+00  1.120805e+00   0.374645787  0.200132866  -0.536143922
    ## 110 149318 -8.511225e-01  5.523098e-01  -0.562243248 -0.811889366   1.348534721
    ## 111  29536  5.048635e-01 -1.253485e+00   1.233691587  1.838676369  -1.510599052
    ## 112  56760 -3.576066e-01  9.064471e-01   1.001369154 -0.460283223   0.577677022
    ## 113 166293  4.719907e-02  8.781789e-01   0.302230380 -0.585009234   0.415469273
    ## 114 110293 -5.604321e-01  2.239672e-01   1.515165337  0.835285134   0.697757703
    ## 115 148068  1.400843e-01  1.019986e+00  -1.026838683 -0.701669631   1.403962180
    ## 116  54719 -8.090949e-01  9.342338e-01   0.944932610 -0.967247196  -0.520957629
    ## 117  80678 -1.360859e+00  1.948469e-01   2.030191491 -0.970126394  -0.435249995
    ## 118 141282  2.011583e+00  1.051928e-01  -1.608585345  0.335710115   0.418423782
    ## 119  32739  1.135656e+00 -5.331906e-01   0.781505832  0.579755536  -1.004386664
    ## 120  74725 -6.781451e-01  1.261277e+00   1.576429820  1.649885241   0.856567198
    ## 121 148877 -8.464899e-01  7.800879e-01  -1.563694261 -0.631671939   1.109343215
    ## 122  99306  1.962469e+00 -3.599301e-01  -0.222852390  0.208402780  -0.401012157
    ## 123 118119  2.052499e+00 -1.208112e-01  -1.245394558  0.189701484   0.137668951
    ## 124 169157 -3.075086e+00 -5.675594e-01  -0.410428178 -0.489773320  -1.640342002
    ## 125  92847  1.333414e+00 -1.407437e+00  -0.882359682  0.328666931  -0.123233555
    ## 126  47920  1.201893e+00  1.325632e-01   0.598673689  0.562885452  -0.611017044
    ## 127 126623 -1.769198e+00  4.524812e-01   0.116810841 -3.145848200  -1.172130428
    ## 128  53344 -2.000396e+00  1.301436e+00   0.199130816 -1.248673011   1.989515742
    ## 129  18022 -8.582587e-01  3.150840e-01   2.484181786 -1.879289817  -0.237476286
    ## 130  68445  1.159066e+00  1.641551e-02   1.130007831  1.071428288  -0.664967049
    ## 131  63463 -5.733378e+00 -3.970107e+00   0.910780543  2.468688212   2.507168380
    ## 132 139649 -1.081244e+00  9.998293e-01   1.610824700  1.002875449   0.128740024
    ## 133  33604 -1.162533e+00  9.060071e-01   1.359105664 -1.772635650   0.420093012
    ## 134  90352  1.106477e-01  7.607689e-01   0.712646354  1.095057603   0.281263647
    ## 135  90315 -3.118259e+00  1.928064e+00   0.435215682 -1.120806890  -0.717235921
    ## 136  31541  1.212009e+00  2.296211e-01   0.255919150  0.653909016  -0.278466147
    ## 137 152972  1.574679e-01  7.694330e-01  -0.519898587 -0.516675064   1.253510037
    ## 138  35504  1.283262e+00 -6.456905e-02  -0.065882065  0.042508987  -0.282300753
    ## 139  46927 -2.092354e+00  1.444260e+00   1.542575719  1.213319264  -1.963341078
    ## 140  13115  7.738209e-01 -6.327591e-01   1.165355771  2.065386346  -0.884607725
    ## 141  56261  1.164205e+00  1.462688e-01   0.265609371  0.542583940  -0.251797546
    ## 142  35452  1.182250e+00 -3.773448e-01   0.357811597  0.628970584  -0.866497528
    ## 143  64328 -1.485148e+00  5.857536e-01   1.052820179  0.628254717   0.490471593
    ## 144 133318  9.517809e-03  7.954545e-01   0.225040149 -0.602866629   0.508091238
    ## 145 143600 -4.681429e-01  5.972341e-01  -0.564267605 -1.895175800   2.535448340
    ## 146  72827 -1.574114e+00  1.903130e+00   1.157589221  1.538652302  -0.065232436
    ## 147  47875  9.913374e-01 -4.535612e-01   0.464315264 -0.279089297   0.079751466
    ## 148 165953 -5.902771e-02  1.027853e+00  -0.657426911 -1.113549350   1.013434420
    ## 149  48025 -9.812448e-01  4.886997e-01   0.150356344  0.867900398   1.193719885
    ## 150  67747  5.831709e-01 -1.271850e+00   0.937895285  0.506191394  -1.393395679
    ## 151  72377 -7.150484e-01  1.143525e+00   0.925032591 -0.383776413   1.353816454
    ## 152  40207 -6.999470e-01  6.760236e-01   2.079747453 -0.216417573  -0.321272576
    ## 153 150629 -3.118927e-01 -6.133155e-01   0.675842815 -0.276259440   2.398068549
    ## 154  33673 -2.674696e+00  8.893171e-02   1.230011004  0.740628574  -0.296536563
    ## 155 144202  1.917008e+00 -1.009731e+00  -2.759173475 -0.915057279   0.309592212
    ## 156  76586  1.299635e+00 -5.661398e-01   0.108296171 -0.825905761  -0.741582389
    ## 157 159831  2.015174e+00 -5.359864e-01  -0.506141817  0.331959918  -0.546343142
    ## 158  93609  2.214016e+00 -5.840305e-01  -1.238673035 -0.702993204  -0.184491521
    ## 159  67153 -4.138552e-01  6.528119e-01   0.625965460 -1.486860785  -0.208617206
    ## 160  48279 -8.299422e-01  4.336018e-01   0.037026120 -1.576890799   3.112217304
    ## 161 112190  1.961965e+00 -4.986258e-01  -0.519509504  0.241051365  -0.377392424
    ## 162  71324 -1.632811e+00  1.017604e+00   1.503123604  0.538405955  -1.533914756
    ## 163 168831 -1.268354e+00  1.771130e+00  -2.092199695 -0.086837982   1.280314975
    ## 164 131356  1.893527e+00 -1.277486e+00  -0.590607745 -0.324482565   0.856818695
    ## 165  73958  1.423808e+00 -3.775681e-01   0.242070053 -0.697072573  -0.763103804
    ## 166 125443  2.053050e+00 -6.224043e-02  -1.067222480  0.409487428  -0.115800447
    ## 167 156505 -4.265742e-01  2.107531e+00   0.138663158  4.410433071   0.796098886
    ## 168 128252  1.978667e-01  1.565890e-01   1.140264140 -0.800159990   0.241697234
    ## 169 106121 -1.241093e+00  9.370220e-01   1.771786615 -1.357937830  -0.073938330
    ## 170  51367 -1.373284e+00  1.138689e+00   1.292812841  0.882135507  -0.657962138
    ## 171  83821 -7.834611e-01  2.778028e-01   1.523711356  1.342395367   0.814007242
    ## 172  49317 -2.871965e+00  2.030094e+00   0.726023719 -0.577497690   0.107638461
    ## 173 155464 -9.017785e-01  7.659580e-01   0.117486010  0.277408468   0.704608718
    ## 174  40321 -9.590946e-01  5.284076e-01   1.083109769 -0.055784922   0.752203330
    ## 175  35284  1.334214e+00 -6.975346e-01   0.567067139 -0.676827897  -1.175570966
    ## 176 126485  1.646966e+00  9.575514e-02   0.021831493  3.477538585   0.613306130
    ## 177  84106  1.250685e+00 -2.674170e-01   1.354316280  0.572027729  -1.477070666
    ## 178 135529 -4.770012e-01  9.516470e-01  -0.553025850 -0.783022247   0.691326367
    ## 179 168757 -5.938225e+00  2.791712e+00  -4.243521008 -1.483604896  -1.245294351
    ## 180  81924 -4.207746e-01  1.042279e+00   1.561025961 -0.029471058  -0.077784350
    ## 181  61115  1.254972e+00 -4.840398e-01   0.392819718  0.001657046  -0.681518838
    ## 182 151971 -5.023550e-01  8.619497e-01  -0.618120887 -0.790276876   0.582704769
    ## 183  48607 -1.893402e+00  7.135598e-01   1.280242015 -0.513232457  -0.851571175
    ## 184 119202  2.071844e+00 -9.809687e-01  -1.355910227 -0.992165250  -0.563328168
    ## 185 141807  2.048635e-01  8.500142e-01   1.421918249  4.424811577   0.367293735
    ## 186  44358  1.152442e+00  4.738855e-02   1.307477038  1.332618659  -0.883447906
    ## 187  33537  1.301332e+00 -5.814182e-01   0.521451671 -0.774068669  -0.985016085
    ## 188 139820 -3.715521e-01  1.008340e+00  -0.025209623 -0.625446135   0.080303405
    ## 189 170088 -1.442674e+00  1.780766e+00  -1.319956594 -0.966095549   0.745734230
    ## 190  66138  1.107107e+00 -1.673324e+00   0.153202167 -1.343151022  -1.561232303
    ## 191 128615  2.014261e+00 -1.274115e-01  -1.012973996  0.325929982  -0.169881945
    ## 192  54794 -2.645887e+00  8.408551e-01  -0.317445475  1.675026196  -1.399599091
    ## 193  42821  1.052496e+00  7.441743e-01  -0.113602525  2.575407403   0.344062547
    ## 194  77371  9.973947e-01 -1.395239e+00   1.158503868 -0.387827150  -2.092896079
    ## 195 164657  1.846726e+00 -1.123322e-01  -1.804236288  0.329857141   0.349709839
    ## 196  70369  1.040681e+00 -4.060935e-01   0.570590498  0.464307907  -0.650606203
    ## 197 127105 -4.799111e-01 -9.725573e-01  -0.390000280 -0.946454508  -1.269422962
    ## 198 159085  8.036103e-01 -4.463475e-01   0.588159239 -0.956774338  -0.983016770
    ## 199 137522 -3.471769e+00 -3.598909e+00   0.465800739  0.768359764   5.373888799
    ## 200 170487  2.098995e+00 -1.210726e+00  -0.574813265 -0.743944549  -1.362361003
    ## 201  57996  8.070502e-01 -5.357136e-01  -0.098805833  1.253575165  -0.360029855
    ## 202 169423  1.688495e+00 -7.514388e-01  -0.848984892  0.028590696   0.395737992
    ## 203 158645 -3.163604e+00 -3.250637e+00   2.782214128  2.120590634  -0.169067390
    ## 204  38295 -1.841637e+00  5.952891e-01   0.501049521 -0.766244870  -0.835496696
    ## 205  61520 -1.021140e+00  3.789692e-01   2.571491694 -0.532546857  -0.179431599
    ## 206  79714 -6.056543e-01  9.843489e-01   1.285845058  0.347946609   0.382060842
    ## 207  45866  1.286008e+00  7.815263e-02  -1.305352172 -0.054094483   2.250596265
    ## 208 160632  1.230882e+00 -1.679696e+00  -1.764674655  0.283384541  -0.254127310
    ## 209  54234 -3.763040e-01  7.727119e-01   1.286331033  1.165357986   0.292659256
    ## 210 111149 -1.456138e-01  1.101585e+00   0.802175890  0.692516052   0.830992616
    ## 211 149059 -1.024396e+00  5.957052e-01   0.806142630  0.804240340   0.079171033
    ## 212 167949  2.111083e+00  1.792638e-01  -1.669960090  0.144261805   0.624253905
    ## 213 133280  1.932479e+00  4.881279e-01  -1.288387510  3.656564632   0.772239401
    ## 214  37360 -4.255211e-01  1.247598e+00   0.895044381  0.795528895  -0.150126029
    ## 215  71225  1.502188e+00 -1.094812e+00   0.270838249 -1.546237688  -1.221617804
    ## 216  33947 -7.340361e-01 -4.996595e-01   1.864176553  1.379820423   0.540870214
    ## 217 109264 -4.990401e-01  7.522399e-01   2.381407689 -0.107221296   0.055775501
    ## 218  88105  2.108345e+00 -2.274118e-02  -2.870251214 -0.229287241   1.197184290
    ## 219 134409 -1.680407e-01  6.544299e-01  -0.130369964 -1.426358634   0.995709631
    ## 220  87390 -1.686834e+00  4.652961e-01  -2.286724397 -2.525937569   0.295719568
    ## 221 142231  2.023752e+00  1.672337e-01  -1.581823108  0.626308032   0.084077763
    ## 222  83059 -3.291300e+00  1.562496e+00   0.122639755  1.296378455  -0.943518806
    ## 223 158989  1.211247e-01  1.010865e+00  -0.523092134 -0.755808605   1.228693047
    ## 224  69654 -1.301944e+00 -8.571578e-01   2.450737825 -1.752033277  -0.276135556
    ## 225 166508 -1.167656e+00  1.402584e+00  -0.850168024  1.136006899   0.121775112
    ## 226 170445  1.183146e+00 -1.586077e+00  -0.895090657  0.592781471  -0.816604769
    ## 227  38056  8.785659e-01 -2.015811e+00   0.122528863 -1.275673885  -1.720609458
    ## 228  66472 -2.662102e+00 -3.457437e+00   2.064667724  1.036797966  -0.561607241
    ## 229 131013  2.270523e+00 -1.218502e+00  -1.903340114 -1.527699866  -0.751103976
    ## 230  62867  1.200942e+00 -2.243045e-01   0.212782402  0.276775029  -0.257219947
    ## 231 168272  1.859607e+00 -1.101859e+00  -1.037737291 -0.335998883  -0.669454123
    ## 232 149683 -1.243436e+00  1.361759e+00   1.337419574 -0.855033013  -0.268885722
    ## 233  75615  1.192218e+00  1.685921e-01   0.138626232  0.460967105  -0.062255952
    ## 234  39051 -3.120725e-01  1.987155e-01   1.352626539 -0.799471972  -0.537771706
    ## 235  35105  1.252907e+00  3.960499e-01   0.315355089  0.687475572  -0.311265084
    ## 236 123738  1.991533e+00 -2.517939e-01  -0.752112475  1.320307689   0.076959978
    ## 237 158766  2.208414e+00 -1.578291e+00  -1.318810134 -1.389831742  -0.996498956
    ## 238 126735  1.917637e+00 -4.683347e-01  -0.255767432  0.556941722  -0.812963639
    ## 239  64244  1.176643e+00  1.330391e-01   0.126779543  0.468812665  -0.082453170
    ## 240  38262  9.819645e-01  4.720953e-02   1.206764044  2.524675041  -0.301263897
    ## 241  54029  1.084039e+00 -1.319291e+00   1.046038240 -0.891594810  -1.506968985
    ## 242 162961  2.062714e+00  1.137733e-01  -1.739473646  0.414062200   0.412272546
    ## 243  86013  1.290823e+00  3.996432e-01  -0.209420507  0.715915377   0.713038361
    ## 244 157290  1.897320e+00 -8.302408e-02  -1.906916649  0.076942535   1.215639944
    ## 245  72255 -1.777569e+00  2.398110e+00   0.189887878  1.011235986  -1.226231526
    ## 246 162059 -1.713927e+00  2.153464e+00   0.739898612  0.967273535  -0.809945773
    ## 247 151746  1.492131e+00 -1.645904e+00  -2.833991325 -0.085901917  -0.260923777
    ## 248  27523  1.429635e+00 -9.556722e-01   0.285789685 -1.518285506  -1.162912032
    ## 249 169536 -3.473175e-01  6.627575e-01  -0.382771315 -0.554116787   1.202893461
    ## 250  89603 -9.199978e-01  1.325687e+00   0.573379376 -0.195699652   0.034158699
    ## 251  62313 -8.187029e-01  8.921309e-02   2.197168291  0.505677414  -0.126827754
    ## 252 165739  1.800850e+00  2.201260e-01  -2.427497080  2.617232552   3.030277181
    ## 253 125037  4.489031e-01 -1.752298e-01   0.606673622 -1.876915289   0.012497522
    ## 254 113630  2.131496e+00 -1.096824e-01  -1.770008868  0.023269718   0.471295522
    ## 255  81828  1.244959e+00 -8.145580e-02   0.313665372 -0.041528262  -0.686196549
    ## 256 157583  1.949438e+00  3.621886e-02  -1.579322945  1.320755213   0.307034115
    ## 257 157038 -1.292589e+00  2.003079e+00  -1.754551167 -0.891704602   0.702191291
    ## 258  34549  1.486332e+00 -5.374774e-01  -0.616151223 -1.039484603   0.071496955
    ## 259  56392 -4.507330e+00  1.533336e+00  -0.574713672 -1.852330908  -1.580105087
    ## 260  78749  1.183063e+00 -3.823166e-01  -1.767513167 -1.179691861   2.150011287
    ## 261 131965  2.039570e+00  5.932807e-01  -2.367412219  0.769417229   0.574944589
    ## 262  74061 -5.735311e-01  7.668192e-01   1.205977723 -0.687676517   0.669890097
    ## 263 126576 -1.292290e+00  2.814376e-01   1.260561329 -0.454388326  -0.347838934
    ## 264 147875  2.030665e+00 -1.284088e+00  -0.350091705 -0.648034035  -1.094408501
    ## 265 134564  1.980868e+00 -2.291704e-01  -0.879212957  0.164553098  -0.410166574
    ## 266  77911 -3.732276e+00  3.131195e+00  -0.961135647 -0.187914366  -2.042346365
    ## 267  58723 -3.761756e+00 -2.973654e+00   1.216834459  1.004886944   1.572241608
    ## 268 136718  2.364546e-01  7.587704e-01  -0.773480373 -0.514605904   1.260813074
    ## 269  66457  1.158908e+00  7.249499e-01  -0.222185190  1.539797197   0.103013646
    ## 270  30039 -6.155507e+00  5.042451e+00  -0.838337645 -1.382924592  -0.545593284
    ## 271  94056 -1.053098e-01  1.123662e+00  -0.422523879  1.056035983   1.174505490
    ## 272   4830  6.206945e-01 -1.108690e+00   0.971537136  0.276851875  -0.452815246
    ## 273  75212  1.223752e+00 -7.444086e-01   0.641712116 -0.338289011  -1.239984485
    ## 274 160776 -1.016759e+00  2.009928e+00  -2.240972364 -0.389833750   1.251211981
    ## 275  26537 -2.854262e+00  7.602876e-01   0.607677246  1.103538491   1.156284696
    ## 276     32 -2.008872e+00  2.198527e+00   0.144241740  1.159432262  -0.815174288
    ## 277  46646 -5.844844e-01  1.293593e+00   1.191508346  0.011018292   0.024710994
    ## 278  31037  3.401947e-01  7.448111e-01   1.265582563  1.160695798  -0.361814298
    ## 279 112882  1.837950e+00 -5.967204e-01  -0.483761562  0.141943369  -0.489952156
    ## 280 166061 -2.652649e+00  2.549303e+00  -0.304193269 -0.329644326  -1.064860663
    ## 281 127894  1.609287e+00 -5.814372e-01   0.008484303  1.655628907  -1.162795077
    ## 282  33663 -1.450335e+00  2.105961e-01   1.133294428 -2.289623901  -0.340782158
    ## 283  82076  1.258237e+00  4.546781e-01   0.335016390  0.679897477  -0.249883057
    ## 284  39870  1.108091e+00  7.273677e-02   0.403150408  1.292482117  -0.177263686
    ## 285  80371  1.410342e+00 -4.712006e-01   0.132596310 -0.913275591  -0.445017094
    ## 286  80005  1.168897e+00  2.524150e-01   0.381788148  1.063210898  -0.368592928
    ## 287 124307  1.892778e+00  3.853009e-01   0.038522997  3.661012334   0.126542675
    ## 288  81708 -4.310696e-01  1.038649e+00   1.575061920 -0.026925253  -0.016033535
    ## 289 134577  2.080683e+00  6.392270e-02  -1.789953783  0.793740009   0.895053949
    ## 290  75021  1.521725e+00 -1.360483e+00   0.050166391 -1.864197046  -0.916364765
    ## 291  92468  1.759077e+00 -4.746422e-01  -0.300981655  1.439678652  -0.539426011
    ## 292  71489  1.411657e+00 -1.251581e+00   0.883097400 -1.021452581  -1.621202450
    ## 293  24862  1.154741e+00  6.023570e-02   0.823156516  0.849649759  -0.523112113
    ## 294  53796  5.567653e-01 -9.722779e-01   1.008417090  1.499482224  -1.137941160
    ## 295  84623  1.222715e+00 -3.775544e-01   0.567249886 -1.106864085  -0.428326757
    ## 296  91460 -3.994579e-01  1.002002e+00   1.789405030  4.847826797   1.107257401
    ## 297 148879  1.899691e+00 -4.035571e-01  -0.805419523  1.358924793   0.030073815
    ## 298 147505  2.053311e+00  8.973465e-02  -1.681835669  0.454211960   0.298310371
    ## 299  48642 -1.148869e+00 -1.370327e-01   1.047798338  0.172192566   1.692773995
    ## 300  57926 -4.337117e-01  1.195248e+00   0.627633152  0.347447076   0.269848462
    ## 301  67739 -1.031278e+00 -1.032008e-01   2.964222925 -1.515721020  -1.524238373
    ## 302  41316  1.278010e+00  4.751097e-01  -0.120012619  1.110908246   0.001114148
    ## 303 164601  1.660756e-02  7.211655e-01   0.155821884 -0.763442289   0.524126533
    ## 304  77952 -1.136775e+00  1.844279e+00   0.128955423  1.006745038  -0.708505883
    ## 305 148465  1.881478e+00 -1.424681e+00  -0.961610895 -1.265361452  -0.112058107
    ## 306  71331 -5.459913e+00  5.339356e+00  -3.242363354 -0.267287664  -2.155951608
    ## 307  54215  1.433535e+00 -1.221115e+00  -0.592950943 -1.721415764  -0.615343299
    ## 308  51388  1.109552e+00 -4.319959e-01   0.563084076  0.535321035  -0.256280349
    ## 309  30303 -2.099830e+00 -1.440325e-01   1.634346890 -0.872671404   1.426993212
    ## 310  72147  1.244300e+00 -4.970748e-01   0.458962688 -0.197711673  -1.088947165
    ## 311 155999 -1.802183e-01  1.025301e+00  -0.369068205 -0.653619822   0.752444934
    ## 312  81074 -8.941063e-01 -4.665365e-01   1.591372639 -1.192388444  -0.130735349
    ## 313  41449 -1.060521e+00  3.343290e-01   2.248949054 -1.486245901  -1.413844699
    ## 314    478 -3.943639e-01  9.624889e-01   1.164111305 -0.110028409   0.221215518
    ## 315 125842  2.010155e+00  2.985726e-01  -2.492837102  0.590059605   0.994711348
    ## 316  33103 -8.401236e-01  1.754789e+00   1.715407063  2.536185791  -0.088251785
    ## 317 159963  2.036568e+00 -1.310986e-01  -1.184529873  0.229355002   0.050988511
    ## 318  65128 -5.177943e-01  1.033806e-01   2.215021257  0.836108676  -0.732709428
    ## 319  84904 -1.610580e+00  8.072377e-01   1.838030840  1.728429620   1.125186693
    ## 320   8610 -6.287958e-01  1.637571e+00  -0.259087024  0.021698910   1.060562230
    ## 321  82650  6.570523e-01 -1.356898e+00   0.072423157 -0.002943194  -0.517098038
    ## 322 120028  1.878311e+00 -1.138466e+00  -1.657854821 -0.664093530  -0.561163277
    ## 323  20783  1.185465e+00 -3.890741e-01   0.990192968 -0.554455155  -1.042981995
    ## 324  34484 -2.513570e+00 -3.226728e-01  -0.123144227 -0.185623551   0.236331496
    ## 325  60193 -7.631806e-01  3.586866e-01   2.127374601  2.414410641   1.215153600
    ## 326 125020 -2.412006e+00  2.795197e+00  -1.780936188  0.886047219   0.503349619
    ## 327 137419 -1.470919e+00  5.608427e-01   0.409841220 -0.223127609   0.676457004
    ## 328 119434 -6.944901e-01  3.511097e-01   1.558926927 -2.472953033   0.061635361
    ## 329 140440  1.870808e+00 -4.734990e-01  -0.243368108  0.402665455  -0.784594113
    ## 330 136811  2.053422e+00 -4.652036e-02  -1.092199040  0.391539766  -0.052377986
    ## 331  49251  1.469506e+00 -6.200352e-01  -0.221780397 -1.013860420  -0.299066666
    ## 332   1422 -4.874244e-01  8.491590e-01   1.458538017  1.102493430   0.392615122
    ## 333 161850 -5.784288e-01  7.650227e-01  -0.942130240 -0.958077905   1.000742479
    ## 334 164150  1.927603e+00 -5.953576e-01  -0.747030072  0.227593431  -0.251898915
    ## 335 146389  2.149217e+00 -8.205813e-01  -1.550611782 -0.791468324  -0.381514679
    ## 336 138374  2.045430e+00  1.820099e-01  -1.692724641  1.299718295   0.556925077
    ## 337  39720  1.018947e+00 -7.389270e-02   0.272303055  1.136632857  -0.543110233
    ## 338   2149  1.030228e+00 -6.554631e-02   0.394997965  0.295957478   0.319119586
    ## 339  43073 -1.375650e-02 -3.954110e-01   2.157806437  0.875812958  -1.514224232
    ## 340  71383 -1.873640e+00 -6.845764e-01  -0.604066147  0.270137558   2.775091006
    ## 341  47045  1.201135e+00  5.769436e-02   0.577186181  0.572927776  -0.703247421
    ## 342  53183 -3.294421e+00  2.012037e+00  -0.638099965 -1.216178730  -1.159611544
    ## 343  24510  1.111137e+00  5.804507e-02   0.017381973  1.249269147   0.478712515
    ## 344  95783 -3.733290e-01 -7.509688e-02   1.569139725 -2.118489062  -0.320586512
    ## 345  74087 -1.825653e+00  1.816481e+00  -0.007962603  0.367608141  -0.901568304
    ## 346 138768  1.590475e+00 -1.069827e+00  -0.426451449  1.621572121  -0.727993538
    ## 347   1540 -7.959906e-01  1.027158e+00   0.724285551 -0.097385871  -0.072521846
    ## 348  66633  1.068276e+00 -1.115399e-01   1.079593963  1.040111445  -0.595620744
    ## 349 137844  1.506199e+00 -2.249417e+00  -1.070650044 -0.883957940  -1.271794104
    ## 350  66980  1.016889e+00 -1.979448e-01   0.715649455  0.923623511  -0.668545993
    ## 351 154698 -6.684902e-01  4.592193e-01   0.324870191 -0.047354109   0.930590907
    ## 352 160056 -6.232725e-01  1.452837e+00  -0.976728219 -1.407319363   1.272034914
    ## 353 123592  2.004735e+00 -8.616859e-01  -1.208193070 -0.762669278   0.399800069
    ## 354  67759  1.159118e+00  2.245626e-01   0.409934200  1.391788606  -0.117461292
    ## 355 132788  1.581827e-01 -3.937085e+00  -3.631226351  0.498912420  -1.084870637
    ## 356 159469  1.976832e+00  1.619458e-01  -1.350047076  0.759081126  -0.348630648
    ## 357  23404 -1.160569e+00 -9.373390e-01   2.067620666 -1.622826319  -2.595437553
    ## 358  67704 -1.374367e+00  3.605908e-01   1.382046665 -0.247017943   0.971973429
    ## 359 130089 -2.787296e-01  1.603146e+00  -0.023562760  2.730092202   1.469221136
    ## 360 156339  2.584676e-01  1.177416e+00  -1.249790311 -0.450162743   0.711805640
    ## 361  29568 -9.827974e+00  5.733219e+00 -10.630811417  2.060822664  -5.622900856
    ## 362 128938 -4.510495e-01  1.364618e+00  -1.470157574 -1.358746184   1.297760118
    ## 363  63993 -3.150128e+00  3.730753e+00  -1.831957351  0.146298531  -0.636662632
    ## 364  71132 -6.284926e-01 -4.195523e-01   1.742067955 -2.587676785  -0.754985918
    ## 365 126646  2.018624e+00 -1.852957e+00  -0.826632203 -1.577474274  -1.418391477
    ## 366  69230 -5.960578e-01  1.448923e+00   0.158731527  0.305992316   0.296198468
    ## 367 156909 -9.064027e-01  5.535063e-01   1.462519001 -1.154893859  -0.522526404
    ## 368  52744 -1.288156e+00  6.350623e-01   0.776408255  0.289187850  -0.737805847
    ## 369 125880 -3.483907e-01  1.199641e+00  -0.873374650 -1.260536476   1.124230001
    ## 370 142285  3.984420e-02  7.118127e-01   0.249243662 -0.573963566   0.245478025
    ## 371  42598  1.301261e-01 -1.589452e-01   0.675770358  1.560976859  -0.050242589
    ## 372 159763  2.023952e+00 -1.201398e-01  -1.086918335  0.423019142  -0.142901392
    ## 373  76914  1.106163e+00 -1.049702e+00   0.826138869  0.124546701  -1.065545670
    ## 374 134514 -1.399361e+00 -4.499784e-01   1.835165719  0.354127729   1.081690455
    ## 375  34882  7.235218e-01 -1.705924e+00   1.311958493  0.252715455  -1.826327505
    ## 376 153095  1.715705e+00 -1.360814e+00   0.352142473  0.845746050  -1.736453435
    ## 377  63647 -1.094511e+00  1.596381e+00   1.054455274 -0.256279856  -0.017120640
    ## 378  48533 -1.321038e+00  4.472232e-01   0.534272678 -0.509824251   1.633673425
    ## 379  55879 -8.546619e-01  1.087996e+00   1.268830472 -0.028802960  -0.532590635
    ## 380 142413  1.928808e+00 -1.081309e+00  -2.342597479 -0.934884308  -0.092359273
    ## 381 121064 -1.212766e+00  1.895607e-01  -0.377412346 -0.379931967   0.538022576
    ## 382  68775  1.088110e+00 -1.387686e-01   1.372240665  1.284381107  -1.089129200
    ## 383 143656  1.883975e+00 -2.009208e-01  -2.370848435  0.611977290   0.257184402
    ## 384  19295 -9.255912e-01  1.504134e+00   1.316442531  0.685463694   0.964491337
    ## 385  81183  8.701653e-01 -1.191976e+00   0.153230271  0.326683096  -0.648765133
    ## 386 147586  2.150996e+00 -7.050897e-02  -2.571992683 -0.338699580   1.002008782
    ## 387  74553 -1.889596e+00 -1.443024e-01   0.192740508  1.967545379   0.590804751
    ## 388 127878 -1.601924e+00 -1.303200e+00   0.826623062 -2.777827571  -0.065328875
    ## 389 123556  2.160688e+00 -1.495806e+00  -0.334390258 -1.426840627  -1.692699207
    ## 390 123467 -7.106563e-01  7.476465e-01   0.550058554 -0.061651784   2.537722446
    ## 391 122742  1.991981e+00 -3.348399e-01  -0.298982459  0.412249562  -0.780400468
    ## 392 171392  2.181622e+00  3.114773e-02  -2.525391140 -0.172394990   0.899295369
    ## 393 147566  2.085899e+00  1.824785e-01  -1.760422710  0.377231796   0.526847389
    ## 394  62762 -1.372022e+00 -8.083251e-01  -0.233484700 -0.915872569   1.561514536
    ## 395 126292  2.286370e+00 -1.466646e+00  -1.019927561 -1.351897124  -0.987040821
    ## 396 128314 -9.052525e-01  1.232731e+00   0.683559275  0.012652376   1.358511211
    ## 397 136079 -5.582898e-01  6.423635e-02   0.878778286 -0.636425296   1.144913337
    ## 398  75157  1.429532e+00 -1.111379e+00   0.155865009 -1.433548079  -1.184087734
    ## 399 156824 -4.790999e-01 -7.753424e-02   0.404027764 -0.666042655   1.264095741
    ## 400  99524 -3.180641e-01  1.285166e+00  -2.421091517  1.400346265   1.157746432
    ## 401  78249 -2.925684e+00 -3.615459e+00  -0.816797723  1.357305941  -4.703545648
    ## 402 133691  1.845377e+00 -6.133237e-01  -1.465959802 -0.086973920   0.602056675
    ## 403  37793 -8.857761e-01  8.635018e-01   0.148509904 -1.590470666   0.051612857
    ## 404  58435 -8.368557e-01  3.783131e-01   0.654166356 -0.910719385  -0.849017173
    ## 405   2625  1.345717e+00 -1.328331e+00   0.624538776 -1.371451244  -1.671558603
    ## 406  42037 -2.562775e-01  7.930207e-01   1.146419838  0.333752268   0.569880407
    ## 407  39333  1.473038e+00 -1.315002e+00  -1.607732022 -2.453200717   1.182910293
    ## 408 133087 -1.353985e-01  1.321319e+00  -0.309917192  1.280498740   0.995281458
    ## 409  31517 -3.946460e-01 -2.766880e-01   1.866060099 -0.743387586  -1.166050503
    ## 410  54023  1.028700e+00  1.237286e-01   1.204479377  2.762143401  -0.681905949
    ## 411  34772 -9.195749e-01 -2.286669e-01   1.815836620 -1.801305780  -1.877123742
    ## 412 145848 -2.926714e-01  8.621597e-01  -1.685425903  0.036193656   2.933736589
    ## 413 132701  2.274438e+00 -5.904332e-01  -2.721125328 -1.132776956   0.505378665
    ## 414  33575 -1.037308e+00  5.340446e-01   2.817314271  0.310641733   0.020676422
    ## 415  55688  1.244819e+00 -6.015205e-01   0.642891719 -0.690740140  -1.159491686
    ## 416  66062 -4.241271e-01  1.003839e+00   1.278886973 -0.100727120   0.237322964
    ## 417 122109 -1.112399e+00 -8.417240e-01  -0.017257459 -1.464624385   1.463253757
    ## 418  77522 -2.635915e-01  1.135931e+00   0.735941551  1.593792710  -0.365747542
    ## 419 129804  1.791803e+00 -1.636258e-01  -2.295408044  1.026397495   0.966209730
    ## 420 148310 -1.780533e+00  2.440129e+00  -2.275577992 -1.563747641   0.722879846
    ## 421  47703 -4.242602e+00  3.599164e+00  -1.191606204 -1.851656700  -1.823437209
    ## 422 127724  4.275760e-01 -3.806555e+00  -0.913445159  0.142946276  -2.235176411
    ## 423 170347 -1.375801e+00  2.073836e+00  -1.649878454 -1.449256827   0.816406139
    ## 424  14137  1.180510e+00 -1.532723e-01   1.076073227  0.822500210  -0.810890950
    ## 425 111643  2.084172e+00 -7.301011e-02  -1.482034977  0.179258004   0.165400572
    ## 426  56178 -9.592486e-01  1.062826e+00   1.285878630 -0.634271211  -0.426257491
    ## 427 124940  1.948766e+00 -6.747144e-01  -2.623888822 -0.823123014   2.235466476
    ## 428 121451  2.296559e+00 -1.489055e+00  -0.854611253 -1.895250523  -0.972907696
    ## 429 169450 -6.882669e-01  1.138649e+00  -0.495103473  0.899796162   0.900568449
    ## 430  55601 -4.462008e-01  9.474255e-01   1.133693270 -0.163162537   0.425228883
    ## 431  75512 -6.850427e-01  1.183173e+00   1.216015399  3.176928118  -0.280602063
    ## 432 111134  2.103431e+00  5.546652e-03  -1.445844937  0.331573391   0.319787209
    ## 433  34169  1.472281e-01 -2.039569e+00   0.586144812  0.592302412  -1.270351995
    ## 434 169711  1.569951e+00 -2.948239e-01  -3.491982316  0.583723405   1.076294905
    ## 435  79463 -2.202675e+00 -8.925929e-01   0.436432446  1.523906789   0.764081817
    ## 436  60980 -5.996144e-01 -4.165952e-01   1.612385504 -2.678515260  -0.303768456
    ## 437 123061  2.112413e+00  6.597460e-01  -2.817815048  0.476289586   1.276957872
    ## 438 161484  1.511034e+00 -2.440701e+00  -0.242735431 -0.461434593  -1.864950857
    ## 439  44050 -2.147298e+00  8.741986e-01  -1.213386816  2.049722471  -5.087206789
    ## 440  60297  4.726037e-01 -1.093629e+00  -0.207801977  0.924707228  -0.734267507
    ## 441 148758 -3.829565e+00  2.936362e+00  -3.799693869 -2.711412493   2.894934869
    ## 442  42211  1.152090e+00 -5.654655e-01   0.092185467 -0.510819406  -0.270639666
    ## 443  48160 -3.373360e-01 -7.143417e-01   2.096708964 -1.497648896  -1.944234242
    ## 444 145620  2.060821e+00  8.049633e-02  -3.272485031  0.106516593   3.098994239
    ## 445 169068 -8.923540e-01  1.372071e+00  -0.092896978 -0.624192239   0.346116154
    ## 446  67781 -8.271468e-01 -2.798842e-01   2.916035972 -1.595125765  -1.066542862
    ## 447 138134  2.162358e+00 -8.030282e-01  -1.985566784 -0.854043122  -0.053192731
    ## 448  52844 -2.658239e+00 -3.890428e+00   1.763330179  0.895355135   1.731253673
    ## 449  14888 -1.642052e+00 -1.141346e+00   1.750667767  2.198730625  -2.147557121
    ## 450  35731 -8.321590e-01  7.811489e-01   1.809976181 -0.159888912   0.270252496
    ## 451 118131 -4.365182e-01  6.194367e-01   0.159033379 -0.824799861   0.943902862
    ## 452 169040  6.593057e-01  8.236680e-01   0.103199690  1.037706072   0.476198616
    ## 453  85685 -1.134880e+00  5.760268e-01   0.837752604 -3.755985379  -0.305642600
    ## 454  79378 -8.649296e-01 -1.136927e-01   2.468061293 -0.203700698  -0.333564120
    ## 455 163880  5.317235e-02  6.421888e-01   0.097700522  1.906111188   1.412519231
    ## 456    406 -2.312227e+00  1.951992e+00  -1.609850732  3.997905588  -0.522187865
    ## 457    472 -3.043541e+00 -3.157307e+00   1.088462780  2.288643618   1.359805130
    ## 458   4462 -2.303350e+00  1.759247e+00  -0.359744743  2.330243051  -0.821628328
    ## 459   6986 -4.397974e+00  1.358367e+00  -2.592844218  2.679786967  -1.128130942
    ## 460   7519  1.234235e+00  3.019740e+00  -4.304596885  4.732795130   3.624200831
    ## 461   7526  8.430365e-03  4.137837e+00  -6.240696572  6.675732163   0.768307025
    ## 462   7535  2.677923e-02  4.132464e+00  -6.560599968  6.348556673   1.329665669
    ## 463   7543  3.295943e-01  3.712889e+00  -5.775935108  6.078265506   1.667359013
    ## 464   7551  3.164590e-01  3.809076e+00  -5.615159011  6.047445102   1.554025957
    ## 465   7610  7.256457e-01  2.300894e+00  -5.329976183  4.007682805  -1.730410590
    ## 466   7672  7.027099e-01  2.426433e+00  -5.234513296  4.416661243  -2.170806216
    ## 467   7740  1.023874e+00  2.001485e+00  -4.769751832  3.819194585  -1.271754227
    ## 468   7891 -1.585505e+00  3.261585e+00  -4.137421983  2.357096252  -1.405043314
    ## 469   8090 -1.783229e+00  3.402794e+00  -3.822742266  2.625368153  -1.976415411
    ## 470   8169  8.573210e-01  4.093912e+00  -7.423893589  7.380244518   0.973366023
    ## 471   8408 -1.813280e+00  4.917851e+00  -5.926129694  5.701500431   1.204392572
    ## 472   8415 -2.514710e-01  4.313523e+00  -6.891437717  6.796796681   0.616296505
    ## 473   8451  3.145966e-01  2.660670e+00  -5.920037380  4.522499605  -2.315027286
    ## 474   8528  4.473956e-01  2.481954e+00  -5.660813931  4.455922812  -2.443779754
    ## 475   8614 -2.169929e+00  3.639654e+00  -4.508497786  2.730668149  -2.122692871
    ## 476   8757 -1.863756e+00  3.442644e+00  -4.468259731  2.805336259  -2.118412477
    ## 477   8808 -4.617217e+00  1.695694e+00  -3.114372201  4.328198553  -1.873256991
    ## 478   8878 -2.661802e+00  5.856393e+00  -7.653616213  6.379741881  -0.060712119
    ## 479   8886 -2.535852e+00  5.793644e+00  -7.618462836  6.395829746  -0.065210105
    ## 480   9064 -3.499108e+00  2.585552e-01  -4.489558073  4.853894351  -6.974521545
    ## 481  11080 -2.125490e+00  5.973556e+00 -11.034727227  9.007146872  -1.689450534
    ## 482  11092  3.782745e-01  3.914797e+00  -5.726872015  6.094140959   1.698875264
    ## 483  11131 -1.426623e+00  4.141986e+00  -9.804103442  6.666273003  -4.749526793
    ## 484  11629 -3.891192e+00  7.098916e+00 -11.426467097  8.607556787  -2.065706210
    ## 485  11635  9.191365e-01  4.199633e+00  -7.535606594  7.426940375   1.118215330
    ## 486  12093 -4.696795e+00  2.693867e+00  -4.475132713  5.467684549  -1.556758075
    ## 487  12095 -4.727713e+00  3.044469e+00  -5.598354267  5.928190802  -2.190769729
    ## 488  12393 -4.064005e+00  3.100935e+00  -1.188498004  3.264632740  -1.903562358
    ## 489  12597 -2.589617e+00  7.016714e+00 -13.705407287 10.343228060  -2.954461035
    ## 490  13126 -2.880042e+00  5.225442e+00 -11.063330280  6.689950658  -5.759923989
    ## 491  13323 -5.454362e+00  8.287421e+00 -12.752811273  8.594341893  -3.106002281
    ## 492  14073 -4.153014e+00  8.204797e+00 -15.031714210 10.330099825  -3.994426097
    ## 493  14152 -4.710529e+00  8.636214e+00 -15.496222004 10.313349373  -4.351340983
    ## 494  15817 -4.641893e+00  2.902086e+00  -1.572938709  2.507298518  -0.871782564
    ## 495  17187  1.088375e+00  8.984740e-01   0.394684329  3.170257575   0.175738797
    ## 496  17220  1.189784e+00  9.422894e-01   0.082334145  3.024049706   0.412406009
    ## 497  17230 -4.693267e-01  1.111453e+00   2.041002725  1.731595163   0.135146842
    ## 498  17520 -5.268053e+00  9.067613e+00 -15.960728135 10.296602790  -4.708241091
    ## 499  17838 -5.187878e+00  6.967709e+00 -13.510931139  8.617895141 -11.214422394
    ## 500  18088 -1.222402e+01  3.854150e+00 -12.466765725  9.648310727  -2.726961321
    ## 501  18399 -1.447444e+01  6.503185e+00 -17.712632369 11.270352333  -4.150142064
    ## 502  18675 -1.233960e+01  4.488267e+00 -16.587072937 10.107273873 -10.420198983
    ## 503  18690 -1.539885e+01  7.472324e+00 -19.026912343 11.165525841  -6.893856280
    ## 504  19762 -1.417917e+01  7.421370e+00 -21.405835744 11.927511869  -7.974280687
    ## 505  20011 -1.472463e+01  7.875157e+00 -21.872317364 11.906169908  -8.348733692
    ## 506  20332 -1.527136e+01  8.326581e+00 -22.338590513 11.885312892  -8.721334356
    ## 507  20451 -1.581918e+01  8.775997e+00 -22.804686461 11.864868080  -9.092360532
    ## 508  20931 -1.636792e+01  9.223692e+00 -23.270630523 11.844776586  -9.462037146
    ## 509  21046 -1.691747e+01  9.669900e+00 -23.736443411 11.824990230  -9.830548229
    ## 510  21419 -1.746771e+01  1.011482e+01 -24.202142233 11.805469211 -10.198045808
    ## 511  21662 -1.801856e+01  1.055860e+01 -24.667741249 11.786180362 -10.564656569
    ## 512  25095  1.192396e+00  1.338974e+00  -0.678876187  3.123672307   0.643244864
    ## 513  25198 -1.590364e+01  1.039392e+01 -19.133602465  6.185968835 -12.538020916
    ## 514  25231 -1.659866e+01  1.054175e+01 -19.818981809  6.017294647 -13.025901050
    ## 515  25254 -1.727519e+01  1.081967e+01 -20.363885974  6.046611747 -13.465033352
    ## 516  25426  1.125336e+00  1.130146e+00  -0.962975335  2.675687859   0.990074518
    ## 517  26523 -1.847487e+01  1.158638e+01 -21.402916813  6.038515416 -14.451158140
    ## 518  26556 -1.917983e+01  1.181792e+01 -21.919173581  6.086235634 -14.708844790
    ## 519  26585 -1.985632e+01  1.209589e+01 -22.464082746  6.115541102 -15.148021521
    ## 520  26833 -2.053275e+01  1.237399e+01 -23.009002914  6.144820978 -15.587296004
    ## 521  26863 -2.120912e+01  1.265220e+01 -23.553932944  6.174077910 -16.026658126
    ## 522  26899 -2.188543e+01  1.293051e+01 -24.098871852  6.203314192 -16.466099123
    ## 523  26931 -2.256170e+01  1.320890e+01 -24.643818777  6.232531823 -16.905611362
    ## 524  26961 -2.323792e+01  1.348739e+01 -25.188772969  6.261732551 -17.345188165
    ## 525  27163 -2.391410e+01  1.376594e+01 -25.733733766  6.290917914 -17.784823662
    ## 526  27187 -2.459024e+01  1.404457e+01 -26.278700587  6.320089265 -18.224512674
    ## 527  27219 -2.526636e+01  1.432325e+01 -26.823672914  6.349247807 -18.664250613
    ## 528  27252 -2.594243e+01  1.460200e+01 -27.368650288  6.378394608 -19.104033404
    ## 529  27784  2.880289e-01  9.656814e-01  -1.459495200  1.921863002  -1.912414030
    ## 530  28143 -2.714368e+01  1.536580e+01 -28.407424455  6.370895347 -20.087877572
    ## 531  28242 -2.787248e+00 -7.134038e-02  -1.505287775  3.361777137  -3.357421669
    ## 532  28625 -2.784818e+01  1.559819e+01 -28.923755945  6.418441747 -20.346228155
    ## 533  28658 -2.852427e+01  1.587692e+01 -29.468732093  6.447591402 -20.786000042
    ## 534  28692 -2.920033e+01  1.615570e+01 -30.013712486  6.476731180 -21.225809654
    ## 535  28726 -2.987637e+01  1.643452e+01 -30.558696821  6.505861787 -21.665654296
    ## 536  28755 -3.055238e+01  1.671339e+01 -31.103684825  6.534983864 -22.105531524
    ## 537  29526  1.102804e+00  2.829168e+00  -3.932870265  4.707691440   2.937966855
    ## 538  29531 -1.060676e+00  2.608579e+00  -2.971679003  4.360089352   3.738853043
    ## 539  29753  2.696141e-01  3.549755e+00  -5.810353445  5.809369574   1.538807934
    ## 540  29785  9.237644e-01  3.440481e-01  -2.880003656  1.721680075  -3.019564779
    ## 541  30852 -2.830984e+00  8.856570e-01   1.199930098  2.861292240   0.321668778
    ## 542  32686  2.879528e-01  1.728735e+00  -1.652173046  3.813544188  -1.090927154
    ## 543  32745 -2.179135e+00  2.021792e-02  -2.182732796  2.572046261  -3.663733112
    ## 544  34256  5.392759e-01  1.554890e+00  -2.066180333  3.241617463   0.184735711
    ## 545  34521  1.081234e+00  4.164139e-01   0.862918741  2.520862807  -0.005020500
    ## 546  34634  3.334995e-01  1.699873e+00  -2.596561167  3.643945269  -0.585067615
    ## 547  34684 -2.439237e+00  2.591458e+00  -2.840125650  1.286243910  -1.777016191
    ## 548  34687 -8.608269e-01  3.131790e+00  -5.052968134  5.420940709  -2.494141416
    ## 549  35585 -2.019001e+00  1.491270e+00   0.005221816  0.817253270   0.973252234
    ## 550  35771 -3.218952e+00  2.708535e+00  -3.263041587  1.361865944  -1.645775675
    ## 551  35866 -2.044489e+00  3.368306e+00  -3.937111391  5.623119731  -3.079232039
    ## 552  35899 -2.857170e+00  4.045601e+00  -4.197298786  5.487198631  -3.070776168
    ## 553  35906 -3.519030e+00  4.140867e+00  -3.628202369  5.505672037  -4.057462506
    ## 554  35926 -3.896583e+00  4.518355e+00  -4.454027240  5.547453347  -4.121458573
    ## 555  35942 -4.194074e+00  4.382897e+00  -5.118363373  4.455229840  -4.812620682
    ## 556  35953 -4.844372e+00  5.649439e+00  -6.730395746  5.252842019  -4.409566150
    ## 557  36170 -5.685013e+00  5.776516e+00  -7.064976729  5.902715452  -4.715563831
    ## 558  37167 -7.923891e+00 -5.198360e+00  -3.000023922  4.420666202   2.272193965
    ## 559  39729 -9.645673e-01 -1.643541e+00  -0.187726849  1.158253110  -2.458335646
    ## 560  40086  1.083693e+00  1.179501e+00  -1.346150391  1.998824401   0.818034176
    ## 561  40276  1.159373e+00  2.844795e+00  -4.050679508  4.777700881   2.948980041
    ## 562  40662 -4.446847e+00 -1.479316e-02  -5.126306996  6.945129976   5.269255348
    ## 563  40742 -2.377533e+00  5.205394e-01  -8.094139455  8.005351473   2.640750214
    ## 564  40892 -2.140511e+00  4.104871e+00  -8.996859402  4.028390951  -5.131359346
    ## 565  40918 -3.140260e+00  3.367342e+00  -2.778931407  3.859700674  -1.159518363
    ## 566  40919 -2.740483e+00  3.658095e+00  -4.110635520  5.340241669  -2.666774760
    ## 567  41116 -3.600544e+00  4.519047e+00  -6.340883535  6.214767278  -5.829557637
    ## 568  41138 -4.595617e+00  5.083690e+00  -7.581014617  7.546032795  -6.949164564
    ## 569  41147 -5.314173e+00  4.145944e+00  -8.532522459  8.344391674  -5.718008208
    ## 570  41164 -5.932778e+00  4.571743e+00  -9.427246928  6.577056438  -6.115218125
    ## 571  41170 -6.498086e+00  4.750515e+00  -8.966557831  7.098854313  -6.958376223
    ## 572  41181 -7.334341e+00  4.960892e+00  -8.451409591  8.174825168  -7.237463730
    ## 573  41194 -7.896886e+00  5.381020e+00  -8.451161615  7.963928087  -7.862419430
    ## 574  41203 -8.426814e+00  6.241659e+00  -9.946469745  8.199613932  -8.213093368
    ## 575  41204 -8.440284e+00  6.147653e+00 -11.683706159  6.702779571  -8.155839297
    ## 576  41227 -9.001351e+00  6.613284e+00 -12.423634724  7.519929232 -10.266254639
    ## 577  41233 -1.064580e+01  5.918307e+00 -11.671042595  8.807369179  -7.975501366
    ## 578  41237 -1.028178e+01  6.302385e+00 -13.271718029  8.925115476  -9.975578311
    ## 579  41243 -1.094074e+01  6.261586e+00 -14.182338784  7.183602086  -9.951363344
    ## 580  41273 -1.168221e+01  6.332882e+00 -13.297109247  7.690771915 -10.889890521
    ## 581  41285 -1.283576e+01  6.574615e+00 -12.788462483  8.786256839 -10.723120917
    ## 582  41305 -1.298094e+01  6.720508e+00 -13.455635943  8.698610093 -11.479551783
    ## 583  41308 -1.368076e+01  6.990389e+00 -13.770001136  8.694896899 -11.426967973
    ## 584  41313 -1.389721e+01  6.344280e+00 -14.281666443  5.581008997 -12.887133162
    ## 585  41353 -1.502098e+01  8.075240e+00 -16.298090538  5.664820053 -11.918153373
    ## 586  41397 -1.497035e+01  8.401421e+00 -16.867238487  8.252334306 -13.565130487
    ## 587  41413 -1.514045e+01  7.378042e+00 -16.356367356  9.194934914 -13.466163290
    ## 588  41505 -1.652651e+01  8.584972e+00 -18.649853185  9.505593515 -13.793818527
    ## 589  41582 -1.048005e+00  1.300219e+00  -0.180400576  2.589842643  -1.164793552
    ## 590  41607 -1.824751e+01  8.713250e+00 -17.880126821  9.249459237 -14.541212524
    ## 591  41646 -3.240187e+00  2.978122e+00  -4.162313937  3.869124379  -3.645256455
    ## 592  41743 -2.144411e+00  1.073499e+00  -2.773662944  1.384393731  -4.015477127
    ## 593  41791 -7.222731e+00  6.155773e+00 -10.826459837  4.180779179  -6.123555101
    ## 594  41851 -1.913973e+01  9.286847e+00 -20.134992105  7.818673310 -15.652207677
    ## 595  41870 -2.090691e+01  9.843153e+00 -19.947726046  6.155788648 -15.142013160
    ## 596  41991 -4.566342e+00  3.353451e+00  -4.572027808  3.616119304  -2.493138109
    ## 597  42247 -2.524012e+00  2.098152e+00  -4.946075210  6.456588410   3.173921377
    ## 598  42474 -3.843009e+00  3.375110e+00  -5.492893197  6.136377643   2.797195461
    ## 599  42985 -4.075975e+00  9.630313e-01  -5.076070198  4.955963358  -0.161436852
    ## 600  42988 -4.423508e+00  1.648048e+00  -6.934387857  4.894600566  -5.078131120
    ## 601  43028 -1.109646e+00  8.110693e-01  -1.138135327  0.935264670  -2.330247878
    ## 602  43369 -3.365319e+00  2.426503e+00  -3.752226527  0.276016598  -2.305870297
    ## 603  43494 -1.278138e+00  7.162416e-01  -1.143279229  0.217804801  -1.293890470
    ## 604  44393 -4.617461e+00  3.663395e+00  -5.297446410  3.880959654  -3.263550913
    ## 605  44532 -2.349223e-01  3.554126e-01   1.972182786 -1.255592713  -0.681386509
    ## 606  45463 -1.476893e+00  2.122314e+00  -1.229469560  1.201848798  -0.343264388
    ## 607  45501  1.001992e+00  4.793838e-02  -0.349001552  1.493958437   0.186938875
    ## 608  45541 -1.519244e+00  2.308492e+00  -1.503599320  2.064100584  -1.000845058
    ## 609  46057 -1.309441e+00  1.786495e+00  -1.371070240  1.214335492  -0.336642222
    ## 610  46149 -1.346509e+00  2.132431e+00  -1.854355318  2.116997706  -1.070378379
    ## 611  46925 -4.815312e-01  1.059542e+00   0.647117403  0.905586234   0.819368240
    ## 612  47545  1.176716e+00  5.570909e-01  -0.490799623  0.756424293   0.249191559
    ## 613  47826 -8.872869e-01  1.390002e+00   1.219685820  1.661425176   1.009227839
    ## 614  47923  3.643770e-01  1.443523e+00  -2.220907374  2.036985086  -1.237055031
    ## 615  47982 -1.232804e+00  2.244119e+00  -1.703825793  1.492535865  -1.192985170
    ## 616  48380 -2.790771e+00 -1.464269e+00   1.031165008  1.921356084  -0.090013820
    ## 617  48533  1.243848e+00  5.245258e-01  -0.538883626  1.209195974   0.479537597
    ## 618  48884 -2.139051e+00  1.394368e+00  -0.612034895  1.049327056  -1.162101908
    ## 619  49985 -1.554216e+00  1.694229e+00  -0.903333638  2.425436085  -2.899787485
    ## 620  50706 -8.461845e+00  6.866198e+00 -11.838269221  4.194210967  -6.923097097
    ## 621  50808 -9.169790e+00  7.092197e+00 -12.354036869  4.243068974  -7.176437755
    ## 622  51112 -9.848776e+00  7.365546e+00 -12.898538231  4.273323073  -7.611991011
    ## 623  51135 -1.052730e+01  7.639745e+00 -13.443114503  4.303402918  -8.048209867
    ## 624  51155 -1.120546e+01  7.914633e+00 -13.987751639  4.333341180  -8.484969523
    ## 625  52814 -1.101847e+00 -1.632441e+00   0.901066557  0.847752886  -1.249090729
    ## 626  52934  1.036639e+00  4.072273e-01   0.757706165  3.161821447  -0.568122226
    ## 627  53031  2.060755e-01  1.387360e+00  -1.045286987  4.228686194  -1.647549174
    ## 628  53076  1.296231e+00  4.174473e-01   0.193963068  0.901644195   0.130531367
    ## 629  53451  3.851085e-01  1.217620e+00  -1.953872165  2.087075991  -1.144225406
    ## 630  53658 -1.739341e+00  1.344521e+00  -0.534379047  3.195291499  -0.416196142
    ## 631  53727 -1.649279e+00  1.263974e+00  -1.050825674  2.237990639  -2.527889238
    ## 632  53937 -2.042608e+00  1.573578e+00  -2.372652433 -0.572676431  -2.097352765
    ## 633  54846 -2.986466e+00 -8.912295e-04   0.605886830  0.338337781   0.685448451
    ## 634  55279 -5.753852e+00  5.776098e-01  -6.312781518  5.159401364  -1.698319850
    ## 635  55311 -6.159607e+00  1.468713e+00  -6.850888147  5.174706273  -2.986703924
    ## 636  55614 -7.347955e+00  2.397041e+00  -7.572355818  5.177819255  -2.854838283
    ## 637  55618 -7.427924e+00  2.948209e+00  -8.678550384  5.185303480  -4.761089973
    ## 638  55760 -6.003422e+00 -3.930731e+00  -0.007045455  1.714669183   3.414666602
    ## 639  56098 -1.229669e+00  1.956099e+00  -0.851197960  2.796986550  -1.913976841
    ## 640  56624 -7.901421e+00  2.720472e+00  -7.885935672  6.348333551  -5.480119036
    ## 641  56650 -8.762083e+00  2.791030e+00  -7.682766857  6.991213965  -5.230695451
    ## 642  56806  1.682781e-02  2.400826e+00  -4.220359821  3.462217464  -0.624142188
    ## 643  56887 -7.548347e-02  1.812355e+00  -2.566980810  4.127548607  -1.628531570
    ## 644  57007 -1.271244e+00  2.462675e+00  -2.851395003  2.324480065  -1.372244890
    ## 645  57027 -2.335655e+00  2.225380e+00  -3.379450386  2.178538229  -3.568263715
    ## 646  57163 -1.036305e+01  4.543672e+00  -9.795897736  5.508003334  -6.037155994
    ## 647  58060 -2.630598e+00  5.125759e+00  -6.092254547  5.527393062   1.605144855
    ## 648  58067 -2.648687e-01  3.386140e+00  -3.454996990  4.367629058   3.336060428
    ## 649  58199  3.403908e-01  2.015233e+00  -2.777329561  3.812024131  -0.461729072
    ## 650  58217 -4.437940e-01  1.271395e+00   1.206177863  0.790371244   0.418935335
    ## 651  58222 -1.322789e+00  1.552768e+00  -2.276920704  2.992117391  -1.947063920
    ## 652  58642 -4.513826e-01  2.225147e+00  -4.953050453  4.342228248  -3.656190190
    ## 653  58822 -4.384221e+00  3.264665e+00  -3.077158017  3.403593597  -1.938075280
    ## 654  59011 -2.326922e+00 -3.348439e+00  -3.513407961  3.175059710  -2.815137034
    ## 655  59385 -7.626924e+00 -6.976420e+00  -2.077911151  3.416754160   4.458758402
    ## 656  59669  3.260066e-01  1.286638e+00  -2.007181287  2.419674796  -1.532901697
    ## 657  59777 -8.257111e+00 -4.814461e+00  -5.365306890  1.204229864  -3.347420097
    ## 658  59840 -3.215382e+00 -3.642231e-01  -1.261883344  3.794949039   0.711205930
    ## 659  60353 -3.975216e+00  5.815733e-01  -1.880372107  4.319241280  -3.024329760
    ## 660  61108 -2.756007e+00  6.838214e-01  -1.390168883  1.501887261  -1.165614356
    ## 661  61646 -1.522305e+00  1.505152e+00   0.372364351  2.286868558  -0.526518744
    ## 662  62059 -1.644403e+00  3.129852e+00  -2.576976822  3.415573363  -0.448525407
    ## 663  62080 -1.599457e+00  2.607720e+00  -2.987192942  3.064156147  -2.497914317
    ## 664  62330  1.140865e+00  1.221317e+00  -1.452955460  2.067575058   0.854742052
    ## 665  62341 -5.267760e+00  2.506719e+00  -5.290924818  4.886134198  -3.343188411
    ## 666  62467 -5.344665e+00 -2.857604e-01  -3.835615782  5.337048027  -7.609908700
    ## 667  63578 -6.391913e-01 -8.559479e-02   1.265451959  1.401165732  -0.260542079
    ## 668  64093 -6.133987e+00  2.941499e+00  -5.593985568  3.258845223  -5.315512219
    ## 669  64412 -1.348042e+00  2.522821e+00  -0.782432259  4.083046652  -0.662280330
    ## 670  64443  1.079524e+00  8.729880e-01  -0.303850487  2.755368851   0.301687623
    ## 671  64585  1.080433e+00  9.628307e-01  -0.278065478  2.743318063   0.412364076
    ## 672  64785 -8.744415e+00 -3.420468e+00  -4.850574732  6.606845786  -2.800546172
    ## 673  65358  1.193916e+00 -5.710850e-01   0.742522414 -0.014587682  -0.624560898
    ## 674  65385 -2.923827e+00  1.524837e+00  -3.018758033  3.289291475  -5.755542374
    ## 675  65728  1.227614e+00 -6.689738e-01  -0.271784826 -0.589439742  -0.604795004
    ## 676  65936 -3.593476e+00  7.814423e-01  -1.822447514  0.605760683  -1.194655867
    ## 677  66037  2.863024e-01  1.399345e+00  -1.682502709  3.864377040  -1.185373402
    ## 678  67150 -1.824295e+00  4.033265e-01  -1.994122006  2.756558164  -3.139064326
    ## 679  67571 -7.584687e-01 -4.541028e-02  -0.168438290 -1.313274814  -1.901762532
    ## 680  67857 -1.739334e+00 -1.304655e+00   0.314102907  0.053740106  -0.058695700
    ## 681  68207 -1.319267e+01  1.278597e+01  -9.906650021  3.320336883  -4.801175932
    ## 682  68357  1.232604e+00 -5.489312e-01   1.087872669  0.894081820  -1.433054899
    ## 683  69394  1.140431e+00  1.134243e+00  -1.429454741  2.012225989   0.622800198
    ## 684  70071 -4.400952e-01  1.137239e+00  -3.227079757  3.242292934  -2.033998153
    ## 685  70229  3.156417e-01  1.636778e+00  -1.519649883  4.028570999  -1.186794267
    ## 686  70270 -1.512516e+00  1.133139e+00  -1.601052480  2.813400968  -2.664503406
    ## 687  70536 -2.271755e+00 -4.576546e-01  -2.589054639  2.230778267  -4.278982503
    ## 688  70828  1.967068e-01  1.189757e+00   0.704882218  2.891387773   0.045555369
    ## 689  71033 -3.170818e+00  1.857346e-01  -3.399851579  3.761155290  -2.148047028
    ## 690  72327 -4.198735e+00  1.941206e-01  -3.917585678  3.920747846  -1.875485747
    ## 691  72824 -1.111495e+00 -2.575752e-01   2.250209635  1.152670903   0.432904474
    ## 692  73408 -2.869795e+00  1.335667e+00  -1.009530198  1.693884729  -0.741479895
    ## 693  74159 -1.548788e+00  1.808698e+00  -0.953509034  2.213085393  -2.015727792
    ## 694  74262 -2.250535e+00  2.365755e+00  -2.955491179  0.089790598  -2.830744567
    ## 695  75033 -4.303302e-01  9.856327e-01   0.645789437  0.317131047   0.616332206
    ## 696  75556 -7.343032e-01  4.355193e-01  -0.530865547 -0.471119607   0.643214387
    ## 697  75581 -2.866364e+00  2.346949e+00  -4.053306688  3.983358880  -3.463185713
    ## 698  75851 -4.793667e+00  3.418911e+00  -5.074444533  4.035986943  -3.527874599
    ## 699  75978 -5.140723e+00  3.568751e+00  -5.896245457  4.164719930  -4.091192639
    ## 700  76575 -5.622469e+00  3.480623e+00  -6.200676594  4.311233668  -5.226286324
    ## 701  76826 -6.616293e+00  3.563428e+00  -7.058901298  4.284345955  -5.096298873
    ## 702  76845  1.141572e+00  1.291195e+00  -1.432900453  2.058202222   0.940823738
    ## 703  76857  1.140208e+00  1.156431e+00  -1.471577967  2.076278405   0.774809059
    ## 704  76867  1.082566e+00  1.094862e+00  -1.367020048  2.012554296   0.708142048
    ## 705  76876 -1.298359e+00  1.079671e+00  -0.180677931  1.287839084   1.858273479
    ## 706  77154 -7.154137e-01  6.085903e-01   1.155500890 -0.267565093  -0.563747541
    ## 707  77171  1.118560e+00  1.291858e+00  -1.298805326  2.135771701   0.772203698
    ## 708  77182 -1.410852e+00  2.268271e+00  -2.297553649  1.871330527   0.248957098
    ## 709  77202 -3.563262e-01  1.435305e+00  -0.813563965  1.993117177   2.055877540
    ## 710  77627 -7.139060e+00  2.773082e+00  -6.757845069  4.446455974  -5.464428185
    ## 711  78725 -4.312479e+00  1.886476e+00  -2.338633561 -0.475242892  -1.185443862
    ## 712  79540 -1.143607e-01  1.036129e+00   1.984405261  3.128243274  -0.740343560
    ## 713  81372 -8.852541e-01  1.790649e+00  -0.945148961  3.853432747  -1.543510075
    ## 714  82289 -1.464897e+00  1.975528e+00  -1.077144539  2.819190640   0.069850228
    ## 715  83934 -4.332224e-01  2.428379e+00  -3.996454397  4.871298541  -1.796307856
    ## 716  84204 -9.378433e-01  3.462889e+00  -6.445103954  4.932198666  -2.233983070
    ## 717  84204 -1.927453e+00  1.827621e+00  -7.019494685  5.348303240  -2.739187880
    ## 718  84694 -4.868108e+00  1.264420e+00  -5.167885427  3.193647845  -3.045621377
    ## 719  84789 -1.430864e+00 -8.025287e-01   1.123320300  0.389760265  -0.281213534
    ## 720  85181 -3.003459e+00  2.096150e+00  -0.487030348  3.069453160  -1.774328684
    ## 721  85285 -7.030308e+00  3.421991e+00  -9.525071773  5.270891009  -4.024630276
    ## 722  85285 -6.713407e+00  3.921104e+00  -9.746678217  5.148262549  -5.151562720
    ## 723  85573 -1.756712e+00  3.266574e+00  -4.153387897  3.924525614  -1.753771843
    ## 724  85576 -2.207631e+00  3.259076e+00  -5.436365470  3.684737394  -3.066400910
    ## 725  85864 -3.365265e+00  2.928541e+00  -5.660999199  3.891159928  -1.840375110
    ## 726  85867 -3.586964e+00  2.609127e+00  -5.568577208  3.631946953  -4.543589711
    ## 727  86376 -6.702381e-01  9.452064e-01   0.610050626  2.640065087  -2.707774626
    ## 728  87202 -4.198202e-01 -1.155978e+00  -2.092515502  2.786750330   0.736296592
    ## 729  87883 -1.360293e+00 -4.580690e-01  -0.700403918  2.737229172  -1.005105974
    ## 730  88672 -3.859881e+00  2.632881e+00  -5.264265310  3.446113147  -0.675231057
    ## 731  88737  1.917827e+00  9.519666e-01  -2.059205513  3.833997712   1.668192191
    ## 732  90676 -2.405580e+00  3.738235e+00  -2.317843041  1.367442441   0.394001128
    ## 733  91075 -1.855061e+00  1.554964e+00  -1.405808829  0.669327346  -0.280229520
    ## 734  91407 -3.951209e+00  2.881805e+00  -6.421490295  2.434181424  -1.327324538
    ## 735  91502  7.379075e-03  2.365183e+00  -2.600287382  1.111601765   3.276441326
    ## 736  91524  1.954852e+00  1.630056e+00  -4.337199681  2.378367292   2.113347861
    ## 737  91554 -5.100256e+00  3.633442e+00  -3.843918624  0.183208447  -1.183997376
    ## 738  92092 -1.108478e+00  3.448953e+00  -6.216971806  3.021051618  -0.529900932
    ## 739  92102 -1.662937e+00  3.253892e+00  -7.040484712  2.266456480  -4.177648913
    ## 740  93742 -3.291125e+00  4.401194e+00  -8.394211623  4.453579569  -4.790054518
    ## 741  93823 -3.821939e+00  5.667247e+00  -9.244963458  8.246146934  -4.368286087
    ## 742  93824 -3.632809e+00  5.437263e+00  -9.136521481 10.307226308  -5.421830295
    ## 743  93834 -3.765680e+00  5.890735e+00 -10.202267631 10.259035977  -5.611448417
    ## 744  93853 -6.185857e+00  7.102985e+00 -13.030455264  8.010823398  -7.885237361
    ## 745  93853 -5.839192e+00  7.151532e+00 -12.816760081  7.031114765  -9.651272168
    ## 746  93856 -6.750509e+00  5.367416e+00 -10.054634953  9.064477918  -7.968118084
    ## 747  93860 -1.085028e+01  6.727466e+00 -16.760583410  8.425831679 -10.252696868
    ## 748  93860 -1.063237e+01  7.251936e+00 -17.681071821  8.204144406 -10.166590752
    ## 749  93879 -1.308652e+01  7.352148e+00 -18.256576112 10.648505446 -11.731476451
    ## 750  93879 -1.283363e+01  7.508790e+00 -20.491952211  7.465779992 -11.575303706
    ## 751  93888 -1.004063e+01  6.139183e+00 -12.972971894  7.740555445  -8.684704528
    ## 752  93897 -1.030082e+01  6.483095e+00 -15.076362720  6.554191153  -8.880251722
    ## 753  93904 -1.132063e+01  7.191950e+00 -13.179082545  9.099551882 -10.094748875
    ## 754  93920 -1.238105e+01  8.213022e+00 -16.962529649  7.116090598  -9.772826359
    ## 755  93965 -1.139773e+01  7.763953e+00 -18.572307456  6.711855252 -10.174215599
    ## 756  94141 -1.351207e+01  8.215177e+00 -16.582605727  6.207368890 -11.318471518
    ## 757  94362 -2.645774e+01  1.649747e+01 -30.177317456  8.904156771 -17.892599693
    ## 758  94364 -1.519206e+01  1.043253e+01 -19.629515252  8.046075046 -12.838166615
    ## 759  94625  1.707857e+00  2.488079e-02  -0.488140039  3.787548155   1.139451485
    ## 760  94952  8.420253e-01 -3.655184e-01  -2.464063331  4.820885964   0.775505434
    ## 761  95559 -1.630865e+01  1.161480e+01 -19.739386297 10.463866275 -12.599145656
    ## 762  95628 -1.751891e+01  1.257212e+01 -19.038538321 11.190894656 -13.554720708
    ## 763  96135 -1.952933e+00  3.541385e+00  -1.310561408  5.955663594  -1.003992663
    ## 764  96291 -3.552173e+00  5.426461e+00  -3.731810320  6.679061807  -2.187543379
    ## 765  96717 -3.705856e+00  4.107873e+00  -3.803655995  1.710314448  -3.582466046
    ## 766  97121 -1.797627e+01  1.286499e+01 -19.575066191 11.345119818 -13.998645603
    ## 767  97235 -1.753759e+01  1.235252e+01 -20.134612814 11.122771433 -14.571080140
    ## 768 100223 -1.964186e+01  1.470633e+01 -22.801237695 12.114671842 -14.898112531
    ## 769 100298 -2.234189e+01  1.553613e+01 -22.865228496  7.043373548 -14.183129192
    ## 770 100501 -6.985267e+00  5.151094e+00  -4.599337773  4.534478601   0.849053979
    ## 771 100924 -2.398475e+01  1.669783e+01 -22.209874816  9.584968620 -16.230438789
    ## 772 101051 -1.465316e+00 -1.093377e+00  -0.059767848  1.064784635  11.095088600
    ## 773 101313 -2.582598e+01  1.916724e+01 -25.390229312 11.125434715 -16.682643920
    ## 774 101597  9.131161e-01  1.145381e+00  -4.602877933  2.091803374  -0.473223859
    ## 775 102114 -2.825505e+01  2.146720e+01 -26.871338780 11.737436136 -17.999630285
    ## 776 102318 -1.020632e+00  1.496959e+00  -4.490937118  1.836727058   0.627318468
    ## 777 102480 -1.929597e+00  4.066413e+00  -4.865184010  5.898602293  -0.552492585
    ## 778 102489 -2.296987e+00  4.064043e+00  -5.957706345  4.680008064  -2.080937600
    ## 779 102542 -1.456876e+00  3.740306e+00  -7.404518446  7.440964036  -1.549877953
    ## 780 102572 -2.870923e+01  2.205773e+01 -27.855811134 11.845012910 -18.983812582
    ## 781 102619 -2.488363e+00  4.359019e+00  -7.776409850  5.364026770  -1.823877232
    ## 782 102622 -2.877176e+00  4.569649e+00  -9.553068767  4.441079354  -3.653961387
    ## 783 102625 -4.221221e+00  2.871121e+00  -5.888715887  6.890952075  -3.404894350
    ## 784 102669 -5.603690e+00  5.222193e+00  -7.516829977  8.117724276  -2.756857953
    ## 785 102671 -4.991758e+00  5.213340e+00  -9.111326298  8.431986352  -3.435515672
    ## 786 102676 -5.552122e+00  5.678134e+00  -9.775527980  8.416295394  -4.409843819
    ## 787 103808 -4.517344e+00  2.500224e+00  -4.013927537  1.189451634  -2.486860636
    ## 788 109297  7.451532e-01  2.809299e+00  -5.825406357  5.835566426   0.512319881
    ## 789 109298 -1.000611e+00  3.346850e+00  -5.534491171  6.835802039  -0.299803294
    ## 790 110087  1.934946e+00  6.506777e-01  -0.286957049  3.987828091   0.316051835
    ## 791 110547 -1.532810e+00  2.232752e+00  -5.923100075  3.386708412  -0.153443340
    ## 792 110552 -2.450367e+00  2.107729e+00  -5.140663040  1.411303574  -1.690779503
    ## 793 110617 -1.101035e+00 -1.674928e+00  -0.573388196  5.617555666   0.765556413
    ## 794 115691 -1.550273e+00  1.088689e+00  -2.393387662  1.008732717  -1.087561887
    ## 795 116067  9.492407e-01  1.333519e+00  -4.855402014  1.835005883  -1.053245292
    ## 796 118532 -5.961457e+00  5.313382e+00  -6.674320456  6.028974991  -1.387559928
    ## 797 118603 -6.677212e+00  5.529299e+00  -7.193274565  6.081321080  -1.636070971
    ## 798 121238 -2.628922e+00  2.275636e+00  -3.745369349  1.226948197  -1.132966495
    ## 799 122608 -2.003460e+00 -7.159042e+00  -4.050976316  1.309579747  -2.058101588
    ## 800 123078 -1.073820e+00  4.156165e-01  -2.273976798  1.536844197  -0.758696869
    ## 801 123525 -5.904921e+00  4.439911e+00  -8.631802213  7.788684407  -4.989579801
    ## 802 125200 -7.691717e-01  1.342212e+00  -2.171454137 -0.151513300  -0.648374467
    ## 803 125612  1.889618e+00  1.073099e+00  -1.678017522  4.173268194   1.015515806
    ## 804 125658  2.244145e-01  2.994499e+00  -3.432458196  3.986518932   3.760232949
    ## 805 126219 -1.141559e+00  1.927650e+00  -3.905356149 -0.073942594  -0.044857819
    ## 806 128471  9.091238e-01  1.337658e+00  -4.484727667  3.245357645  -0.417808776
    ## 807 128519 -4.599447e+00  2.762540e+00  -4.656529871  5.201403067  -2.470387712
    ## 808 128595 -5.313774e+00  2.664274e+00  -4.250707018  0.394707095  -0.391383377
    ## 809 128803 -2.272473e+00  2.935226e+00  -4.871394115  2.419011986  -1.513021771
    ## 810 129095 -1.836940e+00 -1.646764e+00  -3.381168242  0.473353567   0.074242868
    ## 811 129186  2.901552e-01  4.924313e-02  -0.740523666  2.865462770   1.395293916
    ## 812 129222  1.177824e+00  2.487103e+00  -5.330608161  5.324546835   1.150243258
    ## 813 129308  5.468215e-02  1.856500e+00  -4.075450735  4.100097970  -0.800931346
    ## 814 129371  1.183931e+00  3.057250e+00  -6.161997335  5.543972483   1.617041370
    ## 815 129668  7.533560e-01  2.284988e+00  -5.164492004  3.831111999  -0.073622492
    ## 816 129741 -1.396204e+00  2.618584e+00  -6.036770271  3.552453710   1.030091306
    ## 817 129764 -2.434004e+00  3.225947e+00  -6.596281631  3.593160524  -1.079452045
    ## 818 129808  1.522080e+00 -5.194295e-01  -2.581684880  0.774741046   0.206721721
    ## 819 131024  4.697495e-01 -1.237555e+00  -1.767340616  4.833490128  -0.268714559
    ## 820 132086 -3.614278e-01  1.133472e+00  -2.971360440 -0.283072526   0.371451593
    ## 821 132688  4.325545e-01  1.861373e+00  -4.310353062  2.448080475   4.574093675
    ## 822 133184 -1.212682e+00 -2.484824e+00  -6.397185815  3.670562448  -0.863375061
    ## 823 133731  1.176633e+00  3.141918e+00  -6.140444836  5.521820589   1.768514889
    ## 824 133958  5.238199e-01  1.531708e+00  -4.176390134  3.584614831  -1.023954033
    ## 825 134766 -7.965254e-02  3.222010e+00  -3.724201389  6.037345128   0.583394746
    ## 826 134769 -9.677671e-01  2.098019e+00  -5.222929466  6.514573243  -4.187674068
    ## 827 134928  1.204934e+00  3.238070e+00  -6.010324404  5.720846677   1.548399778
    ## 828 135095  2.325125e-01  9.389437e-01  -4.647779879  3.079843688  -1.902655319
    ## 829 135102  1.862102e+00 -1.240519e-01  -1.989751750  0.382608915   0.473031677
    ## 830 135314 -3.158990e+00  1.765452e+00  -3.390168050  0.987409979  -1.509930366
    ## 831 137211  6.305787e-01  1.183631e+00  -5.066282718  2.179902515  -0.703375741
    ## 832 138894 -1.298443e+00  1.948100e+00  -4.509946888  1.305804767  -0.019485927
    ## 833 138942 -2.356348e+00  1.746360e+00  -6.374623621  1.772205446  -3.439293796
    ## 834 139107 -4.666500e+00 -3.952320e+00   0.206093982  5.153524618   5.229468882
    ## 835 139117 -3.975939e+00 -1.244939e+00  -3.707413720  4.544772208   4.050675573
    ## 836 139767  4.679919e-01  1.100118e+00  -5.607145000  2.204713526  -0.578538843
    ## 837 139816 -3.955822e-01 -7.517921e-01  -1.984665722 -0.203459371   1.903967176
    ## 838 139951 -2.921944e+00 -2.280624e-01  -5.877288646  2.201883522  -1.935439873
    ## 839 140293  9.510252e-01  3.252926e+00  -5.039104740  4.632410980   3.014501280
    ## 840 140308 -4.861747e+00 -2.722660e+00  -4.656248090  2.502004702  -2.008346425
    ## 841 141320 -6.352337e+00 -2.370335e+00  -4.875396679  2.335044531  -0.809554897
    ## 842 141565  1.149650e-01  7.667615e-01  -0.494132199  0.116771978   0.868168535
    ## 843 141925  1.203014e-01  1.974141e+00  -0.434086817  5.390793100   1.289683729
    ## 844 142280 -1.169203e+00  1.863414e+00  -2.515135481  5.463680646  -0.297971323
    ## 845 142394 -3.367770e+00  9.924944e-02  -6.148487449  3.401955368   0.458306649
    ## 846 142409 -1.172183e+00  1.661713e+00  -3.049637019  2.555058434   3.669034614
    ## 847 142840 -3.613850e+00 -9.221357e-01  -4.749887364  3.373001328  -0.545207025
    ## 848 142961  4.578454e-01  1.373769e+00  -0.488926009  2.805350941   1.777386060
    ## 849 143354  1.118331e+00  2.074439e+00  -3.837518277  5.448059692   0.071815968
    ## 850 143434 -2.729482e+00  3.312495e+00  -4.242710444  5.036985077  -0.376561161
    ## 851 143438 -5.256434e+00  3.645413e-01  -5.412085016  2.400031491   0.697301199
    ## 852 143456 -2.006582e+00  3.676577e+00  -5.463810784  7.232058275  -1.627859462
    ## 853 144808 -2.405207e+00  2.943823e+00  -7.616654221  3.533373872  -5.417493628
    ## 854 144839 -6.423306e+00  1.658515e+00  -5.866439564  2.052064402  -0.615816843
    ## 855 146022  9.086367e-01  2.849024e+00  -5.647342963  6.009414778   0.216656395
    ## 856 146026  1.894036e+00  1.905806e+00  -3.515730077  4.508913151   2.044466245
    ## 857 146179 -6.767162e-02  4.251181e+00  -6.540388352  7.283657037   0.513540871
    ## 858 146344 -9.972390e-02  2.795414e+00  -6.423855582  3.247513015  -1.632290269
    ## 859 146998 -2.064240e+00  2.629739e+00  -0.748406253  0.694992040   0.418177986
    ## 860 147501 -1.611877e+00 -4.084099e-01  -3.829761676  6.249461761  -3.360921609
    ## 861 148028 -1.053840e+00  4.362801e+00  -6.023533637  5.304534323   1.480738298
    ## 862 148053  1.261324e+00  2.726800e+00  -5.435018912  5.342759009   1.447043018
    ## 863 148074 -2.219219e+00  7.278314e-01  -5.458229947  5.924849847   3.932463824
    ## 864 148468  2.188102e-01  2.715855e+00  -5.111657839  6.310660974  -0.848345065
    ## 865 148476 -1.125092e+00  3.682876e+00  -6.556167949  4.016730951  -0.425570578
    ## 866 148479 -1.541678e+00  3.846800e+00  -7.604114173  3.121458585  -1.254924173
    ## 867 149096  1.184891e+00  3.152084e+00  -6.134779825  5.531252206   1.733866514
    ## 868 149236 -1.370976e+00 -2.546461e-02  -2.774906916  2.650530185   4.511309291
    ## 869 149582 -4.280584e+00  1.421100e+00  -3.908228843  2.942946080  -0.076205336
    ## 870 149640  7.543159e-01  2.379822e+00  -5.137274493  3.818391722   0.043202652
    ## 871 149676  1.833191e+00  7.453328e-01  -1.133008713  3.893555977   0.858163817
    ## 872 150138 -2.150855e+00  2.187917e+00  -3.430515892  0.119476214  -0.173210022
    ## 873 150139 -6.682832e+00 -2.714268e+00  -5.774530457  1.449792375  -0.661836408
    ## 874 150494  1.852889e+00  1.069593e+00  -1.776101273  4.617409969   0.770412982
    ## 875 150949 -2.423535e+00  1.659093e+00  -3.071420549  2.588032647   1.135791170
    ## 876 151029 -3.818214e+00  2.551338e+00  -4.759158316  1.636966959  -1.167900084
    ## 877 151916 -5.488032e+00  3.329561e+00  -5.996296087  3.601719636  -2.023926036
    ## 878 151972 -6.618211e+00  3.835943e+00  -6.316453129  1.844111483  -2.476892421
    ## 879 152036 -4.320609e+00  3.199939e+00  -5.799736054  6.502330236   0.378479253
    ## 880 152058 -3.576362e+00  3.299436e+00  -7.460433197  7.783634177  -0.398548658
    ## 881 152098 -4.124316e+00  3.748597e+00  -7.926506608  7.763241791  -0.769374581
    ## 882 152165 -4.673231e+00  4.195976e+00  -8.392422791  7.743215147  -1.138803489
    ## 883 152307 -5.222968e+00  4.641827e+00  -8.858204209  7.723501994  -1.507034956
    ## 884 152710  5.107548e-02  1.310427e+00   0.733222065  2.620281923   1.402357614
    ## 885 152802  1.322724e+00 -8.439106e-01  -2.096887538  0.759759069  -0.196377115
    ## 886 153653 -5.192496e+00  3.164721e+00  -5.047679060  2.246597360  -4.011781405
    ## 887 153761  1.146259e+00  1.403458e+00  -4.159148191  2.660107057  -0.323216827
    ## 888 153875 -6.136959e-01  3.698772e+00  -5.534941162  5.620486385   1.649262850
    ## 889 154181 -5.496154e-01  2.219075e+00  -3.522024322  0.236995179   1.087468857
    ## 890 154278 -1.600211e+00 -3.488130e+00  -6.459302807  3.246815663  -1.614607699
    ## 891 154309 -8.298347e-02 -3.935919e+00  -2.616708986  0.163310075  -1.400951828
    ## 892 154493 -7.381547e+00 -7.449015e+00  -4.696286524  3.728439150   6.198304347
    ## 893 154599  6.677136e-01  3.041502e+00  -5.845111956  5.967587038   0.213863333
    ## 894 154657 -6.795206e-01  4.672553e+00  -6.814798440  7.143500120   0.928653839
    ## 895 155054 -5.123490e-01  4.827060e+00  -7.973939249  7.334058516   0.367703645
    ## 896 155359 -1.067713e+00  5.262312e+00  -8.438567066  7.316487300   0.008253573
    ## 897 155535  7.111555e-01  2.617105e+00  -4.722363221  5.842969959  -0.600179299
    ## 898 155542  1.868226e+00  1.363077e+00  -1.994934283  4.173515744   1.239750950
    ## 899 155548  1.878230e+00  1.325630e+00  -2.333468993  4.233151214   1.355183996
    ## 900 155554 -1.040067e+00  3.106703e+00  -5.409027199  3.109903424  -0.887237040
    ## 901 155662 -1.928613e+00  4.601506e+00  -7.124052978  5.716088037   1.026578545
    ## 902 155965 -1.201398e+00  4.864535e+00  -8.328823316  7.652399362  -0.167445106
    ## 903 156685 -1.297776e-01  1.415475e-01  -0.894701870 -0.457662245   0.810607981
    ## 904 156710  2.024023e-01  1.176270e+00   0.346379124  2.882137799   1.407133333
    ## 905 157207  1.170756e+00  2.501038e+00  -4.986159233  5.374160058   0.997798392
    ## 906 157284 -2.422450e-01  4.147186e+00  -5.672349096  6.493740596   1.591168156
    ## 907 158638 -5.976119e+00 -7.196980e+00  -5.388315911  5.104798647   4.676532867
    ## 908 159844 -4.081108e-01  3.132944e+00  -3.098029764  5.803892615   0.890608919
    ## 909 160034 -2.349340e+00  1.512604e+00  -2.647497196  1.753792472   0.406327950
    ## 910 160243 -2.783865e+00  1.596824e+00  -2.084843989  2.512985584  -1.446748566
    ## 911 160537  5.675393e-01  3.309385e+00  -6.631268489  6.394573696  -0.054172326
    ## 912 160665 -4.173398e-01  4.700055e+00  -7.521766828  7.671884451   0.260821238
    ## 913 160791  2.132386e+00  7.056078e-01  -3.530758736  0.514779311   1.527174698
    ## 914 160870 -6.442777e-01  5.002352e+00  -8.252738760  7.756914712  -0.216267318
    ## 915 160895 -8.482902e-01  2.719882e+00  -6.199070197  3.044437186  -3.301909553
    ## 916 161154 -3.387601e+00  3.977881e+00  -6.978585065  1.657765845  -1.100499733
    ## 917 163181 -5.238808e+00  6.230130e-01  -5.784506592  1.678889307  -0.364431872
    ## 918 165132 -7.503926e+00 -3.606280e-01  -3.830952283  2.486102856   2.497366775
    ## 919 165981 -5.766879e+00 -8.402154e+00   0.056543246  6.950982945   9.880564026
    ## 920 166028 -9.563904e-01  2.361594e+00  -3.171194607  1.970758779   0.474761354
    ## 921 166831 -2.027135e+00 -1.131890e+00  -1.135194254  1.086962637  -0.010547346
    ## 922 166883  2.091900e+00 -7.574594e-01  -1.192257634 -0.755458093  -0.620323626
    ## 923 167338 -1.374424e+00  2.793185e+00  -4.346572055  2.400731420  -1.688432714
    ## 924 169142 -1.927883e+00  1.125653e+00  -4.518330641  1.749292533  -1.566487292
    ## 925 169347  1.378559e+00  1.289381e+00  -5.004246784  1.411849842   0.442580636
    ## 926 169351 -6.761427e-01  1.126366e+00  -2.213699523  0.468308388  -1.120541044
    ## 927 169966 -3.113832e+00  5.858642e-01  -5.399730211  1.817092473  -0.840618466
    ## 928 170348  1.991976e+00  1.584759e-01  -2.583440645  0.408669993   1.151147061
    ##               V6            V7            V8            V9           V10
    ## 1   -0.412319996   0.027989810  6.374939e-01 -9.211004e-01  -0.506703834
    ## 2    0.298507561   0.790798748 -1.239956e-01  3.941539e-01  -0.503813823
    ## 3   -0.710962242  -0.478698916  1.558632e-01 -7.340673e-01   1.260822508
    ## 4   -0.578713478  -0.844144505  1.634674e-02 -5.297640e-01   0.546259891
    ## 5   -1.263733615   0.884483482 -5.396886e-01  1.939387e-01  -0.029890889
    ## 6   -0.157811104   2.034553195 -5.684424e-01  1.238477e+00  -1.331454509
    ## 7    3.276639233  -1.163402348  7.089906e-01  3.594890e-01   0.279566745
    ## 8    0.264900956  -0.611438181  1.288748e-01 -8.509983e-01   0.955577255
    ## 9   -1.322616612   0.117009989  8.816123e-01  1.730488e-01   0.052979444
    ## 10   0.920017273   0.387762842  4.580832e-01  1.323239e+00  -0.689343241
    ## 11  -0.780106846   0.878764125 -4.500776e-01  1.480450e-01   0.294464250
    ## 12   0.881927966  -0.721391085  9.757569e-01  8.734077e-01  -1.108299921
    ## 13  -0.143423245  -0.877682854  2.454343e-01  2.106866e+00  -1.021955201
    ## 14  -0.284923678   0.279571905  7.632751e-02  3.793382e-01  -0.655520820
    ## 15  -0.462898132   0.768867562 -3.848405e-01 -1.563521e-01   0.132865942
    ## 16  -0.482555355   0.392063689  2.046082e-01 -5.482198e-01  -0.396384618
    ## 17   2.122021321  -0.721224661  7.635422e-01  6.066363e-01  -0.273312504
    ## 18  -0.933345303   0.328064273 -2.955506e-01 -1.631439e-01  -0.028661707
    ## 19   1.113731179  -0.644492733  4.461398e-01  9.914229e-01  -0.247962965
    ## 20  -0.160281616   0.131513096  3.516576e-01  4.821639e-01  -0.637568209
    ## 21   0.387610255  -0.127818525  2.578643e-01 -1.758975e-01   0.149495966
    ## 22  -0.944068213   0.373733317  4.325359e-01 -9.756720e-01  -0.998620597
    ## 23   1.030148337  -1.932888541  6.853217e-01  8.938029e-01   0.469226828
    ## 24  -0.019359177  -0.484113020  1.226153e-01  1.143446e+00   0.198129014
    ## 25  -0.684839002   1.033498334  3.799745e-01 -3.582035e-01  -1.415979902
    ## 26  -0.210774412   0.841655497  9.016521e-02 -2.175855e-01  -0.655446318
    ## 27  -0.448946516   0.161388045 -8.253745e-02  4.326606e-01   0.188383953
    ## 28  -0.353664483   0.704042952  4.172324e-01 -1.142765e+00  -0.630011152
    ## 29  -1.254201162   0.112638152 -2.318595e-01 -3.019609e-03  -0.290223002
    ## 30  -0.315906599   0.521556982 -2.819849e-01  2.589277e+00  -1.683452243
    ## 31  -0.597582335   0.485027811  4.642883e-01  1.343882e-01  -0.947191882
    ## 32  -0.161914710  -0.693332660  1.061990e-01 -6.052508e-01   0.819861128
    ## 33   3.070078964  -0.384580311  7.826909e-01  5.719281e-01  -1.050590545
    ## 34  -0.359776372   0.052200077  8.344429e-02  3.318910e-01  -0.652780934
    ## 35  -1.021593958   0.221136881 -2.381511e-01  3.487570e-01  -0.362221516
    ## 36  -0.148760969  -0.254599228  7.997877e-02  3.325661e-01  -0.024263416
    ## 37   1.183049236  -1.175722233 -2.261859e+00 -2.319736e+00   1.680206509
    ## 38  -1.016557132  -0.070835593 -2.546671e-01 -7.659115e-01   1.103755260
    ## 39   0.933148669  -0.585895878  4.093590e-01  9.508296e-01  -0.266578017
    ## 40  -2.602351242   0.701386110  8.757935e-02  2.515833e-01   0.919564481
    ## 41   0.353595000  -0.895274292  9.028189e-01 -1.259234e+00   0.494493124
    ## 42   0.396619139  -0.857764841  4.046235e-01  1.143619e+00  -0.311026760
    ## 43  -0.354730648  -0.514610582 -3.590198e-02  1.811835e+00  -0.803582131
    ## 44  -0.732143831  -0.380642726  6.389891e-01  8.124622e-03  -1.051009795
    ## 45  -0.764556989   0.259655157 -2.920549e-01  3.297425e-01  -0.411539056
    ## 46  -0.007006736  -1.031907125  7.420278e-01 -8.645297e-01  -0.050989691
    ## 47   0.004203887  -0.354120880  4.310709e-01  9.726968e-01  -1.340685507
    ## 48   0.169139656  -0.068189335  3.643020e-01  1.701421e+00  -0.354780059
    ## 49  -0.314049155   0.659342612  8.079513e-02  6.133864e-01  -0.077413620
    ## 50  -0.780625490   0.099268338  1.832956e+00  5.196646e+00   6.285222449
    ## 51  -1.343609796   0.785002068 -1.300967e-01 -1.870707e+00  -0.194853161
    ## 52   1.839674575   1.329663008  3.585451e-03 -3.474621e-01  -0.140740663
    ## 53   0.324555301  -1.224265913  1.304781e-01  5.866076e-01   0.332307749
    ## 54   0.013824883   0.563968479  2.062457e-01 -7.074207e-01  -0.402061446
    ## 55  -0.038069843   0.593296319  1.749651e-02 -1.394092e+00   0.367929812
    ## 56  -1.111512903   0.765999936 -4.866335e-01  4.689091e-02   0.209912868
    ## 57   0.974533894  -1.964718866  3.750676e-01  1.412595e+00   0.428452732
    ## 58  -1.083456310   1.227479521 -4.378488e-01 -1.271903e+00  -0.592285231
    ## 59   0.084820132   1.212237398 -4.403405e-02 -1.133019e-01   1.149075122
    ## 60  -2.244368906  -1.224062140  6.066910e-01 -1.557227e+00  -0.082558041
    ## 61  -0.610427079   0.795021951  3.478814e-02 -5.538702e-01  -0.572425314
    ## 62   0.716981882   0.114156071  1.004673e+00 -6.174663e-01  -0.196647371
    ## 63   0.058199145   0.473925048 -5.457208e-02 -6.636195e-01   0.114574375
    ## 64   1.310421549  -0.245882687  4.445044e-01  5.284632e-01   0.099308358
    ## 65  -1.455268584  -1.107855088  1.007305e+00 -9.560043e-01   0.021625739
    ## 66  -0.398134945   0.522883972 -1.239907e+00 -9.981234e-02   0.880369913
    ## 67   0.112040865  -1.239022423  2.050849e+00  7.190253e-01   1.884601565
    ## 68  -0.663726720   0.081954827 -1.630989e-01  3.128064e-01   0.239920468
    ## 69  -0.804857164   2.698590875 -8.684154e-01 -1.778918e+00   0.187559148
    ## 70  -0.483233968   0.894074026 -1.590854e-02 -2.962397e-01  -0.252891871
    ## 71  -1.375324438   0.695150172 -1.055531e+00 -5.149447e-01  -1.136786505
    ## 72  -0.608732345   0.727949413 -5.653243e-02 -4.201808e-01  -0.326614667
    ## 73  -0.124160805  -0.587557184  4.777732e-02  1.202943e+00  -0.202658209
    ## 74  -0.816704507   2.331468154 -2.150352e+00  2.535723e+00   5.205894366
    ## 75  -0.540104238   0.653120553 -3.686810e-01  2.372897e-01   0.247432797
    ## 76  -0.678966680   0.955567261  5.635074e-02 -1.060623e+00  -0.657797755
    ## 77   0.723659022  -0.447516866  9.506197e-01 -2.736392e-01  -0.397054855
    ## 78  -0.335289169   1.256119812 -2.569043e-01  1.661468e-01  -1.211653702
    ## 79  -0.814403522   0.973234722  2.510371e-01 -1.028020e+00   0.246824138
    ## 80  -0.725006229  -0.403933058 -1.266488e-01  1.062942e+00  -0.153378399
    ## 81   0.418165495  -0.094790738  9.494321e-01 -4.041795e-01  -0.203408494
    ## 82   0.756208706   3.210651087 -2.979823e-01  1.157260e-01  -1.398915027
    ## 83   0.084784043  -0.194305494 -1.501516e-01  5.190349e-02   0.563900924
    ## 84  -0.249367392  -0.980467259  1.922188e-01  1.218368e+00   0.153484902
    ## 85  -0.667642796   1.054997437 -1.025684e-01 -7.745763e-01  -1.033142259
    ## 86   0.310253114  -0.059047534  1.663106e-01 -1.075360e-01   0.194152547
    ## 87  -0.298373720   0.096310971 -1.152339e-01  5.711553e-01   0.167713539
    ## 88  -0.406753245  -0.144936363  3.699734e-01  5.262014e-01  -0.806744739
    ## 89  -0.368395440   1.319584814 -1.950899e-01  1.348766e+00  -0.218407743
    ## 90  -0.801508947   0.801959311  2.353822e-01 -5.478279e-01  -1.075381611
    ## 91   0.160009727  -0.201997837  3.788268e-02 -1.817507e-01   1.306698940
    ## 92  -0.577415373  -1.376404862  3.961920e-02 -1.788494e+00   1.578969735
    ## 93   0.392314530  -1.142168555 -1.913198e-03 -1.447762e+00   1.699671931
    ## 94  -0.379055680  -2.083029465  8.767726e-02 -3.162689e-01   1.503251701
    ## 95  -0.546801053   0.015617469 -6.863552e-02 -2.927658e-01  -0.130194484
    ## 96   0.043767365  -0.880684360  2.551191e-01  1.209069e+00   0.227418119
    ## 97  -0.352396629   1.719765252 -5.492173e-01  8.362098e-02  -0.587044812
    ## 98   0.033901940   0.007542482  7.991657e-02 -8.600272e-02   0.027209557
    ## 99  -0.830214747   0.808109752  6.745547e-02 -7.936317e-01  -0.914517127
    ## 100  1.200327478   0.721649435  2.513080e-01 -1.568848e+00   0.300656362
    ## 101 -0.839979908  -2.116085355  3.640409e+00  5.694722e-01  -0.666533556
    ## 102 -1.234294223   1.015288044  1.385750e-01 -2.697170e-01  -0.962729491
    ## 103 -0.040893324   0.495310717 -3.470044e-03  3.937441e-01  -0.281546499
    ## 104  0.172896884  -1.052433378 -2.840287e-02 -1.511890e+00   1.595025982
    ## 105  0.104295320   1.088862037  3.380946e-01 -1.252613e+00  -0.513212068
    ## 106 -0.132209600   0.104178922 -1.679669e-01 -7.405087e-01   0.407252104
    ## 107 -0.136938656  -0.553551676 -4.902194e-02  1.239795e+00  -0.216943200
    ## 108 -1.131791679   0.173621503 -5.031818e-01 -1.890114e+00   1.094022757
    ## 109 -0.707480816  -0.157914052  9.675466e-01 -3.994130e-01  -0.861766220
    ## 110  0.787795339   0.416874912  7.597735e-01 -1.933391e-01  -0.824516761
    ## 111  0.542698451  -0.475897422  3.240402e-01  1.335541e+00  -0.373114913
    ## 112  0.185088553   0.516209418  2.137919e-01 -4.391812e-01  -0.116626078
    ## 113 -1.132966556   1.009690695 -2.117435e-01 -5.562006e-02  -0.411098689
    ## 114 -0.051238431   0.243480790 -3.819086e-01  1.939497e+00   0.343525026
    ## 115 -0.964027621   1.466907467 -4.951205e-01 -4.364158e-01  -0.235011817
    ## 116 -1.026394997   0.175327428  5.122120e-01 -3.483788e-01  -0.522003651
    ## 117  0.277984120  -0.332440471  8.095547e-01  5.903100e-02  -1.352203517
    ## 118 -0.573743270   0.071220231 -7.493361e-02  2.205413e-01  -0.217857770
    ## 119 -0.074196365  -0.615907425  1.396714e-02 -8.487492e-01   0.760236250
    ## 120  0.168509221   0.689915750 -9.630951e-01 -1.482996e+00   0.721925402
    ## 121 -1.129430167   0.840252151  1.933188e-01 -3.634920e-01   0.203211502
    ## 122  0.193129065  -0.854062625  6.944119e-02  2.414626e+00  -0.447441367
    ## 123 -0.618769764   0.065569348 -1.522294e-01  3.283917e-01   0.245474085
    ## 124  1.146962356   0.515584530 -2.232289e+00  8.467552e-01  -1.052480055
    ## 125  1.338022811  -0.517307366  4.345683e-01  2.601837e+00  -0.685821106
    ## 126 -0.808709747  -0.103778934 -1.524447e-02 -7.833019e-02   0.131924578
    ## 127 -0.665611453  -0.685322925  8.615706e-01 -1.843765e+00   0.821619562
    ## 128  0.018325337  -5.037860305 -7.080181e+00  2.404260e-01  -0.236528945
    ## 129  0.530777699  -0.159210320  4.884518e-01  2.381678e+00  -2.121544369
    ## 130  0.240762095  -0.620550088  1.841983e-01  4.186527e-01   0.011038649
    ## 131 -0.328359315  -3.785377781 -5.165635e+00  1.793691e-02  -0.052252760
    ## 132  0.616225840   0.179395019  6.768112e-01 -6.004937e-01  -0.784472705
    ## 133 -0.344282772   0.752585125 -2.117043e-01  7.957896e-01   0.833352632
    ## 134 -0.002904313   0.348306326 -1.065808e-03  6.954694e-01  -0.083038026
    ## 135 -0.496084737   0.406899511  1.120043e-02  3.970353e+00   1.872114510
    ## 136 -0.627798065  -0.061786937 -5.809904e-03  1.872236e-01  -0.296142425
    ## 137  0.029066868   0.705097017  1.428404e-01 -2.871929e-01  -0.728337457
    ## 138 -0.623186216   0.001919843 -5.018885e-02  1.790219e-01   0.017327691
    ## 139 -0.191380072  -0.662830086  1.287495e+00 -4.550242e-01  -1.036521819
    ## 140  0.679384020  -0.486774031  2.230192e-01  2.678274e+00  -0.830415305
    ## 141 -0.308431939  -0.184499973  1.601063e-01 -2.121188e-02  -0.118487332
    ## 142 -0.786551509  -0.141097464 -2.615434e-01 -1.142738e+00   0.822285770
    ## 143 -0.759678535  -0.044136597  3.561547e-01 -8.939055e-01  -0.730443377
    ## 144 -1.061849468   0.999766156 -1.975730e-01 -3.711320e-02  -0.412874955
    ## 145  3.690664425  -0.143791160  1.117855e+00  4.665874e-01  -1.207693806
    ## 146 -0.049519221   0.203383891  6.915854e-01 -9.759152e-01   1.109548897
    ## 147  1.623810230  -0.736658243  6.805579e-01  2.412917e-01  -0.182065722
    ## 148 -1.522310618   1.585806291 -3.321322e-01 -5.629641e-01  -0.286391872
    ## 149  0.258383106   0.388663040  1.373756e-01 -4.676938e-01   0.096051038
    ## 150  0.019692875  -0.287881301  5.558018e-02  1.036339e+00  -0.653393490
    ## 151 -0.183300250   1.222512879 -4.193920e-02 -1.157318e+00  -0.557363890
    ## 152 -0.347747121   0.470557231 -3.481550e-02  1.958169e-01  -0.803069832
    ## 153 -1.593917688  -3.734518523 -1.542304e+00 -1.633399e-01   0.993783394
    ## 154 -0.584262085   0.011556099 -1.349835e-01  1.358596e+00   1.269357341
    ## 155 -1.265151226   0.723974836 -6.595467e-01 -1.470159e+00   0.991161729
    ## 156 -0.700362495  -0.322184862 -1.087758e-01 -1.188703e+00   0.768248886
    ## 157 -0.032765486  -0.776135925  1.072750e-01  1.414137e+00   0.027569104
    ## 158 -0.646248112  -0.425559772 -3.524284e-01  8.692646e-01   0.349828531
    ## 159 -0.974977044   1.692655135 -5.963000e-01  5.345957e-01  -0.966958607
    ## 160  2.950344734   0.082189442  9.212496e-01 -6.314516e-01  -0.945418949
    ## 161  0.212714224  -0.714958931  1.471337e-01  1.241728e+00  -0.010113343
    ## 162 -0.117501982  -0.860450647  1.066539e+00 -1.605067e+00   0.561609477
    ## 163 -1.348584757   0.728987098  6.572106e-01 -1.262797e+00  -1.907000727
    ## 164  4.945267357  -2.226846720  1.456245e+00  2.434762e+00  -0.324075863
    ## 165 -0.731959899  -0.367359773 -2.786168e-01 -9.968395e-01   0.635635582
    ## 166 -1.129875177   0.167924065 -3.071659e-01  5.454132e-01   0.065774726
    ## 167  0.395031714   1.107400790 -1.206154e-02 -2.230828e+00   1.871626379
    ## 168  0.703494700   0.185760664  1.257952e-01  3.244031e-01  -0.230630217
    ## 169 -0.110610192   0.027417728  7.076961e-01  1.496821e+00  -1.874132356
    ## 170  0.308502740  -0.293909487  6.566080e-01  2.200907e-01   0.261947439
    ## 171  0.053857808   0.496120782 -1.038754e-01 -1.170757e-02   0.067192137
    ## 172  0.945527737  -2.074897791 -6.433228e+00 -4.442367e-01  -0.409198144
    ## 173  0.326370579   0.022001541  4.983681e-01  5.773190e-02  -1.285000966
    ## 174 -0.203003929   0.410001924  1.925929e-01 -5.610379e-01  -0.256512572
    ## 175 -0.533563228  -0.836240063  5.729232e-02 -4.378737e-01   0.615685857
    ## 176  2.191648803  -0.648921474  6.582411e-01 -7.901004e-01   1.365980882
    ## 177 -0.782013197  -0.699574110 -1.343474e-01 -8.418658e-01   0.720918163
    ## 178 -0.532186338   0.667990482  3.503413e-01 -7.832640e-03  -0.655225412
    ## 179  5.364899913 -10.630712660 -2.273976e+01 -2.566870e+00  -4.579688210
    ## 180 -0.887309502   0.731485065 -7.475016e-02 -4.146566e-01  -0.319980692
    ## 181  0.071238168  -0.486374310  7.311257e-02 -9.677411e-01   0.746299928
    ## 182 -0.363471345   0.672561191  4.384043e-01  7.851302e-02  -0.651966003
    ## 183 -0.311079114  -0.546034128  1.113520e+00 -3.844093e-01  -0.802474332
    ## 184 -0.800754017  -0.411611338 -1.446605e-01 -5.619460e-01   0.974089015
    ## 185  1.422115041  -0.402694941  4.815522e-01 -1.418004e+00   1.739753618
    ## 186 -0.173668492  -0.484410890  2.822957e-02  7.243167e-01  -0.234497027
    ## 187 -0.434288840  -0.702765347  9.326544e-02 -9.831267e-01   0.769150355
    ## 188 -0.917721291   0.542933594  3.762337e-01 -4.379878e-01  -0.461836160
    ## 189 -0.561869985   0.884446410 -3.073556e-02  1.545136e+00   1.816305808
    ## 190 -0.252280499  -0.911355072  5.242267e-02 -1.960374e+00   1.539019232
    ## 191 -0.953292184   0.052589456 -1.685742e-01  3.423772e-01   0.255404789
    ## 192  0.116965870   0.119265608  1.204225e+00 -4.098638e-01  -0.446614878
    ## 193 -0.546751886   0.480735174 -1.324573e-01 -1.337125e+00   0.366346649
    ## 194 -0.438593626  -1.117039349  5.115353e-02 -7.496878e-02   0.564983031
    ## 195 -0.566793530   0.136501892 -1.245912e-01  5.905533e-01  -0.795222265
    ## 196 -0.135437159  -0.307122763 -3.023047e-02  4.855635e-01  -0.169039823
    ## 197 -0.266654162   2.122983720 -4.284205e-01 -1.204221e+00  -0.306466817
    ## 198 -0.161067356  -1.133446472 -7.509875e-01 -6.187259e-01   0.744585518
    ## 199 -3.525315074  -1.228754207 -8.371083e-02 -5.876842e-01   0.060641342
    ## 200 -0.814948292  -1.050794711 -6.336815e-02  4.057578e-01   0.776251413
    ## 201 -0.092655936   0.185092605  1.535207e-01  2.495235e-01   0.060697192
    ## 202  1.742909560  -0.690703532  5.881747e-01  9.592993e-01  -0.174606799
    ## 203  0.677432918  -0.830764571  8.436037e-01  1.195650e+00  -0.658215811
    ## 204 -0.199318651   1.094902030 -3.444988e+00 -4.853720e-01  -0.951999611
    ## 205 -0.677449417   0.508322759 -6.763796e-02  1.668956e-01  -0.470373643
    ## 206 -0.409368957   1.265072390 -3.469529e-01 -6.646126e-01   0.156658256
    ## 207  3.293940660  -0.350997897  7.625076e-01 -1.541721e-01   0.071323922
    ## 208  0.105588985   0.336433331 -4.086111e-02  9.311823e-01  -0.221376965
    ## 209  0.704620158   0.746388540  1.800989e-01 -7.307874e-01  -0.186298118
    ## 210  0.530656497   0.487424418 -6.925840e-03  5.023819e-01  -0.459241808
    ## 211  0.317775696  -0.131578839  6.842782e-01  2.433146e-02  -0.359680618
    ## 212 -0.865006095   0.473886323 -3.970908e-01  2.541586e-01  -0.046463456
    ## 213  0.019253943   0.375385086  1.740581e-02 -1.060019e+00   1.694137797
    ## 214 -0.266101444   0.038744623 -1.808785e+00 -2.343905e-02   0.730037551
    ## 215 -0.322494534  -1.015612811 -1.506831e-01 -1.749647e+00   1.384083419
    ## 216 -0.270753365  -0.785410472  3.481345e-01  5.013463e-01  -0.244151021
    ## 217 -0.297970079   0.447949306 -1.910992e-01  1.671292e+00  -1.114123295
    ## 218 -0.836816778   0.875098224 -4.052640e-01 -3.806592e-02   0.193618325
    ## 219 -1.449630107   1.333346564 -2.159429e-01 -2.673011e-01  -0.776761213
    ## 220 -0.865340968   1.417859691  6.029936e-01 -3.062405e-02  -1.177053362
    ## 221 -1.442763992   0.274630847 -3.132270e-01  7.918729e-01  -0.586452674
    ## 222  1.329620677  -0.827281205  1.202143e+00  8.888963e-01   0.015685271
    ## 223 -0.209746160   0.887510970  5.246680e-02 -3.709755e-01  -0.689799023
    ## 224  0.508910692   0.190802095 -2.884836e-01  7.336842e-02   0.131988380
    ## 225 -1.012540919   0.636751294  6.147171e-01 -6.464556e-01  -0.300502964
    ## 226 -0.149397392   0.080450695 -7.376749e-02  8.728731e-01  -0.187663115
    ## 227 -0.376471699  -0.652398936 -1.198222e-01 -2.084223e+00   1.442744531
    ## 228  0.645132375  -0.350923277  5.998571e-01 -6.028494e-01  -0.026719224
    ## 229 -1.377009706  -0.327901534 -4.294852e-01 -1.960573e+00   1.853062243
    ## 230  0.126697567  -0.229541520  6.661314e-02  4.915743e-01  -0.149643272
    ## 231 -0.201465149  -0.477095311 -5.763119e-02 -3.630425e-01   0.803932470
    ## 232  0.385432941   0.152027444  1.877986e-01  8.148028e-01  -0.471141669
    ## 233 -0.115400075  -0.173231269  1.635685e-01 -4.670155e-02  -0.126172588
    ## 234 -0.863334234   0.553266219 -1.679004e-01 -1.827079e+00   0.270348805
    ## 235 -1.053396158   0.135249149 -2.423143e-01 -6.931904e-02  -0.303458389
    ## 236  0.501153554  -0.412363262  2.051225e-01  1.410217e+00   0.016093378
    ## 237 -0.284392080  -0.923721623 -1.645842e-01 -1.170168e+00   1.590055420
    ## 238 -0.598507487  -0.527542349 -1.078119e-01  1.387943e+00  -0.191420010
    ## 239 -0.115795917  -0.171950766  1.693177e-01 -2.678102e-02  -0.125856243
    ## 240  1.287358792  -0.691801643  5.574298e-01 -1.535567e-01   0.619404804
    ## 241  0.789850866  -1.379949667  4.899549e-01 -2.491121e-02   0.467273275
    ## 242 -0.853308713   0.179406655 -2.219435e-01  5.322848e-01  -0.375019626
    ## 243  0.349515292   0.213943456 -7.506033e-02 -1.635884e-01  -0.091679356
    ## 244  1.140356570  -0.164788977  3.764599e-01  2.412486e-01  -0.355749508
    ## 245  2.370471583  -4.234205532 -1.045702e+01 -1.382862e+00  -2.341490376
    ## 246  0.340993491  -1.446997330 -1.688070e+00 -7.239446e-01  -1.056770160
    ## 247 -1.594200221   1.052585841 -7.387247e-01 -1.085390e+00   0.873688122
    ## 248 -0.407249286  -0.857679688  4.021145e-04 -2.276225e+00   1.559965927
    ## 249 -0.641581359   0.788676492 -1.897205e-03 -2.899930e-01  -0.557309051
    ## 250 -0.773073926   0.672290346 -8.515741e-01  1.538427e+00   0.559768964
    ## 251 -0.560512772  -0.082415468  2.342698e-01  4.589538e-01  -0.627196412
    ## 252  3.919818853  -0.022740608  8.294604e-01 -1.457823e+00   1.513692553
    ## 253 -0.370099300   0.287547347 -3.169878e-01 -6.880545e-01   0.268304990
    ## 254 -0.531878912   0.090602823 -1.987565e-01  9.337269e-01  -0.072385981
    ## 255 -1.152466138  -0.016033088 -2.169811e-01  3.181147e-01  -0.189968971
    ## 256 -0.900029323   0.471315495 -2.048148e-01  1.400443e-01   0.459621776
    ## 257  0.070651030  -0.072354031  1.246123e-01 -3.142997e-01  -0.467815280
    ## 258  0.228002102  -0.381227759 -5.607911e-02 -9.426741e-01   0.740951320
    ## 259 -0.190406738  -1.268984946 -4.310678e+00 -5.495355e-01   1.704800921
    ## 260  2.876223990  -0.056650708  6.139640e-01 -1.990452e-01  -0.168128337
    ## 261 -1.723762590   0.373559686 -3.956697e-01  4.802845e-01  -1.212803429
    ## 262 -0.663899285   0.827357280 -2.234509e-01  1.680863e-01  -0.110015825
    ## 263 -0.559202531   0.233704330  2.396075e-01  5.274525e-01  -0.538601474
    ## 264  0.596206586  -1.481704477  3.484238e-01  2.781293e-01   0.978879127
    ## 265 -0.539332542  -0.603829657  1.198356e-01  1.204900e+00  -0.504991902
    ## 266 -0.579146589  -1.539982585  2.549498e+00 -3.745009e-01   0.276904949
    ## 267 -1.782236872  -0.835267015  6.874225e-01 -5.775640e-01  -0.936097060
    ## 268 -0.504203409   1.020338782 -2.138283e-01  1.516559e-01  -0.358924042
    ## 269 -0.913525596   0.165237361 -9.219467e-02 -1.157351e-01  -0.785030860
    ## 270 -0.497449849   1.143606443 -4.353308e-01  5.173414e+00   8.079414359
    ## 271 -1.076437773   1.360768723 -3.325777e-01  2.388883e-01   0.042799899
    ## 272  2.081273409  -0.938632010  6.744196e-01  2.203733e+00  -0.930252380
    ## 273 -0.639904834  -0.569369664 -6.488403e-02 -4.603748e-01   0.430203273
    ## 274 -1.356291968   0.875473633  6.169679e-01 -1.198720e+00  -1.976197967
    ## 275 -1.135363730   0.135071272  7.474878e-01 -1.579511e+00  -0.486908459
    ## 276  0.182288271  -0.617108302  1.530817e+00 -5.868323e-01   0.129875751
    ## 277 -1.007848240   0.699860179 -7.459593e-02 -2.325929e-01  -0.206045724
    ## 278 -0.250262237  -0.081823130 -8.227366e-01 -3.196881e-01  -0.138567182
    ## 279 -0.014649829  -0.571277122  5.024167e-02  6.288026e-01   0.157688067
    ## 280  1.911522709  -3.615259790 -7.848030e+00  2.306892e-01  -1.397107077
    ## 281 -0.938786073  -0.287846513 -1.248177e-01  8.552584e-01   0.154008610
    ## 282 -1.357650884   0.302030639  4.048129e-01  9.032211e-01  -2.064334729
    ## 283 -1.041167975   0.178479173 -2.810693e-01 -1.735926e-01  -0.322423445
    ## 284  0.119791219  -0.083767897  1.779446e-01  1.623472e-01   0.066675472
    ## 285 -0.131840982  -0.538585382 -4.928935e-02 -8.354128e-01   0.553382735
    ## 286 -0.768806136   0.095892090 -7.139848e-02 -2.114419e-01   0.194929617
    ## 287  0.638060918  -0.414465896  1.734239e-01 -4.912144e-01   1.357618286
    ## 288 -0.915881422   0.694430553 -7.754138e-02 -4.301510e-01  -0.314820712
    ## 289  0.195472342   0.077455512 -1.406453e-02  1.387844e-01   0.519047903
    ## 290  0.933681703  -1.577031809  3.486068e-01 -1.485205e+00   1.601506715
    ## 291 -0.011533488  -0.647535612  7.820034e-02  2.162909e+00   0.023265632
    ## 292  0.121554103  -1.308421271  3.495123e-02 -1.159242e+00   1.107050729
    ## 293 -0.288493350  -0.275298873 -5.614714e-02  1.524526e+00  -0.441563138
    ## 294  0.492624380  -0.446542913  3.231257e-01  5.437247e-01  -0.051668119
    ## 295  0.496258992  -0.684523830  2.348364e-01  1.422465e+00  -0.991631176
    ## 296  1.384183962  -0.000178495  1.565553e-01 -2.636873e-01   0.801913949
    ## 297  0.518261627  -0.349278591  1.927833e-01  1.404618e+00  -0.006049039
    ## 298 -0.953526086   0.152002545 -2.070714e-01  5.873353e-01  -0.362047348
    ## 299 -1.587890985   0.341468732 -1.350143e-01 -7.482601e-01  -0.404466883
    ## 300 -0.029080409   0.386138785  5.639296e-01 -9.709547e-01  -0.466977792
    ## 301  0.447844183  -0.925319386  6.104842e-01  3.196476e-01   0.196245631
    ## 302 -1.204172386   0.524226928 -3.329609e-01 -2.185304e-01   0.057509562
    ## 303 -0.641943237   0.836412090  1.913128e-02 -1.241227e-01  -0.212421629
    ## 304 -0.443678198  -0.421681384  8.647360e-01 -4.486940e-01   0.073763576
    ## 305  1.721404553  -1.265282820  5.622869e-01 -1.128913e-01   0.710464929
    ## 306  3.070693311  -7.547794766 -1.363486e+01 -3.159854e+00  -4.029387474
    ## 307  0.112258808  -0.745545367  8.023226e-02 -2.062109e+00   1.617168791
    ## 308  1.000447141  -0.621917461  2.651931e-01  9.638516e-01  -0.322975732
    ## 309 -1.907512796   0.101832470 -3.080777e-01 -4.935979e-02  -0.515674272
    ## 310 -0.817252565  -0.394711270 -8.450171e-02 -1.186880e+00   0.910292500
    ## 311 -0.420174942   0.956849808 -3.524062e-02 -9.124686e-02   0.169825295
    ## 312  2.752512585   1.140666170 -1.499384e-01  1.273153e+00  -0.073781882
    ## 313 -0.356625285  -0.690335062  6.676840e-01 -4.710594e-01  -0.402172339
    ## 314 -0.526324365   0.505039916  1.477429e-01 -4.994571e-01  -0.311498738
    ## 315 -0.357778618   0.122209336 -8.407089e-02  3.538587e-01  -0.931023816
    ## 316  0.122291792   0.415383328  2.398129e-01 -1.207697e+00   1.652930460
    ## 317 -0.709528065   0.066194474 -1.578390e-01  3.308687e-01   0.246693748
    ## 318  0.390612365  -0.176092409  3.837878e-01  4.803908e-01  -0.446532180
    ## 319  0.207081105   0.287096405  3.151385e-01 -8.932489e-01  -0.283911633
    ## 320 -0.581243165   0.580236766  3.149490e-01  9.026965e-01  -1.922153969
    ## 321  0.947861072  -0.304270961  1.300610e-01  5.642325e-01  -0.277882477
    ## 322 -0.830815477  -0.351832616 -2.218423e-01 -1.220379e-01   0.155932949
    ## 323 -0.323836694  -0.794323388  7.717320e-02  3.247844e+00  -1.471233061
    ## 324 -0.727720376   1.838202264 -4.585595e-01  4.799777e-01   0.402150763
    ## 325  1.641411838   0.210186643  4.378951e-01 -8.281622e-01   0.321812792
    ## 326 -0.102471983   0.049928830  1.563238e+00 -1.816186e+00  -1.268417498
    ## 327 -0.068389486   0.223598815  3.309800e-01 -5.560739e-02  -0.858599533
    ## 328 -0.188076939   0.265571175  4.673004e-02 -1.083579e+00  -0.264415852
    ## 329 -0.484536874  -0.537304741  6.673158e-02  1.076871e+00   0.004030485
    ## 330 -1.080232271   0.193739664 -3.233922e-01  4.964071e-01   0.054323410
    ## 331  0.167133851  -0.694717048  6.723967e-02 -5.076154e-01   0.628900470
    ## 332 -0.057450175   0.709004901 -1.068869e-01  1.268106e-01   0.204084629
    ## 333 -1.754037936   1.759567047 -4.746536e-01 -1.023270e-01  -0.175973577
    ## 334  0.369313830  -0.727171405  2.198842e-01  9.259383e-01   0.268182570
    ## 335 -0.232108332  -1.001202032  3.824931e-02  2.365656e-01   0.086204651
    ## 336 -0.887595058   0.579517176 -3.631422e-01  2.925149e-01   0.254479936
    ## 337 -0.767251419   0.123546109 -2.880999e-02 -5.362301e-02   0.191349548
    ## 338  1.350738931  -0.429964271  5.853160e-01 -1.143703e-01  -0.065299037
    ## 339  1.267945251  -1.063945938  6.166575e-01  2.084349e-01   0.646642209
    ## 340  3.179093967  -0.559795164  1.458278e+00 -5.342798e-01  -0.389545778
    ## 341 -0.833297568  -0.173893531  4.346746e-02  7.814978e-02   0.162277929
    ## 342 -1.035570645  -0.546132933  1.577602e+00 -2.407133e-01  -0.527823525
    ## 343  0.871117263  -0.107407637  1.307596e-01  1.436603e+00  -0.338844542
    ## 344 -0.165851392  -0.237570716 -9.401821e-02  1.399630e+00  -0.066691820
    ## 345 -0.242785664  -0.025618076  1.122969e+00 -1.466947e+00  -0.573530200
    ## 346  0.756864068  -0.753632249  3.958790e-01  1.631951e+00   0.137572902
    ## 347 -0.128722361   0.001560002  7.962628e-01 -4.463839e-01  -0.544332639
    ## 348  0.568370783  -0.731632782  3.684498e-01  3.195351e-01   0.128521089
    ## 349  0.210831811  -0.900841765  1.974719e-01  1.627918e-01   0.823091491
    ## 350 -0.339403492  -0.070331941 -1.324704e-02  4.582519e-01  -0.294688567
    ## 351 -2.275705790   0.722680500 -2.093972e-01  5.022727e-02  -1.088541450
    ## 352 -1.312385561   1.753732035 -5.091543e-01  1.364078e-01   0.843411116
    ## 353  1.697971095  -0.901654516  4.139783e-01 -3.800917e-01   0.692179347
    ## 354 -0.170267214   0.097024071 -3.667561e-02  2.351843e-01  -0.136862270
    ## 355 -1.461879391   1.910204624 -8.796262e-01 -9.910152e-01   0.592770510
    ## 356 -2.049790470   0.259843009 -4.177052e-01  6.306147e-01  -0.457508557
    ## 357  1.120830739   1.367825559 -3.276356e-01  4.863390e-01  -0.395842421
    ## 358 -1.209651315   0.474821457 -1.737086e-01 -3.792655e-01  -0.695150295
    ## 359 -0.274794521   1.588211427 -2.308476e-01 -2.199019e+00   1.227232227
    ## 360 -1.080057903   0.587170600  1.301538e-01 -4.436057e-01  -1.078339938
    ## 361 -1.801498405  -3.920569503  6.667838e+00  1.509848e-01  -0.372890187
    ## 362 -0.780608283   1.176477727  1.693490e-01 -7.048566e-01  -0.419573379
    ## 363 -0.931582049   0.047541760  9.074070e-01  1.212299e+00   3.624540997
    ## 364  0.077901875  -0.435587724  2.829127e-01 -1.967172e+00   0.907036825
    ## 365 -0.075955372  -1.293748022  3.914427e-02 -1.259868e+00   1.667815347
    ## 366 -1.146468916   0.604902694  2.847151e-01 -8.573224e-01  -1.229577407
    ## 367  0.237694725   0.005154096  5.129638e-01 -1.553247e+00  -0.394129732
    ## 368 -0.716100688   0.035953817  4.547812e-02 -1.583399e+00   0.784857740
    ## 369 -0.579119603   0.882120205  1.124672e-01 -2.613026e-01  -0.939662900
    ## 370 -1.090592275   0.851460784 -6.005179e-02  2.785715e-01  -0.355741278
    ## 371  2.162259492  -0.439078198 -3.913686e-01  2.420301e-01  -0.008275522
    ## 372 -1.127751951   0.178493435 -3.032335e-01  5.645092e-01   0.062830722
    ## 373  1.167958088  -1.380569877  5.307156e-01 -2.312072e-01   0.895227906
    ## 374 -0.471623793  -0.802887137  3.935845e-01  1.880216e-01  -1.110406743
    ## 375  0.958383708  -1.192322227  3.845883e-01  2.168173e-02   0.443365574
    ## 376  0.063354067  -1.274177773  1.258174e-01  3.737663e-01   0.873076270
    ## 377 -0.302619802   0.136423805 -1.580159e+00 -1.108245e-02   0.141275038
    ## 378  0.212501980   0.600004206  1.438554e-01 -3.287910e-01  -0.096470523
    ## 379 -0.791272327   0.153393726  5.730512e-01 -6.315836e-01  -0.330558726
    ## 380 -1.288796163   0.372187923 -4.946564e-01 -1.171997e+00   1.036358301
    ## 381 -1.313551506   0.213296875  5.350961e-01  1.291762e-01  -1.456010280
    ## 382 -0.007043868  -0.702042763  2.645369e-01  7.248136e-01  -0.014094796
    ## 383 -2.115311749   1.144218323 -6.771101e-01  2.367648e-01   0.061768118
    ## 384  0.338589000   1.411016965 -8.631682e-01  1.205437e+00   1.164786087
    ## 385  0.951364323  -0.622213210  2.636911e-01 -8.749412e-01   0.884463289
    ## 386 -0.596890534   0.500968813 -2.014560e-01  2.562962e-01   0.230257566
    ## 387 -1.220681578  -0.034446290  1.920035e-01 -5.436066e-01   0.363348034
    ## 388 -0.846971002  -0.774039064  4.415781e-01 -2.212928e+00   0.190147401
    ## 389 -0.797892264  -1.209143210 -2.367479e-01 -1.145950e+00   1.461775430
    ## 390  4.642815364   0.019342675  1.256290e+00 -1.208743e-01  -0.487558798
    ## 391 -0.772542681  -0.539487946 -1.313905e-02  1.365470e+00  -0.094695255
    ## 392 -1.164681754   0.748460490 -4.966667e-01  3.705724e-01   0.030443113
    ## 393 -0.782231789   0.209551496 -2.492564e-01  4.443778e-01  -0.389741039
    ## 394  4.107287448   2.011055755  5.172065e-01 -8.778862e-01  -0.762797256
    ## 395 -0.016124200  -1.108325028 -8.090203e-02 -7.334161e-01   1.327697397
    ## 396 -0.254129337   1.898066341 -3.801802e-01 -9.046010e-01  -0.076523939
    ## 397 -1.153180633   0.794784296 -2.310917e-01 -2.590080e-01  -0.267961456
    ## 398 -0.377859145  -0.894666337 -1.165823e-01 -1.875794e+00   1.389886385
    ## 399 -0.829801702   1.123111100 -4.876205e-01  1.817499e-01   0.334604215
    ## 400  0.219406221   1.681343691  1.455419e-01  1.943360e-01  -1.512677444
    ## 401  2.034991725   6.201364724 -5.134543e-01 -1.120183e+00  -1.766276539
    ## 402  1.089754586  -0.290252940  3.162915e-01  9.458247e-01  -0.174261158
    ## 403  0.165910603  -0.029010439 -1.678299e+00 -1.509720e+00   0.315491041
    ## 404 -0.336878441   0.550696334  4.450390e-01 -6.001902e-03  -1.319354088
    ## 405 -0.455365262  -1.114487241 -1.499101e-01 -1.668635e+00   1.344000012
    ## 406 -0.363724781   1.227555332 -2.622544e-01 -7.811231e-01  -0.058777344
    ## 407  3.055111818  -1.104713780  6.218781e-01 -2.212086e+00   1.500672976
    ## 408 -1.157077618   1.068632053 -1.191513e-01 -4.965381e-01  -1.080909802
    ## 409  0.189756503  -0.082328924  1.291197e-01 -1.540648e+00   0.506262729
    ## 410  0.070243816  -0.310510782  6.833918e-02 -3.544978e-02   0.490713040
    ## 411  0.402664506   1.727539817 -5.173029e-01 -1.147503e+00  -0.165999428
    ## 412  3.585259633   0.170930295  7.610000e-01 -4.609560e-01  -0.003253840
    ## 413 -1.019858685   0.272372289 -3.945851e-01 -1.128920e+00   1.134987355
    ## 414 -0.280928525   0.170679449  1.358910e-01  2.220774e-02  -0.983728509
    ## 415 -0.799114771  -0.530797414 -1.132119e-01 -1.148524e+00   0.729158485
    ## 416 -0.201654583   0.492765868  2.400218e-01 -5.860731e-01  -0.371244636
    ## 417 -2.106040644   0.667058005 -3.361958e-01 -1.523826e+00   0.035357127
    ## 418 -0.745591008   0.162176492  2.953213e-01 -7.519693e-01  -0.088396397
    ## 419 -0.369313443   0.806456691 -2.588616e-01 -3.634020e-01   0.473802335
    ## 420 -0.971785535   0.773863550  6.860842e-01  3.650388e-03   0.785097732
    ## 421 -0.961883141  -1.061139005  1.938125e+00  1.373822e+00   2.104248641
    ## 422 -0.078641554   0.009479556 -3.152482e-01  3.165151e-01   0.088704196
    ## 423 -1.172043111   1.144345890  1.858822e-01  3.582564e-02   0.812297654
    ## 424 -0.078239810  -0.698527577 -1.615608e-02  2.155761e+00  -0.571874411
    ## 425 -0.911928406   0.122464942 -2.565911e-01  8.485339e-01  -0.080928855
    ## 426 -0.771303369   0.262673994  5.353583e-01 -3.661105e-01  -0.700657070
    ## 427  3.478809295  -0.542321315  8.133677e-01  1.160271e+00  -0.343230584
    ## 428  0.286067317  -1.437013234  4.227707e-02 -1.219213e+00   1.669361242
    ## 429 -0.498841366   0.843246753  3.453624e-01 -9.754434e-01  -0.040650438
    ## 430 -0.012252145   0.523007177  2.421800e-01 -6.136321e-01  -0.387357775
    ## 431  0.019668016  -0.085564617  3.554398e-01 -9.451760e-01   0.682290930
    ## 432 -0.740045235  -0.001286086 -3.003771e-01  2.207544e+00  -0.439908205
    ## 433  1.071708944  -0.303789001  3.919721e-01  8.931174e-01  -0.516812851
    ## 434 -0.814310030   0.746138849 -2.416997e-01  6.008493e-04  -1.146890035
    ## 435 -1.593148907   0.016451942  1.856648e-02 -7.686514e-01   0.139917451
    ## 436 -0.223416559  -0.194613763  1.116934e-02 -2.845944e+00   0.819881295
    ## 437 -0.930400838   0.395576286 -3.450288e-01  3.398460e-01  -1.245783609
    ## 438  0.698836687  -1.353540281  2.552965e-01  1.210464e+00   0.356179535
    ## 439  2.392604123   4.345092818 -3.156590e+00 -1.859882e+00  -2.150814146
    ## 440 -0.664377327   0.652469549 -2.291094e-01 -1.460996e-01  -0.124627880
    ## 441  2.920488281   0.333187011 -9.687811e-01  1.738311e+00   3.148594672
    ## 442  0.756301592  -0.790212273  3.924483e-01 -6.217749e-01  -0.064972373
    ## 443 -0.199168521  -0.279867455 -6.542099e-02 -2.329613e+00   0.956798063
    ## 444  3.029389424   0.168870135  5.824863e-01 -3.040133e-01   0.393635437
    ## 445 -0.902462008   0.909337310 -3.577831e-01  2.280545e-01   0.602316282
    ## 446  1.143717054  -0.947968415  7.170272e-01  3.727627e-02  -0.176783273
    ## 447 -0.941197407   0.003681045 -3.777825e-01 -5.257986e-01   0.786670673
    ## 448 -2.247281156  -2.032339116  2.003429e-01 -9.430609e-01   0.623102212
    ## 449  2.080617401   2.533143741 -9.082164e-02  1.792119e+00  -1.222250277
    ## 450  0.047446112   0.917995994  2.156944e-01 -7.607071e-01  -0.684761583
    ## 451 -0.951905301   1.153002162 -2.324188e-01  4.158931e-01  -2.012191181
    ## 452 -0.727709961   0.958248165 -5.845910e-01 -3.557323e-01   0.377423198
    ## 453 -1.114281509   0.404636772  5.255546e-01  1.255661e+00  -2.450348801
    ## 454 -0.429186603  -0.345186442  4.224606e-01  4.547102e-01  -0.689836018
    ## 455 -0.189461392   1.012934630 -3.430419e-01 -1.006005e+00   0.707760110
    ## 456 -1.426545319  -2.537387306  1.391657e+00 -2.770089e+00  -2.772272145
    ## 457 -1.064822523   0.325574266 -6.779365e-02 -2.709528e-01  -0.838586565
    ## 458 -0.075787571   0.562319782 -3.991466e-01 -2.382534e-01  -1.525411627
    ## 459 -1.706536388  -3.496197293 -2.487777e-01 -2.477679e-01  -4.801637406
    ## 460 -1.357745663   1.713444988 -4.963585e-01 -1.282858e+00  -2.447469255
    ## 461 -3.353059548  -1.631734673  1.546124e-01 -2.795892e+00  -6.187890630
    ## 462 -2.513478848  -1.689102200  3.032528e-01 -3.139409e+00  -6.045467798
    ## 463 -2.420168414  -0.812891249  1.330801e-01 -2.214311e+00  -5.134454471
    ## 464 -2.651353112  -0.746579273  5.558631e-02 -2.678679e+00  -4.959492912
    ## 465 -1.732192568  -3.968592618  1.063728e+00 -4.860966e-01  -4.624984954
    ## 466 -2.667553561  -3.878088455  9.113371e-01 -1.661990e-01  -5.009248502
    ## 467 -1.734662427  -3.059245032  8.898048e-01  4.153821e-01  -3.955812344
    ## 468 -1.879437193  -3.513686871  1.515607e+00 -1.207166e+00  -6.234561332
    ## 469 -2.731689012  -3.430559149  1.413204e+00 -7.769415e-01  -6.199881763
    ## 470 -2.730761517  -1.496497056  5.430152e-01 -2.351190e+00  -3.944238498
    ## 471 -3.035137548  -1.713402063  5.612574e-01 -3.796354e+00  -7.454840651
    ## 472 -2.966326542  -2.436652527  4.893279e-01 -3.371639e+00  -6.810813099
    ## 473 -2.278351654  -4.684053936  1.202270e+00 -6.946955e-01  -5.526278060
    ## 474 -2.185040262  -4.716142945  1.249803e+00 -7.183261e-01  -5.390330256
    ## 475 -2.341016828  -4.235253083  1.703538e+00 -1.305279e+00  -6.716720023
    ## 476 -2.332284887  -4.261237197  1.701682e+00 -1.439396e+00  -6.999906634
    ## 477 -0.989908136  -4.577264627  4.722162e-01  4.720170e-01  -5.576022636
    ## 478 -3.131549732  -3.103569746  1.778492e+00 -3.831154e+00  -7.191604246
    ## 479 -3.136371964  -3.104557363  1.823233e+00 -3.878658e+00  -7.297803350
    ## 480  3.628382091   5.431270921 -1.946734e+00 -7.756801e-01  -1.987773188
    ## 481 -2.854415285  -7.810440745  2.030870e+00 -5.902828e+00 -12.840933818
    ## 482 -2.807313947  -0.591118213 -1.234956e-01 -2.530713e+00  -5.153094610
    ## 483 -2.073129006 -10.089931163  2.791345e+00 -3.249516e+00 -11.420450974
    ## 484 -2.985288022  -8.138589417  2.973928e+00 -6.272790e+00 -13.193415067
    ## 485 -2.886722351  -1.341035996  3.639333e-01 -2.203224e+00  -4.137840196
    ## 486 -1.549420289  -4.104214873  5.539341e-01 -1.498468e+00  -4.594951763
    ## 487 -1.529322966  -4.487421960  9.163918e-01 -1.307010e+00  -4.138891214
    ## 488  0.320350587  -0.954940039 -3.277535e+00  2.820829e+00   1.015112880
    ## 489 -3.055115785  -9.301288615  3.349573e+00 -5.654212e+00 -11.853866973
    ## 490 -2.244030791 -11.199975217  4.014722e+00 -3.429304e+00 -11.561949772
    ## 491 -3.179948757  -9.252793938  4.245062e+00 -6.329801e+00 -13.136698369
    ## 492 -3.250013184 -10.415697812  4.620804e+00 -5.711248e+00 -11.797181068
    ## 493 -3.322688735 -10.788372839  5.060381e+00 -5.689311e+00 -11.712186624
    ## 494 -1.040902572  -1.593900740 -3.254905e+00  1.908963e+00   1.077417521
    ## 495 -0.221980885  -0.022988728 -1.087354e-02  8.600436e-01  -0.592472935
    ## 496 -0.214414558   0.053557839 -1.103534e-01  8.837978e-01  -0.554223901
    ## 497 -0.093625175   0.266155170  8.298824e-02  5.802554e-01  -0.164562675
    ## 498 -3.395374852 -11.161057003  5.499963e+00 -5.667376e+00 -11.627193556
    ## 499  0.672247839  -9.462532811  5.328704e+00 -4.897006e+00 -11.786811656
    ## 500 -4.445610382 -21.922811034  3.207923e-01 -4.433162e+00 -11.201400086
    ## 501 -3.372098209 -16.535806811 -1.443947e+00 -6.815273e+00 -13.670545126
    ## 502  0.130669812 -15.600323304 -1.157696e+00 -5.304631e+00 -12.938929311
    ## 503 -2.120936575 -14.913329985 -7.212141e-01 -7.175097e+00 -14.166794660
    ## 504 -2.202710277 -15.471612280 -3.565949e-01 -6.380125e+00 -13.348277654
    ## 505 -2.262846420 -15.833442782  7.787367e-02 -6.356833e+00 -13.261651708
    ## 506 -2.324306949 -16.196418660  5.128818e-01 -6.333685e+00 -13.175198079
    ## 507 -2.386893207 -16.560368108  9.483486e-01 -6.310658e+00 -13.088890918
    ## 508 -2.450444374 -16.925152044  1.384208e+00 -6.287736e+00 -13.002709301
    ## 509 -2.514828873 -17.290656675  1.820408e+00 -6.264903e+00 -12.916636109
    ## 510 -2.579938008 -17.656787996  2.256902e+00 -6.242149e+00 -12.830657200
    ## 511 -2.645681199 -18.023467673  2.693655e+00 -6.219464e+00 -12.744760787
    ## 512 -1.184322994   0.397585526 -2.534985e-01  4.111350e-01  -0.859862302
    ## 513 -4.027030447 -13.897826598  1.066225e+01 -2.844954e+00  -9.668788912
    ## 514 -4.128779003 -14.118864847  1.116114e+01 -4.099551e+00  -9.222825507
    ## 515 -4.166647245 -14.409447985  1.158080e+01 -4.073856e+00  -9.153368038
    ## 516 -0.243317741   0.316192138  1.229601e-01 -1.143343e+00  -0.369908966
    ## 517 -4.146523506 -14.856123673  1.243114e+01 -4.053353e+00  -9.040396249
    ## 518 -4.308887607 -15.357951807  1.285717e+01 -3.999861e+00  -8.928655661
    ## 519 -4.346724083 -15.648507472  1.327680e+01 -3.974162e+00  -8.859194059
    ## 520 -4.384490669 -15.939002694  1.369642e+01 -3.948455e+00  -8.789723363
    ## 521 -4.422194594 -16.229443725  1.411600e+01 -3.922741e+00  -8.720244515
    ## 522 -4.459842126 -16.519835985  1.453556e+01 -3.897022e+00  -8.650758329
    ## 523 -4.497438720 -16.810184192  1.495511e+01 -3.871297e+00  -8.581265516
    ## 524 -4.534989152 -17.100492477  1.537463e+01 -3.845567e+00  -8.511766697
    ## 525 -4.572497620 -17.390764470  1.579414e+01 -3.819832e+00  -8.442262418
    ## 526 -4.609967826 -17.681003372  1.621363e+01 -3.794093e+00  -8.372753160
    ## 527 -4.647403049 -17.971212019  1.663310e+01 -3.768351e+00  -8.303239351
    ## 528 -4.684806204 -18.261392934  1.705257e+01 -3.742605e+00  -8.233721370
    ## 529 -1.130281900  -2.814262660  6.486632e-01 -6.952626e-01  -3.465689086
    ## 530 -4.666313063 -18.709478821  1.790357e+01 -3.722279e+00  -8.120961738
    ## 531  0.565834734   0.303653111  9.669145e-01 -2.245882e+00  -3.651426569
    ## 532 -4.828202466 -19.210896418  1.832941e+01 -3.668735e+00  -8.009159386
    ## 533 -4.865613418 -19.501084075  1.874887e+01 -3.642990e+00  -7.939642419
    ## 534 -4.902997397 -19.791248405  1.916833e+01 -3.617242e+00  -7.870121943
    ## 535 -4.940356329 -20.081391075  1.958777e+01 -3.591491e+00  -7.800598208
    ## 536 -4.977691964 -20.371513594  2.000721e+01 -3.565738e+00  -7.731071441
    ## 537 -1.800903718   1.672734122 -3.002403e-01 -2.783011e+00  -1.884842440
    ## 538 -2.728395073   1.987616355 -3.573447e-01 -2.757535e+00  -2.335933327
    ## 539 -2.269219136  -0.824203333  3.510699e-01 -3.759059e+00  -4.592389806
    ## 540 -0.639736123  -3.801324502  1.299096e+00  8.640655e-01  -2.895251668
    ## 541  0.289966026   1.767760449 -2.451050e+00  6.973581e-02   3.245086401
    ## 542 -0.984744666  -2.202317927  5.550879e-01 -2.033892e+00  -2.734155555
    ## 543  0.081567933   0.268048731  6.604374e-01 -2.374027e+00  -3.582810020
    ## 544  0.028329559  -1.515521380  5.370354e-01 -1.999846e+00  -2.133176443
    ## 545  0.563341023  -0.123372252  2.231219e-01 -6.735976e-01   0.644549730
    ## 546 -0.654658532  -2.275788510  6.752290e-01 -2.042416e+00  -2.834871160
    ## 547 -1.436138771  -2.206056242 -2.282725e+00 -2.928850e-01  -3.717449929
    ## 548 -1.811286628  -5.479117451  1.189472e+00 -3.908206e+00  -7.060746182
    ## 549 -0.639268313  -0.974072997 -3.146929e+00 -3.159074e-03  -0.121653218
    ## 550 -1.852981871  -3.069958214 -1.796876e+00 -2.133561e-01  -3.551984009
    ## 551 -1.253474118  -5.778879593  1.707428e+00 -4.467103e+00  -6.067798150
    ## 552 -1.422686338  -5.651313740  2.019657e+00 -5.015491e+00  -6.319707509
    ## 553 -0.905945283  -6.652031121  2.634524e+00 -4.679402e+00  -6.546241815
    ## 554 -1.163406658  -6.805053053  2.928356e+00 -4.917130e+00  -6.600460622
    ## 555 -1.224644707  -7.281327867  3.332250e+00 -3.679659e+00  -7.524368337
    ## 556 -1.740767456  -6.311698529  3.449167e+00 -5.416284e+00  -7.833555571
    ## 557 -1.755633324  -6.958679143  3.877795e+00 -5.541529e+00  -7.502112191
    ## 558 -3.394483429  -5.283435335  1.316189e-01  6.581764e-01  -0.794993882
    ## 559  0.852221660   2.785162546 -3.036085e-01  9.400055e-01  -1.965308809
    ## 560 -0.771419408   0.230307373  9.368313e-02 -1.675943e-01  -1.959809258
    ## 561 -2.010361110   1.744085818 -4.102865e-01 -2.450198e+00  -2.042167863
    ## 562 -4.297177163  -2.591241685  3.426706e-01 -3.880663e+00  -3.976525139
    ## 563 -3.381586453  -1.934372441  5.623221e-01 -3.104027e+00  -3.051210355
    ## 564 -4.153568445  -9.360095134  1.922075e+00 -4.026180e+00 -13.691315133
    ## 565 -0.721552488  -4.195341543 -5.983458e-01 -2.870145e+00  -5.290610062
    ## 566 -0.092781815  -4.388698809 -2.801327e-01 -2.821895e+00  -4.466284163
    ## 567 -2.478094956  -9.938411559  2.830086e+00 -5.659162e+00 -11.298156492
    ## 568 -1.729184930  -8.190191720  2.714670e+00 -7.083169e+00 -11.141277690
    ## 569 -3.043536081 -10.989184667  3.404129e+00 -6.167234e+00 -11.435623996
    ## 570 -3.661797896 -10.894078555  3.709210e+00 -5.859524e+00 -12.981619145
    ## 571 -2.822126187 -10.333405656  4.031907e+00 -6.648778e+00 -11.634414425
    ## 572 -2.382710665 -11.508841858  4.635798e+00 -6.557760e+00 -11.519860926
    ## 573 -2.376819975 -11.949723457  5.051356e+00 -6.912076e+00 -11.589748311
    ## 574 -2.522046051 -11.643028274  5.339500e+00 -7.051016e+00 -12.265323844
    ## 575 -3.716264156 -12.407313056  5.626571e+00 -6.232161e+00 -13.386683440
    ## 576 -2.113207542  -9.984287448  5.541941e+00 -7.383705e+00 -13.215172300
    ## 577 -3.586806344 -13.616797054  6.428169e+00 -7.368451e+00 -12.888158288
    ## 578 -2.832513464 -12.703252659  6.706846e+00 -7.078424e+00 -12.805683190
    ## 579 -3.860820337 -13.547302195  7.096472e+00 -6.294029e+00 -13.608143163
    ## 580 -2.792360038 -12.561782581  7.287122e+00 -7.570322e+00 -12.835737683
    ## 581 -2.813535856 -14.248846531  7.960521e+00 -7.718751e+00 -13.074068166
    ## 582 -2.681518651 -14.019290758  8.218191e+00 -7.930900e+00 -12.695947404
    ## 583 -2.919844990 -14.594561588  8.622905e+00 -8.090697e+00 -12.780633936
    ## 584 -3.146176367 -15.450467217  9.060281e+00 -5.486121e+00 -14.676470250
    ## 585 -4.246957124 -14.716667552  9.435084e+00 -6.795398e+00 -15.124162814
    ## 586 -2.782437559 -14.263735349  9.643419e+00 -7.701499e+00 -14.226698058
    ## 587 -2.958430501 -16.165538624  1.007525e+01 -7.901821e+00 -13.009402806
    ## 588 -2.832404299 -16.701694296  7.517344e+00 -8.507059e+00 -14.110184442
    ## 589  0.031823406  -2.175778099  6.990717e-01 -1.140208e+00  -3.226787107
    ## 590 -1.911563593 -18.014660131  5.522162e+00 -9.283925e+00 -14.557159053
    ## 591 -0.126270561  -4.744729737 -6.533104e-02 -2.168366e+00  -4.758304075
    ## 592  0.747235432   1.525637535  1.929878e-01 -2.431862e+00  -4.155837834
    ## 593 -3.114136482  -6.895112283  5.161516e+00 -2.516477e+00  -6.403370552
    ## 594 -1.668347707 -21.340478099  6.418997e-01 -8.550110e+00 -16.649628160
    ## 595 -2.239565648 -21.234463154  1.151795e+00 -8.739670e+00 -18.271168174
    ## 596 -1.089999713  -5.551433070  4.477831e-01 -2.424414e+00  -5.699922283
    ## 597 -3.058806055  -0.184709665 -3.904200e-01 -3.649812e+00  -4.077585409
    ## 598 -2.646162265  -1.668931064 -2.617552e+00 -3.945843e+00  -4.565252094
    ## 599 -2.832662998  -7.619765445  1.618895e+00 -2.992092e+00  -7.142199114
    ## 600  0.010848878  -3.409095679  1.409291e+00 -3.260672e+00  -7.781352763
    ## 601 -0.116106055  -1.621986235  4.580279e-01 -9.121892e-01  -2.961995702
    ## 602 -1.961578319  -3.029283445 -1.674462e+00  1.839613e-01  -4.980928438
    ## 603 -1.168951525  -2.564181983  2.045325e-01 -1.611155e+00  -1.250285537
    ## 604 -0.918546734  -5.715262078  8.310396e-01 -2.457034e+00  -5.726817096
    ## 605 -0.665731671   0.059110490 -3.153277e-03  1.122451e+00  -1.481245861
    ## 606 -1.317703909  -1.528142159 -6.209534e-01 -1.213040e+00  -2.975267248
    ## 607  0.190965634  -0.001112104  1.471402e-01  5.804153e-01  -0.792937991
    ## 608 -1.016897033  -2.059730520 -2.751664e-01 -1.562206e+00  -2.755796928
    ## 609 -1.390119865  -1.709109107  6.677476e-01 -1.699809e+00  -3.843911040
    ## 610 -1.092671357  -2.230986064  1.036425e+00 -1.895516e+00  -3.364010580
    ## 611 -0.091183566   0.504134851  1.610643e-01 -7.650538e-01  -0.550545348
    ## 612 -0.781870547   0.228750322 -4.084008e-02 -4.321106e-01  -0.585778431
    ## 613 -0.733908445   0.855828927  7.684623e-05 -1.275631e+00  -0.433393926
    ## 614 -1.728161006  -2.058582365  3.588951e-01 -1.393306e+00  -3.505790326
    ## 615 -1.686110213  -1.864611505  8.561223e-01 -1.973535e+00  -3.942382904
    ## 616 -0.483871348   0.780731181 -3.487758e-01  6.091335e-01   0.225933803
    ## 617 -0.197428734   0.049166260  3.779163e-02  1.281186e-01  -0.552902906
    ## 618 -0.768219363  -1.997237401  5.749965e-01 -9.808318e-01  -2.495619245
    ## 619  0.133028430  -0.286225919  5.559446e-01 -1.394918e+00  -2.892611761
    ## 620 -3.221146649  -7.553496511  6.015618e+00 -2.466143e+00  -6.246243185
    ## 621 -3.386618049  -8.058011947  6.442909e+00 -2.412987e+00  -6.134906887
    ## 622 -3.427045237  -8.350808153  6.863604e+00 -2.387567e+00  -6.065782363
    ## 623 -3.466996536  -8.643192792  7.284105e+00 -2.362097e+00  -5.996595920
    ## 624 -3.506561173  -8.935243032  7.704449e+00 -2.336584e+00  -5.927359169
    ## 625  0.654936785   1.448867826  2.330791e-02 -1.367424e-01  -0.150129183
    ## 626  0.202180695  -0.689804056  4.113799e-01  3.367692e-01  -0.283731284
    ## 627 -0.180896946  -2.943677575  8.591555e-01 -1.181743e+00  -3.096503599
    ## 628 -0.371634021   0.158125824 -2.026687e-01 -7.951176e-02  -0.045088248
    ## 629 -0.576887995  -2.582865044  6.432300e-01 -1.191233e+00  -3.095093607
    ## 630 -1.261960668  -2.340990939  7.130036e-01 -1.416265e+00  -2.996669302
    ## 631 -0.889939993  -2.355253671  8.546586e-01 -1.281243e+00  -2.705011382
    ## 632 -0.174142123  -3.039519787 -1.634233e+00 -5.948086e-01  -5.459601894
    ## 633 -1.581954437   0.504206377 -2.334026e-01  6.367677e-01   1.010291022
    ## 634 -2.683286440  -7.934389138  2.373550e+00 -3.073079e+00  -7.145136625
    ## 635 -1.795054146  -6.545071526  2.621236e+00 -3.605870e+00  -8.122161075
    ## 636 -1.795238774  -8.783234735  4.371566e-01 -3.740598e+00  -8.332863063
    ## 637 -0.957095016  -7.773379785  7.173093e-01 -3.682359e+00  -8.403150262
    ## 638 -2.329582623  -1.901511690 -2.746111e+00  8.876732e-01  -0.049232856
    ## 639 -0.044934446  -1.340739354 -5.555480e-01 -1.184468e+00  -3.245108597
    ## 640 -0.333059240  -8.682376445  1.164431e+00 -4.542447e+00  -7.748480050
    ## 641 -0.357387951  -9.685621456  1.749335e+00 -4.495679e+00  -7.864506479
    ## 642 -1.294302586  -2.986028314  7.518827e-01 -1.606672e+00  -5.974925438
    ## 643 -0.805895446  -3.390134545  1.019353e+00 -2.451251e+00  -3.555834927
    ## 644 -0.948195687  -3.065234362  1.166927e+00 -2.268771e+00  -4.881142927
    ## 645  0.316813584  -1.734948003  1.449139e+00 -1.980033e+00  -5.711504740
    ## 646 -0.133492972 -11.724346118 -3.198346e+00 -4.767842e+00  -9.332128069
    ## 647 -2.319883530  -3.207075706 -1.482583e+00 -5.074871e+00  -6.778331382
    ## 648 -2.053918423   0.256890458 -2.957235e+00 -2.855797e+00  -2.808455543
    ## 649 -1.152021971  -2.001958668  5.486811e-01 -2.344042e+00  -3.076698585
    ## 650 -0.848375715   0.917690595 -2.355114e-01 -2.856921e-01  -0.867899742
    ## 651 -0.480288250  -1.362387523  9.532422e-01 -2.329629e+00  -3.393553481
    ## 652 -0.020121466  -5.407553618 -7.484361e-01 -1.362198e+00  -4.170622887
    ## 653 -1.221081278  -3.310316609 -1.111975e+00 -1.977593e+00  -3.288203565
    ## 654 -0.203362788  -0.892144430  3.332257e-01 -8.020052e-01  -4.350685312
    ## 655 -5.080407799  -6.578947868  1.760341e+00 -5.995087e-01  -4.001742085
    ## 656 -1.432803315  -2.459529640  6.177378e-01 -1.125861e+00  -3.236784016
    ## 657 -1.331601475  -1.967892799  1.295438e+00 -1.674415e+00  -3.426052253
    ## 658 -1.316475603  -5.165140556  6.252784e-01 -1.582301e+00  -3.252633595
    ## 659  1.240792771  -1.909558926  6.607176e-01 -2.752611e+00  -3.550385399
    ## 660 -0.131206949  -1.478740616 -2.469216e-01 -1.005227e-01  -2.301109897
    ## 661  0.998593092  -1.087556040 -2.724840e-02 -5.330013e-01   0.169572798
    ## 662 -1.241892856  -1.991652188  1.002665e+00 -2.809071e+00  -4.153692406
    ## 663 -0.541102793  -2.277786228  1.268166e+00 -1.997331e+00  -3.834775088
    ## 664 -0.981222508   0.325713914 -3.772141e-02  1.132194e-01  -2.126972750
    ## 665 -1.100085485  -5.810509186  1.726343e+00 -7.492772e-01  -4.834827995
    ## 666  3.874668032   1.289629924  2.017419e-01 -3.003532e+00  -3.990551332
    ## 667  1.009794578   1.301998971 -1.352575e-01 -4.315209e-01  -0.091353373
    ## 668 -0.637327763  -4.476488045  1.695994e+00 -1.606743e+00  -5.117259210
    ## 669 -0.598775713  -1.943551851 -3.295790e-01 -1.853274e+00  -3.162135955
    ## 670 -0.350283737  -0.042847968  2.466246e-01 -7.791756e-01  -0.157696446
    ## 671 -0.320778353   0.041289549  1.761703e-01 -9.669516e-01  -0.194120467
    ## 672  0.105511584  -3.269801299  9.403779e-01 -2.558691e+00  -3.624774714
    ## 673  0.832161660  -0.833350425  2.728970e-01  1.169425e+00  -0.371671945
    ## 674  2.218276303  -0.509995432 -3.569444e+00 -1.016592e+00  -4.320535776
    ## 675 -0.350284967  -0.486364541 -1.080863e-02 -7.949442e-01   0.264544749
    ## 676 -0.517194857  -1.722523123  1.288895e-01  1.496331e-02  -2.856116830
    ## 677 -0.341732014  -2.539380123  7.683784e-01 -1.547882e+00  -2.659717947
    ## 678  0.408185015  -1.209045119  1.095634e+00 -1.447225e+00  -3.951002892
    ## 679  0.739433415   3.071892046 -4.834222e-01  6.182028e-01  -1.769059721
    ## 680  0.071259557   0.694861853 -3.132699e-01 -6.493771e-01   0.517567880
    ## 681  5.760058556 -18.750889158 -3.735344e+01 -3.915397e-01  -5.052502367
    ## 682 -0.356797142  -0.717492197  3.166531e-03 -1.003967e-01   0.543187256
    ## 683 -1.152923087   0.221159478  3.737177e-02  3.448593e-02  -1.879643821
    ## 684 -1.618414648  -3.028012599  7.645548e-01 -1.801937e+00  -4.711768684
    ## 685 -0.789813100  -2.279807405  4.729879e-01 -1.657635e+00  -2.894990401
    ## 686 -0.310371094  -1.520895164  8.529958e-01 -1.496495e+00  -4.056292616
    ## 687  0.388610080   0.102485457  8.131278e-01 -1.092921e+00  -5.032028386
    ## 688  1.245730460  -1.198714200 -2.421616e+00 -1.232089e+00   0.324238886
    ## 689 -1.598944425  -2.519565318  1.316215e+00 -2.400106e+00  -4.993417038
    ## 690 -2.118932927  -3.614445110  1.687884e+00 -2.189871e+00  -4.684233325
    ## 691  1.254126028  -0.584162790 -6.096816e-01  1.014602e+00   0.334532824
    ## 692 -0.796772696  -2.614241877  1.066636e+00 -1.135497e+00  -3.943336821
    ## 693 -0.913456845  -2.356012983  1.197169e+00 -1.678374e+00  -3.538650232
    ## 694 -0.844461639  -0.174061561 -4.071381e-01  1.742163e-01  -2.998926256
    ## 695 -1.347462457   1.078233569 -1.615183e-01 -4.928559e-01  -1.039637697
    ## 696  0.713832370  -1.234572071 -2.551412e+00 -2.057724e+00   0.166830708
    ## 697 -1.280953016  -4.474763835  1.216655e+00 -2.309829e+00  -5.515507073
    ## 698 -1.923241502  -5.065980838  1.996885e+00 -3.097379e+00  -6.447202274
    ## 699 -1.989960291  -5.472436017  2.422821e+00 -2.909735e+00  -6.287803351
    ## 700 -1.341763573  -5.220940942  2.682844e+00 -2.921484e+00  -6.561257411
    ## 701 -1.768617804  -4.937554127  2.748460e+00 -3.796760e+00  -6.825490104
    ## 702 -0.958273876   0.391154204 -9.251920e-02 -3.282859e-02  -2.155302544
    ## 703 -1.002531952   0.264947929  1.316226e-02  2.488353e-01  -2.100666513
    ## 704 -0.807712094   0.151952109  1.583525e-01  9.872284e-03  -1.925278124
    ## 705 -2.223695002   0.525166986 -9.687431e-02 -1.688934e-01  -2.544410419
    ## 706 -0.618897960   0.698307699  6.983690e-02 -1.333410e-01  -1.025335393
    ## 707 -1.147291442   0.390577698 -1.070723e-01 -3.833927e-02  -2.152518891
    ## 708 -1.208798691  -1.358648491  1.102916e+00 -1.317364e+00  -4.626919164
    ## 709 -0.543578666   0.487691067  8.544881e-02 -5.363520e-01  -2.231209328
    ## 710 -1.713401451  -6.485365409  3.409395e+00 -3.053493e+00  -6.260705515
    ## 711 -2.112078898  -2.122792966  2.725651e-01  2.902731e-01  -3.833741385
    ## 712  1.548618696  -1.701283955 -2.203842e+00 -1.242265e+00   0.269561809
    ## 713  0.188582114  -2.988382756  1.344059e+00 -2.294535e+00  -1.886175990
    ## 714 -0.789043630  -1.196101449  6.736544e-01 -1.363724e+00  -2.932895258
    ## 715 -0.586867590  -4.654542618  1.285230e+00 -2.743539e+00  -5.638941313
    ## 716 -2.291561121  -5.695593929  1.338825e+00 -4.322377e+00  -8.099119398
    ## 717 -2.107219296  -5.015847976  1.205868e+00 -4.382713e+00  -8.337706974
    ## 718 -2.096166099  -6.445610183  2.422536e+00 -3.214055e+00  -8.745972612
    ## 719 -0.055123492   1.326232489  1.956998e-01 -5.468896e-01  -0.713473761
    ## 720  0.251804210  -4.328776495 -2.425478e+00 -9.852220e-01  -3.995210734
    ## 721 -2.865681618  -6.989194734  3.791551e+00 -4.622730e+00  -8.409664876
    ## 722 -2.099388657  -5.937766832  3.578780e+00 -4.684952e+00  -8.537757687
    ## 723 -1.005786779  -4.313217176  1.560712e+00 -3.295674e+00  -6.213355249
    ## 724 -0.671323195  -3.696178394  1.822272e+00 -3.049653e+00  -6.353887252
    ## 725 -1.800887204  -5.558679353  2.402322e+00 -2.848923e+00  -5.995676127
    ## 726 -0.157899075  -4.089127668  2.417305e+00 -3.239901e+00  -5.822449156
    ## 727  1.952610561  -1.624608199 -5.229908e+00  2.102020e-01  -2.069904390
    ## 728 -0.167291966   1.600027463 -1.174269e-01 -7.969538e-01  -0.133950251
    ## 729  2.891399364   5.802537353 -1.933197e+00 -1.017717e+00   1.987861863
    ## 730 -1.904959454  -3.291040869 -9.857660e-01 -1.168114e+00  -3.936294170
    ## 731  0.769825697   0.232463057  1.079406e-01 -1.143646e+00   0.541699243
    ## 732  1.919937985  -3.106942048 -1.076440e+01  3.353525e+00   0.369936028
    ## 733  1.178652467  -3.459979066 -2.815155e+00  1.242229e+00  -4.156354480
    ## 734 -2.122105987  -3.227533330 -6.855605e-01  7.759851e-01  -4.723091746
    ## 735 -1.776140785   2.114530881 -8.300841e-01  9.004900e-01  -3.376177067
    ## 736 -1.583850667   0.653745453 -1.928920e-01  1.217608e+00  -2.829098205
    ## 737  1.602139333  -3.005953152 -8.645038e+00  1.285458e+00  -3.717481375
    ## 738 -2.551375175  -2.001742908  1.092432e+00 -8.360984e-01  -4.095648973
    ## 739 -0.746924828  -0.248336733  1.091157e+00 -3.071375e-01  -5.567947051
    ## 740 -4.240182082  -9.219000996  1.974030e+00 -2.912943e+00 -13.540168238
    ## 741 -3.450734518  -8.427378029  2.305609e+00 -5.338079e+00 -12.011160884
    ## 742 -2.864815149 -10.634087621  3.018127e+00 -4.891640e+00 -11.235047911
    ## 743 -3.235375579 -10.632683478  3.272716e+00 -5.268905e+00 -11.182125455
    ## 744 -3.974549994 -12.229607779  4.971232e+00 -4.248307e+00 -12.965481204
    ## 745 -2.938427258 -11.543207169  4.843627e+00 -3.494276e+00 -13.320788867
    ## 746 -2.263798414 -10.317566181  4.237666e+00 -5.324109e+00 -11.092392369
    ## 747 -4.192171044 -14.077086010  7.168288e+00 -3.683242e+00 -15.239961959
    ## 748 -4.510343770 -12.981606156  6.783589e+00 -4.659330e+00 -14.924654774
    ## 749 -3.659167406 -14.873657965  8.810473e+00 -5.418204e+00 -13.202577148
    ## 750 -5.140999477 -14.020563705  8.332120e+00 -4.337713e+00 -15.563791339
    ## 751 -3.837429167 -11.907701904  5.833273e+00 -5.731054e+00 -12.438944906
    ## 752 -4.471672420 -14.900688600  3.840170e+00 -4.358441e+00 -14.533161687
    ## 753 -2.440115070 -14.184337295  4.452503e+00 -6.241960e+00 -12.618162687
    ## 754 -3.666835648 -16.147363374  2.078706e+00 -4.250657e+00 -16.746044105
    ## 755 -4.395918331 -15.893788243  2.083013e+00 -4.988837e+00 -15.346098847
    ## 756 -2.997206988 -17.640470406  4.034908e-02 -5.620232e+00 -15.123752180
    ## 757 -1.227903881 -31.197328549 -1.143892e+01 -9.462573e+00 -22.187088562
    ## 758 -1.875858746 -21.359738279 -3.717850e+00 -5.969782e+00 -17.141513641
    ## 759  2.914672622  -0.743357878  6.991359e-01  1.008471e+00   0.912806096
    ## 760 -0.614784605   1.368024400 -5.262617e-01 -1.213564e-01  -0.357616193
    ## 761 -1.202393318 -23.380507821 -5.781133e+00 -7.811022e+00 -16.303537659
    ## 762 -0.411923613 -23.189396512 -5.301412e+00 -8.630390e+00 -16.255611749
    ## 763  0.983049268  -4.587234602 -4.892184e+00 -2.516752e+00  -2.850323902
    ## 764  2.433940238  -8.748110359 -1.210828e+01 -2.856359e+00  -5.665861799
    ## 765  1.469728940  -9.621560127 -1.191310e+01 -3.222968e-01  -6.625692494
    ## 766 -0.354899961 -23.783469882 -4.872353e+00 -8.504285e+00 -16.601196966
    ## 767 -0.381622313 -23.928661029 -4.724921e+00 -8.603038e+00 -15.231833365
    ## 768  0.840777704 -28.011292910 -1.191964e+01 -8.960922e+00 -18.913243335
    ## 769 -0.463144880 -28.215112054 -1.460779e+01 -9.481456e+00 -20.949191554
    ## 770 -0.210701203  -4.425229685 -5.134525e+00  6.932057e-02  -0.870997151
    ## 771  2.596332658 -33.239328167 -2.156004e+01 -1.084253e+01 -19.836148852
    ## 772 -5.430971283  -9.378025171 -4.464561e-01  1.992110e+00   1.785921772
    ## 773  3.933699430 -37.060311455 -2.875980e+01 -1.112662e+01 -23.228254836
    ## 774 -2.085435833  -1.671244032  9.439659e-02  3.377988e-01  -4.431809632
    ## 775  6.065901352 -41.506796083 -3.898726e+01 -1.343407e+01 -24.403184970
    ## 776 -2.735568610  -1.546273642  4.598216e-01 -6.827414e-01  -4.363102193
    ## 777 -1.555962453  -3.833623441  6.795116e-01 -3.463765e+00  -6.683689229
    ## 778 -1.463272160  -4.490846869  1.029246e+00 -1.593249e+00  -8.993810884
    ## 779 -1.661696862  -5.757213014  1.615011e+00 -2.194881e+00  -6.807135251
    ## 780  6.474114627 -43.557241571 -4.104426e+01 -1.332015e+01 -24.588262437
    ## 781 -2.445140136  -4.964220921  1.484890e+00 -2.947899e+00  -7.175349722
    ## 782 -1.877980853  -3.514353488  1.547608e+00 -2.503304e+00  -7.575634380
    ## 783 -1.154393968  -7.739928010  2.851363e+00 -2.507569e+00  -5.110727642
    ## 784 -1.574564622  -6.330343250  2.998419e+00 -4.508167e+00  -7.334377084
    ## 785 -1.827565102  -7.114302844  3.431207e+00 -3.875643e+00  -6.868508507
    ## 786 -1.506235355  -6.899838797  3.750443e+00 -3.879793e+00  -6.864164360
    ## 787 -1.413475533  -4.245706623  1.265087e+00  1.168828e+00  -4.513906605
    ## 788 -0.615622343  -2.916575644  7.767100e-01 -1.878832e+00  -4.546935751
    ## 789  0.095951299  -2.440419274  1.286301e+00 -2.766438e+00  -4.458007701
    ## 790 -0.099449376  -0.021483158 -1.723273e-01  5.087300e-01   1.072955206
    ## 791 -1.419747681  -3.878576339  1.444656e+00 -1.465542e+00  -5.208334593
    ## 792 -0.736427203  -3.657946348  1.944906e+00 -7.883875e-01  -5.624677134
    ## 793  0.440607081   1.934740475 -1.019788e+00 -1.932442e-01   1.783739060
    ## 794 -1.104602382  -2.670503277  1.476547e-01 -9.786255e-01  -3.514133305
    ## 795 -2.562826070  -2.286985626  2.609065e-01 -8.953655e-01  -4.542612038
    ## 796  0.670638452  -4.128987489 -4.765894e+00 -1.005259e+00   0.453505060
    ## 797  0.500609583  -4.640770208 -4.339840e+00 -9.500364e-01   0.566679814
    ## 798 -1.256353100  -1.752419849  2.817364e-01 -1.792343e+00  -4.820779414
    ## 799 -0.098620927   2.880082727 -7.274840e-01  1.460381e+00  -1.531607982
    ## 800 -1.670380514  -2.377139814  9.037045e-02  4.847264e-03  -2.776747144
    ## 801 -1.200143965  -7.674059940  4.125761e+00 -5.315778e+00  -4.891156203
    ## 802 -0.973504253  -1.706657654  3.137445e-01 -1.982302e+00  -3.158127006
    ## 803 -0.009389035  -0.079705604  6.407082e-02 -7.145167e-01   0.042227520
    ## 804  0.165640146   1.099377709 -6.545571e-01 -2.435416e+00  -2.276733073
    ## 805 -1.756998959  -1.217416499  3.645633e-01 -2.770148e+00  -3.216187718
    ## 806 -0.762118684  -2.506348775  6.941639e-01 -4.675559e-01  -4.565260095
    ## 807 -0.357617930  -3.767188827  6.146644e-02 -1.836200e+00  -1.470645258
    ## 808  0.683526140  -5.133671002 -7.907790e+00  2.154747e-01  -2.297734259
    ## 809 -0.480624709  -2.126136159  1.883507e+00 -1.297262e+00  -5.487424583
    ## 810 -0.446750698   3.791906549 -1.351045e+00  9.518596e-02  -0.084499559
    ## 811 -0.535163279   0.142543370 -2.227697e-01 -1.463691e+00   1.713537553
    ## 812 -1.281842535  -1.171994200  4.137782e-01 -2.659840e+00  -2.971695236
    ## 813 -0.292501825  -2.317431394  1.189747e+00 -7.862379e-01  -3.906745733
    ## 814 -1.848006038  -1.005508101  3.399367e-01 -2.959806e+00  -3.851721665
    ## 815 -1.316595946  -1.855495444  8.310791e-01 -1.567514e+00  -3.224559148
    ## 816 -2.950358272  -1.528505745  1.893190e-01 -1.433554e+00  -5.569142496
    ## 817 -1.739740708  -0.047420092  3.014245e-01 -1.779434e+00  -5.836452727
    ## 818 -1.431020118   0.757011225 -4.444180e-01  9.979210e-01  -1.429490124
    ## 819 -0.512759848   1.140148806 -3.412730e-01 -1.046351e+00   0.085661535
    ## 820 -0.574679962   4.031513441 -9.343978e-01 -7.682547e-01  -2.248115174
    ## 821 -2.979912075  -2.792379134 -2.719867e+00 -2.767037e-01  -2.314747164
    ## 822 -1.855854731   1.017731580 -5.447038e-01 -1.703378e+00  -3.739659479
    ## 823 -1.727185698  -0.932428910  2.927965e-01 -3.156827e+00  -3.898240381
    ## 824 -0.502471058  -1.891965557  8.784169e-01 -1.541942e+00  -2.649405830
    ## 825 -0.691346180  -1.799884833 -2.627781e+00 -4.001338e+00  -2.271525784
    ## 826  2.114178350   0.948701404 -2.448427e+00 -3.203666e+00  -3.074034313
    ## 827 -2.321063516  -0.781880413  7.661924e-02 -2.976249e+00  -4.070256863
    ## 828 -1.041407520  -1.020406645  5.470686e-01 -1.105990e+00  -3.520127674
    ## 829 -0.674516988   0.298621349 -2.824157e-01  8.020525e-01  -0.989431190
    ## 830 -1.280952131  -2.719556967  7.183248e-01 -1.660183e+00  -3.841098151
    ## 831 -0.103614288  -3.490349858  1.094734e+00 -7.174185e-01  -5.179934714
    ## 832 -0.509237784  -2.643397619  1.283545e+00 -2.515356e+00  -4.501314810
    ## 833  1.457811065  -0.362577080  1.443791e+00 -1.927359e+00  -6.564659262
    ## 834  0.939039595  -0.635032860 -7.045065e-01 -2.347857e-01   4.031435051
    ## 835 -3.407678617  -5.063118270  1.007042e+00 -3.190158e+00  -4.250716511
    ## 836 -0.174199761  -3.454201472  1.102823e+00 -1.065016e+00  -5.416036889
    ## 837 -1.430288633  -0.076547897 -9.922604e-01  7.563072e-01   0.217630454
    ## 838  0.631140872  -1.245106342  1.511348e+00 -1.899987e+00  -6.428230830
    ## 839 -1.349570138   0.980940272 -1.819539e+00 -2.099049e+00  -1.347556715
    ## 840  0.615421803  -3.485679585  1.878856e+00 -1.116268e+00  -5.112971088
    ## 841 -0.413647316  -4.082308171  2.239089e+00 -1.986360e+00  -5.165330818
    ## 842 -0.477982006   0.438495665  6.307334e-02 -1.862071e-01  -0.159325099
    ## 843  0.280590024   0.221962545  6.782675e-02 -1.387054e+00  -0.045125330
    ## 844  1.364918040   0.759218697 -1.188610e-01 -2.293921e+00  -0.423783865
    ## 845 -1.571629715  -1.358707981  6.724094e-01 -3.188001e+00  -4.937426660
    ## 846 -3.162998372  -5.985640116 -2.179935e+00 -1.120292e+00  -3.075558310
    ## 847 -1.171301063  -4.172314571  1.517016e+00 -1.775833e+00  -3.754053923
    ## 848  0.100492060   1.295015920 -1.358572e-01 -1.695822e+00   0.955004467
    ## 849 -1.020509228  -1.808574442  5.217435e-01 -2.032638e+00  -2.732791967
    ## 850 -1.532461753  -3.449158503  1.856839e+00 -3.623334e+00  -5.653637686
    ## 851 -1.998755028  -5.835532442 -8.877394e-01 -2.828287e+00  -4.614507733
    ## 852 -0.996754564  -4.299832586  2.268867e+00 -3.651067e+00  -4.400929850
    ## 853 -0.112631915  -1.329371563  1.709417e+00 -2.322716e+00  -6.540988520
    ## 854 -3.372265711  -5.036555755  2.643106e+00 -2.274630e+00  -7.049229360
    ## 855 -2.397014424  -1.819307886  3.385270e-01 -2.819883e+00  -4.063098108
    ## 856 -1.420467618   1.057480446 -2.811573e-01 -1.832604e+00  -0.628463195
    ## 857 -2.635066388  -1.865910810  7.802716e-01 -3.868248e+00  -4.851487194
    ## 858 -2.766664931  -2.312222883  9.610141e-01 -1.896001e+00  -4.919348299
    ## 859  1.392519734  -1.697801219 -6.333065e+00  1.724184e+00  -0.887241636
    ## 860  1.147963582   1.858424853  4.748583e-01 -3.838399e+00  -1.445374957
    ## 861 -2.193821452  -1.530816792  6.268574e-01 -4.037021e+00  -6.285424475
    ## 862 -1.442584085  -0.898701854  1.230620e-01 -2.748496e+00  -3.202436122
    ## 863 -3.085984237  -1.677869988  8.650746e-01 -3.177260e+00  -3.419207384
    ## 864 -0.882445799  -2.902079247  9.391622e-01 -3.627698e+00  -1.873330853
    ## 865 -2.031209870  -2.650137222  1.131249e+00 -2.946890e+00  -4.816400800
    ## 866 -2.084875078  -2.385026925  1.471140e+00 -2.530507e+00  -5.175659680
    ## 867 -1.816861466  -0.916696278  2.655683e-01 -3.158014e+00  -3.890169242
    ## 868 -3.289343682  -0.118841442 -1.427945e-02 -9.327728e-01  -2.637441745
    ## 869 -2.002526000  -2.874154733 -8.560054e-01  9.636742e-01  -3.235438957
    ## 870 -1.285451374  -1.766683621  7.567107e-01 -1.765722e+00  -3.263006726
    ## 871  0.910234501  -0.498200069  3.447032e-01 -6.679391e-01   0.398155065
    ## 872  0.290699944  -2.808987620 -2.679351e+00 -5.566849e-01  -4.485482704
    ## 873 -1.148649740   0.849686456  4.334270e-01 -1.315646e+00  -2.796331539
    ## 874 -0.400859126  -0.040970122  8.951046e-02 -2.177053e-01  -0.373927300
    ## 875 -1.892387970  -2.588418177 -2.226592e+00 -1.670173e+00  -3.508924662
    ## 876 -1.678413388  -3.144732472  1.245106e+00 -1.692541e+00  -4.759931374
    ## 877 -1.737393477  -4.396859320  2.283941e-01 -1.675884e+00  -3.991784632
    ## 878 -1.886718108  -3.817494617  6.134702e-01 -1.482121e+00  -4.868747268
    ## 879 -1.948246371  -2.167859505 -7.282067e-01 -1.977238e+00  -3.473410704
    ## 880 -1.968441448  -3.110476193 -3.284037e-01 -1.574363e+00  -2.497560547
    ## 881 -2.031170883  -3.474549465  1.071214e-01 -1.551352e+00  -2.411272015
    ## 882 -2.094899155  -3.839486567  5.430534e-01 -1.528448e+00  -2.325113441
    ## 883 -2.159483574  -4.205164096  9.793343e-01 -1.505637e+00  -2.239066261
    ## 884  0.528488565   1.086014426 -1.464227e-01 -1.724333e+00   1.280166743
    ## 885 -1.166352579   0.482534011 -3.497908e-01  1.045007e+00  -1.474974161
    ## 886 -0.638908045  -2.873463269  1.576318e+00 -2.861986e+00  -2.120457595
    ## 887 -1.836070894  -1.623740059  2.595625e-01 -1.132044e+00  -3.356473714
    ## 888 -2.335144981  -0.907188473  7.063621e-01 -3.747646e+00  -4.230983836
    ## 889 -1.235530093  -0.279312557  5.143607e-01 -1.849927e+00  -3.443819022
    ## 890 -1.260374549   0.288223322 -4.896399e-02 -7.349746e-01  -4.441484081
    ## 891 -0.809419405   1.501579840 -4.709996e-01  1.519743e+00  -1.134453811
    ## 892 -6.406266634  -5.831451560  1.457175e+00 -6.462026e-01  -4.029128720
    ## 893 -1.462922811  -2.688760584  6.777635e-01 -3.447596e+00  -4.707570806
    ## 894 -1.873013153  -2.306689021  9.937019e-01 -4.944054e+00  -5.576418796
    ## 895 -2.055129085  -2.935855700  1.431008e+00 -4.544722e+00  -5.258095976
    ## 896 -2.125936320  -3.307035735  1.869838e+00 -4.522584e+00  -5.172811823
    ## 897 -1.646313177  -2.785198366  5.403677e-01 -3.044029e+00  -3.926509940
    ## 898 -0.746186462   0.572730531 -1.312348e-01 -1.551839e+00   0.228849431
    ## 899 -0.853508247   0.716025068 -1.649098e-01 -1.502345e+00   0.259411077
    ## 900 -2.497521836  -2.073347463  6.398181e-01 -3.013331e+00  -5.954838297
    ## 901 -3.189073067  -2.261896902  1.185096e+00 -4.441942e+00  -6.646153830
    ## 902 -2.767694650  -3.176420951  1.623279e+00 -4.367228e+00  -5.533443029
    ## 903 -0.504723385   1.373587546 -2.094765e-01  2.084940e-01  -1.613618339
    ## 904 -0.504354582   1.438537460 -3.956035e-01 -1.555142e+00   1.081513989
    ## 905 -1.259004478  -1.237688814  3.584256e-01 -2.612489e+00  -3.064729531
    ## 906 -1.602522595  -0.950462840  7.229029e-01 -4.128505e+00  -3.963224472
    ## 907 -5.566870060  -4.291180422  8.765306e-01 -1.075478e+00  -3.272569174
    ## 908 -0.501474323  -0.440054283  5.918283e-01 -3.267693e+00  -2.223070328
    ## 909 -2.188494005  -0.686934748 -5.479844e-01 -9.952811e-02  -1.672345978
    ## 910 -0.828495614  -0.732262192 -2.033289e-01 -3.470457e-01  -2.162061390
    ## 911 -2.396535281  -2.792488873  5.148110e-01 -3.541780e+00  -5.334753871
    ## 912 -2.646692540  -2.854432162  9.587831e-01 -4.588536e+00  -6.120715087
    ## 913 -1.716268269   1.132790930 -5.742144e-01  1.289039e-01  -1.000805052
    ## 914 -2.751496118  -3.358857315  1.406268e+00 -4.403852e+00  -5.945633973
    ## 915 -1.992116836  -3.734901766  1.520079e+00 -2.548788e+00  -4.533515403
    ## 916 -3.599486973  -3.686650710  1.942252e+00 -3.065089e+00  -7.509557474
    ## 917 -0.477294698  -4.276131504 -6.951729e-01 -2.971644e+00  -5.529131231
    ## 918  1.332436943  -6.783963787 -1.541538e+01  4.655116e-01  -0.394381262
    ## 919 -5.773191769  -5.748879199  7.217429e-01 -1.076274e+00   2.688669933
    ## 920 -1.902598204  -0.055177812  2.778312e-01 -1.745854e+00  -2.516628002
    ## 921  0.423796702   3.790880171 -1.155595e+00 -6.343357e-02   1.334414062
    ## 922 -0.322077027  -1.082510657  1.172000e-01 -1.409267e-01   0.249310803
    ## 923  0.111136093  -0.922037621 -2.149930e+00 -2.027474e+00  -4.390841922
    ## 924 -2.010494231  -0.882849831  6.972111e-01 -2.064945e+00  -5.587793782
    ## 925 -1.326535934  -1.413169956  2.485255e-01 -1.127396e+00  -3.232153175
    ## 926 -0.003346296  -2.234739296  1.210158e+00 -6.522499e-01  -3.463890879
    ## 927 -2.943547791  -2.208001920  1.058733e+00 -1.632333e+00  -5.245983838
    ## 928 -0.096694744   0.223050267 -6.838388e-02  5.778294e-01  -0.888721676
    ##               V11           V12           V13           V14           V15
    ## 1   -1.3260239222   0.606050260  1.5143523996   0.613776557  0.8739882579
    ## 2    0.4899333770   0.556489979 -0.2528829904   0.330891782 -0.0036388236
    ## 3   -0.8512820387  -0.908310808 -1.4938766869   0.506772852  0.6720234746
    ## 4    0.1941177955  -0.632492902 -0.3022815205  -0.156488659  1.2657339582
    ## 5   -0.8233439621   0.151768810 -0.0308945618   0.810760667  0.4426447952
    ## 6    1.1199784173   0.126585029 -0.6611682983  -0.298459609  1.2810341588
    ## 7    1.0095953279  -3.173121501  1.8701127441   1.413721205  0.2605183919
    ## 8    0.1208957102  -0.234401947 -0.4952872493   0.347390249  0.5586512814
    ## 9   -1.1997771044   0.565841285  0.1795577629   0.641442233 -0.4390248117
    ## 10   1.7203577769  -2.672109575 -0.5746381114   2.240629533 -0.7594193351
    ## 11  -1.9663811084  -0.350486669 -0.5069619554   0.722195229 -0.2873837831
    ## 12  -0.2898796444  -0.465648504 -1.2611448873  -1.034236014  0.0951830514
    ## 13   0.8527460631   1.312577470 -0.9585493035  -0.387597124 -0.9223759831
    ## 14  -1.0442424546  -0.023357561 -0.0498740896  -0.184821170  0.1295057280
    ## 15   0.3130961360   1.103273796  1.0261075180   0.747022957 -0.3143236981
    ## 16  -0.1081622920   0.621523675  0.2428382217   0.036602366  0.0912365991
    ## 17   1.6688056992   1.560057245 -0.3642111017  -0.017038864 -0.1580891669
    ## 18  -0.2326138564   0.687551456  0.9389346156   0.210963067  0.8212791108
    ## 19   0.3243703658   1.645697723 -0.1385336864  -0.598889226 -2.3984737544
    ## 20  -1.6630004083  -0.739366963 -0.8288542946   0.216035490  1.5013901564
    ## 21   1.6085840249   0.647216784 -0.8040656611   0.734459521  0.8273240545
    ## 22   1.4186988256   0.709183786 -0.1871316486   0.108008497 -0.1970402275
    ## 23  -0.0106585189  -0.915806038 -2.8290774055  -0.631686516 -1.4908218121
    ## 24  -1.2426959528  -0.712539125 -1.5686962528   0.365624126  1.0790696907
    ## 25   0.7766452582   0.403374782 -0.6125050821  -1.220791886 -1.5733053543
    ## 26   0.4634408991   0.204789777 -0.3343898393  -0.783049879 -0.8614671248
    ## 27   0.7277400505   0.623152175 -0.9727285957   0.638888551 -0.7666731906
    ## 28   0.9622911174   1.127087350  0.4772646822   0.811973995 -0.5490741166
    ## 29   0.2389426385   0.385011940  0.2498085605  -0.281399542  1.0627243124
    ## 30  -1.0030841849   0.432235204 -0.2124870006   0.358782506  2.1102177332
    ## 31  -1.5112393288  -0.394644396 -0.5963890637  -0.916538523 -0.8714805927
    ## 32   0.8113526371   0.327104631 -0.6679022579   0.148044169 -0.2924345004
    ## 33   0.4584489984  -0.123277216 -0.4670064463  -2.425260679  0.0343053058
    ## 34  -0.8195108697  -1.601967036 -2.3687895235  -0.873137868  1.6691035858
    ## 35   1.0902918411   0.614328469 -0.0584502281  -0.640635580  0.2392033930
    ## 36  -0.3188972363  -0.047226586 -0.4558393095   0.280598415  1.5251910865
    ## 37   1.3171088557  -1.300709439 -1.6841781452   0.524000128  1.3644031933
    ## 38   0.5126117839   0.172046698 -1.0903665232   0.701552414 -1.1293955553
    ## 39   0.6337500826   1.806693374 -0.0942785777  -0.590596344 -2.4295390824
    ## 40  -1.3784213987  -0.070086773 -0.7326013565   1.151733995 -0.7851941663
    ## 41   0.2687099133   0.019805284 -0.6553562809   0.447291627  0.1479868028
    ## 42   0.8907232929   0.622512506 -1.4898869369  -0.046482648 -0.5944247482
    ## 43   0.3500764659   1.599708829  0.9101667952   0.029248252  0.4743792238
    ## 44  -0.2131029454  -0.381898894 -1.5709801193   0.589171806  0.1019346901
    ## 45  -0.5104578731   0.709893984  1.2477064259  -1.084046277  0.0896969244
    ## 46   0.3444221335   0.408514981  0.3167308597  -0.045165469 -0.0816470563
    ## 47   1.6216571111   0.641413151 -0.9890151638   0.405545600  1.7194719922
    ## 48   1.3693049360  -1.868666150  1.5477629634   1.470496009 -0.5769587627
    ## 49   0.0116612684   0.394414990 -1.0237287028   0.203985364 -1.4241484801
    ## 50   1.3720181258  -1.060346894  2.3171375290   1.374747401 -0.5268125115
    ## 51   1.1862250822  -0.591755601 -1.0640158731   0.275731836 -1.3706490700
    ## 52   1.1367467552  -0.222543494 -0.7866415639  -0.983581603 -1.3105664765
    ## 53   2.3171547209  -2.282745086  2.2740006163   1.129183268 -0.0816454970
    ## 54   1.4942696783   0.833337528  0.5367663430  -0.316484895  0.4183649728
    ## 55   0.8828632785  -0.687738632 -1.1641611824  -2.412914981 -1.5488633468
    ## 56  -0.8026633245   0.853112751  0.7352564962   0.345849585 -0.6222719153
    ## 57  -1.9773205494   0.842999865  1.0782051866  -1.668894855 -1.6901689062
    ## 58   1.0837239585   1.126898565  1.0722669834   0.521924766 -0.3193559726
    ## 59  -0.6157196129  -0.113050129 -0.6041616053   0.475748154 -0.7577081699
    ## 60   0.5012188951  -0.057817851 -1.4244332831   0.875545511 -1.0475455346
    ## 61  -0.3401887030  -0.618841641 -1.3068639360   0.609471745  1.1119687205
    ## 62   0.4719357847   0.772243298 -0.5415062979   0.817015237 -0.3968529182
    ## 63  -0.3639108337   0.511033916  1.3729323983   0.142459321  1.6979613275
    ## 64   0.9377200263   0.509641219 -1.3928272506   0.766480516  0.1207389178
    ## 65   1.5599678970  -0.528036116 -1.1069723648  -0.608048622  0.1810897668
    ## 66  -0.3638042646  -0.622140829 -0.6776867662   0.137578662  1.7866749975
    ## 67  -2.1870485138  -0.053745936 -0.2035632986   1.202805530  1.4419573128
    ## 68   0.6029498887   0.980642868  0.0148519035   0.457291128 -0.6271311668
    ## 69  -1.5224923957  -0.666353977  0.2639145606   0.318144638 -1.2195833486
    ## 70   0.4859784038   0.777783951  0.1101640760   0.266573867 -1.0405532113
    ## 71  -0.7766806649   0.838740703  1.1558020416  -0.134533359 -1.0273855793
    ## 72  -0.7797290094  -0.009705229  0.5045803768   0.037672552  0.8437437742
    ## 73   0.9469559160   1.221871546  0.4174672310  -0.031327527  0.0954154258
    ## 74   2.2842428070  -0.903988746  0.3867272344  -2.619943980  1.3440475200
    ## 75  -1.8703803642   0.101656404  0.1535145582   0.427595226 -0.4659031986
    ## 76   1.6812468165   0.984156499  0.0336317356   0.533290993 -0.3276778380
    ## 77   0.3937942886   0.190086396 -0.7845844969   0.409517677  0.4537662735
    ## 78   0.8718330869   0.648944837  0.3192430117  -2.225311347 -1.3953251226
    ## 79   0.1575187742  -0.092437913 -1.3214638659   1.542537191 -0.2840120852
    ## 80  -0.5292645438   0.909043602  0.9331984075  -0.106535000  0.9348304706
    ## 81  -1.4128246710   0.861754772  1.2693360272   0.263868728  0.1120382736
    ## 82   0.3280004455  -2.530585824  2.8783502669   1.137419693 -0.6625065871
    ## 83  -1.4261037534  -0.526778960 -0.0004843204   0.020601884  0.9487346143
    ## 84   0.7893450724   0.703767994 -0.5846840211   0.021779317  0.1385955393
    ## 85   1.0384644204   0.195671985 -0.0017203882  -0.623236846  0.4832236960
    ## 86   0.7422587688   0.268337600 -0.7798210249   0.690009267  0.6374200818
    ## 87  -0.0706255149   0.194406116 -0.6626406051   0.779528217  0.3035397104
    ## 88  -0.6143374404   0.821733348  0.8590976646  -0.608411270 -0.3977675517
    ## 89  -0.4695940004   0.187189129 -0.1517225615  -0.723617040  0.0033702564
    ## 90   0.7668346295   0.564821223  0.1936858915  -0.349275739 -0.0529379959
    ## 91  -1.7692268422  -0.224731724 -0.5712752941  -0.240768713 -1.9094540658
    ## 92   1.3284713432  -0.364383282 -0.0166046225  -0.191045646  0.2180914101
    ## 93  -0.1571824856  -0.246859841  0.9454674197  -0.380264567 -0.7894973864
    ## 94  -1.2046542319  -1.091421447  0.0413021077  -0.898284201  0.5440701154
    ## 95   1.4381605110   1.124826050  0.7807259715  -0.226561570  0.3535748052
    ## 96   0.4408101381   0.126181616 -1.5938750326   0.276836689 -0.1671370462
    ## 97  -0.2607628044   0.754173889  0.5700143474  -0.770124022 -1.4558556451
    ## 98   1.4624894172   1.652530620  0.4357122863   0.136090460 -0.8866827993
    ## 99   1.4498848637   0.244449496 -0.8292985836  -0.080094019 -0.1915055830
    ## 100 -1.6355077740  -0.489278905  1.1311073908  -0.535239175 -0.2203893925
    ## 101 -1.2667756775   1.432156597 -0.9598030815   2.222762695 -2.2275463808
    ## 102 -1.2362027985   0.222656258 -0.0370807670   0.607374506 -0.3957665729
    ## 103  0.7757517842   1.439517405  0.7996470110  -0.381644866 -1.1803236675
    ## 104  0.3174607026   0.332914175  1.3725553122  -0.494607997 -1.0838719338
    ## 105 -0.3661984334  -0.288010271 -0.3539232238   0.935802470  1.8420366200
    ## 106 -0.6449337061   0.364539049  0.4860130988  -0.095137952 -0.1665568250
    ## 107 -0.8014610520   1.200945580  1.5981878846  -0.504223828  0.3626530442
    ## 108 -0.2722241543  -0.699222914  0.1011591014  -0.130317756 -0.3186154946
    ## 109 -0.5651557679  -0.581907464 -1.4634317121   0.540652424  1.2900417653
    ## 110 -0.9875064317   0.233040391 -0.2300229752   0.570948169  0.1087094836
    ## 111  0.3593394719   1.160861251 -1.2026927129  -0.447011533 -2.3487371247
    ## 112 -0.0544699663  -0.239798194 -0.6246561698   0.417993901  0.3355790820
    ## 113 -0.9056721189   0.181839736  0.0846971722   0.097405537 -0.4124886976
    ## 114  1.7171616122  -1.644073821  2.0618050539   0.785695366 -1.3650112481
    ## 115 -1.0028389580   0.379759540  0.7311938986   0.218063618 -0.8653259639
    ## 116  0.9294807149   0.096940736 -1.4295919815   0.832590842 -0.2555116559
    ## 117  0.8614306095   1.675880366  0.6604434290  -0.312748467 -1.5788319430
    ## 118  1.3409747872   0.914339846 -0.0551787826  -0.631638464 -0.4610122789
    ## 119 -0.4024245962   0.128084824  0.7386282895  -0.057608372  1.2766066663
    ## 120  0.8468248577   0.524791869  0.7087274708   0.051404544 -0.5297164251
    ## 121  0.4854355956   0.638953801 -0.5166942757   0.873567284 -1.4672101349
    ## 122  1.8771039712  -0.954075708  2.4379812900   1.253791012 -0.9287490384
    ## 123  0.5160766467   0.921206281 -0.0292975940   0.461818856 -0.6155118802
    ## 124 -1.2816059537   0.359521584  0.2506288140   0.003062122  0.3420162689
    ## 125  1.8291374007  -1.908568006 -0.1346329497   2.019953931 -0.3951175167
    ## 126  1.4962901019   0.597321701 -0.7982987750   0.645805421  0.4254141442
    ## 127 -2.3970582516  -2.029890204 -1.2055180414   0.215475945 -0.6061617653
    ## 128  0.6616713116   1.448123177 -0.7172558005   1.214078723 -0.0277745513
    ## 129  2.1845788186  -2.034055381 -0.1655942423   1.687113104 -0.0907521907
    ## 130  0.8474819570   1.373508997  0.8893466563  -0.246422065 -0.1461409094
    ## 131  0.9988956584   0.369476473 -1.4397933546   0.917275098  1.7571386210
    ## 132  0.0805729481   1.326920896  0.4767959860   0.074344401 -1.5601203983
    ## 133  0.5785746530  -0.237350918 -0.8657682378  -0.359952287  0.0794490100
    ## 134  1.3501009175  -3.179597544 -0.4016661007   2.439792549  0.1526931266
    ## 135  0.1016164020  -2.326210133  1.6921982716   0.335468293 -0.4080420257
    ## 136 -0.0234602450  -0.236468890 -0.8403866337  -0.050857669  1.4607331671
    ## 137  0.2066078117  -0.459751399 -1.3626455638  -0.481599151 -0.4127314608
    ## 138  0.7085401410   0.273835268 -1.2344007220   0.560687079 -0.4507140887
    ## 139 -0.8750115104   0.499145662  0.1185732082   0.609192985  0.7693989186
    ## 140 -0.3383430388  -2.082688836  0.3397033094   0.939492262 -2.3046297287
    ## 141  1.6148302445   0.270978593 -1.2632322012   0.213617496  0.7980121363
    ## 142 -0.5713060417   0.110272940  0.7882090156   0.105913950  0.7469253145
    ## 143  0.0878268352   0.710625156  1.3077644559  -0.192100173  1.4244387750
    ## 144 -1.0543474106   0.106131401  0.0670584879   0.100661835 -0.3933326488
    ## 145  0.1982289745  -0.102481926 -0.2837802535  -1.288289761  0.8604159798
    ## 146  0.3935284732  -0.454402386 -1.5793758962   0.798811093 -0.1529248805
    ## 147  1.9999819018   0.777444589 -0.6941144068   0.537162137  1.7762203056
    ## 148  0.7557173445   0.230833587 -1.0823365949   1.064118531 -0.8116193334
    ## 149 -1.4038108976   0.092287187  1.2909864900   0.068233944  1.2929928214
    ## 150 -0.1545478371   1.162810531  0.6683464276  -0.584682458 -0.1169511371
    ## 151  0.0986352885   0.253913423  0.0992712686   0.523967864 -0.5085610117
    ## 152 -0.8502428108   0.626299739  0.8836101783  -0.694643474 -0.6859151406
    ## 153 -1.4118703050   0.371722482  0.0210689807   0.210736943 -0.3828530906
    ## 154 -0.7529159733  -0.014719196 -0.0833930877  -0.832055909  0.4348997274
    ## 155  0.7486502399   0.445298679  1.0122134090   0.434700448 -1.3042762906
    ## 156  1.2310625785  -0.063729473 -0.1823685036   0.252152341 -0.0335530498
    ## 157 -1.4329466851  -0.275033080 -0.7061936749  -0.109831308  0.5147575517
    ## 158  0.1771574329  -2.676488354  2.2274281601   1.177295772 -1.1974959211
    ## 159  0.2328349029  -0.024699649 -0.8920899271   0.397710412  1.4746729440
    ## 160 -0.1364151725  -0.329621497 -0.1724816554  -0.236325842  0.6118141066
    ## 161  0.1238177447   1.009266186  0.2616353114  -0.059229829 -0.1364899212
    ## 162 -0.5641485927  -0.332583592 -0.1424544411   0.634818080  1.7175684958
    ## 163  1.0672634709  -0.471666105 -1.6817204769  -1.385611924 -0.7811984848
    ## 164 -0.9916202716   0.816865996 -0.2036393675  -1.145462117 -1.0877092519
    ## 165 -0.7180557079   0.402916549  1.7406792109  -0.223020427  0.8013733303
    ## 166 -0.6967633729   0.439500711  0.0242603686   0.288272939  0.0913526097
    ## 167 -0.7833028398  -0.091749010  0.8187769653   0.348331448 -0.2056914698
    ## 168  0.4579459890   0.413906823  0.3768664781  -0.333592069  0.3118058830
    ## 169  0.7681515713  -2.910597561 -0.8453509098   2.181751923 -1.4528231526
    ## 170  0.3142084782   0.721629399 -0.2089824125   0.046603331 -0.4698961002
    ## 171 -1.0126910504  -0.578628875 -1.0991952450  -0.009209950  0.4470868914
    ## 172  1.3110832821   1.296811991 -0.4540548705   0.834306661 -0.1020235820
    ## 173 -1.5092838309   0.494352224  0.9687259597  -1.494412226 -0.7325339923
    ## 174  1.4616773349   0.696108295  0.2982067252  -0.311922168  0.4894456405
    ## 175 -0.2928338943  -1.361025297 -1.4228129436   0.122656569  1.3058406380
    ## 176  0.9531789486   1.019635267  0.6070319293   0.041801547 -0.3263930368
    ## 177  0.0538278314   0.768308409  1.5902783523  -0.322964850  1.0370082781
    ## 178 -2.3012180986  -0.622049401 -0.9142964233   0.612252507 -0.5257893622
    ## 179 -1.8600487474   2.347549445 -1.5054757554   3.210588802 -2.2941620162
    ## 180 -0.3790384587   0.009201984  0.1444315423   0.120292687  0.8358833037
    ## 181  1.0571691302   1.075210844  0.4104002647   0.002365114 -0.9206080186
    ## 182 -2.2049282412  -1.043472666 -1.8256501344   0.788218888 -0.4578440828
    ## 183  0.5218318680   0.554914413 -0.2955831483   0.654514240  0.1956609170
    ## 184  0.6732112209  -0.465042321 -1.0449820063   0.423470586 -0.4273763259
    ## 185  0.2426224007  -0.276952060 -0.4317672851   0.171670287 -0.0519548207
    ## 186 -0.6014015548   1.069516388  1.1375195152  -0.579221393  0.0572711966
    ## 187  1.7772224947  -0.153098089 -0.7857890128   0.288051827  0.4067342220
    ## 188  0.6608794748   0.872624077 -0.3000803997   0.686764941 -1.1182842648
    ## 189 -1.7150454111  -1.139987582 -1.4101404830   0.151589066  0.8535760944
    ## 190  1.2142853594  -0.878662231 -1.0265494502   0.284420461  0.3159810764
    ## 191  0.9656607970   0.923321487 -0.4783721267   0.565211259 -0.6189636910
    ## 192 -1.1031645487   0.272544746 -0.5674864003   1.004817182  0.3185700747
    ## 193  1.6198745245   0.746606500  0.5549665474  -0.667835901 -0.3593502346
    ## 194 -0.7589457046  -0.642235913 -0.3544203873  -0.198848016  1.5804493637
    ## 195  1.4372974307   1.463945662  0.7640418797  -1.584943403 -0.8741527099
    ## 196 -1.0208217933   0.122880869  0.7327306819  -0.121546264  1.4375487583
    ## 197 -1.1672469143  -0.297754936  0.5017615261   0.096034981 -0.0326985873
    ## 198  0.8988569441  -0.069869307 -0.2797557358  -0.037219496 -0.1959607742
    ## 199  0.6481702947   0.121854280 -1.0479685709   1.200256440  0.4866792893
    ## 200 -1.2533437959  -1.451881892 -1.4952713511  -0.139752467  0.3768700599
    ## 201  0.8239537680  -0.346560365 -2.9532475056   1.066030156 -0.0950129106
    ## 202  1.2013494018   1.281108668 -0.0106726943   0.292002538  0.8411060155
    ## 203 -0.3106345870   0.450839863 -1.2787700997  -0.735230362 -1.6193049604
    ## 204 -0.3841306908  -0.219499591 -0.8382110945   0.488620342  0.4141706308
    ## 205 -0.5467813653  -0.506441688 -0.5053449211  -0.291604883  0.9487961797
    ## 206  0.0886009378  -0.025541151  0.1652395479   0.166551871  1.4027899245
    ## 207 -0.1619142320   0.028964719 -0.0217566747   0.526760667  1.1051831915
    ## 208  0.2929420634   0.671434698 -0.9027386979   0.528692717 -0.7172897150
    ## 209  0.8370443694   0.951070722  0.2731780812   0.146616664 -0.5239475074
    ## 210  1.3437077600  -2.826434661  0.9201629528   2.333161634  0.9975975733
    ## 211 -2.2719766275  -1.105342087 -1.0046282000   0.507420108  1.3205506404
    ## 212 -0.6675505803   0.901054019  1.3282857209   0.337468300  0.5795067440
    ## 213 -0.2650325897  -0.739891353 -2.2970416950   0.998957639 -1.7720338828
    ## 214  1.2443643112   0.217495066 -1.1654419284   0.478369350  0.3990144027
    ## 215 -1.2417190316  -0.909993357  1.2062552692  -0.530620987  0.7778912512
    ## 216 -0.7661172776   0.412282776  0.1282838435  -0.323271677  0.4263602326
    ## 217  0.1328292248  -3.136021900  0.6211233756   1.461684051  0.7609827906
    ## 218  0.2804751385   0.420476934 -0.6618030161   1.123179320 -0.6523388074
    ## 219  0.6613966355   0.530627310 -0.8729228020   0.805060092 -1.2819411670
    ## 220 -0.1784798632   0.306065258 -1.4183108656   1.388905310 -1.7135791472
    ## 221 -0.4812169735  -0.007295887 -0.6747986710  -0.766056897  0.3638875479
    ## 222 -2.2850855464   1.343145027  1.2254265852  -0.552936836 -1.5784727258
    ## 223  0.8328277119   0.777186929  0.5560183527  -0.951087366 -0.9289819288
    ## 224 -1.1614286737  -1.040892658 -0.4406367864  -1.519099386 -0.9550199784
    ## 225 -1.6785428929  -0.747930117 -1.9396703458   1.451915495 -0.1664351725
    ## 226  0.6951949676   1.328275582  0.4906972013   0.211959150 -0.0347689551
    ## 227  0.9762606330  -0.308757011  0.3838579018  -0.032751689  0.0395339286
    ## 228  0.0841409506   0.094962837 -1.5504698250  -0.317393624 -1.8073441813
    ## 229  0.6466869439  -0.545332315 -0.6498485681   0.407978403 -1.3960443936
    ## 230  0.2357686908   0.976797445  0.1313471370  -0.075112352 -0.9913248213
    ## 231  0.6755554480   0.903093058  0.4031664706   0.204841696 -0.4126163019
    ## 232 -0.5679385672   0.111845691  1.4565101061  -2.344863071  1.0914108493
    ## 233  1.3777285071   0.332367697 -0.8889656872   0.130292939  0.7866517716
    ## 234  0.4376440724  -0.592007095  0.0450791329   0.081891750  0.8552396119
    ## 235  0.0439207214   0.591293305  0.9003717716  -0.421287684  1.0207636518
    ## 236 -2.1273573254  -0.213113280 -1.3719380500  -0.060785979 -0.9313258708
    ## 237 -0.1912248464   0.114833265  0.4013255626  -0.574217036 -2.5900493281
    ## 238 -0.8969491703   0.779463120  0.6910707581  -0.311483411  0.4139066464
    ## 239  1.3543444694   0.259413781 -1.0253310234   0.159528969  0.8027777248
    ## 240  1.0334920513   0.581600741 -0.8707851989   0.073752798 -0.5382707871
    ## 241  1.1116385210   0.417004889 -0.8831359312  -0.124509998  0.1044386458
    ## 242 -0.6555601647   0.004307891 -0.1599447656  -0.798963542  0.2359479381
    ## 243 -1.1131361911   0.677552647  1.7607566412   0.034028361  1.1287697834
    ## 244  1.5827010790   1.015021060 -0.0284751825  -0.572471057  0.3435678131
    ## 245 -1.3083634027   1.152103235  0.2882595296   1.577750677  1.2683513933
    ## 246 -1.1695193125   0.996752256  0.8309054660   0.754164785  0.5185792281
    ## 247 -1.1428875608  -0.605908357 -0.6661733994   0.931713982 -0.2047166840
    ## 248  1.5249870474  -0.195007804 -0.0019227179   0.160391620  0.0861803483
    ## 249  1.2550143356   0.873409009  0.3674615008  -0.941773586 -0.9183873445
    ## 250  0.1070397736   0.355483658  0.9069144723  -2.407366759  0.6266118817
    ## 251 -1.2225885789  -0.617304337 -1.3360823564  -0.100570089  0.0593959894
    ## 252 -0.2283452362  -0.277220610 -0.1705605460   0.761568650  0.0206277262
    ## 253 -0.8710249488  -0.503776136  0.9319360235  -0.878772769 -0.4294227527
    ## 254 -1.5369662182  -0.410675155 -0.6909922552   0.605578315  1.0490451431
    ## 255 -0.3010550458  -0.034979317 -0.3522859614   0.333344947  1.1906524321
    ## 256  0.5705507077   0.319560024 -1.7171761210   1.009025712 -1.0744728229
    ## 257  0.2994303886   1.363786702  1.3470116296  -0.579382828 -1.0189386583
    ## 258 -0.2913302925  -0.222165895  0.5052999582  -0.160921389 -0.8359134195
    ## 259  1.1857661942  -0.165831786 -1.4566978653   0.595377828 -0.6225034739
    ## 260  0.0817066614   0.094132329 -0.0394336317   0.636754215  1.0299344374
    ## 261  0.2769039993  -0.118631216  0.1486036237  -2.563148250  1.0986020213
    ## 262 -1.3487325870  -0.908709960 -0.9381383300   0.032220054  0.5635200978
    ## 263 -1.3018442331  -0.280708082 -0.4526675106  -0.177904747  0.0543979553
    ## 264 -0.2192253550  -0.233725489 -0.8338471066  -0.033104373  0.1354939776
    ## 265  0.8217414287   0.301448779 -1.0702488042  -0.902421646  0.2448902213
    ## 266  0.8040896136   1.018017709 -1.0396352426   1.937050355  0.4076767406
    ## 267 -0.4627395369   0.030686530 -0.3702595626   0.827537523  1.3617013050
    ## 268  0.1419981588  -0.658298069 -1.2158079159  -0.659942065  0.2292849237
    ## 269  0.6832203212  -0.023566247 -0.1635354531  -1.612792860  1.3688116433
    ## 270  0.3163819231   0.444908952  0.2861378952  -2.680980393  0.1567640501
    ## 271  1.6219515485  -2.600990208  0.3278640408   2.683528218 -1.0709586877
    ## 272  2.7984204853  -0.904016330  1.4061695041   1.255426896 -0.6193087234
    ## 273  0.0722428533  -0.171328226 -0.2503630102  -0.359214917 -0.2494924545
    ## 274  1.0540626933  -0.261809938 -1.3867252536  -1.488357254 -1.1362926834
    ## 275 -1.0282498969  -0.340462960 -0.9072141202   1.098027819 -0.5779812041
    ## 276  0.6584794483   1.265049218 -0.2065181664   1.013603212 -0.5494135096
    ## 277 -0.1570744495   0.041548036  0.3284786388  -0.541341971  0.8883200529
    ## 278  0.2590110304   0.457433873  0.0695885024   0.445144558  1.5505631426
    ## 279  1.0983169518   1.475760726  1.2001700238  -0.095650160  0.1066338211
    ## 280 -1.4813304345   0.979371128 -0.0754616816   0.865089001  0.5186264606
    ## 281 -0.1176125565   0.466487151 -0.2000149652   0.130178727  0.7967156060
    ## 282 -0.0696237141   1.313694733  0.1959486927   0.121482490 -0.2140118106
    ## 283  0.1662151489   0.959050567  1.5871687908  -0.558536466  0.9445055530
    ## 284  1.0128192826   0.756549630 -1.0837010702   0.437005593 -0.6989043461
    ## 285 -0.6228525329  -0.501052357  0.7134826706  -0.218188156  1.0435384689
    ## 286  1.3969207695   0.661371760 -0.6539508785   0.706112322  0.2638353404
    ## 287 -1.3696301914  -0.097532824  0.4182667713  -0.194360870 -0.3806290148
    ## 288 -0.3558117484   0.086835327  0.2820517314   0.093533330  0.8214660918
    ## 289 -0.1994508868   0.127205191 -0.4784081612   0.745216550  0.0165153692
    ## 290 -0.4283583435  -1.601794164 -0.9049060878  -0.038732608  0.7034683512
    ## 291  1.1846516005  -2.446924789  0.2895560747   1.925215977 -0.4674529627
    ## 292 -0.7462706076   0.149901340  1.3069586771  -1.211738011 -1.1976389755
    ## 293  2.0574704069  -1.168847396  1.8318182585   1.452858063 -1.5987540057
    ## 294  1.2403233213   0.706436138 -0.8940607525   0.212944012  0.2501593173
    ## 295  0.9264544862   1.913692066  1.2931518107  -0.231922078  0.7457435116
    ## 296 -0.8601813021  -2.514164483  2.8464197632   0.851513572 -1.6963469918
    ## 297 -2.1266762129  -0.211155168 -1.3634623041  -0.050023112 -0.9253453646
    ## 298 -0.5895980404  -0.174712053 -0.6211270614  -0.703512784  0.2719566102
    ## 299 -0.8047729377   0.075486633  0.4343494623   0.356166988  0.5487050306
    ## 300  1.0635777485   0.966467577  0.0730754985   0.841013615  0.1102492268
    ## 301  0.0839619928  -0.568138948 -1.8003831569  -1.136015539 -2.4592135499
    ## 302 -0.4289536632   0.004824522 -0.4016880193   0.642930398  0.7254920905
    ## 303  0.3074854914   0.163594593 -1.0651997140   0.498534028 -0.9651933644
    ## 304 -0.1688100023   0.132391790  0.0686653943   0.269790078  2.0003190371
    ## 305  0.9305451821   0.125317266 -0.0602844855  -0.143802009  0.4169017114
    ## 306 -0.8656049490   3.314576245  0.6827534505   4.000226899 -0.7734996602
    ## 307  0.1423708370  -1.534183269 -1.5634546121   0.533537566  0.1673620569
    ## 308 -1.7047624854   0.246920901  0.6208356280  -0.515260359  0.5252766336
    ## 309 -0.4541366084   0.438311794  0.6635582884  -0.216132734 -0.1019796747
    ## 310  1.2945730225   0.356520997 -0.3714499169   0.503222568  0.2204853447
    ## 311  0.8285723602   0.576798588 -0.1690251526   0.561773475 -0.1260651718
    ## 312  1.7565343976   1.181234651  0.1512377078  -1.351686815 -0.4421829485
    ## 313 -1.2967484530  -0.836922931 -0.8495681260  -0.623653606 -1.2057437536
    ## 314  1.0490409356   0.055268911 -0.7547533588  -0.081797714  0.3120335590
    ## 315  1.1654413545   0.847028580  0.5838660415  -2.641748425 -1.0597950222
    ## 316  1.1290493573  -0.153598345 -0.7491761740   0.479598197  0.7004471290
    ## 317  0.6408360547   0.925540502 -0.1453705739   0.489507769 -0.6168574010
    ## 318 -0.9228346490  -0.823873682 -1.7930482792  -0.021335611  0.9624940316
    ## 319  0.9574212094   0.151273106 -0.9331824961  -1.203328931 -1.6931927264
    ## 320 -0.8697778067  -3.824983440  0.4931502413   0.430356592 -0.3828668939
    ## 321 -0.2299869521   1.016914810  1.4947641382  -0.395835965  0.0447659859
    ## 322 -1.0290774147  -1.084895032 -0.2744123443  -1.379681795  0.2619304469
    ## 323  0.5611636934  -2.195146101  0.6168082120   1.508406316  0.8942633023
    ## 324 -0.9793434655   0.257715988  0.8176198914  -0.552312716 -0.4968373186
    ## 325  0.3597497464   0.413179137  0.0294786146  -0.247528024  0.5453900555
    ## 326  0.2835700377   1.023162921  0.5958366913   0.136665602 -0.7862347060
    ## 327 -1.6245079146  -0.070876557 -0.1305394152   0.276917847  0.1227339685
    ## 328 -0.9182682989  -0.509246818  0.9066268405  -0.766878558 -0.5047472793
    ## 329  0.8971751701   1.015008948 -0.4438412432   0.261656372 -0.0036509386
    ## 330 -0.6994441798   0.604405678  0.3906398870   0.214245477  0.0580252541
    ## 331 -1.4966549372  -1.637052673 -0.8443474523   0.042883595  1.2134840498
    ## 332 -0.7064966140  -0.316312593 -0.9818592998  -0.086944858  0.1429150905
    ## 333 -0.9315316157  -0.422239192 -0.8923747365   0.787195532 -0.0169350608
    ## 334  0.1398423244   0.240025900 -0.6091964669   0.257209904  0.3372449358
    ## 335 -1.0934956326  -1.504665962 -0.6088434030  -1.969177565  0.5773372117
    ## 336 -1.3598769123   0.099848708 -0.3741698160   0.523217892 -0.4302780150
    ## 337  1.2100364399   0.078280832 -1.7425344472   0.942970876  0.3944934551
    ## 338  2.3070416384   1.017644885 -0.6221367725   0.711076161  1.5063185094
    ## 339  0.0083328906   0.309815562 -1.5121001858  -0.739145921 -2.1304634226
    ## 340 -0.6328053054   0.113201048  0.0027702686   0.596374615  0.5736067862
    ## 341  1.3200491156   0.047692923 -1.8317493647   0.851474903  0.5388013912
    ## 342 -0.6013925984   1.137744874  0.9432419981   0.958023240  0.4268726787
    ## 343  1.0134997480  -1.834504527  1.3322289995   1.698907880 -1.4067119182
    ## 344 -0.5305918723  -4.434670911 -0.0869076139   0.710710572 -1.1475233337
    ## 345  1.3476088995   1.407238306  0.5883155304   1.342666038  0.3331043203
    ## 346 -0.2431311984   0.180698742 -2.3010706419   0.022085735 -1.7568911770
    ## 347  0.8645351940  -0.504910158 -2.2221073615   0.565197885  0.6918523660
    ## 348  1.0944030586   0.625358421 -0.2892508784   0.238481773  1.0305560812
    ## 349  0.1329127780  -1.309981192 -2.1584639824   0.316078445  0.2618847514
    ## 350 -0.0917675273   0.823115703  0.1614008831  -0.102122393  0.0700922560
    ## 351  0.2216954349   0.338760069 -0.2652317468  -0.967238847 -0.1873612769
    ## 352  0.6002983872   0.206795621 -0.7111885976   0.555675444 -0.7412374294
    ## 353  0.7930005545   1.398542874  1.4088620367  -0.023034386  0.3112328826
    ## 354 -0.4740880962   0.724128501  0.2349179606  -0.006276156 -0.1336716428
    ## 355 -1.3728062085  -1.209839162 -1.6973354116   1.314090715 -0.0384287307
    ## 356  0.3344229099   0.001529854 -0.8296851203  -0.539481748  0.9072795564
    ## 357  1.5803787278  -2.951874118  1.4220143727   0.863177655 -0.4691946447
    ## 358 -0.4644761985   0.104783627  0.6788717744  -0.651289307  0.6043533156
    ## 359  0.3963447569   0.129240170 -0.3348891600   0.579915814 -2.3609939917
    ## 360 -0.5707539734   0.443418106  1.2743159933  -0.873578999  0.1104687388
    ## 361 -1.7013955572   1.175196355 -0.5124963517   0.861061257  1.4030777830
    ## 362  0.1170016279   0.393731598 -0.4148124799   1.197943117 -0.8061510777
    ## 363  0.7953282304   0.839370040 -0.1864478438   0.512033962  0.1091482715
    ## 364 -0.3073512557  -1.312561576 -1.2242568966  -0.405225392 -1.6190842006
    ## 365  0.6650283886  -0.297753540  0.0009773148  -0.250258311 -0.6253691963
    ## 366  0.2501383221  -0.009888978 -0.1330377265  -0.721701827  0.8055662496
    ## 367  0.1321528988   1.166254839  1.3289058008  -0.016298726 -1.0103352895
    ## 368 -0.8003664028  -0.094084579  0.6680719676   0.308685555  0.9531932052
    ## 369 -1.6878922104   0.476497220  1.2858821388   0.377675151 -0.2325139640
    ## 370 -1.2414764313  -0.981778202 -2.1363360717   0.543915868 -0.1211855548
    ## 371 -0.5129099098   0.546898620  0.0535872602  -0.011192973 -0.0405103569
    ## 372 -0.7200471206   0.366835116 -0.1108569612   0.319093740  0.1083591592
    ## 373 -0.1798216331  -0.798438725 -1.3934946658   0.260531922  1.2974690931
    ## 374 -0.6582833826  -0.676024729 -0.4116007828  -1.472240245  1.4620218496
    ## 375  0.7126441930   1.227137928  0.5057424045  -0.731313612 -1.1191995116
    ## 376 -1.2703186421  -0.044035206  0.3296079052  -0.475084868  0.7610485035
    ## 377 -0.8677435228  -0.171366797 -0.0950154788   0.129085777  0.8839682388
    ## 378  0.0946347815   0.408639089  0.1897018079   0.239682951 -0.2797605555
    ## 379  1.1029131840   0.148513798 -1.3790546933   0.939988749  0.2446759810
    ## 380  0.7624623714  -0.270265269 -0.3669253554   0.671547449 -0.6862048960
    ## 381  0.7684522260   0.196885534 -1.5347129370  -0.794458592 -1.0850424007
    ## 382  0.9697954023   0.861303861 -0.7947558600  -0.009047510 -0.4587827343
    ## 383 -0.6372581837  -0.336255853 -1.6591958332   1.221687577  0.0054983140
    ## 384  2.4599590703  -2.261846431  1.3197665059   1.234399776 -0.1546369311
    ## 385  0.0818092429  -0.124584012 -0.1937593928   0.314485965  0.8129856403
    ## 386  0.0941306631  -0.291525831 -1.8290715609   1.312531143  0.0168374958
    ## 387 -0.2117147407  -0.573465684 -1.4160803528   1.103045247  1.8791399134
    ## 388 -2.1229612318  -1.609974078 -0.2481636326  -0.103244656 -0.5861536562
    ## 389 -0.7412860013  -0.070347726  1.6524070741  -0.811359378 -0.0557339380
    ## 390 -0.5730649454   0.180040963 -0.4734267159  -0.123181374 -0.3895712642
    ## 391 -0.8699971172  -0.153572036 -1.0648060660   0.291092666  1.1540458147
    ## 392 -1.6195009525  -0.409944095 -0.5669943744   0.888660448  0.5666768781
    ## 393 -0.6396870791   0.306873657  0.4900191638  -0.932997858  0.1729721136
    ## 394 -0.1186509978  -0.526850558 -0.0309541436   0.208721823  1.0643662735
    ## 395 -1.7179961613  -0.356980840  0.4659425099  -0.942769486 -2.0190848390
    ## 396  0.4627948099   0.504700966 -0.1019765372   0.259999696 -1.4088408432
    ## 397  0.7221560329   0.228941044 -1.0849630863   0.397240026 -0.9890247536
    ## 398 -0.5647754849  -0.937032478  0.5166811511  -0.245939818  0.8712841972
    ## 399  0.6218472348   0.358605705 -0.3882773615  -0.107124615 -0.9281683527
    ## 400  1.1503151011  -2.085766336  2.5809437738  -0.346739505 -1.5297325316
    ## 401 -0.0567884574  -0.855611463 -0.6950344876   0.684859105  1.5515456736
    ## 402  0.6761813603   1.165444937 -0.3188857101   0.336253038 -0.3056236933
    ## 403  0.6883319746  -0.575382896  0.1142119728  -0.955319948 -0.0769077877
    ## 404  1.6070404147   0.960969377 -0.4896508407   1.036108374  1.4859350349
    ## 405 -0.7935379866  -0.648823324  1.3913432612  -0.650212866  0.8473638387
    ## 406  0.9404388011   0.332307663 -0.2380324572   0.351654186 -0.0669131878
    ## 407 -0.2360722380  -1.051907221  0.4266178128   0.010737325  0.5522074086
    ## 408 -0.4783994215  -1.483679526 -2.0090485737  -1.475912336  0.9389871999
    ## 409  0.8223939848   0.147087975  0.5878226506  -0.041809224  0.8433685553
    ## 410 -0.9594382879   0.225914998  0.2192764242  -0.361047798 -0.4422562295
    ## 411 -0.5603561700  -0.920782690  0.8075826557  -0.918496897  0.4595481367
    ## 412  0.4983360068  -0.462181459 -0.0516102046  -0.532091889  2.5681375225
    ## 413  0.4341960356  -0.762357563 -1.2552307719   0.829551114 -1.0446377735
    ## 414 -0.7776092233   0.873550403  1.2381174173  -0.755360352 -0.5002724153
    ## 415  1.8114124030   0.596792846  0.7667828537  -0.082243945 -0.0023278131
    ## 416  1.6255976529   0.443112233 -0.4588457335  -0.110735858  0.4971679714
    ## 417 -1.5817830879  -0.990694615 -1.0943804695   0.578059824 -1.0317820641
    ## 418 -0.6534687726  -0.258207914 -0.3508346311   0.813391990  1.5965115003
    ## 419  0.5141479608   0.520646164 -0.3925394510   1.065051737 -0.3432328248
    ## 420  0.1479761125   0.623086215 -0.3842066066   1.206208641 -0.7860811772
    ## 421  0.6163984319   1.174795037  0.0544668844   0.489782295 -0.1113686874
    ## 422 -0.8435604866   0.807719337  1.9675172038  -1.093040205 -1.0367970113
    ## 423  0.6909016761   0.541316898 -0.6594278858   0.980040409 -0.7745371312
    ## 424 -0.1332593700  -2.377702295  1.8013450168   1.107203065 -0.0420728102
    ## 425 -0.8896438031  -0.051190939 -0.5518989050   0.594853253  0.9714838511
    ## 426  0.9296599581   0.775525576 -0.6825161945   0.513250178 -0.9998589124
    ## 427 -0.2980423037   0.633722385 -0.3254534484   0.309831752  0.0930197358
    ## 428 -0.3727557830  -0.143587071  1.2264156858  -0.592594508 -0.5977896510
    ## 429  0.0784887288   0.371194209 -0.7532923448   1.107450336 -0.9630523323
    ## 430  1.3881824125   0.504662795 -0.0803648276  -0.190758836  0.4882534935
    ## 431 -0.9031901205   0.650134026  1.2346715204  -0.115871865 -0.1485420958
    ## 432 -0.2167652936  -2.839115656  0.7457862613   2.091712436  0.2931746194
    ## 433  1.2532640609   1.147548510 -0.6088031941  -0.084948041 -0.4209235092
    ## 434  1.2955168829  -0.215718716 -0.6107882234  -2.539897073  0.0058389782
    ## 435 -0.0224967425   0.932562128  1.8227295118   0.274698015  1.5708995085
    ## 436  0.3954829095  -0.397012350  1.1749646920  -0.443251060 -0.6912246963
    ## 437 -0.2581656788   0.206740067  1.4980263020  -2.866074353  1.0194740432
    ## 438 -2.0198523711  -0.210620486 -0.5057680727  -0.881185140 -0.8309849928
    ## 439  0.1004011485  -0.499377531 -0.4018194730   0.088227192  1.7441547811
    ## 440  1.3262070567   1.059774979 -0.1769254453   0.496890790 -0.6065237505
    ## 441  0.0722695845   0.115300070 -0.4397378155   0.137210435  0.4358725205
    ## 442  1.0555773537  -0.279963051 -0.3915979275  -1.154825298  1.1600961158
    ## 443 -0.5379020682  -1.020126443  1.1178815033  -0.721537409  1.0586800033
    ## 444  0.0046252528   0.166726630 -0.3004174521   1.012574953  0.6024290328
    ## 445  0.7784686139   0.149519138 -1.0017990067   0.551129366 -0.0378930618
    ## 446  0.1118217521  -0.210671623 -1.0263959961  -1.203979025 -2.2782976361
    ## 447 -0.9814045833  -0.742115576 -0.4011857433   0.118450210 -0.6197333465
    ## 448 -0.5517945080  -0.201278306  0.7743481055  -0.470072240  0.4246597005
    ## 449 -0.1817779409  -2.335893774  1.1702991777   0.550485745 -1.9764789756
    ## 450  1.2727231872   0.668351773 -0.3412667398   0.388808859 -0.2412932671
    ## 451  1.4431423776   0.349414775 -0.8141678462  -2.309466734 -0.7649788806
    ## 452 -0.4444749155   0.147922264  0.8968022939   0.270445728  1.8118669851
    ## 453  1.3678550421   1.745094631 -0.5246013043   0.552709784 -0.4990000276
    ## 454  0.6204671728   0.054382804 -1.9021480011  -0.015820480 -1.0178171574
    ## 455  0.4795704403   0.224940457 -0.2935669059   0.651672405  0.0647170526
    ## 456  3.2020332071  -2.899907388 -0.5952218813  -4.289253782  0.3897241203
    ## 457 -0.4145754483  -0.503140860  0.6765015446  -1.692028933  2.0006348391
    ## 458  2.0329121576  -6.560124295  0.0229373235  -1.470101536 -0.6988260686
    ## 459  4.8958442235 -10.912819319  0.1843716858  -6.771096725 -0.0073261826
    ## 460  2.1013438650  -4.609628391  1.4643776248  -6.079337193 -0.3392373727
    ## 461  5.6643947086  -9.854484823 -0.3061666583 -10.691196212 -0.6384981927
    ## 462  6.7546254481  -8.948178579  0.7027249981 -10.733854103 -1.3795198568
    ## 463  4.5607201055  -8.873748362 -0.7974835996  -9.177166370 -0.2570247751
    ## 464  6.4390533516  -7.520117393  0.3863516674  -9.252307247 -1.3651884150
    ## 465  5.5887239147  -7.148242636  1.6804507410  -6.210257747  0.4952821178
    ## 466  4.6757294187  -8.167188052  0.6385592822  -6.763334391  1.2968602561
    ## 467  3.5720548156  -7.186451591  0.1472423408  -5.249304549  1.6783336736
    ## 468  5.4507460669  -7.333714067  1.3611933241  -6.608068252 -0.4810694249
    ## 469  4.3667134863  -8.243262434  0.3457611652  -6.590550297  0.2655760959
    ## 470  6.3550777366  -7.309747984  0.7484505506  -9.057992529 -0.6489449091
    ## 471  7.3880551229 -10.475228650 -0.3793145268 -11.736729121 -2.0869889522
    ## 472  7.6200890584 -10.285282516 -0.3424443341 -11.543497916 -1.3349879033
    ## 473  6.6624368307  -8.525464925  0.7427454469  -7.678668010  0.5930704591
    ## 474  6.4541875249  -8.485346574  0.6352814088  -7.019901559  0.5398140713
    ## 475  6.3536123192  -8.601648263  0.4499300382  -7.506169374 -0.4380817818
    ## 476  6.3162096772  -8.670818005  0.3160239919  -7.417712065 -0.4365374714
    ## 477  4.8023227613 -10.833164469  0.1043038757  -9.405423062 -0.8074778687
    ## 478  7.1029885879  -9.928699943 -0.0674983775 -10.924186865 -1.6979139771
    ## 479  7.0716691368 -10.001046368 -0.2079345409 -10.860698045 -1.6908250412
    ## 480  4.6903956655  -6.998042432  1.4540119864  -3.738023334  0.3177420626
    ## 481 12.0189131816 -17.769143463 -0.4310356582 -19.214325490 -0.9624645747
    ## 482  4.6540884225  -7.839538998  1.3718185803  -9.634689713 -0.7395968558
    ## 483 10.8530116482 -15.969207521  0.5466900846 -14.690729134  0.9123373999
    ## 484 11.6197234754 -17.631606314 -0.3551943895 -18.822086742 -1.2831290856
    ## 485  4.5701128076  -7.629169589  1.7339174635  -9.440374995 -0.0233533499
    ## 486  5.2755058508 -11.349028550  0.3745492351  -8.138694884  0.5485708942
    ## 487  5.1494087830 -11.124018607  0.5430677655  -7.840942205  0.7436339448
    ## 488  3.1871868974  -7.004327284  0.8727106055  -6.220605353 -0.9044446296
    ## 489 11.6692047358 -17.228662239  0.0555715665 -18.493773355 -0.3041721135
    ## 490 10.4468468145 -15.479052483  0.7344416081 -13.883778549  0.8214395454
    ## 491 11.2284702796 -17.131300945 -0.1694010568 -18.049997690 -1.3662356610
    ## 492 11.2779207278 -16.728339332  0.2413676826 -17.721638354 -0.3872999234
    ## 493 11.1524905986 -16.558197141  0.3026445224 -17.475921283 -0.4123932673
    ## 494  3.3385021604  -6.542610345  1.0995363818  -3.266475545  1.0147279044
    ## 495  1.0727283997  -2.547556814  1.2359500568  -0.330305804 -1.0223543746
    ## 496  0.1549210089  -2.776756575  1.6412066261  -0.456076957 -1.2404143924
    ## 497  0.5003763700  -3.021760182  0.9242132603   1.500793775 -0.4002826442
    ## 498 11.0270590938 -16.388054167  0.3639214865 -17.230202161 -0.4374875592
    ## 499  9.3690790577 -15.094163149  1.2563770070 -11.852161304  0.2744298142
    ## 500  9.3287992566 -13.104933466  0.8884807877 -10.140200337  0.7134653926
    ## 501 10.5452629546 -15.022699634  0.1716330851 -15.066374363 -0.2595759396
    ## 502  8.8056819672 -13.556130130  1.1654637603  -9.809881502  0.3699872792
    ## 503 10.2777688628 -14.985433733  0.3451792341 -14.666388973 -0.3463525420
    ## 504 10.1875873242 -14.563979755  0.5937593576 -14.491598346 -0.2813925040
    ## 505 10.0637897463 -14.394766802  0.6548887235 -14.248315827 -0.3053607614
    ## 506  9.9398197417 -14.225455704  0.7160336652 -14.004776176 -0.3294478482
    ## 507  9.8157031745 -14.056061184  0.7771918464 -13.761017962 -0.3536359398
    ## 508  9.6914609821 -13.886595159  0.8383613758 -13.517072408 -0.3779106071
    ## 509  9.5671102952 -13.717067379  0.8995407059 -13.272965062 -0.4022600443
    ## 510  9.4426652654 -13.547485900  0.9607285583 -13.028717026 -0.4266744987
    ## 511  9.3181376854 -13.377857434  1.0219238678 -12.784345887 -0.4511458433
    ## 512  1.1240593825  -3.763873848  0.3679759668  -0.971757538 -0.0138825132
    ## 513  7.3944194105 -11.635630048  1.4232769255  -8.640459100 -0.6747195389
    ## 514  6.3293646762  -8.952190712 -0.1383636913  -9.825054421  0.0572235922
    ## 515  6.2108830117  -8.778572032 -0.0613674967  -9.574662299  0.0492885399
    ## 516  1.8344941980  -0.491243315 -1.1299070046  -2.238622244  0.2784687417
    ## 517  5.9662025072  -8.463966130  0.0786921318  -9.092532652  0.0108220974
    ## 518  5.8492930662  -8.261649837  0.1538291483  -8.829359228  0.0088788388
    ## 519  5.7308155374  -8.088033512  0.2308249693  -8.578973273  0.0009466367
    ## 520  5.6123471080  -7.914422366  0.3078199684  -8.328600888 -0.0069792946
    ## 521  5.4938868365  -7.740815864  0.3848142305  -8.078240669 -0.0148996037
    ## 522  5.3754339072  -7.567213541  0.4618078293  -7.827891399 -0.0228148529
    ## 523  5.2569876096  -7.393614992  0.5388008291  -7.577552018 -0.0307255319
    ## 524  5.1385473219  -7.220019864  0.6157932860  -7.327221600 -0.0386320691
    ## 525  5.0201124978  -7.046427847  0.6927852494  -7.076899329 -0.0465348410
    ## 526  4.9016826551  -6.872838664  0.7697767627  -6.826584487 -0.0544341798
    ## 527  4.7832573670  -6.699252074  0.8467678647  -6.576276436 -0.0623303799
    ## 528  4.6648362539  -6.525667860  0.9237585895  -6.325974612 -0.0702237028
    ## 529  2.3516191062  -3.826705319 -0.7675698491  -4.852866915  1.6648630042
    ## 530  4.4199434558  -6.210941121  1.0638373951  -5.843528381 -0.1088364492
    ## 531  0.9872770220  -3.834781569 -0.9613541425  -3.182168085 -1.1260022842
    ## 532  4.3030958176  -6.008660007  1.1389688288  -5.580447120 -0.1107371158
    ## 533  4.1846736894  -5.835075215  1.2159596453  -5.330143782 -0.1186311382
    ## 534  4.0662550729  -5.661492423  1.2929501445  -5.079845681 -0.1265227404
    ## 535  3.9478397173  -5.487911486  1.3699403492  -4.829552443 -0.1344120954
    ## 536  3.8294273950  -5.314332276  1.4469302799  -4.579263728 -0.1422993600
    ## 537  3.7925653599  -1.460471123 -1.3127762764  -7.077360606 -0.2994663166
    ## 538  2.1115168671  -2.591949650 -1.9505893547  -7.311579817  0.0777527658
    ## 539  5.4793915101  -5.657857541 -2.6480760899 -10.384889741 -0.2259464078
    ## 540  3.0281622310  -2.549177308 -1.5604318247  -2.971316758  1.0788953965
    ## 541  0.6752878534  -0.677096334 -0.2227411281  -0.913786971  1.6033319231
    ## 542  2.3684336728  -3.656802651 -0.1695346833  -4.744413330  0.7651928293
    ## 543  1.9758210618  -3.500542265  0.1706810271  -2.735940374  1.6702511035
    ## 544  2.7561719454  -2.242864368  0.6152042469  -4.652803807  0.2221638155
    ## 545  1.1205051771   1.237773313  0.3423621144   0.038371979 -1.1945030885
    ## 546  2.8020051262  -4.392732438 -1.3696714497  -5.327286652  0.1631589615
    ## 547  3.9073991278  -7.220003686 -1.2117393457  -9.657626594  0.9275178995
    ## 548  4.7299741544  -8.629054408  1.1787979629 -11.182063414  0.4452431127
    ## 549  0.0243696058  -1.073812240 -2.0027691092  -0.895433621  1.7489363011
    ## 550  3.7797495390  -6.984771477 -1.1409771886  -9.377878452  0.9214635067
    ## 551  3.8397877107  -8.277840948  1.4939149081  -8.416681212  0.7922979282
    ## 552  3.7796022078  -8.077093646  1.4408888558  -7.891908779  0.5304527884
    ## 553  3.5382020401  -8.042285142  1.6975093837  -8.051048902  1.0704573965
    ## 554  3.3678456082  -7.888977520  1.6826034667  -7.628651916  0.7792459957
    ## 555  2.9543443655  -7.099825343  1.5203689080  -7.687802767 -0.2250020210
    ## 556  3.6573495650  -7.781447953  1.5077204268  -8.604758868  0.0105926876
    ## 557  3.6767031869  -7.642983067  1.6898651227  -8.299659531  0.6469963648
    ## 558  3.2660660158  -2.719184951 -0.1241039633  -5.274865819  0.6385750026
    ## 559  0.1597435223  -0.490697387 -1.1819769934  -1.958875970  1.1527432862
    ## 560  2.8043347798   0.312424169 -0.5959764673  -4.662576955 -0.2995150200
    ## 561  1.8440931702  -2.425932815 -1.5842751344  -7.208311579  0.4598283833
    ## 562  5.7300843556  -5.031868148 -1.1354391093 -10.787050570  0.2654562305
    ## 563  5.7357626592  -4.747692501 -0.7842232845 -10.716339302  0.6565169781
    ## 564  9.0952881525 -14.168120882 -0.1404553898 -17.620634352  1.4827404110
    ## 565  3.8237620841  -7.492276169 -1.5243674799  -8.527678622 -0.3912145245
    ## 566  3.9697998123  -7.346716786 -1.1633117696  -8.225568912  0.8250018319
    ## 567  7.2529532268 -14.275091847  0.1889032837 -14.555957211 -0.3382885475
    ## 568  7.3818593743 -14.468655096  0.2996599094 -13.602211466  0.5161126997
    ## 569  7.6745338342 -14.296091426  0.5269389237 -15.445025812  0.9916512013
    ## 570  8.2652946006 -14.154165361  0.4798427820 -16.337595945  1.6460452796
    ## 571  6.8775708985 -13.697685618  0.4630404018 -13.044182400 -0.3092292657
    ## 572  6.4558281329 -13.380221874  0.5452792268 -13.026863859 -0.4535945436
    ## 573  6.6452007261 -13.542095596  0.8882479188 -12.623316046  1.0027551956
    ## 574  7.0477327400 -13.742952842  0.8216272231 -14.107464263  1.0204709398
    ## 575  7.6055594744 -13.351815018  0.8270496428 -14.979477349  1.5328130844
    ## 576  6.8951813681 -13.279699711  0.7552635788 -13.417012272 -0.2107743621
    ## 577  6.8549531139 -13.211695030  1.0356123433 -13.778955640  1.0404483553
    ## 578  6.7860583020 -13.064239894  1.1795245106 -13.694873040  0.9514791762
    ## 579  7.1271653215 -12.611003687  1.2145563978 -13.906296824  1.5140699063
    ## 580  5.8047078520 -12.156239494  1.1849846628 -10.468677087 -0.4167431971
    ## 581  5.9751306669 -12.375374654  1.4722051186 -11.470692506  0.9931432537
    ## 582  5.5893616319 -11.960866033  1.5386707619  -9.887214067  0.6339792179
    ## 583  5.3306013515 -11.898065148  1.5959929020  -9.560169271  0.5964037629
    ## 584  5.1483517211 -10.749591995  1.5854889675  -9.798012077  0.7332769991
    ## 585  6.0376662294 -11.676722237  1.3269787863 -12.381605895 -0.0716392778
    ## 586  5.6109987966 -11.793978835  1.5983865300 -11.606970376 -0.0441804438
    ## 587  4.8282353322 -11.058401424  1.9981255755  -8.592047402  0.5403853092
    ## 588  5.2992363496 -10.834006481  1.6711202533  -9.373858584  0.3608056416
    ## 589  3.4908049504  -2.454339210  0.1771894852  -5.160608097 -0.6861142343
    ## 590  4.3924355857 -10.380073122  1.6118209013  -6.917639506  0.1588410374
    ## 591  3.4710978465  -6.533106769 -0.9834686696  -6.073989003  1.1254067090
    ## 592  4.3547746303  -2.872400109  0.4688754623  -4.555666748  0.6996714930
    ## 593  3.1860579481  -6.855570981 -0.8377226148  -8.036614875  0.9569696438
    ## 594  4.8181524471  -9.445314783  1.3170562933  -7.243460974  0.8309102910
    ## 595  4.6773492966  -9.583566437  1.0351658236  -8.199690029 -0.3265188825
    ## 596  3.5868241834  -6.636229049 -1.1281764439  -7.245549595  0.6383255362
    ## 597  4.3892135026  -5.849558007 -0.7509643746 -11.583898387  0.8387500530
    ## 598  4.0972160744  -5.450916439 -0.9656927522 -10.904458873  0.5269477645
    ## 599  6.2786658455  -9.407060948 -1.8188739608 -10.739571891  0.2242060837
    ## 600  6.2833770785  -9.656606360 -1.6191348548 -10.732108925  0.2882207645
    ## 601  2.7789678964  -2.379599611  0.9189627840  -4.015786780 -0.5300097906
    ## 602  3.6578816667  -6.409822355 -1.0873102948  -8.509433187  1.4325717873
    ## 603  3.3673610231  -4.583095845 -0.8069396837  -5.208305222  0.5258236822
    ## 604  3.5257259872  -6.489111560 -1.0127906317  -7.052310952  0.7844454033
    ## 605 -0.4738521059   0.922184420  0.9135138398  -0.499349579  0.9338984429
    ## 606  3.5322196024  -3.682639870 -1.1547774863  -5.165229227 -0.2400914255
    ## 607 -0.9841717281  -0.567380065 -1.1055922497  -1.381213542  0.4054904212
    ## 608  3.4382482599  -3.521529360 -0.9187612884  -4.452100249  0.4993137539
    ## 609  2.9625987914  -3.956045201 -1.5392319194  -4.634631393 -0.2484034907
    ## 610  2.8870478766  -3.784459658 -1.2889042585  -3.985626008  0.5318382980
    ## 611  1.4802860203  -0.105359123 -0.8048414548  -0.819901849  0.9168817665
    ## 612  2.2819393759   0.654566719 -0.3103794323  -1.043705225  0.5137503993
    ## 613  0.1411789934  -0.460348456 -0.3678679310  -1.168339418  0.1521076306
    ## 614  2.8143141548  -3.342175996 -0.1502693666  -4.820322244  1.1026095579
    ## 615  2.4016506803  -3.394265618 -0.1451617250  -4.329795513  0.7587608858
    ## 616 -1.0645341945  -0.787372738 -1.5764695845  -0.126756362  0.3670122819
    ## 617 -0.6683588313  -0.818803694 -0.8838320911  -0.994580854  1.5068487502
    ## 618  2.5555891531  -3.530436270 -1.0162338153  -3.455196577 -0.0563633864
    ## 619  1.9622359746  -2.385262661  0.4583372023  -3.724543454  1.8439783251
    ## 620  2.9439852198  -6.512389337 -0.6981759468  -7.543645932  0.9264552840
    ## 621  2.8266712282  -6.309842777 -0.6230023858  -7.279869219  0.9242332260
    ## 622  2.7078564051  -6.136034466 -0.5459760958  -7.028980273  0.9160685746
    ## 623  2.5891035399  -5.962261422 -0.4689554027  -6.778183722  0.9079466220
    ## 624  2.4704010156  -5.788517031 -0.3919392571  -6.527462241  0.8998593623
    ## 625  1.1723550829  -0.795988137 -2.1369509658   0.578869439  1.4121147225
    ## 626  0.9031174529  -0.158825775 -1.9307090190  -2.376653641 -1.8101724847
    ## 627  3.2009123555  -2.450832429  0.3166198786  -6.397170315 -2.1297283654
    ## 628 -0.8767375917   0.687487636  1.5747177026  -0.007791230  0.9483156918
    ## 629  2.8211448880  -3.100545658  0.1461993743  -4.510123915  0.5744900227
    ## 630  0.7474781358  -4.596611512 -1.9277597019  -4.757074499 -1.2478148189
    ## 631  1.1744750601  -4.381919567 -1.2266658982  -2.953824253  1.9941611939
    ## 632  2.3785372249  -2.330270647 -0.2462327249  -4.058523436 -0.3169831466
    ## 633  0.0045179680   0.044396639  0.4208530624  -0.931614225  1.1479736828
    ## 634  6.2500675898  -9.150822685 -1.3761414696  -9.982584707  1.2022169676
    ## 635  6.0290325632  -9.225855308 -1.5467586310 -10.309334189  0.3080623935
    ## 636  5.7631894088  -8.707879292 -1.7169494877  -9.577194393  0.1463690026
    ## 637  5.7052055072  -8.640746412 -1.6029250190  -9.466139123  0.1373241869
    ## 638 -0.1531310267  -0.814309846 -1.8947955862  -0.831482609  1.3872886952
    ## 639  2.3383232700  -2.820042188  0.7633257732  -5.419589869  0.0414327925
    ## 640  5.2665858727  -8.679678803 -1.1663656782  -8.107974661  0.7013647610
    ## 641  4.7741477401  -8.245897882 -1.3334918601  -7.974250523 -0.3987989475
    ## 642  3.2649221516  -5.095031624  0.3078075413 -10.018106061  0.2732826366
    ## 643  4.3157110301  -5.479801677 -2.1155031560  -5.588785577  0.2801080693
    ## 644  2.2551474887  -4.686386898  0.6523746685  -6.174288348  0.5943796080
    ## 645  1.8372149016  -4.540341645  0.7478458603  -6.284314070 -0.1288872165
    ## 646  5.2241240042  -7.743122345 -1.5603432004  -7.140418033  1.1859778256
    ## 647  5.7252545915  -6.949171825 -3.1277950120 -11.090424813 -0.8002725694
    ## 648  1.8475035111  -2.414822338 -2.5732245424  -6.599126130  0.1234703827
    ## 649  3.5547375503  -3.969689300 -0.9321836895  -5.876067321 -0.0667394504
    ## 650  0.6149771203  -0.657947691 -0.9484120317  -1.635144524  1.2632031631
    ## 651  3.1284396922  -3.570393731 -0.5951981915  -3.988415055  0.9959058424
    ## 652  3.7623063823  -4.226224695 -2.0466940301  -4.710497936  0.2175723600
    ## 653  2.8860914766  -6.848977934 -1.0292345906  -7.460102983  0.0559060176
    ## 654  3.0642458413  -2.718730632  0.0687877027  -5.586873126 -0.9660764777
    ## 655  2.6117618432  -2.210691055  0.1453232206  -5.281678131 -1.4462273155
    ## 656  3.5617949640  -3.914679352 -2.1249564560  -4.458895030  0.5782018754
    ## 657  0.1445626901  -4.283529310 -0.2408950415  -3.657490404  0.9231047678
    ## 658  5.0407512027  -5.314442204 -0.6564706620  -6.233043671  0.9747712793
    ## 659  4.8389641870  -6.040235379 -0.3744604837  -6.221945359  0.9587584437
    ## 660  1.9144540449  -3.830998011  0.7194885668  -6.353019858  1.4387161924
    ## 661  2.7903963906  -2.316951892 -0.7825317871  -3.431737963 -0.5277017608
    ## 662  1.3981851185  -5.653269702 -1.2336941384  -6.332512815  0.4587148002
    ## 663  1.1133540926  -4.976921041 -0.7955309347  -4.549556744  1.8137632540
    ## 664  0.9159470289  -0.468571538 -0.5233772937  -4.861866581  0.4210133539
    ## 665  3.5460122500  -7.751285478 -1.1956939450 -10.018502446  0.2963961221
    ## 666  4.9860138036  -6.116383223  0.0423244062  -6.043393144  1.8214010318
    ## 667 -0.9210519957  -0.613815529  0.1989130555   0.016087005  2.2062637752
    ## 668  3.9772222396  -8.066096093 -0.7608875918  -8.660770959  2.2202379897
    ## 669  1.5331938379  -4.635126948 -1.1135727340  -6.435870738 -0.2931919479
    ## 670  1.9285676320  -0.935863286 -2.4314464452  -1.633472024  0.5343366527
    ## 671  2.1400568155  -0.276308753 -1.1913057375  -1.880275403  0.3982719563
    ## 672  5.2678616038  -5.583696926  0.1738955924  -6.679976755  2.0841991431
    ## 673 -0.2456768233   1.375941350  0.8704568313  -0.819319365 -1.5509035620
    ## 674  1.2772022647  -3.701749667 -0.9718695935  -4.857776672  0.0906055221
    ## 675  0.8812680504  -0.324703761 -0.2632638005  -1.112735439 -0.5408495033
    ## 676  1.6182616204  -3.581374501  0.6237069199  -6.160456620  0.4947333991
    ## 677  2.6215881261  -3.462361885 -1.0543001733  -4.955823304 -1.4002744582
    ## 678  2.5607002016  -3.411982911 -0.7816696199  -4.732767535 -0.4339736549
    ## 679 -0.6514140140  -0.005423291 -0.5171943414   0.217469760  0.8835588245
    ## 680 -1.3565577561  -0.238382664  0.0695790191  -0.431690054 -0.6747238893
    ## 681  4.4068055236  -4.610756477 -1.9094879697  -9.072710934 -0.2260744509
    ## 682 -1.0394170684   0.285261609 -0.2060065678  -0.498521776 -1.0641084507
    ## 683  2.1916650800  -0.487849733 -1.7571370994  -4.461051207 -0.4028842057
    ## 684  4.6632545848  -5.171734608 -1.6240702479  -6.713878195  0.6079862677
    ## 685  1.6019853828  -2.824945625  1.2692036355  -5.591363689 -0.9748272871
    ## 686  1.5537558680  -2.743551359  1.5045406706  -5.428788499 -0.0160171234
    ## 687  3.5103475603  -2.227398406  0.6568239280  -5.199185743 -0.1283110296
    ## 688 -1.2739350803  -0.868686295 -1.1819446550   1.027584172  1.6881323448
    ## 689  4.4471917663  -5.293760103 -1.4195782993  -6.425276214  0.9140830342
    ## 690  4.3769066636  -5.007440828 -1.3047450780  -6.192475264  1.0024042674
    ## 691  0.8263748444   0.196886900 -1.8859934981  -0.472025637 -0.5781413957
    ## 692  2.5728302158  -2.288663500  0.5620841777  -5.338714210 -1.8532518914
    ## 693  3.1020899272  -3.993373054 -1.9374106233  -3.822894106  0.8309701107
    ## 694  2.6051685752  -3.677716614 -0.9612546362  -4.629480997  1.9421816128
    ## 695 -0.3956077225  -0.664683930 -0.9857004967  -0.660967627  0.3441052029
    ## 696  0.5675519546  -0.858059916 -0.4480679300  -0.264936239  0.8888629392
    ## 697  3.0483920345  -7.128250768  0.0059558009  -7.495187336 -0.2510880219
    ## 698  3.7755784394  -7.498783455  0.2313069221  -8.658268432  1.3433076567
    ## 699  3.7043121519  -7.267549534  0.3292699381  -8.458974956  1.3544872009
    ## 700  3.2112638784  -6.861660514  0.3264618346  -7.755253664  0.5046521926
    ## 701  3.2595947541  -6.943890618  0.4359751631  -6.947010309  1.0110040653
    ## 702  1.0804386161   0.044415321  0.4411765901  -5.053824765  0.3151852567
    ## 703  0.7632048408  -0.944916479 -1.4190344714  -4.683619696  0.5192823012
    ## 704  2.6059370235  -0.309218873 -1.7675489156  -4.429194587 -0.1719397946
    ## 705  0.7069995742  -0.835695915 -1.0221173166  -4.993991789 -0.4557068432
    ## 706  1.5006291808  -0.417898027 -1.5902951515  -1.074999041  0.2882342977
    ## 707  1.3527525896   0.092848590  0.2731526946  -5.012202359  0.3035524153
    ## 708  3.3576397365  -2.998814904 -1.9567863455  -6.766633371 -0.8698995924
    ## 709  0.2739835702  -0.535839387  0.5956772217  -5.141823493  0.7323667643
    ## 710  2.3941676655  -6.163537380  0.6028505209  -5.606346429  0.2066217337
    ## 711  2.5097814039  -3.611231365 -1.0242995299  -5.291924955  1.6498073489
    ## 712  1.2934182966   0.933215784 -0.1353259606   0.521483690  0.3868841881
    ## 713  1.9042835931  -4.002414031 -0.9369555884  -2.801915423  0.7983280528
    ## 714  0.3013688300  -3.246141437 -0.4656401476  -5.429361879 -1.3049829164
    ## 715  4.3437434980  -6.655734470 -0.2931086628  -9.505140826 -1.3961490593
    ## 716  7.1829670088  -9.445943382 -0.3146199675 -12.991465582 -0.1363589320
    ## 717  7.1903060894  -9.424844460 -0.2232926464 -12.875494223 -0.0719181607
    ## 718  5.4160419091  -8.164125077 -0.1650106072 -10.193530355 -1.8952103103
    ## 719  1.2746290775   0.023682135 -1.5216961082   0.727242086  0.1665046624
    ## 720  1.8806015529  -2.325179128  0.6364586630  -5.417499345 -0.2629094648
    ## 721  6.3090440060  -8.576761433  0.2467467169 -11.534046018 -0.3642651388
    ## 722  6.3489793034  -8.681608932  0.2511793848 -11.608002257 -0.3515687820
    ## 723  4.5229919754  -6.652844053 -0.1246852339  -8.604486202  0.1103593148
    ## 724  4.3514814971  -6.453247330 -0.0614319368  -8.442872857 -0.3288206398
    ## 725  4.2213041078  -6.119667353  0.0230135706  -8.149321784 -0.3013519313
    ## 726  3.3691860515  -5.691925786  0.2873713908  -5.413786691 -0.5555499249
    ## 727 -0.5298057283   0.999716276 -0.0861120244  -1.508557140 -0.4506694923
    ## 728  1.0330320122  -0.284367998 -1.2082798097  -0.429211904  0.2489453605
    ## 729  0.5041161412  -0.863430853 -0.1844500765  -1.016915535 -1.5594101983
    ## 730  4.2565644873  -6.600654481  0.0056263765  -9.958530919  2.4713579038
    ## 731  0.3019772297   0.257697091  0.7091476520  -2.420715975 -1.9977401808
    ## 732  0.3821721768  -2.262932535  1.2106045832  -1.880618705 -0.9341604163
    ## 733  2.2094407519  -5.122314109  0.6612971062  -3.103937720 -2.4108697319
    ## 734  4.9216574470  -8.752770087  1.5201977420  -8.552022732  0.3519759577
    ## 735  2.0568120599  -3.984256730  1.0219679134  -5.967904969 -1.1516083149
    ## 736  3.5045677925  -3.918199880 -1.0157919297  -4.704508705 -0.9685530755
    ## 737  3.7192115392  -5.034029747  0.9189991582  -4.220365918 -1.0504999574
    ## 738  2.4259507160  -6.067261945  1.8324327202  -3.501702516 -1.0040082046
    ## 739  2.7127394038  -5.948403127  2.1447260549  -4.211755924 -0.1371955388
    ## 740  9.4130395398 -18.047596571 -0.7281865540 -15.393044831  0.3494850490
    ## 741  8.6212550814 -18.683714633 -0.9620734284 -15.297656186 -2.4141321234
    ## 742  8.7887836671 -18.553697010 -0.3395334077 -15.623187330 -0.1889785742
    ## 743  8.8794756677 -18.431131028 -0.2328220277 -15.021657300  0.1411862529
    ## 744  8.6883079924 -17.182918430  0.0695769150 -14.116156007  0.9590317174
    ## 745  8.4602444490 -17.003289446  0.1015566149 -14.094451660  0.7470307664
    ## 746  7.1540828422 -17.150405251 -0.0953992278 -11.030110364 -1.9590545548
    ## 747  8.0307084076 -16.060305763  0.2705297423 -14.952981039 -0.2410948837
    ## 748  8.3891423345 -16.465503942  0.3385169598 -14.224403603  0.5565844726
    ## 749  6.3572273634 -15.531611180  0.6596948754 -11.412329595 -2.4475760743
    ## 750  7.6108197884 -15.592323223  0.5047890552 -13.247888596 -0.7965260301
    ## 751  6.9898856133 -16.218610393  0.2533033658  -8.728645065 -0.1475290911
    ## 752  7.5887408802 -15.835718808  0.1350353834 -11.567005917 -0.0770269124
    ## 753  6.0573193048 -15.717606647  0.1178174096  -8.653743357 -2.5025537763
    ## 754  7.4258009631 -15.564837601 -0.4263384204 -14.029537674 -1.6818892301
    ## 755  7.0931820726 -14.979754757 -0.1482880452  -9.935679582 -1.1619993902
    ## 756  5.9346573434 -14.175030163 -0.1690444917  -6.552553684 -1.5700072332
    ## 757  4.4199966015 -10.592305005 -0.7037958037  -3.926207316 -2.4002464921
    ## 758  5.9023997550 -13.580147257 -0.4514068994  -8.334762587 -2.0251446865
    ## 759  0.7653541420  -2.043868387  1.0015473387   1.322887254 -2.7200769618
    ## 760  0.6377666854  -2.818882270  2.3605954142  -1.108698940 -1.4499510947
    ## 761  5.5736252730 -13.635216125 -0.4832362439  -7.352792243 -2.3926841890
    ## 762  5.7021114798 -13.812404090 -0.3068153225  -7.013607372 -1.1822903538
    ## 763  2.0246740805  -6.487746413  2.3764625998  -2.241516346 -0.5668697349
    ## 764  2.5408355321  -6.147053583  1.7968906416  -4.328988995 -0.1500373748
    ## 765  2.1752303056  -4.381077558  2.0632987095  -0.673818453  1.4007632166
    ## 766  5.5692581750 -13.932248753 -0.2048548274  -7.581023247 -1.0136123900
    ## 767  4.4170546101 -12.893520209 -0.1096284564  -3.547229894 -2.1778239976
    ## 768  4.9712487070 -12.686307522 -1.0714431134  -7.383370040 -2.9924302993
    ## 769  4.7395824711 -11.924954690 -1.5014110258  -3.836781466 -2.7203288382
    ## 770  5.2008151688  -7.802227790  0.0339124180  -7.560131972 -1.0146519347
    ## 771  3.2232327029 -10.895133506 -1.5234524033   0.116303256 -3.0988053270
    ## 772  1.3685845342  -1.471696588 -0.7247588462   3.442421996 -0.9574032249
    ## 773  3.7859770496 -10.522494013 -2.6574963639  -3.792795436 -4.4989446768
    ## 774  4.8719798598  -5.395221177  1.3325021562  -4.603796692 -0.7773135481
    ## 775  3.4093226199  -9.608267356 -3.0763179682   0.240060358 -2.8366769320
    ## 776  4.2870208796  -5.403768357  1.2602623090  -3.087742907 -1.3174880630
    ## 777  7.0212777482 -11.102493382  1.7117550736 -10.447632938 -2.7990761592
    ## 778  7.8644674003 -10.649839651  1.8265910165 -12.913631737 -0.7663303147
    ## 779  6.8257931621 -10.399749118  1.8364488398 -11.872845098 -3.6423925983
    ## 780  3.4819518034  -9.128341286 -3.0082554783   0.796579962 -2.1022985205
    ## 781  7.3436521086 -10.178995535  2.1421506307  -8.983052168 -0.4737460334
    ## 782  7.0021357740  -9.959193994  2.1220720607  -8.884871772 -1.6448574782
    ## 783  5.3508898842  -9.299807227  2.7931403573  -6.106551925 -2.1069470258
    ## 784  7.1887239652 -10.655180912  2.5946798608 -10.242859083 -0.1911581192
    ## 785  7.1506251695 -10.262984045  2.7330849358 -10.127524781 -0.2627836210
    ## 786  7.0510650896 -10.137529194  2.8154398146  -9.909168054 -0.2622295515
    ## 787  3.6947112399  -6.110443233 -0.1105216361  -4.184555281 -0.9310346342
    ## 788  4.6987954959  -8.507785707  0.7182493093  -7.628710147 -2.2495784830
    ## 789  4.6965333919  -8.762113340  0.9858266011  -7.320272022 -1.0638404577
    ## 790 -0.4275673915  -2.777649224  1.6371401488   1.571080121 -1.4453668182
    ## 791  4.5463010689  -7.761194032  1.1595403217  -5.231611457 -0.1716421195
    ## 792  3.5196423449  -7.221590004  1.2017280233  -3.811428143 -1.7014280221
    ## 793  0.6069108220  -2.854275428  2.3794730455   1.268146617 -0.2836190894
    ## 794  3.5860424237  -4.053357848  0.8624260250  -5.765823022 -0.5075110295
    ## 795  2.4537103782  -4.925198695 -1.2004119709  -6.424372113  1.0148974913
    ## 796  4.4317363366  -5.142736603 -1.1816920655  -8.755449133  1.0043556881
    ## 797  4.3150759159  -4.938283600 -1.1057097373  -8.490812622  1.0036257460
    ## 798  4.1760976235  -4.619009880  0.7426895415  -7.636962888 -0.1978876052
    ## 799 -1.3943282617  -0.220718798 -1.5309904315   1.075247654  0.3883832093
    ## 800  1.4284125455  -4.482678636 -0.7095193850  -4.610138440  0.6213381687
    ## 801  4.2204185935  -6.408300677  1.3280032674  -5.853545071 -0.9283370669
    ## 802  1.3415717222  -3.299472370  1.2476473736  -6.393372843 -0.0532048016
    ## 803 -0.4084027811  -0.929370066  0.0586863167  -3.512845461 -0.1646129741
    ## 804  1.8460785172  -1.961194600 -0.8511663872  -7.122316384  1.0196049004
    ## 805  1.2867492568  -3.272704585  1.0422156228  -5.748943252 -0.6681365266
    ## 806  3.6126453464  -3.171936612 -0.4000458186  -8.569454155 -1.6148312071
    ## 807  2.1439306333  -5.839736057  0.1530116309  -6.177364566 -0.7854156537
    ## 808  1.9614809315  -1.944441128 -0.6925589650  -4.346347631 -1.7269292329
    ## 809  2.9692400737  -3.055405131 -0.3165436822  -7.862808526 -2.2028004047
    ## 810 -1.5423048682  -0.540534807 -0.2720790391   0.712541143  0.2349798449
    ## 811 -1.1275734918  -0.708657346  0.2721860774   0.274710292  0.2351919518
    ## 812  3.6195974073  -4.478044065 -2.0341844861  -7.772197355 -1.2014927495
    ## 813  2.9934888336  -3.613125030 -1.7272971295  -7.364638820 -0.3432959350
    ## 814  4.5133552601  -4.858311644 -1.7657341931 -10.006243430 -0.4393944169
    ## 815  3.3695582496  -3.468545258 -1.3835710679  -6.020578971  0.3096018553
    ## 816  3.2775462853  -5.896936563 -1.2771017597 -11.221804771  0.3260267860
    ## 817  3.2494311465  -5.961487660 -1.1551757286 -11.002289062  0.5882863840
    ## 818 -0.3754661887   0.179094864 -0.1484485514  -2.135154864 -0.0449156938
    ## 819 -0.0001211803  -0.054380843  0.7361916033  -2.306644788 -0.4640033644
    ## 820 -0.4824093840  -0.690550364  0.1812749052  -2.372551798 -0.0068680461
    ## 821  2.2229600578  -0.407550443 -1.6523115385  -5.871438078 -0.4276469571
    ## 822  1.7381240138  -2.844449336  0.7658639612  -4.799737135 -0.0113354116
    ## 823  4.7767203301  -4.152498850 -0.4729908030 -10.258851478 -0.5334555916
    ## 824  3.2116336530  -2.587499314  0.2570709511  -4.683806432  0.9126365451
    ## 825  1.5138981794  -3.682943464  0.1858304261  -4.692787638  0.2474956749
    ## 826  1.0514863026  -3.474862967  0.5733713003  -5.254253463 -0.3263883267
    ## 827  3.3941520860  -4.547742185 -0.0636341961 -10.516465211  0.0532673835
    ## 828  2.0741164546  -3.010771738  0.6949776896  -5.387831227  1.7786697978
    ## 829 -0.4911123183   0.903819350  1.3665243152  -1.886162307 -0.1948613021
    ## 830  3.2828246099  -3.906876946  0.9513677040  -4.898183035 -0.5519996089
    ## 831  2.8670285001  -5.259996133 -1.2771514223  -6.989272314  0.6900096133
    ## 832  2.0930750099  -5.418888944 -1.2470137120  -3.828268179  0.3990503366
    ## 833  2.4507521593  -5.694074283 -1.1554690852  -7.132150639 -0.0596278100
    ## 834  0.9836467789  -0.578913743 -0.1998142144  -0.729706928  1.2667134558
    ## 835  4.1821623680  -4.563674738  1.1825027514  -6.964972476  1.1153395141
    ## 836  4.4979289729  -5.019609888 -1.0196912013  -7.914989007  0.6696480234
    ## 837  0.2539313374  -0.075707112 -2.2155253588   1.065262428 -0.7755030076
    ## 838  4.2291542640  -5.292314146 -0.8880867588  -7.672250312  0.5475710320
    ## 839  2.6805781799  -1.462945176 -0.7053970681  -7.445295834 -0.5613315220
    ## 840  2.9456208658  -4.102127411 -0.4772029734  -4.350434468  0.0007163435
    ## 841  2.9429391649  -4.125722242 -0.4330943953  -3.518131170  0.4207773871
    ## 842  1.2003036071   0.281744447 -0.6238440305  -0.658246292 -0.1558883566
    ## 843  0.1958391039  -0.629085831  0.6812223384  -4.715521240 -0.2878762910
    ## 844  2.3758756169  -3.244827299 -0.5566190047  -5.152474834  0.0509063245
    ## 845  3.7239325344  -4.603176149  1.0191795522  -6.619090200 -0.5902239558
    ## 846  0.9974487632  -3.191914768 -1.5148567758  -4.267296067 -0.1595539752
    ## 847  2.7315757823  -3.489374647  1.5473800891  -4.571267731 -0.5179585687
    ## 848 -1.7022284014  -0.240056328  0.4564750526   0.139567417 -1.8909744925
    ## 849  3.2740753099  -3.541687479 -0.9468201799  -7.555728941 -1.7670051458
    ## 850  3.9423158259  -5.883724361 -1.3478137371  -9.266120185 -2.4595563247
    ## 851  2.7106966787  -3.158155698  1.0836091479  -3.415151948 -1.0538700318
    ## 852  4.2132128939  -5.737815168 -0.8756933667  -8.893726336 -0.5624337774
    ## 853  4.4045784816  -4.938159290 -0.7409846401  -7.462960627  0.5349676597
    ## 854  3.9168025570  -4.481279599 -1.0274793275  -7.113872782 -0.8527752299
    ## 855  2.9411900927  -6.151362191 -1.9895285353  -9.150951006 -0.6042899988
    ## 856  2.2207949138  -0.741087888 -0.7637193779  -4.942612416 -1.3782906202
    ## 857  3.8711653039  -6.887637357 -1.7905001484 -11.272315870  1.3013631488
    ## 858  3.0536544182  -5.297811402 -1.4660297285  -7.035880051  1.9971323723
    ## 859 -1.5942577920  -0.338775118 -0.9780645134  -3.688825995 -1.4870834227
    ## 860  2.1139000141  -3.259702465 -0.3153473394  -1.808102973 -0.7418772952
    ## 861  3.5361454821  -7.959628432 -1.6734285234 -12.457998931 -0.2138847902
    ## 862  1.9913606347  -3.986416216  0.5772068418  -8.485794506 -0.7947822747
    ## 863  3.6931739422  -3.978439755 -1.7185908746  -8.636297394 -0.2429648215
    ## 864  1.2038344845  -6.172629918 -0.9213642967  -3.941965105 -0.0622213829
    ## 865  2.1557960855  -6.285124952 -1.1256253878  -6.800098088  0.9249337786
    ## 866  1.9271857492  -6.011154692 -1.1956014421  -6.745561316 -0.0580910806
    ## 867  4.7365938427  -4.162115192 -0.4566967795 -10.266758108 -0.5830182630
    ## 868  2.8587765016  -0.553179939 -0.7584919980  -6.455029103 -0.1540208823
    ## 869  3.3418019275  -7.562860429 -1.8474434272 -11.571423430 -1.4815558418
    ## 870  3.5927968321  -2.772348806 -0.0745336542  -6.281093649  0.1659780091
    ## 871  0.6138677391  -0.022561483  0.4520180652  -2.969200956 -0.9649670343
    ## 872  1.9039992616  -2.644219451 -0.9822728774  -4.691150855 -0.6930797525
    ## 873 -0.4487937256  -2.562602408  0.2434825268  -1.181669391  0.0043728717
    ## 874 -0.6884539069  -1.463206924 -1.6241550001  -4.252465599 -1.3401758839
    ## 875  1.4393220657  -1.959927496  0.7674845525  -4.978118014  1.0585659804
    ## 876  3.6422569394  -4.165202323  0.7030698751  -7.624316314 -1.4980123996
    ## 877  3.7370376813  -6.150186670  0.2887380710  -8.760694519  2.3446511947
    ## 878  3.3597219395  -5.969161634  0.0511899909  -8.724495602  0.7763906030
    ## 879  4.5691937521  -9.321152631 -1.5925176512 -14.266836280  0.4677766025
    ## 880  4.6041704779  -9.001914856 -1.2763240055 -13.969470625  1.2569453532
    ## 881  4.4800352700  -8.832509725 -1.2151641403 -13.725684613  1.2327444151
    ## 882  4.3557700197  -8.663030576 -1.1539925280 -13.481704675  1.2084538573
    ## 883  4.2313933045  -8.493487981 -1.0928108467 -13.237558514  1.1840864825
    ## 884  0.4708647196   0.435541959  0.5949741161  -0.142098601 -2.0247346028
    ## 885  0.0110374511   0.483019028  0.6652728201  -2.332793382  0.8205784338
    ## 886  1.8635957367  -3.620251763 -1.4807144218  -1.583343405 -1.2304692221
    ## 887  3.6464778375  -3.002684156 -0.6475005803  -5.945003313  0.1746500374
    ## 888  4.4363190742  -4.503801216 -0.9543613459  -9.861372108 -0.5053290374
    ## 889  2.2140002868  -1.689835510  0.4984376781  -4.393633602 -1.2708413953
    ## 890  2.9443748424  -3.805468984 -2.1022268349  -6.106183273 -0.6417363997
    ## 891  0.7138776464   0.979674835 -1.3390312662   0.984993219 -0.3824709683
    ## 892  2.5289430206  -3.308721722 -1.9920096890  -5.757216082 -0.8773234743
    ## 893  4.9058198650  -5.208373972  0.5324983805 -10.544011400 -1.3074187011
    ## 894  5.7836544389  -6.117703911  0.6141648410 -12.451498657  0.4271524778
    ## 895  5.7163186950  -5.810407126  0.7232926463 -12.289133385  0.3787731963
    ## 896  5.5911261935  -5.640417458  0.7845100386 -12.043805616  0.3538148749
    ## 897  2.7578374585  -5.437353689  0.5468789074  -8.396572967 -0.8107157389
    ## 898  1.7018947081   0.144622386  0.1040876838  -2.954166723 -1.3741219259
    ## 899  1.4512375481  -0.280523277 -0.7567940724  -2.757556767 -1.7062850176
    ## 900  3.5863945093  -5.517147627  0.6836524750  -8.560422876  1.8560949159
    ## 901  3.8278680907  -6.518648822  0.2511365485 -12.456705953 -0.6491664821
    ## 902  4.1064045586  -6.331824796  0.6717848149 -12.156586584  1.0202517848
    ## 903 -0.8001504164   0.501435242  1.1180143263  -1.816384147 -0.6812825508
    ## 904 -1.5142047363  -0.973764043 -0.7099284788   0.141331684 -1.4703141523
    ## 905  3.4806017033  -3.735152954 -0.5947776017  -8.229951992 -1.4760940361
    ## 906  4.4694673117  -4.655071325 -0.4417755889 -10.149812923  0.6127977141
    ## 907  1.1682159435  -2.134731971  1.1283132199  -4.566009860 -0.1269504068
    ## 908  0.7570632144  -3.501803779  0.2467422241  -6.065621880  0.3395831342
    ## 909  2.1729761006  -3.103476671  0.2175605446  -6.034402576  0.1644007059
    ## 910  1.9661234570  -3.127455945  0.5065741010  -5.926131041  0.9310907956
    ## 911  3.6187372126  -7.101488298 -0.4421252919 -11.081616843 -0.8137216385
    ## 912  4.5484945837  -7.836252798 -0.2426747707 -13.202505344  1.0924415992
    ## 913 -0.1911905370  -0.323794190 -0.5724559509  -1.422063853  0.3146841990
    ## 914  4.4759046727  -7.607261464 -0.1607231199 -13.010749370  1.0943818192
    ## 915  2.2880217465  -5.267204670  0.3947999504  -4.287995777  1.3152794667
    ## 916  2.9896264066  -5.993632424 -0.1647400777  -8.388442969  0.0421289980
    ## 917  1.7795045031  -4.836159014 -1.2089918751  -2.901189811  0.2818783056
    ## 918  1.0541072121   0.974058516 -1.4536680187  -1.425619905  0.5776615753
    ## 919 -1.4751453156  -0.050467984  0.1135004754   0.984343616  0.3639687392
    ## 920  0.8740516254  -2.513104329  0.0215750823  -3.565118845  0.4611534283
    ## 921  1.0320155822  -0.722023082 -1.5332396524   0.334119448  0.2974788603
    ## 922  1.3383175460  -0.329759340 -0.3635911889  -1.824839105 -0.2296785137
    ## 923  1.2419581755  -3.910521500 -1.2256362992  -4.697996871  0.6761083041
    ## 924  2.1157951772  -5.417424082 -1.2351226314  -6.665176895  0.4017006867
    ## 925  2.8584658157  -3.096914898 -0.7925324362  -5.210140846 -0.6138032639
    ## 926  1.7949689686  -2.775021540 -0.4189501437  -4.057162377 -0.7126159686
    ## 927  1.9335195368  -5.030464797 -1.1274545750  -6.416627976  0.1412372343
    ## 928  0.4911402417   0.728903320  0.3804280455  -1.948883349 -0.8324981363
    ##               V16           V17           V18           V19           V20
    ## 1     0.144921070 -2.509464e-01  0.2706930434  0.6496699582 -0.1739942172
    ## 2     0.677959921 -5.776092e-01  0.5334042962  0.3644258773  1.8888009502
    ## 3    -0.839054103  6.728221e-01 -0.4990738734  0.7102261044 -0.1654261951
    ## 4     1.032942273  5.559320e-01 -1.7497992603 -0.0539761299  0.1097616630
    ## 5    -0.546080290 -3.466241e-01 -0.4269907211  0.1628155640 -0.0120176877
    ## 6     0.563676855 -1.341802e+00  1.4018369425  0.2700499049  0.0848564299
    ## 7     1.534299414  4.462223e-01 -0.8647547622  0.7649041207  0.2327631248
    ## 8    -0.540240611 -8.462378e-01  2.4622194597 -0.8365792055 -0.1868656384
    ## 9     0.132089532 -3.186971e-02 -0.7136845288 -0.1882474436  0.0845721969
    ## 10   -0.659775268  9.560802e-01 -1.0421580482 -0.9969245813 -0.2870779689
    ## 11   -0.378765724 -5.671321e-01 -0.1011014509  0.1439350167 -0.1201020098
    ## 12    1.891262564 -2.844300e-01  1.9876356472 -0.4612935147 -0.1262465345
    ## 13   -0.584101991  2.669586e-01  0.2347630464  1.1912376226  0.1463361740
    ## 14    0.497850490 -6.558509e-01 -0.3168062997 -0.2309137844 -0.1318300967
    ## 15   -0.164107843 -9.543291e-01  0.2025494645  0.7697774278 -0.0072204277
    ## 16   -1.230061260  7.896703e-01 -0.9228828920  0.3517268172 -0.0349860415
    ## 17   -1.457490161  1.002651e+00 -1.7111039207 -1.2570747927 -0.1927625804
    ## 18    0.009324847 -4.781673e-01 -0.2597171847 -0.3884232409 -0.0132416730
    ## 19   -0.862926072  3.950375e-01 -0.7723415656  0.5894955387 -0.1947627053
    ## 20    0.136804104 -4.970611e-01  0.6519868705  1.0265704156  0.0184352206
    ## 21   -0.205786336 -1.181277e-01 -0.3834483243 -0.9164654655 -0.1642803133
    ## 22    0.889605937 -2.201521e-01  0.2837100970 -0.4793256211 -0.1675330513
    ## 23    0.689028754  9.609869e-01 -0.8685209841  1.0626202789 -0.1622736664
    ## 24    0.544679816 -6.751344e-01  0.5512404259 -0.8314492845 -0.0630620829
    ## 25    0.395558537  1.108382e+00  0.9292012677 -0.0318754425  0.1977966036
    ## 26    0.702812292 -6.157270e-02  0.4998539039  0.2669876963  0.0280037726
    ## 27    0.010668695 -3.930810e-01 -0.2364859554  0.4610275484  0.0052933134
    ## 28    0.083372064 -5.647024e-01 -0.1768565255 -0.8637492816 -0.2864170035
    ## 29    0.381142647  1.089541e-01 -0.4131082021 -0.3249987446 -0.0654634632
    ## 30   -0.370246884 -4.534861e-01  0.8460841852  0.5443655084  1.6389076436
    ## 31    0.091300914  9.277919e-01  0.3657105452 -0.0358763733 -0.1995529451
    ## 32   -1.253637700 -1.861684e-01  1.9852127595 -0.9438992264 -0.2855472173
    ## 33    0.268706580  1.955884e+00  0.5101434342 -0.4038520745 -0.0327123145
    ## 34    0.662224860  1.007560e+00  0.7153967870 -0.6515295519 -0.2592428687
    ## 35    0.531857122  1.123969e-01  1.1560789685 -0.0504644733 -0.1033013538
    ## 36    0.177423439 -2.933458e-01 -0.1687327949 -1.0019330942 -0.0522721000
    ## 37   -0.794778785  1.405723e+00  0.3294328394  1.3490530943 -0.7703210464
    ## 38   -1.904148767  8.857650e-02  1.0203400605 -0.3755947958 -0.6761200009
    ## 39   -0.958842655  5.097620e-01 -0.8845786633  0.4548886298 -0.1321817194
    ## 40   -0.509481345 -1.646935e-02 -0.5695188738 -0.2426601788 -0.7528410690
    ## 41   -1.643847002  3.404982e-01  2.0587869311 -0.2999982338  0.0799552499
    ## 42   -0.144496038  2.510991e-01 -0.4133783604  0.3906344786 -0.0811289116
    ## 43    0.071779642 -9.526000e-01  0.4900382554  1.4577773953 -0.1080654082
    ## 44   -0.610679242 -9.126072e-01  2.5700629401 -1.2086256165 -0.6336952962
    ## 45    0.307663640  3.505726e-01 -0.3987022342  0.1042317908 -0.0783904887
    ## 46   -0.344871410 -6.622919e-01  2.3460935100 -1.3064859613 -0.4875427144
    ## 47   -0.612427842  3.457737e-02  0.2685690103  0.1702550692 -0.0496453120
    ## 48    0.230162342  1.077823e-01  0.8124617492  0.6489632891 -0.3619610167
    ## 49   -0.426528450 -3.960959e-01  0.2903617685  0.4474850638 -0.2203571542
    ## 50   -0.022196727  1.679664e-01  0.7517590164 -0.3553527614  1.9498248260
    ## 51    1.436292102 -3.143144e-01 -1.0951126452 -0.6443269556 -0.0873261275
    ## 52    0.930232312  5.335608e-01 -1.8541973321 -0.1945616705 -1.0780696597
    ## 53    1.438153287  6.772962e-01 -0.4622507843  0.0584227343  0.1481000103
    ## 54    0.360822921 -1.100099e-01 -0.0413595543 -0.0399372186  0.1384734658
    ## 55    1.633750672  1.604262e+00  1.2388840394 -1.3670482323 -0.2115269305
    ## 56   -0.780524643 -1.820462e-01 -0.7738145879 -0.2434403394 -0.1548806266
    ## 57   -1.631757947  2.525472e-01  1.2161413520 -0.4800420794 -0.5099224898
    ## 58   -2.574398412 -1.770324e-01  1.3531915542  0.1888354563 -0.4575233468
    ## 59   -0.603163648 -5.749030e-01  0.5838192915  1.5196615035  0.2345629387
    ## 60   -1.330835803  2.567290e-01  1.4419201084 -0.7202072286  0.2240954988
    ## 61   -0.146006177  5.365030e-02 -0.9958603830 -0.0478754510 -0.1059513385
    ## 62   -0.586532427  5.939273e-01  0.0242037533  1.0897328253 -0.2027741974
    ## 63   -0.806220997  1.433628e-01 -0.2595115556  1.0771734421  0.1156353365
    ## 64   -0.787347233  3.546024e-01 -1.4989311999 -0.4869577118 -0.2871029328
    ## 65    2.142898064  1.142046e+00 -0.5712117961 -0.2818943230 -0.8993129823
    ## 66   -0.778454729  1.316321e-01  0.2398252265  1.3842947312 -0.0484715950
    ## 67    0.197834353  3.425934e-01  0.2150747395  0.3759757280  0.2320532115
    ## 68    0.207664374 -6.784137e-01 -0.2411148330  0.5802715648 -0.1861981996
    ## 69   -2.026897052 -4.512394e-01  0.8859695316 -1.3639997767  0.0580886973
    ## 70    0.155231653 -7.548112e-01 -0.2270735457  0.2835277981 -0.0173601305
    ## 71    0.253612644 -7.375195e-01 -1.0241514230 -1.0129109907 -0.4178143239
    ## 72    0.326612468 -5.963449e-01 -0.2333797485  0.1974777416  0.1600265324
    ## 73    0.412374956 -6.949136e-01  0.5471826955  0.2712698224  0.2193452505
    ## 74    1.163160999 -9.600325e-01 -2.9205489382  0.6941406612 -6.4851343469
    ## 75   -0.442994561 -5.205985e-01 -0.3445269713  0.1661871729 -0.2293395888
    ## 76    0.059954651 -4.369878e-01 -0.7841770258 -0.6164557831  0.0963574331
    ## 77    0.462797319 -3.911131e-01  1.2367807359  0.7184494776  0.2219913291
    ## 78    0.219460400  8.738829e-01  1.0565276116  0.0505452405  0.1474470784
    ## 79   -0.464205466 -2.559106e-01  0.5876174759  0.8410051260 -0.1106877546
    ## 80    0.236661985 -6.463682e-01 -0.2843551462 -0.2631149497 -0.1888489863
    ## 81   -0.727450578  5.617349e-01 -0.5155937481  0.8978283856  0.1109761349
    ## 82    1.527759209 -4.703831e-01  0.4426914078 -2.1902054866  2.1180548845
    ## 83   -0.734321138 -4.623107e-01  1.7593636862 -1.2256418785  0.8026130175
    ## 84    0.867507655 -8.098465e-01  0.5798170820 -0.0650817187 -0.1351831384
    ## 85    0.173183129  3.306655e-01  1.3710253804  0.7633770591  0.3247222788
    ## 86    0.300068194 -6.457360e-01  0.3146804601 -0.2937502839 -0.0973791070
    ## 87    0.324850784 -1.060446e+00  0.8579711155  0.3577060690 -0.1235639919
    ## 88   -0.120254344 -3.705658e-02 -0.8118997630 -0.8707991705  0.0766443689
    ## 89    0.676050919 -3.211315e-01 -0.4919611174 -0.5638636524 -0.6924292599
    ## 90    0.462225989  2.196869e-01  1.3362052327  0.2003012660 -0.0431625041
    ## 91    0.390981135 -3.592135e-01 -0.4270069335 -1.2650908513 -0.3559237209
    ## 92    0.091007283  2.105344e-01  0.6472441798 -0.1575421428 -0.2960451944
    ## 93   -0.171171426 -9.964623e-02  0.7035086461  0.4088858022 -0.2419014507
    ## 94    0.182418820  1.967317e-01  0.7415543954 -0.5870007523 -0.4218349194
    ## 95    0.743330657 -4.086543e-01  0.2317907189  0.2110875491 -0.0227524951
    ## 96    0.558005308 -6.305989e-01  0.7214444631  0.0306929250 -0.2645235370
    ## 97   -0.477302201 -5.567326e-01 -1.2868845206 -2.0821253695 -0.1712142098
    ## 98   -0.620053804  4.791623e-02 -0.6361366832 -0.1576315556 -0.1572592261
    ## 99    0.848618196 -3.993499e-01  0.3030676698 -0.5978785516 -0.1339673685
    ## 100   2.062846433 -1.774472e+00  0.3435160042 -2.2872025733 -0.0939524859
    ## 101   1.035973891  4.280880e-01  0.3934151448 -0.5714636094 -0.1418935781
    ## 102  -0.365462003 -1.908463e-01 -0.4750095492  0.1557926875 -0.1226790673
    ## 103  -0.094416981 -7.593895e-01 -0.0999689852  0.5134578324 -0.3645072033
    ## 104  -0.164885148  1.865120e-02  0.0200392480  0.6514884691 -0.1499850775
    ## 105  -0.940545648  7.638966e-01 -0.2060832973  1.1663348107  0.4129527654
    ## 106  -2.147816111  6.060181e-01  0.2455020637 -1.0262673397  0.2200746232
    ## 107   0.046783641 -6.223889e-01  0.0702990076 -0.2142398806 -0.0902589768
    ## 108  -1.040401456  1.103735e+00 -0.3349362079 -0.3538787408  0.9317861473
    ## 109   0.431795032  7.074998e-01 -0.2965147715 -0.4335396808 -0.2636248181
    ## 110  -0.907791718  5.475223e-01 -1.8309564708 -0.8988459301 -0.0469556935
    ## 111  -0.540837571  3.190528e-01 -0.0152041799  0.5668734969  0.3548236234
    ## 112   0.829078397 -1.091855e+00  0.6852714933  0.8214446809  0.0974218251
    ## 113  -0.099927882 -4.095334e-01 -0.7616419957 -0.0872240011 -0.0390213778
    ## 114  -0.097479114 -2.772557e-01  1.0477934909 -0.3452427887 -0.0476699375
    ## 115  -0.699461615 -3.221605e-01 -0.7780839227 -0.1260172262 -0.1904365683
    ## 116   0.687450769 -5.112020e-01  0.4077279764 -0.4229852692 -0.1722382139
    ## 117   0.073429912  5.908717e-03 -0.5572890839 -0.2786315021 -0.0228803761
    ## 118   0.402133659  3.439397e-01  0.0403965311  0.2034121513 -0.1578628340
    ## 119  -1.372130770 -9.048272e-02  1.0953426565 -1.7848345213 -0.4008731043
    ## 120   0.659158493 -1.068195e+00  0.5283043009 -0.4955681802 -0.1504693860
    ## 121  -0.276388483 -3.635241e-01 -0.0765306697  0.1541140729 -0.3060626369
    ## 122   0.185492100  2.491556e-02  0.7308699836 -0.1133445703 -0.1781553164
    ## 123   0.234394389 -7.066077e-01 -0.2076139625  0.6156272085 -0.2034260042
    ## 124   0.585758915 -3.003458e-01  0.2441516928 -0.3195783921 -1.4842247462
    ## 125  -0.712264653  1.119677e+00 -0.9949581771 -0.6394114502  0.0820956965
    ## 126   0.486213882 -5.427132e-01  0.0082207768  0.1002197466 -0.1520445500
    ## 127   0.041296726  4.516857e-01  0.4207275363 -0.4840193972 -0.2889760221
    ## 128   0.799591482 -2.155881e-01 -0.4852347374 -0.8595786421 -0.6438177542
    ## 129  -0.639070063  7.619161e-01  0.3237985161 -0.5860094335 -0.3117519362
    ## 130   0.494759250 -8.237184e-01  0.7094563658 -0.0224804922 -0.0658555726
    ## 131   0.816318098 -2.914661e-01  0.7194133169  0.6678358239 -0.7352190452
    ## 132  -0.579539047  1.908260e-01 -0.1108046773  1.6811427597  0.0350760541
    ## 133   0.855722654 -1.410725e+00  0.4143389134 -0.1538106136  0.4872621107
    ## 134  -0.921370406  1.243330e+00  0.7015210019  3.0421198519  0.0975965945
    ## 135  -0.155786772  1.076549e-01  0.2680226050 -0.8092534869  0.8727537715
    ## 136   0.235227467  3.328647e-01 -0.6196380915 -0.5497343575 -0.1791663659
    ## 137   0.532749920  1.958526e-01  0.6785494387  0.8205517746 -0.0985663773
    ## 138   0.050419322 -2.532537e-01 -0.0147046237  0.7219062203 -0.1501479915
    ## 139   0.195146823  4.979282e-01  0.3428289155  0.4074573750 -0.4125544994
    ## 140  -1.105055130  1.572154e+00 -0.9175375070 -0.0725511627  0.0046531587
    ## 141   0.448727865  1.005206e-01 -0.0811988120 -0.2576376754 -0.1846084122
    ## 142  -1.364792596 -2.392254e-01  1.2370635884 -1.2660959991 -0.3472151967
    ## 143  -0.098596241  6.587436e-01 -0.5840426847  0.2906500041  0.0053282164
    ## 144  -0.051352596 -4.634402e-01 -0.6997421493 -0.0297168973 -0.0048857659
    ## 145   0.551646894  2.659416e-01  0.2648154499 -0.4797261589  0.0928788049
    ## 146   1.073883608 -7.584696e-01  0.4984648248 -0.3902715028  0.0947219783
    ## 147  -0.395958533  4.578143e-01 -1.6679568173 -1.1806584786 -0.1555568198
    ## 148  -0.465828333 -5.759371e-01  0.1650084849  0.1237557937 -0.1218352461
    ## 149  -0.099837455 -4.835133e-01 -0.0265773667  0.5306574952 -0.2160016001
    ## 150  -0.503053882  5.090459e-01 -1.2230176427 -0.1133914962  0.5192892915
    ## 151   0.447649272 -1.195438e+00  0.5885670350  0.0145139247 -0.0651942230
    ## 152   0.080056532 -4.329377e-01 -0.2034174893 -0.5963595061 -0.0887347772
    ## 153  -1.144134040 -3.875406e-01  0.9436647083 -2.4120901764 -1.2481111622
    ## 154  -0.335793907 -4.042989e-03 -0.0289782531  0.8225270268 -0.5589490908
    ## 155   0.363811708  7.544724e-02 -1.2117338247  1.1489731474  0.4117533241
    ## 156   1.367339548 -2.934379e-02 -1.2831390916  1.1985222448  0.1732876318
    ## 157   0.476812453 -5.642798e-01  0.2429944291 -0.1343749922 -0.2076506658
    ## 158   0.547428491  1.140502e+00 -1.7229781137  0.6775516819 -0.0331530586
    ## 159  -1.385801107  2.512120e-01 -0.6061301722  0.3083973271 -0.1651000082
    ## 160   1.005648708 -7.727112e-01  0.2630724885 -0.4740092637  0.1008142517
    ## 161   0.420177932 -9.639886e-01  0.8296819193  0.2572453641 -0.1731512569
    ## 162  -1.684603640  8.201700e-01  1.4127364564 -0.7476262856 -0.2992844934
    ## 163   0.634478388  2.432421e+00  1.4908459749  0.2752846248 -0.2357392266
    ## 164  -0.227461609  2.170802e-02 -0.2321034277  0.1116991004 -0.1532596453
    ## 165  -1.134611410 -3.097933e-01  0.8755347370 -0.5826603186 -0.3686029589
    ## 166  -0.162076670 -2.296315e-01 -0.9541223724  0.0744164570 -0.2326032581
    ## 167  -0.965716476  6.088632e-01 -0.5818980974  1.2084791313  0.2147501933
    ## 168   1.154951702 -1.523935e+00  0.9112226668 -0.3740476592  0.0634220054
    ## 169   1.032443755 -2.638140e-01  0.9346906503 -1.0877324059 -0.5329036942
    ## 170  -0.077929757 -1.584871e-01  0.8862510157  0.9368758965  0.0693634869
    ## 171  -1.145362399  4.616487e-01 -0.4656554356  0.9392398625  0.0112011825
    ## 172   0.524277721 -1.665087e-01 -0.2809006387 -0.7961490334 -0.6551449646
    ## 173  -0.188324916  1.232013e+00 -0.2422883242  1.6243333921 -0.0047046387
    ## 174   0.382742659 -1.157849e-01  0.0003717796  0.0115110405  0.0236687221
    ## 175   1.050638489  5.539685e-01 -1.5702281463  0.1239215410 -0.0377815726
    ## 176   0.651708784 -4.845930e-01 -1.2563565253 -2.3247067480 -0.2524056718
    ## 177  -1.159822955 -2.722107e-01  1.2016460994 -1.5398236101 -0.4509665521
    ## 178   0.110506544 -4.037517e-01 -0.3055201813  0.3581638680 -0.1447271211
    ## 179   1.194678647  2.527572e+00 -2.5289065258 -2.4874045454  4.7078525197
    ## 180   0.193896923 -3.995970e-01 -0.3645927895  0.0113490709  0.1151377839
    ## 181  -2.002044078  3.381653e-01  0.6578885091 -0.6172413899 -0.5176078565
    ## 182   0.140591396 -3.092273e-01 -0.1782218537  0.3099396237 -0.1303989558
    ## 183   1.411770272 -8.107451e-01  0.8161495784 -0.5201245664 -0.1181595554
    ## 184   1.174552384  3.707889e-04 -1.1752325781  1.0792712364  0.0133642742
    ## 185   0.260664757 -1.792067e-01  1.0625388078  1.1190074357  0.1479767881
    ## 186   0.026104320 -3.109835e-01 -0.1979938323 -0.2192303320 -0.0620093784
    ## 187   1.057503860  3.856782e-01 -1.4869583229  0.4604954805 -0.0005277595
    ## 188   0.211988358 -4.413639e-01 -0.2087358897  0.1657026216 -0.1366622921
    ## 189  -0.281514486 -7.767678e-01  0.5182606125  0.1706395202  0.6976042237
    ## 190  -0.422449034  6.229726e-01  0.2027882154 -0.3584013381 -0.0947183829
    ## 191   0.084613637 -4.797408e-01 -0.3507219811  0.4037810056 -0.2464644747
    ## 192  -0.691673753  1.124856e+00 -0.5153913493  0.6817432732 -0.3201105872
    ## 193   1.002252320  7.393448e-02  0.4810232522 -0.8202524397  0.0438642371
    ## 194  -0.570945599 -2.833419e-01  1.7619776712 -1.2114327482 -0.1551168864
    ## 195   0.200726580  8.777985e-01  0.4889010464  0.3608015070  0.0667774678
    ## 196   0.900794540 -8.938470e-01  0.2547408613 -0.0996023200  0.2242837915
    ## 197  -1.752636189 -2.810001e-01  1.5446587067 -1.3544522356  0.3765079996
    ## 198   0.939548230  3.131228e-01 -0.5807119609  1.3257064007 -0.0344534685
    ## 199  -1.173787838 -1.485829e-01  0.6558179819  1.5700112936  1.2740481224
    ## 200   1.277093186  1.843029e-01 -0.8844005419  0.4980837179 -0.0633867857
    ## 201  -0.360765490  9.165182e-02 -0.0087986787 -0.1823649222  0.0055896022
    ## 202  -0.419486742 -5.745487e-02 -1.0294677805 -0.7541075206 -0.1112953878
    ## 203  -0.506458270  2.902555e-01  1.2339634214  1.0635236420  1.6531241223
    ## 204   0.544529019 -4.298833e-01 -0.9138532891 -0.3291270554 -0.0976682487
    ## 205   0.733715692 -8.507756e-01  0.1571547649 -0.8662481773  0.2770081614
    ## 206  -0.849085896  2.881138e-01 -1.0852403962  0.8815134051  0.2775810099
    ## 207   0.374759342 -9.269118e-01  0.2912516096 -0.0008713943 -0.0039444382
    ## 208  -0.140417269 -3.962781e-01  0.1038277672  0.6035358235  0.5871355155
    ## 209  -0.928074780  2.266936e-01 -0.1009103274  1.1447707258  0.1588397394
    ## 210  -1.012825493  8.244680e-01  1.6914229795  2.9892586665  0.2145678702
    ## 211   0.423666792 -4.985703e-01  1.1259114772  0.5841217828  0.1975664427
    ## 212  -0.187835389 -6.550587e-01 -0.5533446228  0.1316827052 -0.1309254488
    ## 213   0.661765330 -7.243720e-01 -0.0108962956 -0.8949796208 -0.4364008959
    ## 214  -0.075000307 -4.157790e-01  0.6269619140  0.4924406891 -0.4156018074
    ## 215   0.085936941  3.313237e-02  0.1904035878  0.1242740881 -0.1849518691
    ## 216  -0.608182392  2.227519e-01 -0.2199026531  0.1561113765  0.1827427288
    ## 217  -0.285702326  4.133630e-01  0.5584301560  0.4766196088 -0.0126146614
    ## 218  -0.585426475 -5.326575e-01  0.1970756838  0.7355590448 -0.1707119635
    ## 219   0.374153382 -1.236489e+00  0.0372121883 -1.1210258978 -0.3643399683
    ## 220   0.138039412 -5.256125e-01  0.0533345511 -0.4350179423 -0.3024145413
    ## 221   0.075740105  6.865644e-01 -0.0495173673 -0.0047563347 -0.2425479815
    ## 222  -0.962083230  9.949970e-01 -0.8823777830  1.3943692824 -0.5935979490
    ## 223   0.547919257  7.447786e-03  0.2378655368  0.1214611940  0.0713037924
    ## 224   0.791527512  2.125548e-01 -1.8065440025  0.0104948384 -0.2278510291
    ## 225  -0.871443604  6.221637e-01 -0.0658519171  0.7700715277 -0.3121786258
    ## 226   0.485570600 -9.024729e-01  0.3610409227  0.2399084629  0.6437351483
    ## 227   0.072488712  1.189096e-01  0.4471665366  0.1604072705  0.3290475339
    ## 228  -2.780125615  1.378866e+00  1.0637411659  1.0338531842  1.0907298219
    ## 229  -1.093318641  6.355938e-01  0.0952426589  0.3053122862 -0.4627744796
    ## 230   0.041716671 -4.216543e-01  0.1402045122  0.9109197884 -0.0060827201
    ## 231  -1.436429550 -4.227672e-01  1.6346543617 -0.5584546168 -0.3450018635
    ## 232   1.346866461 -1.072030e-01  1.0351772892 -0.8048449597  0.4207101305
    ## 233   0.528459486 -3.608451e-02 -0.0144507810 -0.1362530293 -0.1588139714
    ## 234   0.118158322  1.087315e+00 -2.2550229445  1.0501030984  0.3072946635
    ## 235   0.448807064 -4.561961e-02 -0.3795743714 -0.2016354246 -0.0329664986
    ## 236  -0.191458987 -1.982701e-01 -0.5488092370  0.4279307250 -0.3384669364
    ## 237  -1.198577815  7.018977e-01  0.1626520475  0.7746736690 -0.3611254431
    ## 238   0.035547146 -5.264420e-01  0.1863345518 -0.2797858982 -0.1245025275
    ## 239   0.535979570 -2.797291e-02  0.0040959619 -0.1378810206 -0.1513304196
    ## 240   0.087069704  2.123232e-03 -0.4572873806 -1.3002126545 -0.2382663186
    ## 241  -1.660007890  5.950552e-01  0.6306184856 -0.8913652591 -0.4505143113
    ## 242   0.344104357  4.677448e-01 -0.2597389221  0.0719714760 -0.1525336295
    ## 243   0.156154981 -7.527035e-01 -0.5019398290 -0.1284928765 -0.0270056761
    ## 244  -0.347269980  1.016913e+00 -1.1687725162 -0.7096093532 -0.2212569638
    ## 245   1.034220782 -1.423709e-01  0.6330215269 -0.2720839540  2.5996676515
    ## 246   0.182078137  1.807394e-01  0.0637443878  0.3310542031  0.5727198137
    ## 247  -2.243551641  2.805482e-01  0.7022555873 -0.8335549844  0.1043974495
    ## 248  -0.511122713  6.089653e-01 -0.4920695625 -0.0438097189 -0.3653061171
    ## 249   0.415878758  1.733340e-01  0.1005316032 -0.0234509947 -0.0225552655
    ## 250  -0.178875476  3.029553e-01  0.5512822634 -0.4716292931  0.0667534953
    ## 251   0.472941942 -5.985918e-01  0.4935917131 -1.0012902321 -0.0895067771
    ## 252   0.813161847 -9.492549e-01 -0.8026013848 -1.6463844765 -0.0715329332
    ## 253   0.984061711 -1.291951e-01 -1.3233721482  0.4004777741  0.1829910240
    ## 254  -0.013591823 -7.627104e-01  0.4549727139  0.0940000716 -0.2593578591
    ## 255   0.282844845 -2.784905e-01 -0.4340125026  0.0672397149 -0.0415289011
    ## 256  -0.501929657 -2.687128e-01 -0.0230998878  0.0383895599 -0.3494541403
    ## 257   0.865207629  1.975564e-01  0.3559295347  0.1953113950 -0.0428766483
    ## 258   1.212796347 -2.732192e-01 -0.6371636271  1.7506214391  0.1498142203
    ## 259   1.574866714  5.217169e-01 -1.2308318687  0.3584306105 -1.4788589261
    ## 260   0.328569375 -6.447560e-01 -0.5352279670  0.6883276057  0.2480816730
    ## 261   0.704456878  1.816205e+00  1.1601071267 -0.7714726311 -0.1489736594
    ## 262   0.660455145 -1.049904e+00 -0.0643784709 -0.3995099084 -0.0187435967
    ## 263   0.282787230 -4.263720e-01  0.2452864837 -0.2965165113 -0.0192678551
    ## 264  -0.386663521 -7.926234e-01  2.2366490782 -0.5096934361 -0.5389628940
    ## 265   0.946394519  1.453496e-01  0.9941577029  0.2668901308 -0.2348562111
    ## 266   0.637491078  6.336137e-01 -0.1035577233 -0.2643156345 -0.1302462553
    ## 267   0.131756487  1.825199e-01 -0.5128032967 -0.3028976617  1.2359963079
    ## 268   0.493475658 -2.164586e-01  1.6520721665  0.4845316387  0.1069995094
    ## 269   0.399963587  1.458190e+00  0.2459653251 -1.2509201252 -0.1331313096
    ## 270  -1.628415425 -2.254510e-01 -1.5397351646 -0.7236364894  3.3806470709
    ## 271  -0.766107075  4.514075e-01  0.7722837150  0.5899454544  0.0413456823
    ## 272  -1.190606294  1.840386e+00 -1.7662517756 -1.0582884163  0.1314312054
    ## 273   0.143924027  1.163693e+00 -2.3591432243  0.3670121161  0.0786797910
    ## 274   0.775946125  2.223599e+00  1.3967705871 -0.1790800675 -0.3538201537
    ## 275   0.905948203 -4.138060e-01 -0.8612254370 -2.7142548720 -0.8467142033
    ## 276  -0.708034201  8.696432e-01 -0.0956595595  0.9443954091  0.0287401249
    ## 277   0.354133458 -1.413363e-02 -0.1798257012 -0.1193276837  0.2476732588
    ## 278  -0.372730717  1.894434e-01 -0.8149556488 -0.2470675133 -0.1925565220
    ## 279   0.703724205 -9.623150e-01  0.4590490303 -0.1203945634  0.0716354691
    ## 280   0.072651604  5.225270e-01  0.5984144366  0.4504487752  0.4772362513
    ## 281   0.170864926 -3.968170e-01  0.0191991309 -1.2120624901 -0.0044161855
    ## 282  -1.106494933  5.348126e-01 -1.1232418547 -0.3189565519  0.0496221585
    ## 283   0.414548788 -8.301269e-02 -0.4663773692 -0.2075105639  0.0054518326
    ## 284  -0.485049452  6.250764e-02 -0.3749039768 -0.0740241309 -0.2215722203
    ## 285   1.132088837  1.889919e-01 -1.8818386158  0.6006653964  0.1054084243
    ## 286   0.212025297 -5.719701e-01  0.3442964780 -0.1573467438 -0.1548739033
    ## 287   1.387100300 -1.075673e+00 -0.2888581730 -1.4976671773 -0.2260061211
    ## 288   0.185173290 -4.063198e-01 -0.3834144344  0.0124333121  0.1167503857
    ## 289   0.530205143 -1.251467e+00  0.9546695848  0.0980024993 -0.2399883209
    ## 290   0.337693337 -1.296439e-01  1.1251393604  0.1499994540 -0.3811795000
    ## 291   0.800468072 -3.378104e-01  1.4150555508 -0.5758474387 -0.1750287247
    ## 292  -1.474845581  1.366822e+00 -0.9043580178 -0.0807904538 -0.3180556889
    ## 293  -0.024251422  4.755985e-01 -0.0142200191  0.3336045041 -0.0913261160
    ## 294   0.205369910 -2.608617e-01  0.3451115096 -0.7088912266  0.3178854437
    ## 295  -0.346358175 -5.056009e-01  0.3587944572  0.8048436076 -0.0359033445
    ## 296  -0.804587352  1.237584e+00  0.2860740959  1.4961219206  0.3740627585
    ## 297  -0.185188721 -1.968606e-01 -0.5390688963  0.4127469131 -0.2416536150
    ## 298   0.318688063  5.493651e-01 -0.2577858579  0.0162561280 -0.1874207884
    ## 299  -0.009704146 -4.248507e-01 -0.5139704426  0.4560677829  0.3145212303
    ## 300   0.155724290 -3.501866e-01 -0.3617921859  0.0323532073 -0.2284591577
    ## 301   0.949104959  5.051107e-01 -0.6059474180  0.7353820866  0.2118693161
    ## 302  -0.198659625 -2.896293e-01 -0.3219160216 -0.2503409539 -0.1821306145
    ## 303   0.245866751 -7.157836e-01 -0.0193233560  0.3326192445 -0.0784857506
    ## 304  -0.314847775  1.190854e+00 -0.3089389651  1.0841043010  0.1194014259
    ## 305   0.833922195  4.442572e-01 -2.0506737415  0.0890462618  0.0880972292
    ## 306   2.193405499  4.684800e-01  0.4606514015  0.0460829435  3.2134678018
    ## 307  -0.309139172  3.768711e-01  0.1278745653  0.3995079920 -0.3436295349
    ## 308   0.271571120 -4.224613e-01 -0.2559728330  0.1055223332  0.0694154200
    ## 309   0.820572831 -1.096441e+00 -0.5880344016 -1.6058151721 -0.8225740166
    ## 310  -1.084678420 -2.489727e-01  1.3201789247 -0.6330413199 -0.4572049792
    ## 311  -0.295501540 -6.886285e-01  0.5927446714  0.2275539924  0.1339884851
    ## 312  -1.028466080  2.429094e-01 -2.0945159998 -1.3260275544 -0.4823233934
    ## 313   1.037758359  6.891938e-01 -1.3234802156  0.3928029017  0.0148372397
    ## 314   0.712748003 -3.120427e-01  0.5684556804  0.3285151136  0.0912810552
    ## 315   0.595671704  1.606806e+00  1.1114755149  0.1220190631 -0.0397791724
    ## 316   0.212455720 -2.458073e-01  0.4577741447  1.0869124899  0.4809275327
    ## 317   0.193282022 -6.445686e-01 -0.2472057338  0.5564649896 -0.2090939502
    ## 318  -0.525903151  5.745219e-01 -0.1301145381  0.7249389510  0.0929178970
    ## 319   0.528277800  5.480584e-01  0.1451766098 -1.3039965771 -0.3434729023
    ## 320   1.434977256  9.431055e-01  1.5736851629 -1.2170142864 -0.3898418741
    ## 321   1.058722177 -1.184995e+00  0.8791420482  0.7688552041  0.7572756439
    ## 322   1.716815579  9.495617e-01 -1.0440301859  0.8832970168  0.3445883327
    ## 323  -0.715941285  9.193327e-01  0.1345455818 -0.0005178069 -0.2403324582
    ## 324   0.053781077 -7.149087e-01 -0.5243319387 -0.1891523464 -0.5811945992
    ## 325  -1.560751409  1.210408e+00 -2.5024995965 -1.4534587112  0.0780325512
    ## 326   0.045849215  1.622935e+00  1.0058103085  1.1202164798 -0.2162256481
    ## 327  -0.888034275  4.055905e-01 -0.0193263958  1.5441331914 -0.1355454856
    ## 328   1.183456201 -9.181305e-02 -1.3755134848 -0.0491114499  0.2539694272
    ## 329   0.385859965 -7.324691e-01  0.3299919175  0.1710155463 -0.1814077822
    ## 330  -0.157784095 -2.753765e-01 -0.9740024784  0.0977825223 -0.2025537814
    ## 331   1.278579036  9.472934e-02 -1.2216985878  0.6956054674  0.0026752094
    ## 332  -1.160533970  4.366120e-01 -0.6554741242  0.4225414714  0.0749624158
    ## 333  -0.686623634 -3.248154e-01 -0.3993283271 -0.2322934133  0.0399292660
    ## 334   0.967941559 -1.134777e+00  0.9763865194  0.0616978890 -0.1044614956
    ## 335   1.883568190  1.205788e+00 -0.1585608277  0.4065010580  0.0593014311
    ## 336  -0.561528090 -2.811044e-01 -0.5104851302 -0.0695173018 -0.2926151136
    ## 337   0.273916428 -5.066884e-01  0.4953585424 -0.1745610742 -0.0682871298
    ## 338  -0.729906219  5.740327e-01 -1.8951679270 -1.3744501812 -0.2687237232
    ## 339  -2.505190220  9.434683e-01  1.4124231979  0.0118193877 -0.6633894728
    ## 340  -0.065589737 -2.526134e-01  0.3653304115  1.0393173971  0.8065874250
    ## 341   0.534690949 -4.836573e-01  0.1350127461  0.1071971803 -0.2182582621
    ## 342   0.536885244  2.823538e-01 -1.0548281934 -1.0150693497 -0.1425126218
    ## 343  -0.194396565  3.393658e-01  0.2193327446  0.1451284247 -0.1031061834
    ## 344   1.328238075  5.416437e-01 -0.3867451903 -0.4430435395  0.0309258566
    ## 345  -0.188178725  5.176634e-01 -0.2121034589  0.5629065888 -0.3882620671
    ## 346  -0.264892763 -1.148776e-01  0.7208563274  0.0104283021 -0.0899440855
    ## 347   0.616253689  2.205968e-01  0.3466580030 -0.0275734516 -0.1248149295
    ## 348   0.707017693 -7.786670e-01  0.5678324026 -0.6635699466 -0.1091267587
    ## 349   1.972097989 -3.204896e-01 -0.2523017450  0.8812969850  0.5606721698
    ## 350  -0.576526962  3.828263e-01 -1.2973662478 -0.2855832953  0.0025881677
    ## 351  -0.207427738  8.634927e-01 -0.2709240951 -0.5815736260  0.1478339049
    ## 352  -0.507363848 -8.744965e-01  0.1843574960  0.1906730078  0.3721550092
    ## 353  -2.118058751  6.520684e-02  0.4694122350 -1.3593189846 -0.5930507888
    ## 354  -0.739879604  2.428025e-01 -1.0224152253 -0.3711554510 -0.1605796854
    ## 355  -2.041701940  3.141264e-01  1.0725199456 -0.9789677071  1.4501842324
    ## 356  -0.025094960  7.596239e-01  0.2926736069 -0.7671681919 -0.2603927478
    ## 357   0.092271361 -4.942095e-01  2.6392134516 -1.8638789814 -0.4414059373
    ## 358   0.724190735 -3.914561e-01 -0.1785967043 -0.5015128959 -0.1796882690
    ## 359   0.342777724 -8.210008e-01 -0.2445904762 -1.3483862727 -0.3377660176
    ## 360   0.619766381  2.928090e-01  0.3504421956 -0.7695915799 -0.2004681622
    ## 361   2.749203393  5.529122e+00  1.7495727077 -1.1911909033 -0.0186999639
    ## 362  -0.169378027 -6.933015e-01  0.3779926485  0.3668821172 -0.1633836009
    ## 363  -0.268497323 -2.704758e-01  0.3996565561  0.4220580343  1.4099354047
    ## 364  -0.184231613  2.184815e-01  0.5995744869  0.2660807021 -0.2371411103
    ## 365  -0.251629129  2.611837e-01  0.5129446918 -0.0337965342 -0.2030238588
    ## 366   0.066317474  1.361500e+00 -0.1878018806 -0.6758985885 -0.1890829956
    ## 367  -1.539886332 -1.175101e-01  1.4777062248  0.4280940387 -0.3848135754
    ## 368  -1.580505789  1.374841e-01  1.2528358105 -0.2079172902 -0.7704408141
    ## 369   0.271258620 -1.070735e+00  0.2091860401 -0.7030001086 -0.3007134718
    ## 370  -0.051513432 -2.297888e-01 -0.5752674603 -0.1410472185 -0.1866872743
    ## 371  -0.577746911  8.839718e-02  0.6153599262  2.0757702449  0.3217494943
    ## 372  -0.153633324 -2.213124e-01 -0.9341414186  0.0705527340 -0.2108644844
    ## 373  -0.391776655 -7.884648e-01  2.6378656654 -1.1928718341 -0.4536325641
    ## 374   0.554908563  8.449547e-01  0.9453513604  0.5974536468  0.4804006301
    ## 375  -1.648792574  3.457486e-01  1.1685570286 -0.7611659398 -0.0302192642
    ## 376  -0.793407383 -5.405554e-01  1.9288959505 -1.7406598617 -0.3534774854
    ## 377   0.343276005 -4.467613e-01 -0.1576892565  0.1408344954 -0.1019528666
    ## 378   0.609204520 -1.318275e+00  0.3345572030 -0.8374162324 -0.1910810188
    ## 379   0.522120667 -3.232996e-01  0.2844854787  0.2285430294 -0.0666760893
    ## 380   0.759707894  1.996231e-02 -1.0290109764  1.0113663524  0.2815982083
    ## 381   0.376172634  1.018238e+00  0.8256779998  0.1219777797  0.1728772646
    ## 382   0.248732020 -3.842541e-01  0.4541374494  0.0050469973 -0.1675311761
    ## 383  -1.085851544  1.385176e-01 -0.4749642115 -0.0938057588 -0.1451173788
    ## 384  -0.692484078  1.556703e-01 -0.1247175930  0.5121825142  0.1149062635
    ## 385  -0.739730772 -7.047824e-01  2.1019178157 -1.0821906196 -0.1037510377
    ## 386  -0.166226394 -6.544548e-01  0.3705170801  0.6451538505 -0.3052695544
    ## 387  -1.046070864  8.786318e-01 -0.3771547748  1.1147259383 -0.7041259189
    ## 388  -0.067081989  2.984155e-01  0.2882681976 -0.4934174717 -0.1115856138
    ## 389  -0.298150771  3.210679e-01 -0.3055059723 -0.1703005135 -0.2552367382
    ## 390  -0.981027819  2.868874e-01 -0.7066839010  1.1959529285  0.2411196281
    ## 391   0.330384315 -5.321934e-01 -0.0392240055 -0.2496252445 -0.3168621629
    ## 392  -0.368110867 -5.380035e-01 -0.2166028953  0.4290508268 -0.2429591046
    ## 393   0.343344065  3.939096e-01 -0.3055186183  0.1090323792 -0.1249311152
    ## 394   0.417271635 -9.354322e-01  0.2914224292  0.4057716644  1.2956716203
    ## 395  -1.381162476  1.096036e+00 -0.8340600129  0.6317066685 -0.4335826391
    ## 396  -0.617845694 -6.157877e-01  0.0285492572  0.2326937712  0.1002100853
    ## 397  -0.012772982 -6.505681e-01 -0.1364907253  0.2713998674  0.0302671382
    ## 398  -0.597799305  6.581809e-01 -0.3760464931 -0.4805309551 -0.2695015116
    ## 399   0.064080009 -9.874360e-01 -0.1449090755  0.3955645783  0.0010015131
    ## 400   0.366110571  2.111289e+00  1.8678668057  0.7640942499  0.5734583113
    ## 401  -0.124395710  1.924640e-01 -0.0475350984 -0.1611233637  3.4980012588
    ## 402  -1.131953031  3.123567e-01 -1.0549562025 -0.2782767305 -0.2417571282
    ## 403   1.848971788  4.720769e-01 -0.4590311277  1.5067334175 -0.2979177311
    ## 404  -1.080403064  4.396732e-01  0.5117492597  1.4205249885  0.2193819790
    ## 405   0.052048115  1.408997e-01  0.2484644733 -0.2131508303 -0.0672897693
    ## 406   0.052086006 -8.176244e-01  0.4659107124  0.2669663499  0.1010154542
    ## 407  -0.311467204  1.154358e-01  0.2554657678  0.1970437787 -0.0777023756
    ## 408   0.266364849  1.746880e+00  0.9197576929  0.3041280871 -0.0574095197
    ## 409  -1.055900035 -2.201043e-01  2.0677493194  0.6113837500  0.0242320728
    ## 410   0.444708094 -3.660265e-01 -0.1522218463 -0.8836094453 -0.0337815146
    ## 411   1.693266419 -4.275946e-01 -1.0418262653  0.0171817031  0.2616368960
    ## 412  -0.998421709  1.254383e+00  0.8706024356  2.1585372605  0.3376717020
    ## 413   0.443567290  2.091676e-01 -0.9705058166  1.1979059975 -0.0790631046
    ## 414   0.131560951 -5.090987e-01 -0.1143996381 -0.6444461705  0.0559365882
    ## 415   1.334395346  1.384514e-02 -1.2326953179  0.8079638685  0.2262706702
    ## 416   0.308962957  6.229332e-02 -0.0342123875 -0.1552941606  0.0586442500
    ## 417  -1.921797730 -2.998954e-02  0.8437056509 -1.2011634931 -0.2840780977
    ## 418  -0.508878481  3.551823e-01  0.3156325532  1.3323206018 -0.0125526846
    ## 419  -0.081809380 -8.011455e-01  0.2352738915 -0.0246262469  0.0273866942
    ## 420  -0.167386782 -5.295632e-01  0.3181386684  0.1742447083  0.3922942703
    ## 421   1.133241655 -5.425904e-01  0.2093855228 -0.6639620813  0.9434764769
    ## 422   0.857850307  4.412843e-01 -1.6486114249  0.6178769323  1.8361256212
    ## 423  -0.414464736 -5.131493e-01  0.1470965703  0.0312123143  0.3326323550
    ## 424   0.454555145  2.351189e-01  0.4572697566 -0.2525287466 -0.0587734777
    ## 425  -0.222424633 -5.321361e-01  0.2032858367 -0.1620272559 -0.2647382984
    ## 426   0.271912180 -2.704939e-01 -0.0595998151 -0.0115817584 -0.1219844830
    ## 427  -0.910967189 -1.365543e-01 -0.4000192798  0.1760411033 -0.1103668900
    ## 428   0.321169634 -3.986373e-01  0.7008978535  0.6556415671 -0.3230632975
    ## 429  -0.772662585  2.819021e-02  0.3522084140  1.1591846765  0.0381904838
    ## 430   0.390629617 -7.452141e-02  0.0388049165 -0.0373492558  0.1252334521
    ## 431  -0.286891403  3.370364e-01 -0.4189002886  1.1315332754 -0.1596991163
    ## 432  -0.165445741  1.369843e-01  0.6000356544 -0.2744490584 -0.3249678196
    ## 433  -0.387494322  4.821972e-01 -0.8826073519 -0.1883006083  0.8072549949
    ## 434   1.293150482  1.804421e+00  1.6886909498 -0.0372764559  0.4494700613
    ## 435  -0.305660789  2.343567e-01 -0.6694346158  2.0090507651 -0.6674450478
    ## 436   0.123316913 -2.735404e-01  0.7465906942 -0.0772420748 -0.1340308395
    ## 437   0.953266691  1.394805e+00  1.3834572719 -0.4131972425 -0.0112746814
    ## 438  -1.372248278  2.776783e-01  1.0827504244 -0.3769826754  0.0024166869
    ## 439   0.202219899  1.430631e+00  0.7411675158  0.9833462163  0.9226586726
    ## 440  -0.138580687 -1.755743e-01 -0.1193311507  0.2183902328  0.6653313788
    ## 441  -0.947717185 -4.740842e-01 -0.6778855452 -0.4694681950  0.3457994201
    ## 442  -0.236744806  2.649540e+00 -3.2648930919 -1.0096917166 -0.0506992821
    ## 443  -0.366106595  5.457955e-01  0.5803928634  0.3583846466  0.1844459413
    ## 444  -0.484613870 -5.236955e-01 -0.5422201421 -0.5077977594 -0.2131813712
    ## 445  -0.328104518 -6.518280e-01  0.6795454839  0.3594648228 -0.1310949932
    ## 446   0.740389659  6.295720e-01 -1.0211685061  0.5275257790  0.1065653916
    ## 447   0.265648648  5.934921e-01 -1.8283423108  0.8642004756  0.0288716019
    ## 448  -0.035253644  1.240186e+00 -1.2999243701  2.7068706464  1.6364755954
    ## 449  -1.600204370  1.752837e+00 -0.5025928755  0.9291934281  1.6582250735
    ## 450   0.503907263 -8.292244e-01 -0.2079354812 -1.2252456700 -0.0834797271
    ## 451  -0.198121474  1.535937e+00  1.3209989023  0.2682963931  0.1241453754
    ## 452  -1.506094335  3.166704e-01  0.2467587712  2.3302527519  0.2613265962
    ## 453  -0.763831163 -1.640067e-01 -0.2634683364 -0.3824809896 -0.2795986055
    ## 454   0.609936730 -5.495214e-01  0.4742832962 -0.5291016386 -0.1320275003
    ## 455  -1.633010882  3.552050e-01  0.4786530395  2.9416157433  0.3396779621
    ## 456  -1.140747180 -2.830056e+00 -0.0168224682  0.4169557050  0.1269105591
    ## 457   0.666779696  5.997174e-01  1.7253210075  0.2833448301  2.1023387926
    ## 458  -2.282193829 -4.781831e+00 -2.6156649448 -1.3344410667 -0.4300218672
    ## 459  -7.358083221 -1.259842e+01 -5.1315486284  0.3083339458 -0.1716078786
    ## 460   2.581850954  6.739384e+00  3.0424931783 -2.7218531222  0.0090608364
    ## 461  -2.041973791 -1.129056e+00  0.1164525212 -1.9346657389  0.4883782211
    ## 462  -1.638960115 -1.746350e+00  0.7767440979 -1.3273566355  0.5877432190
    ## 463  -0.871688490  1.313014e+00  0.7739138726 -2.3705994506  0.2697727760
    ## 464  -0.502362191  7.844266e-01  1.4943046074 -1.8080121587  0.3883074282
    ## 465  -3.599540209 -4.830324e+00 -0.6490901202  2.2501232488  0.5046462261
    ## 466  -3.811758410 -3.754128e+00 -1.0491774023  1.5541972635  0.4227431292
    ## 467  -2.641473109 -1.312059e+00 -0.3917160510  1.1182635518  0.2041376840
    ## 468  -2.602477870 -4.835112e+00 -0.5530260891  0.3519489434  0.3159572589
    ## 469  -3.028452388 -4.214486e+00 -1.2136078477 -0.2654215954  0.3640891150
    ## 470  -1.073116527  1.524501e+00  1.8313635328 -0.0897244112  0.4833026505
    ## 471  -2.442353983 -3.535524e+00  0.1303603066 -2.0714504884  0.5766560333
    ## 472  -2.689284204 -3.204383e+00  0.0865218957 -1.3144950967  0.6327099638
    ## 473  -4.478136595 -5.844266e+00 -1.1027307946  2.1773856865  0.5627056433
    ## 474  -4.649864299 -6.288358e+00 -1.3393123224  2.2629847876  0.5496129709
    ## 475  -3.694515998 -6.304753e+00 -1.2675871279  0.3579870305  0.5007790540
    ## 476  -3.652801960 -6.293145e+00 -1.2432482913  0.3648104822  0.3609240037
    ## 477  -7.552342204 -9.802562e+00 -4.1206288347  1.7405072904 -0.0390459343
    ## 478  -2.379420985 -2.775114e+00  0.2737989713 -1.3821875635  0.3990968603
    ## 479  -2.360856724 -2.760097e+00  0.2981034156 -1.3855578017  0.4087041845
    ## 480  -2.013542681 -5.136135e+00 -1.1838221165  1.6633940135 -3.0426257572
    ## 481 -10.266609184 -1.550339e+01 -5.4949276425 -0.4104809774  1.4937746455
    ## 482  -0.663203946  8.919345e-01  0.9786756391 -2.0054766027  0.4404394409
    ## 483 -12.227189278 -1.858737e+01 -6.9207618606  3.1669989069  1.4106776526
    ## 484 -10.031096657 -1.522701e+01 -5.3220089806 -0.5017506535  1.3826189997
    ## 485  -1.233958282  1.632009e+00  1.3157345645 -0.2871888553  0.5354346632
    ## 486  -6.653594347 -1.024676e+01 -4.1910662666  0.9914861223 -0.1589705487
    ## 487  -6.777069240 -9.931765e+00 -4.0930211220  1.5049248591 -0.2077594451
    ## 488  -3.075091885 -5.044736e+00 -1.7180826852 -0.6624620684 -0.5318976933
    ## 489 -10.629497390 -1.444121e+01 -5.1054864660  1.2702262705  1.4888551924
    ## 490 -11.911483017 -1.810300e+01 -6.8378345388  3.1269293521  1.1914439664
    ## 491  -9.723565309 -1.474490e+01 -5.2473011063 -0.5746751438  1.3058619148
    ## 492 -10.322016678 -1.395909e+01 -5.0307100383  1.1972663350  1.4126248802
    ## 493 -10.222203170 -1.379915e+01 -5.0085851786  1.1620261478  1.4342397794
    ## 494  -4.842383078 -5.269876e+00 -2.3446685742  1.9592240541 -0.4656792203
    ## 495   0.335642173  2.298998e+00 -0.1620962095 -1.5327072125 -0.2173581922
    ## 496   0.857322736  1.707024e+00  0.5263491965 -0.8651046098 -0.1434346450
    ## 497   0.706047713  1.151143e-01 -0.0467958789 -2.5485215920 -0.3587086656
    ## 498 -10.122391921 -1.363921e+01 -4.9864572584  1.1267843782  1.4558781953
    ## 499 -10.688241792 -1.838881e+01 -6.8988401330  2.3828079355 -0.6237372916
    ## 500 -10.098670655 -1.750661e+01 -8.0612079897  1.6068700326 -2.1471814395
    ## 501  -8.668739272 -1.280414e+01 -5.1166204953  0.5792003580  1.1012498702
    ## 502  -9.505210445 -1.754203e+01 -6.7926377872  2.0693774493 -0.0855014634
    ## 503  -8.333242496 -1.260260e+01 -4.8766834208  0.6046259107  1.1115022146
    ## 504  -8.264128844 -1.243980e+01 -4.8370790260  0.7531498479  1.0027643417
    ## 505  -8.161632445 -1.228096e+01 -4.8185863934  0.7197876821  0.9964687557
    ## 506  -8.059419407 -1.212201e+01 -4.7997101325  0.6862271636  0.9931210173
    ## 507  -7.957447226 -1.196295e+01 -4.7805077876  0.6524980453  0.9922789493
    ## 508  -7.855681494 -1.180382e+01 -4.7610259421  0.6186244128  0.9935845987
    ## 509  -7.754094059 -1.164460e+01 -4.7413027096  0.5846259727  0.9967450952
    ## 510  -7.652661665 -1.148533e+01 -4.7213695755  0.5505190041  1.0015185020
    ## 511  -7.551364931 -1.132600e+01 -4.7012527775  0.5163170733  1.0077032053
    ## 512   1.457578528  2.611450e+00  1.2919552295 -1.5638152267 -0.1854546501
    ## 513  -7.695568788 -1.368414e+01 -4.7774060394  1.2683427732  1.5015654107
    ## 514  -7.541687424 -1.425960e+01 -5.0350515238  1.4322679297  1.5349198211
    ## 515  -7.418487279 -1.410277e+01 -5.0164233352  1.3903143605  1.5449704671
    ## 516   1.477925603  1.776454e+00  1.1492821858 -1.3069946447 -0.1388136975
    ## 517  -7.186375508 -1.379747e+01 -4.9584939097  1.3211665598  1.5779244011
    ## 518  -7.070953009 -1.362972e+01 -4.9588298158  1.2720913635  1.5729496149
    ## 519  -6.947746067 -1.347290e+01 -4.9402108289  1.2301425519  1.5829295549
    ## 520  -6.824524172 -1.331608e+01 -4.9216120867  1.1882042078  1.5927539314
    ## 521  -6.701288870 -1.315927e+01 -4.9030314950  1.1462752483  1.6024388370
    ## 522  -6.578041503 -1.300246e+01 -4.8844672388  1.1043547349  1.6119982190
    ## 523  -6.454783237 -1.284566e+01 -4.8659177372  1.0624418503  1.6214442247
    ## 524  -6.331515094 -1.268886e+01 -4.8473816070  1.0205358793  1.6307874832
    ## 525  -6.208237973 -1.253206e+01 -4.8288576325  0.9786361934  1.6400373352
    ## 526  -6.084952665 -1.237527e+01 -4.8103447413  0.9367422379  1.6492020223
    ## 527  -5.961659873 -1.221848e+01 -4.7918419833  0.8948535218  1.6582888445
    ## 528  -5.838360219 -1.206170e+01 -4.7733485140  0.8529696085  1.6673042904
    ## 529  -2.617361330 -4.835558e+00 -1.9217516014 -0.3858841752  0.3309168942
    ## 530  -5.606597326 -1.175626e+01 -4.7149467613  0.7835775938  1.7038876455
    ## 531  -2.954300813 -5.841218e+00 -2.9104302979 -1.1558501727  1.6073967719
    ## 532  -5.491073262 -1.158854e+01 -4.7154201712  0.7345734930  1.6978562632
    ## 533  -5.367775276 -1.143176e+01 -4.6969244437  0.6926884120  1.7068890620
    ## 534  -5.244471520 -1.127497e+01 -4.6784365293  0.6508073707  1.7158618243
    ## 535  -5.121162405 -1.111819e+01 -4.6599558698  0.6089300805  1.7247788384
    ## 536  -4.997848304 -1.096141e+01 -4.6414819592  0.5670562798  1.7336439937
    ## 537   2.493227604  6.244987e+00  2.9717493641 -2.6799502422 -0.0308799718
    ## 538   2.290607832  6.509272e+00  2.5699714263 -3.2861500821 -0.0890615370
    ## 539  -0.701882909  4.097358e-01  0.9396036940 -1.9542999554  0.3105246334
    ## 540  -4.702011825 -4.908099e+00 -1.5088732903  3.0016850594  0.1708717656
    ## 541  -1.024502092  2.204521e-02 -0.9406846741  0.4847589402 -1.0169231435
    ## 542  -2.479513556 -4.931112e+00 -2.5474025274 -0.9026898029  0.2652502103
    ## 543  -4.046292997 -5.079479e+00 -2.5868570454  1.6692605534  1.9579602999
    ## 544  -0.775666020 -3.218618e+00 -0.3709925010 -0.6861967236  0.3027345789
    ## 545   0.085406344 -2.191432e-01 -0.6942592154 -0.6552699282 -0.1652487768
    ## 546  -1.359389477 -5.095183e+00 -0.3380126562 -0.1241379890  0.3293417141
    ## 547  -4.738662343 -9.276636e+00 -3.0819610810  0.1777456345  0.5135298646
    ## 548  -6.532982278 -1.338925e+01 -4.4804125668  0.4320542065  1.0857595201
    ## 549  -0.103527024  2.031196e+00  0.3175983568  0.3867606369 -1.0299646825
    ## 550  -4.652071195 -9.094231e+00 -3.0825499541  0.1323307764  0.5716544882
    ## 551  -7.862658826 -1.457084e+01 -5.1853855329  2.4143895686  1.1120276189
    ## 552  -7.954070045 -1.426506e+01 -5.7710644259  2.8921696850  0.9816087216
    ## 553  -7.395956925 -1.432963e+01 -4.8680102365  2.4638430391  0.8987232341
    ## 554  -7.533404329 -1.406462e+01 -4.9601086331  2.5257849242  0.7572867266
    ## 555  -8.520850166 -1.327730e+01 -5.2537046198  3.6233318140  0.5790984977
    ## 556  -6.553973257 -1.262439e+01 -4.9899097385  1.6454138933  0.2845550904
    ## 557  -6.792795060 -1.207593e+01 -4.8821796427  2.6266245248  0.2994886322
    ## 558  -2.995830378 -4.698433e+00 -1.7118712247  3.0252609919 -2.1698108922
    ## 559  -1.341269358  2.498325e+00  0.7778310889  1.4060448737  1.7844493350
    ## 560   1.001413919  3.902825e+00  1.6196092268 -0.8880874642 -0.0601585380
    ## 561   2.375296198  6.443510e+00  2.5918460630 -2.8897703583 -0.0592643858
    ## 562  -1.104217463  6.364127e-01  0.9941013687 -0.6887205551 -0.1080064227
    ## 563  -1.142467272  9.185908e-01  1.1263352233  0.0929734779 -0.6347470809
    ## 564 -13.256833091 -2.167548e+01 -7.8234688155  2.9951111673  1.7255776165
    ## 565  -6.180879910 -9.698839e+00 -4.8182333408 -0.8715816885  0.0777814455
    ## 566  -6.772867034 -8.815785e+00 -4.5688592841  1.1265986565  0.1853252698
    ## 567 -12.146540144 -2.516280e+01 -9.0428451507  0.7875788078  1.3847427621
    ## 568 -13.563272956 -2.381564e+01 -9.1705572189  3.6695234398  1.6821603064
    ## 569 -12.391346003 -2.254165e+01 -7.9867206535  2.9925544633  1.1500173187
    ## 570 -12.375397443 -2.109061e+01 -7.7245457359  3.8563747425  1.1910783046
    ## 571 -12.317579724 -2.401910e+01 -9.3351930791  1.9518904954  0.5683380107
    ## 572 -13.251541979 -2.288400e+01 -9.2878322140  4.0382305049  0.7233138222
    ## 573 -12.432279143 -2.324160e+01 -8.4952989380  3.4539425552  0.6459743219
    ## 574 -11.847887192 -2.167399e+01 -7.7840424745  3.2043092170  0.5638693528
    ## 575 -11.771496968 -2.029922e+01 -7.6091100602  3.8091364357  0.5207323302
    ## 576 -10.922655127 -2.190649e+01 -8.8298197660  1.8524669887 -0.6453938579
    ## 577 -11.290327808 -2.057802e+01 -7.5478241122  3.1184797656 -0.0461701790
    ## 578 -10.954285728 -2.058359e+01 -7.5172621317  2.8723536361 -0.2476475277
    ## 579 -11.125739059 -1.971626e+01 -7.5527890897  3.6567870707 -0.0883420819
    ## 580 -10.999792349 -2.260887e+01 -9.4987459210  2.1027354069 -1.0093199377
    ## 581 -10.973804618 -2.104763e+01 -7.9907832844  3.3310390294 -0.6172961505
    ## 582 -11.350243855 -2.171019e+01 -8.8594519250  3.6297136504 -0.8433026197
    ## 583 -11.380995603 -2.153680e+01 -8.4853558052  3.5176105096 -0.8773916099
    ## 584 -12.675268899 -2.074066e+01 -8.1536680262  5.2283417901 -1.0252279239
    ## 585  -9.484143370 -1.967188e+01 -7.6432728209  1.1690644180 -0.9957868578
    ## 586  -9.286954737 -1.989973e+01 -7.5180508492  1.2435174994 -1.6146600560
    ## 587 -10.615745297 -2.101790e+01 -8.3973930520  3.1835589497 -1.6914819878
    ## 588  -9.899246541 -1.923629e+01 -8.3985519949  3.1017353689 -1.5149234353
    ## 589  -3.996144713 -4.695563e+00 -1.8191609635  1.4610803189  0.6449933085
    ## 590 -10.328242388 -2.025456e+01 -8.5345225638  3.2565344806 -0.5263682883
    ## 591  -7.718042405 -9.855927e+00 -5.1939078928  2.0426976878 -0.2240434629
    ## 592  -2.901140776 -4.674667e+00 -2.2136232522  1.2961650498  0.6681237871
    ## 593  -4.557193454 -7.654704e+00 -2.5884186436  0.7185550093  0.8188594168
    ## 594  -9.533257050 -1.875064e+01 -8.0926487734  3.3267582750  0.4272034315
    ## 595  -8.376438821 -1.848546e+01 -7.5899739586  1.1266403653  0.3966548451
    ## 596  -6.856810039 -8.851879e+00 -4.5918832274  0.9369398589 -0.2491277267
    ## 597  -1.461029247  6.356942e-01  0.3901501491 -1.0630276160 -0.1627974823
    ## 598  -1.139754130  8.356396e-01  0.3897742312 -1.4396079761  0.0547955667
    ## 599  -8.427082796 -1.350215e+01 -4.7129675886  2.6304759946  0.0588612515
    ## 600  -7.898122052 -1.357068e+01 -4.5902353001  2.5901733841 -0.5622641036
    ## 601  -2.815887768 -6.649358e+00 -1.7461132839  1.0834067637  0.1688905725
    ## 602  -5.369890011 -9.069079e+00 -2.8198065963  0.8745429481 -0.0368373257
    ## 603  -1.815914313 -5.249111e+00 -2.1966912068  2.1090526447  0.8172031555
    ## 604  -6.773976750 -8.588070e+00 -4.5108917454  1.1478778383 -0.3464558327
    ## 605  -0.560471963 -2.077065e-01  0.3805415711  0.5743359609 -0.0034694628
    ## 606  -2.404927288 -5.671739e+00 -0.9942939512 -0.2899358914  0.2768934663
    ## 607   0.279890165  1.132160e+00  0.0929929763 -0.2989204554  0.0160036733
    ## 608  -2.907903434 -5.248646e+00 -0.9368146526  1.1601202943  0.1750185522
    ## 609  -2.058551384 -5.635494e+00 -0.7752712043 -0.2393096541  0.2534641291
    ## 610  -2.603702650 -5.157596e+00 -0.6960100046  1.2859613947  0.2219186692
    ## 611   0.201150539  8.748906e-01  0.9414766785  0.2726984762 -0.0425151329
    ## 612   0.399177755  1.143130e+00  0.2560138865 -0.5112100881 -0.1027718048
    ## 613   0.882231824  6.018049e-01 -0.3040923915 -2.1917642780 -0.2683473711
    ## 614  -3.157032672 -4.405850e+00 -2.5588278512 -0.2088095367  0.3109796028
    ## 615  -3.199046476 -4.480184e+00 -2.4856687835 -0.0958074337  0.2078893017
    ## 616  -0.635321655  3.270064e-01  0.0438160872  1.4035737623 -1.3767207795
    ## 617   0.624003315  7.057943e-01  0.5241817743 -0.7317498460 -0.1715406547
    ## 618  -2.467737030 -7.140326e+00 -1.2712800216 -0.0017219235  0.0254265126
    ## 619  -2.013923708 -2.815600e+00 -1.0922902723  0.1396872304 -0.0323343863
    ## 620  -4.330294806 -7.339698e+00 -2.5509008539  0.6459048955  0.8147101559
    ## 621  -4.215537135 -7.171672e+00 -2.5503366846  0.5963643205  0.8166516618
    ## 622  -4.092884494 -7.014622e+00 -2.5309672589  0.5540274991  0.8323980660
    ## 623  -3.970130032 -6.857613e+00 -2.5117356820  0.5117619515  0.8470852231
    ## 624  -3.847292842 -6.700637e+00 -2.4926161073  0.4695543140  0.8609117418
    ## 625   0.013466807 -1.192044e-01  0.8268143738  0.4768599025  1.2302778405
    ## 626   0.935791272  1.591506e+00  1.2594707138 -0.6668063943 -0.2194610742
    ## 627  -2.427373075 -4.448472e+00 -1.2122204499  0.0346554556  0.3514843996
    ## 628   0.377153837 -8.681827e-01 -0.0671351121 -0.0644929661 -0.0295593501
    ## 629  -2.123462785 -6.053319e+00 -0.7063761527  0.5727910282  0.3805452123
    ## 630  -2.535494251 -5.356465e+00 -1.9747485246 -0.6037262692  0.0027494785
    ## 631  -3.304253910 -5.585794e+00 -1.6437040317  2.1186333253  0.0874063699
    ## 632  -2.522660811 -5.603400e+00 -0.7784399638  0.8602117936  0.8255659853
    ## 633   0.483063094 -1.524708e-01 -0.2856662588  0.0725498490 -0.7642742583
    ## 634  -7.565042492 -1.369147e+01 -4.3668672814  2.1814227356  0.0175000205
    ## 635  -7.787326221 -1.282218e+01 -4.3676771334  2.6439842213 -0.2898295860
    ## 636  -7.586490786 -1.250393e+01 -4.3756310124  2.4651948704  0.0731637914
    ## 637  -7.303242729 -1.244804e+01 -4.3328342413  2.3520296403 -0.1230850159
    ## 638   1.188906759  5.368173e-01 -0.0514032140 -1.2313855017 -4.1281858287
    ## 639  -4.180757788 -4.285071e+00 -2.5364515728  0.9302391858  0.2704592552
    ## 640  -6.288305758 -1.375313e+01 -4.3292392742  1.5042502109 -0.6147193445
    ## 641  -7.244340474 -1.276009e+01 -4.8247402230  2.8369929950 -0.4603895386
    ## 642  -3.562533517 -4.377106e+00 -1.7926349435  0.0802807730  0.5904183826
    ## 643  -3.507230624 -7.457095e+00 -2.3254916819 -0.4404345529  0.3385981405
    ## 644  -4.849692387 -6.536521e+00 -3.1190938816  1.7154944198  0.5604780757
    ## 645  -3.563239406 -7.368321e+00 -2.6929531579 -0.4505501135  0.2740271227
    ## 646  -6.598704066 -1.212054e+01 -3.9040915335  2.5134304006  0.9243961797
    ## 647  -1.707500854 -1.685473e+00  0.2748908387 -2.0288853524  0.9488637701
    ## 648   2.572371808  6.609366e+00  2.5306697996 -3.6026572509  0.4825130221
    ## 649  -1.495280775 -4.424757e+00 -0.6193232374 -0.2311983863  0.3267728031
    ## 650   0.068532887  1.325794e+00  0.3763834808 -1.0961964541 -0.0672050359
    ## 651  -2.843784559 -4.826246e+00 -0.7038834649  2.1522145577  0.9884934869
    ## 652  -4.739083784 -6.131887e+00 -2.4258706386  2.3141648695  0.7243812677
    ## 653  -4.334783427 -9.874560e+00 -3.5419033293 -0.5315587605 -0.1415332688
    ## 654  -2.502049417 -4.460495e+00 -0.8705257450  0.5956291040  3.2091709439
    ## 655  -2.342178262 -4.442082e+00 -0.8122021970  0.2547718332  2.8752603037
    ## 656  -2.343304119 -5.071450e+00 -0.7491723208  0.2284227484  0.2226666832
    ## 657   0.844221069 -3.948312e+00 -1.8075159916  0.1058788272 -1.2339872572
    ## 658  -6.890317745 -9.516411e+00 -4.3641277149  2.5985775564  0.0196261689
    ## 659  -5.073642635 -1.044101e+01 -3.7555246009  0.6064529650 -0.6565114343
    ## 660  -3.297239075 -4.862971e+00 -2.0023539992  1.5452332696 -0.1069446396
    ## 661  -3.293557690 -3.901499e+00 -2.5254404818  0.7951442202  0.4128006207
    ## 662  -1.983609608 -4.929162e+00 -1.0384820853 -0.1338191999  0.2901868294
    ## 663  -3.177460496 -5.369349e+00 -1.1214541570  2.0305919379  0.2253329304
    ## 664   0.866440625  4.082485e+00  1.1963181517 -1.1007635835 -0.0677120412
    ## 665  -7.477161727 -9.785800e+00 -4.2439771480  2.6811267817 -0.2860432697
    ## 666  -4.745462468 -1.007565e+01 -3.6045960155  1.4352938102 -1.5621623578
    ## 667  -0.527140628 -5.727868e-02  0.4486556611  1.1644354652  0.8613067511
    ## 668  -6.237596542 -1.104638e+01 -4.1558793683  1.7636108744 -0.8150863158
    ## 669  -2.849930795 -3.654258e+00 -1.1830341366  0.8316238530  0.3488962850
    ## 670   1.425605504  1.593529e+00  1.1784678458 -1.5686975579 -0.2521150990
    ## 671   1.367433022  1.522662e+00  1.0263174826 -1.5770704784 -0.1726586446
    ## 672  -4.348891446 -8.760786e+00 -3.4075485442  1.3329300655 -1.8183154248
    ## 673   0.125852996 -3.972455e-01  0.2723770654  1.2260221979  0.0629081292
    ## 674  -2.801501990 -4.186808e+00 -1.6484060937  1.1764455978 -0.4470394067
    ## 675   1.533411248  8.314426e-01 -0.4733466766  1.1901212635  0.2737992099
    ## 676  -2.543105040 -5.301273e+00 -2.1927127060 -0.0947237927 -0.4782193134
    ## 677  -2.339942917 -5.197794e+00 -1.3428144979  0.2888469105  0.2703601551
    ## 678  -3.306088412 -4.231022e+00 -0.7508097704  2.6908083420  1.5036883048
    ## 679  -1.173977649  2.433471e-01 -0.3423007282  0.6870562024 -0.0324998989
    ## 680  -1.654161807 -5.396060e-02  0.5207458402 -1.5545617470 -1.4639943240
    ## 681  -6.211557482 -6.248145e+00 -3.1492466947  0.0515761185 -3.4930499152
    ## 682  -2.156036543  5.647611e-01  0.8378566596 -0.7289897603 -0.5762740614
    ## 683   1.410917109  3.619251e+00  2.2849267627 -0.4439991919 -0.0997118778
    ## 684  -3.275207499 -6.823831e+00 -1.3553086862  0.2912507390  0.8958408440
    ## 685  -2.737795496 -4.961534e+00 -2.2247974078 -0.1361172698  0.3888848491
    ## 686  -3.727187613 -3.946300e+00 -1.6802864432  2.3036126101  1.2495855627
    ## 687  -3.943520999 -3.820522e+00 -0.5708210939  2.7833834077  2.2857579027
    ## 688   0.256215978  1.206083e-01 -0.0523460171  0.3298266319  0.6466164071
    ## 689  -3.515197166 -6.347453e+00 -0.9035121658  1.1916062677  1.9322537251
    ## 690  -3.316934301 -6.188834e+00 -1.0404126485  1.2330443495  1.0033504306
    ## 691  -1.243007101  5.704598e-01 -0.1590567662  0.4071875274 -0.5106135919
    ## 692  -2.619233054 -5.231425e+00 -1.2348537367  0.3973080371  0.0115572105
    ## 693  -2.475358854 -5.211875e+00 -0.4138716782  0.9332621646  0.3907859638
    ## 694  -4.180378884 -4.605686e+00 -2.5571841367  1.5895822019 -0.4865371418
    ## 695   0.530852434  2.781419e-01  0.3555299169 -1.0812232612 -0.1152819354
    ## 696   0.838120617  1.425060e+00 -0.1696641826  2.5020273886  0.8645361206
    ## 697  -7.568014899 -1.222363e+01 -4.8951751935  2.5249674694  0.2700131541
    ## 698  -6.173330860 -1.070802e+01 -3.5048941916  1.8840442573  0.3421223430
    ## 699  -5.976220982 -1.057601e+01 -3.5251612469  1.7379760050  0.3189951997
    ## 700  -6.276616786 -1.061171e+01 -4.1467737451  2.1972877132 -0.1562885227
    ## 701  -6.166837425 -1.056435e+01 -4.0101881531  2.1435017952 -0.5416341558
    ## 702   0.821195362  4.027366e+00  1.0779789803 -1.1072758550 -0.0059125766
    ## 703   0.908454083  4.133667e+00  1.3062045251 -1.0947164743 -0.1250972583
    ## 704   1.055513936  3.970587e+00  1.7622485717 -0.8808121028 -0.1357894043
    ## 705   1.315365139  3.639200e+00  1.6234260947 -1.5393388316 -0.1875719936
    ## 706   1.377768509  2.238869e-01  1.3110726751 -0.8960717864 -0.1869779929
    ## 707   0.731437272  4.152086e+00  0.9853582884 -1.2292543184 -0.0235757903
    ## 708  -0.868576752 -3.703333e-01  0.4192655073 -0.8948936572  0.0580625903
    ## 709   0.623912781  4.090983e+00  1.6340840281  0.2608473304  0.3032616484
    ## 710  -6.525081040 -1.140837e+01 -4.6939777364  2.4312744916 -0.6169493012
    ## 711  -3.262658419 -4.333118e+00 -1.8630489679 -0.0095336066 -0.7187056149
    ## 712   0.059868954  3.063394e-01  0.2650519889  0.2237182883  0.7328524984
    ## 713  -2.348739926 -6.426734e+00 -0.8273084252  1.7323817579  0.3700405206
    ## 714  -0.953623443 -2.727343e+00 -0.6364369187 -0.9271240727  0.0489688003
    ## 715  -4.850323159 -1.000625e+01 -2.7895981123  0.9825838589  0.7450293299
    ## 716  -6.367524375 -1.273439e+01 -3.8451296589  1.0076672599  1.1295322813
    ## 717  -6.299961398 -1.271921e+01 -3.7401761558  0.8440596103  2.1727090563
    ## 718  -7.360474609 -1.466877e+01 -4.8771190048  1.3856095278  0.6673097799
    ## 719   0.348037592 -5.663901e-01  0.2160161018 -1.1403756425  0.6341842899
    ## 720  -3.818085664 -3.700707e+00 -1.9914120781  0.7799505352  0.6925373808
    ## 721  -5.452494658 -1.188757e+01 -3.5635848101  0.8760187682  0.5456980406
    ## 722  -5.363566433 -1.193909e+01 -3.5836028702  0.8974024061  0.1357113227
    ## 723  -5.945480612 -9.338938e+00 -2.3901925882  3.1841523564  0.8747201035
    ## 724  -5.440795183 -9.518038e+00 -2.4380686183  2.3367536205  0.8083363604
    ## 725  -5.375282893 -9.315001e+00 -2.4875625691  2.3559189116  0.5763794653
    ## 726  -5.863661713 -1.114324e+01 -3.4809395529  2.7943334041 -0.1785339643
    ## 727  -0.328387383  2.270271e+00  0.7096660177  0.9131759971  1.4749287568
    ## 728  -0.928921530  1.679986e+00  1.3153388659  2.2261606556  1.2753580265
    ## 729   1.154312686 -2.043858e+00 -0.1516988071 -0.9435135355 -1.4934013747
    ## 730  -5.700141420 -6.388093e+00 -1.8381593010  2.9931735071 -0.1786263584
    ## 731   1.506696150  7.998325e-01  1.1627642616 -1.1506273571 -0.1159223176
    ## 732   0.235476804  3.321569e+00  1.9062708295 -0.5813556906 -2.1408738227
    ## 733  -2.425015632 -3.960623e+00 -0.1250177880  0.6003972884  0.5369164580
    ## 734  -4.534029753 -6.465409e+00 -1.8903099791  0.1748401856 -0.6693514290
    ## 735   1.679739601  5.586115e+00  2.7891311559 -2.2410748530 -0.0063884973
    ## 736   1.854771917  6.024397e+00  3.5312503348 -1.1962847756 -0.2306400954
    ## 737  -1.691045268 -2.372423e+00  0.4500985033  0.4078050608 -2.8063019229
    ## 738  -3.501281103 -2.941876e+00 -1.1257410620  0.3737148079 -0.0685983809
    ## 739  -2.639409140 -2.532355e+00 -0.6670670272  0.6260592934 -0.8421783365
    ## 740 -13.303887577 -2.129791e+01 -7.2623107948  2.2742255099  1.4908691419
    ## 741 -12.105602245 -2.133820e+01 -8.0454357156  0.1560150328  1.1152468889
    ## 742 -12.427961363 -2.015905e+01 -6.8888910928  2.5860932167  1.3540647952
    ## 743 -12.186362498 -2.016557e+01 -7.0516513533  2.5008272080  1.1941373215
    ## 744 -12.375333738 -1.871676e+01 -6.5220145732  3.5179549404  0.4839301085
    ## 745 -12.661695723 -1.891249e+01 -6.6269748452  4.0089207423  0.0556838814
    ## 746 -12.448562342 -2.266791e+01 -9.2646087330  1.9823563172 -0.0812198922
    ## 747 -11.866730637 -1.548699e+01 -5.7486515575  4.1300314273 -0.6468182730
    ## 748 -11.683998044 -1.584162e+01 -5.7531997528  3.8130407928 -0.8101464816
    ## 749  -9.833743021 -1.817462e+01 -7.2699045931  0.6237966118 -1.3762984086
    ## 750 -10.342328359 -1.604452e+01 -5.8821359384  1.5736978964 -1.3789232236
    ## 751 -14.129854517 -2.068503e+01 -8.4844490573  4.8512551366 -0.0822752254
    ## 752 -11.467429737 -1.917300e+01 -6.9698563842  2.2117558327  0.1132443633
    ## 753 -11.231364455 -2.158515e+01 -9.0908920610  1.1711887325 -0.1809347253
    ## 754 -11.133760976 -1.583359e+01 -5.7485329384  2.2710815067  0.5377949000
    ## 755 -11.646833829 -1.838385e+01 -7.4027308042  1.9416968101  0.2088791300
    ## 756 -12.172553018 -2.016454e+01 -8.6688151969  2.5508703438  0.6945744653
    ## 757  -6.809889940 -1.246231e+01 -5.5010507838 -0.5679395729  2.8122411093
    ## 758 -10.196333694 -1.727099e+01 -7.0790957793  1.5176953312  1.6574761842
    ## 759  -0.153429555  7.526106e-01 -0.7557858243 -1.9125626338 -0.3680142550
    ## 760   1.573739874  2.202652e+00  0.8004714666 -2.1035829033  0.9449150981
    ## 761  -8.728142394 -1.759885e+01 -7.4559563294  0.1697407056  2.1060338712
    ## 762  -9.360179193 -1.666837e+01 -7.1474677772  2.1620986455  2.5285123297
    ## 763  -3.936856918 -6.127194e+00 -2.3285519614  1.5621295523  1.9650297834
    ## 764  -1.839004665 -3.354638e+00 -0.2823072612  0.1485923939  3.7926667268
    ## 765  -4.254926620 -5.160213e+00 -1.3025259160  2.5944527612  3.6396027407
    ## 766  -9.014813281 -1.609203e+01 -6.0501476626  1.6610293587  2.4199211301
    ## 767 -10.402068327 -1.837202e+01 -8.3140923121  2.4358234855  1.9251025961
    ## 768  -8.582308650 -1.380657e+01 -6.0003587978  1.4226316133  3.8346490479
    ## 769  -8.880105672 -1.582514e+01 -6.7504252235 -0.1291881443  4.1000190483
    ## 770  -1.507929601 -3.410275e+00  0.1075440074 -1.3755176564 -1.2469963185
    ## 771  -7.606424575 -1.810826e+01 -7.5118661579 -1.2432851572  5.8045506607
    ## 772  -1.626129419  1.418215e+00 -1.4179168284 -1.6517661807 -1.4576102136
    ## 773  -6.557873102 -1.286724e+01 -5.8045904755 -1.2544653445  7.9073781272
    ## 774  -2.571891561 -3.003615e+00 -0.2254413884  0.2522735650  0.5089847483
    ## 775  -6.594756992 -1.295753e+01 -5.6135219500 -0.2488284022 10.4407179813
    ## 776  -3.537056284 -3.172028e+00 -0.5298850845  1.0116264503  0.4434885321
    ## 777  -6.261740850 -1.164716e+01 -3.4514361419 -0.5075365143  0.9424889420
    ## 778  -7.954097709 -7.809635e+00 -1.7185111721  3.8329851423  1.2649537967
    ## 779  -6.044103144 -9.750776e+00 -3.1163722541  0.0653171492  0.5295573309
    ## 780  -5.660366441 -1.303278e+01 -5.3923491741 -0.9010408178 11.0590042934
    ## 781  -8.143342262 -9.785677e+00 -3.0865195054  3.3797133358  0.7922746841
    ## 782  -7.189926187 -1.044511e+01 -3.2719075376  1.4822774347  0.1056151102
    ## 783  -6.250629215 -1.356633e+01 -4.1927801967  0.5105701818 -0.2278815061
    ## 784  -5.504334070 -8.697777e+00 -1.9342253619  1.9587499315  0.2275259947
    ## 785  -5.190271038 -8.655711e+00 -2.0244434744  1.5604792801  0.0981321925
    ## 786  -5.022557177 -8.522975e+00 -2.0129390616  1.5257324410 -0.0647801372
    ## 787  -1.614718326 -3.353180e+00  0.2367620370  0.0038960221 -1.0561622601
    ## 788  -2.558668617 -5.304624e+00 -0.9569384490 -0.5853218520  0.4028367539
    ## 789  -3.371124711 -4.251158e+00 -0.4300572999  1.8040133391  1.1898137923
    ## 790   0.908649625 -1.220161e-01 -0.1041099172 -1.6840216339 -0.3444521238
    ## 791  -4.719829498 -4.847992e+00 -1.1343287451  3.5277382722  0.5208398660
    ## 792  -3.571408379 -7.311407e+00 -1.7543546066  0.7954488675 -0.1304382033
    ## 793  -0.482192056  8.110054e-01 -0.3579106846  0.7527609216 -0.0383141473
    ## 794  -4.921865144 -9.567268e+00 -2.7729741112  1.7170952684  0.3274904573
    ## 795  -3.129421322 -5.147039e+00 -1.8154365293  0.5204969231  0.4246083288
    ## 796  -2.065025612 -3.261435e+00  0.1252233203  0.9369661547 -1.1165809212
    ## 797  -1.949123261 -3.093013e+00  0.1240870747  0.8867469661 -1.1186867554
    ## 798  -4.140395667 -7.604715e+00 -1.5245053974  1.2713219314  0.1331064748
    ## 799  -0.660655352  9.332100e-02  0.3357422216  0.0575510394  3.9732170273
    ## 800  -4.602712645 -7.524188e+00 -2.5571041223  1.3191236294  0.2270506211
    ## 801  -7.085798176 -1.262386e+01 -4.7452238380  2.7978915018 -0.5056079618
    ## 802  -3.258046341 -3.348896e+00  0.2173307143 -0.9174079549 -0.1022942521
    ## 803   2.069853025  1.733382e+00  1.5807579621 -2.3351853761 -0.1535702647
    ## 804   2.131014672  6.443649e+00  1.5925381914 -3.6819035523 -0.2008462807
    ## 805  -3.865975683 -2.705052e+00 -0.1039071814 -0.7168114246 -0.2505829398
    ## 806  -2.388207562 -2.577464e+00  0.0926786705  0.0982420129  0.4455733841
    ## 807  -4.510522737 -1.034051e+01 -4.1811775535 -0.3761356825 -1.0006577223
    ## 808  -1.820404772 -3.746490e+00 -0.8803993114  0.2679145515 -2.6429018472
    ## 809  -1.932316475 -2.284748e+00  0.3247872674 -0.1831809037 -0.0310853104
    ## 810  -0.907666066 -3.910514e-01 -0.0347191590  1.0842864667 -1.5861436973
    ## 811  -0.463552709  4.729954e-01 -0.4478991563  1.7909241465  0.2475795346
    ## 812  -0.509866916 -1.304370e+00  0.4959995346 -1.3598438929  0.1560572390
    ## 813  -2.577634726 -5.153021e-01  1.1091071085  2.3511513824  0.5095587267
    ## 814   0.574874857  1.884696e-01  1.6492235205 -1.7967386312  0.2525711638
    ## 815  -2.617766960 -1.310752e+00  0.4096599097  2.3622907759  0.2857922599
    ## 816  -2.497341110 -1.588336e+00  0.1202893725  0.1701438794  0.0317945560
    ## 817  -2.365801464 -1.351900e+00  0.1894984868  0.5240075241 -0.2805332664
    ## 818  -0.055332753  1.778697e+00  0.6718324342 -0.0020211579  0.3611689323
    ## 819   1.672722800  1.165737e+00  0.5613227074 -2.3007976757  1.2773150340
    ## 820   0.146399069  1.759314e+00  1.0830395144 -0.3910477424 -0.0258616225
    ## 821   1.679872964  5.540865e+00  3.2076369636 -0.5594798108  0.3188528921
    ## 822  -2.693168082 -3.166955e+00 -1.0678001200 -0.5591322023  2.9083739459
    ## 823   0.458138563  1.676018e-01  1.4043108943 -1.8740152633  0.3295679726
    ## 824  -2.695789729 -3.035166e+00  0.0968148690  2.0656919705  0.6218038346
    ## 825  -2.681881051 -2.286145e+00 -1.0488449028  0.9948295004  1.1985372847
    ## 826  -1.848598114 -2.421536e+00 -0.8333448532  0.9521410572  1.1472039204
    ## 827   0.204079529  4.629293e-01  0.8411690590 -2.2213838534  0.3381607085
    ## 828  -2.935804425 -2.285573e+00 -0.2370556328  1.9797470710  1.1416150755
    ## 829   0.086836724  9.785462e-01  0.0556039737  0.1891308356  0.1507269877
    ## 830  -4.787448417 -9.334329e+00 -2.5802486555  1.5015734617 -0.0225898632
    ## 831  -5.496187610 -5.995337e+00 -3.9905149276 -0.1671997979  0.3573693080
    ## 832  -6.366499509 -7.550968e+00 -4.9027666663  0.1528920320  0.2504154373
    ## 833  -4.596576583 -5.522088e+00 -3.5290655977 -0.6633708009  0.1948098146
    ## 834  -0.842769191 -1.029461e-01 -0.5975972128  1.0601542261 -2.2861370803
    ## 835  -4.997331655 -6.419539e+00 -1.1835927397  3.5697325775  2.1094026874
    ## 836  -4.472014334 -5.856998e+00 -2.2431779038 -0.1738140245  0.5895747217
    ## 837  -0.911031402 -2.237683e-01  0.7683698340  0.4775214524 -1.0277158962
    ## 838  -4.307059802 -5.701174e+00 -1.7728029372 -0.1931322058  2.2307348722
    ## 839   3.139655659  5.665429e+00  3.7903162118 -1.9097986195 -0.3814437223
    ## 840  -5.037449410 -8.078010e+00 -3.4579811870  0.5966964301  0.2855587858
    ## 841  -5.104479839 -7.925389e+00 -3.4157067151  0.7183323737  0.1868983045
    ## 842   0.056227349  6.536621e-01  0.3346553117  1.0289268365  0.0621985704
    ## 843   0.497433587  3.871618e+00  1.4923940374  0.5060399785  0.2056912322
    ## 844  -1.022045026 -1.646505e+00  0.1264597527  1.8190127640 -0.2993741129
    ## 845  -4.472547557 -7.165044e+00 -1.5811281913  1.9303797089  0.2646987038
    ## 846  -2.916462628 -2.819906e+00 -1.4172308009  0.3251949429  0.5481063635
    ## 847  -4.239086531 -9.250635e+00 -2.4348278030  1.9873862769 -0.3205409728
    ## 848   0.372379299 -7.982011e-01 -0.6970616084 -1.4971522329 -0.2647047523
    ## 849  -0.642332574 -2.114429e+00  0.1833274128 -1.1777925065  0.1635134851
    ## 850  -2.205644483 -5.709087e+00 -1.1139366402 -1.5565302121  0.1642877925
    ## 851  -3.751030733 -9.261641e+00 -2.9551652220  0.6065707972 -0.4036849104
    ## 852  -3.281783735 -4.368986e+00 -0.7367337647  1.6092929042  0.4744140528
    ## 853  -4.671031532 -4.472403e+00 -0.9820899002  3.4900685927 -0.3387067312
    ## 854  -3.839559974 -5.056010e+00 -1.3444975308  1.1484729388 -0.7135158108
    ## 855  -1.952290400 -2.892555e+00 -0.9120579607 -1.5637399671  0.2419212947
    ## 856   2.244215779  3.702065e+00  2.2568115324 -1.9500603319 -0.1601634794
    ## 857  -1.689836478 -5.293254e-02  0.7787348248  0.2421624960  0.5194039725
    ## 858  -3.836188525 -2.515183e+00 -0.8645787119  2.3289269736  0.3408983157
    ## 859   0.526946021  2.347023e+00  1.6912200047 -0.7361106931 -1.4244861996
    ## 860  -2.986920088 -4.815863e+00 -1.2716545883  1.7016850237  2.4256774298
    ## 861  -1.424131271 -3.243686e+00  0.0599561085 -1.8023324970  0.5315740932
    ## 862  -0.666134267 -1.372629e+00 -0.1043129559 -1.4669105532  0.3133316345
    ## 863   1.174884173  2.134606e+00  2.5943648330 -1.2575889799  0.9647718037
    ## 864  -5.241049108 -8.152829e+00 -3.2434364450  1.9354837541  0.3816819389
    ## 865  -5.239604388 -5.017908e+00 -1.3493088591  2.8686028949  0.4522413109
    ## 866  -4.325132247 -5.558067e+00 -1.5805312203  0.9719063217  0.1147600878
    ## 867   0.513470571  1.136654e-01  1.4886203594 -1.8055767139  0.3364418657
    ## 868   1.323832975  5.580209e+00  3.2936875938 -0.1376935821  0.7012918266
    ## 869  -6.180918699 -9.285186e+00 -3.9217221413  0.6855108603 -0.1989627800
    ## 870  -2.679171246 -1.385557e+00  0.2490567485  2.3534526932  0.3696629618
    ## 871   2.464449802  6.712406e-01  1.9210210917 -1.6169265883 -0.0855787549
    ## 872  -2.553250783 -3.483436e+00 -0.0648523857  1.4903287628  0.5321454219
    ## 873  -2.302595690 -4.446192e+00 -1.7104631485  0.9827286327 -1.9285271925
    ## 874   1.732872800  2.934481e+00  1.4371965724 -1.9233089386 -0.2883921912
    ## 875  -0.997477451 -8.172614e-01  0.3077976004  0.0284845309  0.3538982028
    ## 876  -4.079992330 -6.717177e+00 -1.8875496491  1.3567482897  0.1644534460
    ## 877  -5.145637851 -6.293479e+00 -2.1371866009  2.9952447545 -0.5511211149
    ## 878  -3.939383740 -7.164430e+00 -2.4346718163  0.2352268411 -0.9538269825
    ## 879  -4.066208555 -6.626968e+00 -1.4371584194 -0.2154097385 -0.2636860680
    ## 880  -4.491629225 -5.969987e+00 -1.2746662200  1.1477836122 -0.1814552012
    ## 881  -4.389687678 -5.810920e+00 -1.2554224018  1.1140330504 -0.1819785833
    ## 882  -4.287959839 -5.651765e+00 -1.2358892553  1.0801328930 -0.1802787306
    ## 883  -4.186415178 -5.492536e+00 -1.2161081130  1.0461045110 -0.1766732481
    ## 884   0.828290651 -1.274787e+00  0.1342773052 -1.4894342171 -0.1258769970
    ## 885   0.461206284  1.468052e+00  0.8917055352 -0.4579340399  0.5854347368
    ## 886  -1.202297970 -6.167637e+00 -2.6515390486  0.0135877062 -1.8504699635
    ## 887  -1.475639514 -3.082274e+00  0.2247397551 -0.3009307785  0.2848306556
    ## 888   0.269281953  5.913191e-01  1.7959919205 -1.0852079102  0.3547729330
    ## 889  -1.176628722 -1.859739e+00 -0.0494185885  0.3852626095  0.1265179435
    ## 890  -1.555962896 -2.084067e+00  0.3942473529  0.0833797228  3.1893548959
    ## 891  -0.934615796 -5.241943e-02  0.5111617843  0.7319987491  1.8786116025
    ## 892  -1.547551823 -1.909017e+00  0.5005674704  0.1367224382  2.9327581257
    ## 893  -2.133593488 -5.133081e+00 -1.0621507808 -0.8193710963  0.5584245976
    ## 894  -2.255853807 -2.128027e+00  0.7067650290  1.0938257975  0.8720055313
    ## 895  -2.020734381 -2.039703e+00  0.6581829273  0.8325740350  0.8041013303
    ## 896  -1.920558168 -1.879928e+00  0.6797512992  0.7976413755  0.8214045487
    ## 897  -3.014293978 -6.642482e+00 -2.5840282244 -0.8203544148  0.4610319438
    ## 898   1.889417455  1.714511e+00  1.1046599110 -1.5974402131 -0.1722164975
    ## 899   1.587460391  1.929350e+00  1.1894329029 -1.5301623260 -0.2261080206
    ## 900  -4.878248088 -4.335998e+00 -1.3771910852  3.0573817419  1.0543901030
    ## 901  -1.283145000 -2.718560e+00 -0.0854660891 -2.0973854712  0.3287959830
    ## 902  -2.110863433 -1.558545e+00  0.1959918263  0.5024534511  0.5970255499
    ## 903   0.105946450  7.165913e-01  0.2052846227 -0.0513220971  0.3994470413
    ## 904   0.418268346 -8.281925e-01 -0.3022394234 -1.7623500244 -0.2062387911
    ## 905  -0.408470613 -1.481283e+00  0.3476266922 -1.1644143713  0.2276182555
    ## 906  -0.288725034  1.404507e+00  2.1320813786  0.7071999457  0.5620298611
    ## 907  -2.826985512 -2.865750e+00 -0.9129342382  0.4211440827  3.1363381321
    ## 908  -1.005723154  3.343156e-01  0.4212613624  1.2471426309  0.4995679131
    ## 909  -2.383826034 -4.166479e+00 -1.7723969470 -0.5262807942 -0.0934214754
    ## 910  -2.499307194 -3.712752e+00 -1.1421334372  0.6262408557 -0.5150009201
    ## 911  -3.302898611 -5.848903e+00 -1.8506354642 -1.0186385954  0.5554117805
    ## 912  -3.040862552 -3.008958e+00 -0.1599669537  0.7880862955  0.8320347894
    ## 913  -0.387162054  1.734742e+00  0.6109759880 -0.2212672077 -0.2111671791
    ## 914  -2.864039118 -2.871342e+00 -0.1806135602  0.6477088805  0.8165578589
    ## 915  -6.469186891 -8.713920e+00 -3.7050697433  3.5310028978  0.3195762713
    ## 916  -4.137500501 -6.103903e+00 -1.9729278028  0.4340725369 -0.0043005898
    ## 917  -5.539544320 -7.290960e+00 -4.8977660357 -0.3086089920 -0.2032698022
    ## 918   0.836807763  2.844950e+00  1.6135095582  2.5483126550 -0.4751264428
    ## 919  -0.674356809  3.274635e-01  0.0914699151  1.3079407376  2.4932238913
    ## 920  -2.015712746 -1.731413e+00 -0.4658147271  0.5276197556  0.1908769034
    ## 921  -0.429391896 -8.246442e-01  0.4896677988  0.8733442715  0.0338040155
    ## 922   1.842346923  1.194212e+00  0.0374666576  0.4230993691  0.0374375438
    ## 923  -2.077065542 -2.683521e+00 -0.5259493326  0.4223424448  0.6501958557
    ## 924  -2.897825117 -4.570529e+00 -1.3151472139  0.3911670409  1.2529667347
    ## 925  -2.155296885 -3.267116e+00 -0.6885054627  0.7376572175  0.2261379455
    ## 926  -1.603014747 -5.035326e+00 -0.5069998837  0.2662723203  0.2479677526
    ## 927  -2.549498236 -4.614717e+00 -1.4781379413 -0.0354803665  0.3062707404
    ## 928   0.519435549  9.035624e-01  1.1973147180  0.5935088469 -0.0176522567
    ##               V21          V22           V23           V24           V25
    ## 1     0.189378760  0.355332429 -1.339148e-01 -0.4255910396  2.682213e-02
    ## 2     0.513196784 -0.632525610 -9.362004e-01 -0.2440719028 -4.437460e-02
    ## 3     0.018121647  0.395493316 -2.240515e-01  0.1101107812 -7.921192e-01
    ## 4     0.156564763  0.252023166  9.323654e-02  0.3991377429  1.070953e-01
    ## 5     0.187124336  0.432904070 -1.163108e-01  0.6393533998  4.495725e-01
    ## 6     0.015138197  0.702549880  1.622677e+00 -0.0318624253  9.243249e-01
    ## 7    -0.274924358 -0.901033657  7.661003e-02  0.9202909051  3.550991e-01
    ## 8    -0.149108390 -0.409576961 -2.477504e-01 -0.5599678714  3.682392e-01
    ## 9    -0.278395842 -0.648695728  2.181325e-01  0.0280675715 -2.909905e-01
    ## 10   -0.316660728 -0.524482238  1.511717e-01 -1.5426962690 -7.969234e-01
    ## 11    0.122470157  0.352730917 -2.826122e-01 -0.9905862944  7.031207e-01
    ## 12    0.317558115  0.659786759  3.410447e-01 -0.3325079216 -1.160073e+00
    ## 13    0.025481317  0.126832789 -1.401989e-01  0.6110427312  2.463635e-01
    ## 14   -0.205402906 -0.608086831  2.353369e-01  0.7090049293 -8.414809e-01
    ## 15    0.167536358  0.502072158 -2.000417e-01 -0.2098541235  5.804762e-01
    ## 16   -0.004719082  0.227355713  2.068723e-02  0.6306002277  4.919719e-01
    ## 17    0.154757153  0.755280021 -3.877478e-02 -0.9800671727  3.111142e-01
    ## 18    0.059362366  0.185081032 -1.407622e-01  0.4253499063  7.064168e-01
    ## 19   -0.388866760 -0.650072211  3.727826e-02 -0.2968694364  4.480521e-01
    ## 20   -0.234876222 -0.476364392 -7.968112e-02 -0.7970153829  4.919787e-01
    ## 21    0.192520664  0.517701891 -8.555215e-02 -0.2883993967  4.578392e-01
    ## 22   -0.105202322 -0.691107143 -1.428450e-01  0.4818284106  1.653823e-01
    ## 23    0.164350217  0.703347387 -8.031165e-02 -0.3146230428  3.263072e-01
    ## 24    0.281957716  0.549950291  7.640244e-02  0.6159204266 -1.853923e-01
    ## 25    0.198468000  0.522219831  7.607410e-02 -0.0176249716 -1.604886e-01
    ## 26   -0.328833682 -0.850984707  2.349187e-02 -0.0222272951 -3.900892e-01
    ## 27   -0.136483938 -0.622144390  2.128369e-01  0.7588609070 -2.867492e-01
    ## 28    0.212010581  0.474091290 -1.960539e-01 -0.2888805662  1.638853e-01
    ## 29   -0.259090856 -0.768822934  1.596844e-01  0.6741872745  1.715455e-01
    ## 30    0.749549955  0.351855233 -7.244808e-01  0.3299539355 -2.498253e-01
    ## 31    0.151514714  0.544582928 -2.385430e-01  0.4424302476  1.624640e-02
    ## 32   -0.095937656 -0.055247699 -1.623780e-01  0.5120910058  3.837224e-01
    ## 33   -0.008134541  0.142709289 -2.333479e-02  0.5630234318  2.327523e-01
    ## 34   -0.123146981 -0.360511106 -1.691663e-01 -0.9797810678  6.474564e-01
    ## 35    0.244237170  0.756629269 -1.085256e-01 -0.4073061563  2.504374e-01
    ## 36    0.194945819  0.459492399 -1.055483e-01  0.0768655386  4.089037e-01
    ## 37    1.798778709 -1.779407231  2.812714e-01 -0.1094888164  7.564154e-01
    ## 38   -0.335331300 -0.411643248  1.604439e-01  0.0054452114 -3.463177e-02
    ## 39   -0.346063877 -0.598506833  4.659386e-02  0.0208670913  3.951610e-01
    ## 40    0.243401270  1.491227087  2.305154e-02  0.1839174281 -1.036678e+00
    ## 41    0.055849243  0.208607405  1.052247e-01 -0.3585768974 -1.687635e-01
    ## 42   -0.139314453 -0.325663227  6.831146e-02  0.0698939932 -3.143545e-02
    ## 43   -0.234923260 -0.451760680  2.262835e-01 -1.0323051034 -2.845976e-01
    ## 44   -0.148614547 -0.121430548 -2.754216e-01 -0.5546316070  2.697039e-01
    ## 45   -0.362074838 -0.878347443  3.072884e-01  0.4105320195 -1.978467e-01
    ## 46    0.109103025  0.548880409  1.411875e-01  0.0138609396 -1.060276e+00
    ## 47    0.437906325  1.291117487 -5.811904e-02 -0.2519247007 -6.372638e-01
    ## 48   -0.369616338 -0.547724629  3.815831e-02 -0.5689879739  3.624295e-01
    ## 49    0.231016080  0.782068222 -3.532737e-01 -0.3550948125 -1.728633e-01
    ## 50   -0.804191444  0.812055697 -3.333482e-01 -0.5320790976  2.604479e-01
    ## 51    0.294480195  0.536545600 -4.476824e-01  0.5437707371  8.585436e-01
    ## 52   -0.186948215  0.606398381  4.308840e-01 -0.2315441836  4.435715e-01
    ## 53    0.318861341  1.043941386 -1.560189e-01 -0.3012575817  3.414479e-01
    ## 54   -0.199348997 -0.484805292 -3.864498e-02 -0.3470131159 -2.030610e-01
    ## 55   -0.169749629 -0.488972989  3.025107e-02 -0.0203393754  2.180543e-01
    ## 56    0.086167205  0.433840769 -4.983616e-02  0.0215731171  4.765552e-01
    ## 57   -0.288664660  0.185892617  2.592697e-01  0.7346883969 -4.457321e-01
    ## 58   -0.108122216  0.219770537 -3.851565e-01  0.7588296370  4.180013e-01
    ## 59   -0.111697807  0.429705272 -3.082348e-01 -1.4402901045  3.705702e-01
    ## 60   -0.032814745 -0.893258674 -1.723607e-01  0.5523294121  3.311745e-01
    ## 61   -0.386465037 -1.281825830  1.567695e-01 -0.0054492451 -5.506601e-01
    ## 62    0.035519120  0.129481237 -1.442898e-01 -0.2895419981  1.779926e-01
    ## 63    0.164364054  0.698579609  9.704549e-02 -0.2446646434 -1.253460e+00
    ## 64   -0.136641238 -0.329345020  3.307477e-01 -0.7504987058 -4.404154e-01
    ## 65   -0.139243956  0.486646758  1.749952e+00  0.0852672297  3.707611e-01
    ## 66    0.800348874  0.149387516 -1.642908e-01  0.3785077946 -7.123678e-02
    ## 67    0.143291104  0.076238813  1.413664e-01 -1.3689178382  2.980473e-01
    ## 68   -0.258641014 -0.665588360  2.727449e-01 -0.4070735031 -2.806089e-01
    ## 69    0.156356424  0.683128120  3.863016e-01  0.5126303660 -3.126759e-01
    ## 70   -0.228954681 -0.498576389  9.389429e-03 -0.4695708159 -4.778809e-01
    ## 71    0.444124227 -1.106693762 -3.671502e-01  0.0822536048  7.092620e-01
    ## 72   -0.250013553 -0.657008093 -6.641404e-02 -0.1243683342 -1.105102e-01
    ## 73    0.381862821  0.910125849  1.010727e-02  0.8781883319 -3.440448e-01
    ## 74   -3.294920312  1.583979040  2.541446e+00 -0.6724578613  3.069751e+00
    ## 75   -0.009272984  0.194001809 -1.444560e-01 -1.0507241078  6.015167e-01
    ## 76   -0.169401171 -0.826036828  2.165055e-01  0.1842057313 -1.982854e-01
    ## 77    0.137271576  0.200665000 -1.535527e-01  0.7060232086  3.631221e-01
    ## 78    0.031794000  0.336924231  1.828559e-01  0.0545367136 -9.762685e-01
    ## 79    0.337669637  0.963758607 -4.651375e-02 -0.4208057130 -7.791745e-01
    ## 80   -0.126627390 -0.194171686  3.741316e-01 -0.0386420783 -4.323645e-01
    ## 81   -0.057954369  0.071964948 -6.718525e-02 -0.7334994377  2.763287e-01
    ## 82    0.551218260 -0.121631693  2.349311e+00  0.2345472456 -2.212008e-01
    ## 83    0.196770652 -0.445983001 -3.232633e-01  0.4985369485 -6.757207e-01
    ## 84    0.107125251  0.309724612  2.934581e-01  0.0879218435 -6.785750e-01
    ## 85    0.282660455  0.625855211 -3.933324e-01 -0.3998313353  1.094084e+00
    ## 86    0.123352830  0.258859113 -2.085837e-01 -0.6727429550  6.167880e-01
    ## 87    0.275895409  0.750005263 -1.888193e-01 -1.1393448217  3.007247e-01
    ## 88    0.155109244  0.680534179 -2.321026e-01  0.2109614989 -2.070799e-01
    ## 89   -0.517829569 -0.213789100 -8.800365e-01  0.9052621072 -1.635731e-01
    ## 90    0.419848777  1.068569822 -2.390066e-02 -0.4041904967 -2.930953e-01
    ## 91    0.078334217  0.500861928  8.752552e-02  0.0418162078  8.859538e-02
    ## 92   -0.161621575 -0.203394151  1.233146e-01  0.4956740123  7.129854e-02
    ## 93   -0.053587734  0.235061252  2.440591e-02 -0.3456723960  9.620690e-04
    ## 94    0.068885049  0.701118505  2.687178e-01  0.0215331821 -5.606586e-01
    ## 95   -0.251099859 -0.729983935  6.977432e-02 -0.0210189307  2.496795e-01
    ## 96    0.222900750  0.724382304  1.821071e-01  0.7834185578 -2.936184e-01
    ## 97   -0.043308536  0.355207519 -1.833983e-01  0.4119518221 -2.865489e-02
    ## 98   -0.033144061  0.155050236 -5.158433e-02  0.2506939891  6.315465e-01
    ## 99   -0.182540732 -0.712453888 -1.811616e-01  0.4626588278  1.549015e-01
    ## 100   0.029193873 -0.113968101 -1.748717e-01  0.1809162484  4.490476e-01
    ## 101   0.230039542  0.226992073  2.064814e-01  0.3033958489  4.687703e-01
    ## 102  -0.064061807 -0.098959962 -1.394908e-01 -0.0996798169 -5.881699e-02
    ## 103  -0.339494461 -0.462898767  4.070513e-01  0.7369621794 -7.452109e-01
    ## 104  -0.578372672 -1.417687134  4.086555e-01  0.0224973539 -5.581926e-01
    ## 105   0.212288272  0.217154261  1.901993e-01  0.0464368373  4.425829e-01
    ## 106  -0.261388931 -0.895801680 -2.817362e-01  0.1042240044  2.688088e-01
    ## 107   0.203505577  0.870202782  1.288609e-01  0.7970019017 -2.361986e-02
    ## 108   0.214773346 -0.345721216 -5.869987e-01  0.9536360189  3.609882e-01
    ## 109  -0.163858733 -0.737460870  3.033090e-01 -0.0238142439 -2.196256e-01
    ## 110  -0.019804753 -0.071547384  2.537334e-01 -0.5592827772 -9.515465e-01
    ## 111  -0.059365742 -0.424279454 -2.378535e-01  0.5538987135  3.372971e-01
    ## 112  -0.275741787 -0.798621182 -1.955957e-01 -1.0717492040 -7.063241e-02
    ## 113  -0.260385593 -0.563315394  6.456052e-02  0.0417689294 -4.887833e-01
    ## 114   0.185448895  1.525292084  3.370749e-03 -0.0127314796 -1.151280e+00
    ## 115   0.265539032  0.843557294 -1.648171e-01  0.7063689564 -4.109175e-01
    ## 116   0.237925635  0.572038700 -5.413070e-02  0.5826735463 -5.255629e-01
    ## 117   0.021854516 -0.024956632 -1.640813e-02  0.1133752408 -1.954298e-01
    ## 118  -0.295590775 -0.775420053  3.565541e-01  0.6102548208 -3.269060e-01
    ## 119  -0.129273395  0.044810561 -5.164220e-02  0.0627605275  3.519417e-01
    ## 120   0.973787312  0.549328165 -8.634061e-02  0.0277076921 -6.839066e-02
    ## 121   0.161391313  0.853378790  3.890944e-01  0.8221193827 -1.883824e-01
    ## 122   0.088358760  0.739114751  1.765389e-01  0.7067149517 -1.956069e-01
    ## 123  -0.269812183 -0.681927635  2.698694e-01 -0.4951828304 -2.681478e-01
    ## 124   2.163871659 -0.096734581 -9.561267e-01  0.7669218887 -2.876747e-01
    ## 125  -0.300844519 -1.028181036  2.794208e-01 -1.5503177770 -8.856080e-01
    ## 126  -0.203607379 -0.675731469  1.610480e-01  0.5031771270  1.264081e-01
    ## 127  -0.273139883 -0.496219116 -3.989513e-01 -0.7878246455  9.572142e-01
    ## 128   3.147273181 -2.400830503 -3.095160e+00 -0.6856207758 -2.131764e-01
    ## 129  -0.022601263  0.288664686 -2.892438e-01 -0.3749676444  2.571449e-01
    ## 130   0.161773875  0.661701665 -1.501974e-01  0.0558129677  5.557984e-01
    ## 131  -3.269618925  1.384652621  1.621842e+00 -0.2524103595  7.668095e-03
    ## 132  -0.406249656 -1.155650857 -1.387751e-01 -0.4444657850  3.814184e-01
    ## 133  -0.306915655 -0.316677999 -2.640415e-01 -0.4924024347 -1.965707e-01
    ## 134  -0.316390214 -0.643935987  2.903752e-02 -0.4805951704 -8.782268e-01
    ## 135  -0.380505305  0.568109240  1.422242e-02 -0.0568503309  3.452942e-01
    ## 136  -0.271903231 -0.801360563  1.712475e-01  0.0052774103  1.078021e-01
    ## 137  -0.276605403 -0.850894798  6.769011e-03 -0.2451878668 -6.007237e-01
    ## 138  -0.072518980 -0.146658168 -1.236279e-01  0.0713408046  5.241756e-01
    ## 139   0.189335738  0.295073180  1.637467e-02  0.7158689741  2.428406e-01
    ## 140  -0.481577128 -1.076852726 -3.150667e-02 -0.0003336325  3.082144e-01
    ## 141  -0.222770623 -0.718522161  1.603065e-01 -0.0600515570  6.568042e-02
    ## 142  -0.235572069 -0.329309344 -1.064791e-01  0.3749284665  5.473078e-01
    ## 143  -0.096056358 -0.469249281 -2.084505e-01  0.0953870194 -4.209669e-01
    ## 144  -0.262001219 -0.599639120  7.601331e-02 -0.1174848073 -4.637587e-01
    ## 145  -0.267980390 -0.748250461  3.703289e-02  0.5612913862 -1.339261e-01
    ## 146  -0.197385028 -0.726416145 -2.755221e-02 -0.0970493154 -1.584587e-01
    ## 147  -0.033306672 -0.104697781  2.091639e-01 -1.3244339359 -3.281817e-01
    ## 148   0.294433262  1.000354631 -2.479829e-01  0.1130028022 -3.764456e-01
    ## 149   0.019897050  0.435076474  3.349340e-01 -1.3017152947  1.114973e-01
    ## 150  -0.000102758 -0.333166004 -1.150627e-01  0.5216756483 -6.255525e-02
    ## 151   0.036152195  0.012447262 -6.177240e-01 -0.8359483288  9.872652e-01
    ## 152   0.188828277  0.710748044 -2.493512e-01  0.4827172992 -5.920202e-02
    ## 153   1.254974124 -1.609739477 -5.135338e+00  0.8607503631 -9.583114e-01
    ## 154  -0.267825652  0.827096791  5.577184e-01  0.4766098936 -3.924642e-02
    ## 155   0.627242424  1.486894594 -4.829127e-01 -0.3490443722  7.645849e-01
    ## 156  -0.230237807 -0.986469074  1.156691e-01 -0.0402766443  1.854167e-01
    ## 157   0.125861851  0.501696538  1.719799e-01  0.5051864945 -2.535338e-01
    ## 158  -0.182835419 -0.224862593  2.578995e-01  0.5588557889 -6.077534e-02
    ## 159   0.082993950  0.498406023 -2.438527e-01  0.4272420020  3.213358e-01
    ## 160  -0.235382803 -1.048813868 -7.950003e-02  0.9309947119  1.073445e-01
    ## 161   0.208570092  0.825116540  1.578303e-02 -0.7726665410 -4.319591e-02
    ## 162  -0.122867840  0.027815733 -8.615547e-02  0.3597036631  2.093848e-01
    ## 163   0.164263456  0.103099035 -3.862672e-01  0.5672226967  2.848426e-01
    ## 164   0.063051767  0.659183873  2.235493e-01  0.7958878778 -3.516795e-01
    ## 165  -0.469380556 -0.810630098  3.417453e-02 -0.0635923242  2.527749e-01
    ## 166  -0.283685671 -0.678501330  3.472072e-01 -0.0141939017 -3.147301e-01
    ## 167   0.103020942  0.592500164 -1.072946e-03  1.1224352336 -5.361603e-01
    ## 168   0.136223960  0.448934295  8.925531e-02  0.1273178022 -7.500738e-01
    ## 169  -0.290030117 -0.890273503 -2.663402e-01 -0.5764288197  4.423748e-01
    ## 170   0.212672658  0.819150623 -2.867243e-01  0.0282006409 -1.141774e-01
    ## 171  -0.075144053  0.090511112 -2.462902e-01 -0.2771576747 -1.088623e-02
    ## 172   2.420729178 -1.188241577  8.030675e-01 -0.0792980391 -4.764097e-01
    ## 173  -0.528473338 -1.422338714 -4.060246e-02 -0.0408189041 -6.967284e-01
    ## 174  -0.223862912 -0.462113092  4.225619e-01 -0.3637325874 -1.089131e-01
    ## 175   0.121984454  0.195945771  5.693746e-02  0.0489003490  2.308044e-01
    ## 176  -0.056056919 -0.123287009  4.359808e-01 -1.4904096639 -7.750177e-01
    ## 177  -0.290557671 -0.295396684  1.535648e-01  0.8862189481  1.898495e-01
    ## 178  -0.179849906 -0.497012631  1.155593e-02 -1.0804996245 -7.138321e-01
    ## 179 -10.829075686  6.090513506  1.340620e+00 -0.3082171670 -1.902337e+00
    ## 180  -0.223648786 -0.585531960 -1.123161e-02  0.3687596573 -1.883336e-01
    ## 181  -0.433613144 -0.583595625  3.153436e-02  0.0437943837  3.780646e-01
    ## 182  -0.166787769 -0.583791137  1.525593e-01  0.0384391107 -7.460907e-01
    ## 183   0.114673852 -0.074610506 -1.016083e-01  0.0092855636  3.351553e-01
    ## 184  -0.001417499 -0.243297106  2.147230e-01 -0.4418644126 -2.723169e-01
    ## 185   0.422587733  1.291834265  1.953865e-01  0.6974406069 -1.705961e+00
    ## 186  -0.049321817  0.116819397 -1.795574e-02  0.4262764401  4.520564e-01
    ## 187   0.039875687 -0.055306209  1.370871e-01  0.1986068804  1.522675e-01
    ## 188  -0.157617060 -0.477783276  1.423364e-01  0.0616136287 -5.144553e-01
    ## 189   0.038452179  0.923649311 -1.217681e-01  0.0060402268 -4.351889e-01
    ## 190  -0.045021921 -0.202287605 -8.890444e-02 -0.0354415160  1.931761e-01
    ## 191  -0.237059554 -0.605115510  3.453906e-01  0.0653338327 -3.555333e-01
    ## 192  -0.027553846  0.063220618 -3.703296e-01  0.1372204433 -9.030053e-02
    ## 193  -0.017334461 -0.194170865 -1.410852e-01  0.2408860010  5.823229e-01
    ## 194  -0.073178641 -0.122937641 -6.407668e-02  0.4212527270 -1.316765e-01
    ## 195  -0.151400392 -0.358550449  1.370994e-01  0.6654351727 -7.350129e-02
    ## 196   0.020354446 -0.181716551 -1.553769e-01 -0.4153782235  2.299613e-01
    ## 197   0.297067456  0.698180854  8.167032e-01 -0.0563424710  1.407687e-01
    ## 198   1.204600555  1.014171678 -1.950781e-01  0.0669334307 -1.657424e-01
    ## 199   0.538000957  0.374512215  1.463371e-01 -0.3950697462  1.381344e+00
    ## 200   0.316029427  0.834910823  1.053488e-01 -0.1180102853 -2.034318e-01
    ## 201   0.122804252 -0.025384138 -2.589711e-01 -0.0181158100  5.841607e-01
    ## 202  -0.071168207 -0.215446991  3.841868e-01 -0.7791571149 -6.577086e-01
    ## 203   0.575375530  0.743709592  1.020278e+00 -0.0354440773  7.704918e-01
    ## 204  -1.883877764 -0.892471181 -8.160091e-01  0.4516017702 -6.229550e-01
    ## 205  -0.013573519 -0.008143889 -1.245338e-01  0.3818170569  2.056681e-01
    ## 206  -0.354290624 -0.743352378  6.514325e-02  0.0845616873 -2.722681e-01
    ## 207  -0.021115518 -0.179552357 -1.414193e-01  1.0043762515  8.451995e-01
    ## 208   0.100045843 -0.495986795 -1.547757e-01  0.3015829397 -2.204051e-01
    ## 209   0.087090101  0.397265494 -6.868640e-02 -0.2752444132 -1.686148e-01
    ## 210   0.132353158  0.713403913 -6.485357e-01  0.0698413987  7.680490e-01
    ## 211   0.269644040  0.603526599 -2.278182e-01 -1.1064561371  1.133597e-01
    ## 212  -0.148670922 -0.312862147  2.101022e-01  0.4805918852  9.985706e-02
    ## 213  -0.013963153 -0.069727020  6.672675e-02 -0.5433848352  1.590790e-01
    ## 214   1.694256806  0.007548136  3.589744e-01  0.4693401358 -1.189515e+00
    ## 215  -0.406922425 -0.873915155  1.644391e-02 -0.6657677459  2.685673e-01
    ## 216   0.244729876  0.702956670  3.624536e-01  0.0844043501 -1.290582e+00
    ## 217  -0.315706801 -0.495422349 -2.859098e-01 -0.1450179226  2.144886e-01
    ## 218   0.295252288  0.915231537 -2.792524e-01  0.1767842026  8.011901e-01
    ## 219   0.103192322  0.174253175  1.278704e-01  0.0193600039 -1.053178e+00
    ## 220   0.235920870  0.692599369  2.377119e-01  0.3497299918 -1.522164e-01
    ## 221  -0.326088070 -0.810010076  2.769482e-01 -0.1094279166 -1.389704e-01
    ## 222  -0.196259292 -0.563740556  4.444594e-03 -0.9156930063 -1.684149e-01
    ## 223  -0.301530328 -0.718240677  3.711124e-02  0.1329809035 -3.985716e-01
    ## 224  -0.152737443  0.211507528 -6.034703e-01 -0.4184523809  3.201105e-01
    ## 225   0.181868401  0.496103638 -3.148504e-02 -0.0447404908  2.994441e-02
    ## 226   0.054897499 -0.641578503 -2.197977e-02 -0.3565469856 -5.625363e-01
    ## 227  -0.234662970 -1.097757412 -1.176793e-01 -0.0592957212  1.166114e-02
    ## 228  -0.276534597 -1.393886013  1.427413e+00 -0.0910280233  1.441201e-01
    ## 229   0.020877419  0.500015580 -2.058059e-02  0.0889232540  3.082229e-01
    ## 230  -0.081388171 -0.072208475 -2.212385e-01 -0.4153623376  6.126372e-01
    ## 231  -0.247225728 -0.400622245  1.841950e-01  0.7271807176 -2.235159e-01
    ## 232   0.022440333  0.338154139 -3.568076e-01  0.4271164204  5.688153e-01
    ## 233  -0.242048359 -0.751146879  1.184531e-01 -0.3776095495  1.208080e-01
    ## 234   0.009405836 -0.195087088  2.701006e-01  0.5854186062 -5.198948e-01
    ## 235  -0.278883045 -0.776942992  1.186677e-01  0.3583568490  2.387780e-01
    ## 236  -0.588002585 -1.476180105  4.125624e-01 -0.0971590036 -2.636462e-01
    ## 237  -0.051768644  0.534373259 -9.260932e-02 -0.4359713298  3.028424e-01
    ## 238   0.274595629  1.021976677  5.035955e-02  0.0985198316 -3.990640e-02
    ## 239  -0.238779143 -0.766659427  1.130470e-01 -0.3809712008  1.132046e-01
    ## 240   0.174726085  0.653666960 -4.735707e-02 -0.2853320747  3.252775e-01
    ## 241  -0.356907272 -0.536335769  1.593357e-01 -0.2670868578 -2.971918e-01
    ## 242  -0.365852461 -0.978088510  3.287736e-01  0.4759508116 -2.619557e-01
    ## 243  -0.197984822 -0.447718034 -1.369865e-01 -1.3062980908  6.720671e-01
    ## 244  -0.237919443 -0.520314222  3.790381e-01 -0.8229450145 -4.875146e-01
    ## 245  -5.003274496  1.796366628  2.411431e-01 -0.4670512932  1.248857e+00
    ## 246  -1.290533179  0.245984503  2.512165e-01 -0.0718557517 -3.145473e-01
    ## 247   0.126588529  0.091953915 -4.298434e-01 -0.0100908439  3.280385e-01
    ## 248  -0.567206626 -1.364314517  3.070975e-01 -0.0857639825 -3.030341e-02
    ## 249  -0.280320059 -0.607775470  4.352788e-01  0.6062748952 -3.952568e-01
    ## 250   0.390532576  1.368292572 -7.764875e-02  0.3469448968 -8.086737e-01
    ## 251   0.285028410  0.701581954 -1.702211e-02  0.3612587950 -3.330753e-01
    ## 252  -0.112023935 -0.670881018  2.651740e-01  0.6312404883 -1.341871e-01
    ## 253   0.221676589  0.791562278 -4.768161e-02  0.6980883998 -3.112077e-01
    ## 254   0.206958796  0.705651816 -6.508525e-02  0.0318490879  4.191102e-01
    ## 255  -0.098707849 -0.352600319  2.639019e-02  0.4416268634  2.155739e-01
    ## 256   0.100800960  0.383101439 -9.646262e-04 -0.0128945214  3.651925e-01
    ## 257   0.450375104 -1.082743323  1.610509e-01 -0.2958672115 -1.635675e-01
    ## 258   0.097268271  0.302092313 -4.016945e-01 -1.3264381291  9.955781e-01
    ## 259   4.127640989 -2.163341733  5.938849e-01  0.4126975763  2.493554e-01
    ## 260  -0.377821639 -1.496090845  2.802736e-02  1.0339001476  3.278869e-01
    ## 261   0.105344131  0.522984644 -4.290415e-02 -0.2229051160  2.488099e-01
    ## 262  -0.292422970 -0.709159193 -1.099840e-01 -0.5060058130 -4.942242e-01
    ## 263   0.205810184  0.847943293 -5.103301e-02  0.0972498648 -5.756143e-02
    ## 264  -0.343596017 -0.585239682  3.425919e-01  0.0345707166 -6.983990e-01
    ## 265  -0.246357681 -0.619726291  3.353817e-01 -0.4970400987 -5.225040e-01
    ## 266  -0.015336815 -0.577186262  3.296626e-01  0.1511968727 -9.324602e-02
    ## 267   0.227156983 -1.103118136  3.768607e-01  0.0435323966  4.370492e-03
    ## 268   0.192747229  0.857319950 -3.178627e-01 -1.1382631532 -2.795269e-01
    ## 269   0.008535180  0.133531386 -6.755865e-02  0.2459761298  5.594013e-01
    ## 270  -0.969749500 -0.298098930  5.091800e-02  0.0420503497  1.108418e+00
    ## 271   0.206658256  0.905809454 -1.088356e-01  0.0051651260 -4.159275e-01
    ## 272  -0.099827085 -0.096019451  4.137759e-02 -0.9980154071 -3.037079e-01
    ## 273   0.066901258  0.203703323  2.423893e-02  0.6432829340  3.905514e-01
    ## 274   0.178558593  0.237351415 -2.511424e-01  0.5646958623  3.435960e-02
    ## 275   0.085309472 -0.200251006 -9.400991e-01  0.1091457019  6.474290e-01
    ## 276   0.094917307  0.294982704  1.108138e-02  0.0152493721  3.421064e-02
    ## 277  -0.294277958 -0.696347527 -1.408059e-02  0.3316725214 -1.027852e-01
    ## 278   0.563700220 -0.838496815  3.380787e-02  0.3529699845  7.845842e-01
    ## 279   0.281663618  0.770337455  1.441741e-01  0.8492525224 -3.569969e-01
    ## 280   0.950307787  0.966005926  7.477108e-01  0.5810684300 -6.594235e-01
    ## 281   0.373621216  0.902300574  1.295043e-01  0.8583235948 -2.810385e-01
    ## 282  -0.003697117  0.004054189 -1.874651e-02  0.6512392189  1.611531e-01
    ## 283  -0.272241136 -0.718130308  1.134721e-01  0.3826176822  2.623164e-01
    ## 284  -0.056016301 -0.017942249 -7.081938e-02  0.0025265918  6.011128e-01
    ## 285  -0.155353943 -0.536323418  6.562103e-02 -0.7868952663  2.727824e-01
    ## 286   0.092039001  0.228201900 -8.794740e-02  0.5339071144  6.089247e-01
    ## 287  -0.215322205 -0.637533747  4.454888e-01  0.4898626417 -5.130741e-01
    ## 288  -0.223636245 -0.575656646 -2.345720e-02  0.3732678130 -1.851936e-01
    ## 289   0.223164345  0.688187818 -1.066265e-01 -0.3610312466  4.318570e-01
    ## 290  -0.110706260  0.008836786 -1.820282e-01 -1.7646321996  4.085729e-01
    ## 291   0.144807748  0.505743446  3.290194e-02 -0.4476158669 -1.832377e-01
    ## 292  -0.247838156  0.072236102 -2.086979e-02  0.1324527042  4.893524e-01
    ## 293  -0.387946655 -0.763544461  6.205477e-02  0.2998833015  2.647208e-01
    ## 294   0.365579644  0.583790584 -2.672559e-01  0.2420038947  2.490768e-01
    ## 295   0.024842540  0.387084340 -1.410927e-01 -0.7874456111  5.563753e-01
    ## 296   0.027130560  0.555028018 -2.080602e-01  0.4878666875 -4.831806e-03
    ## 297  -0.559196633 -1.501271149  3.706607e-01 -0.0947196084 -2.811018e-01
    ## 298  -0.361158037 -0.984261949  3.541981e-01  0.6207093385 -2.971379e-01
    ## 299  -0.201772559 -0.988438118  1.764174e-01 -0.1317156798 -2.188436e-01
    ## 300  -0.267562456 -0.972668229  2.400047e-01 -0.4015698937 -5.708849e-01
    ## 301   0.142391212  0.745553032 -3.949468e-01  0.3516139030  5.184311e-01
    ## 302  -0.002010040  0.034531829 -1.397908e-01  0.3955908165  8.393907e-01
    ## 303  -0.240253047 -0.603910179  3.627738e-02 -0.3718521810 -5.110633e-01
    ## 304  -0.102368586 -0.407425764  2.776392e-01  0.0276273520 -7.774708e-01
    ## 305   0.085424013  0.089751127  3.379905e-01 -0.9773542427 -6.739218e-01
    ## 306  -7.664489019  1.179546433  1.606930e+00 -1.1577754001 -9.693951e-01
    ## 307  -0.493399465 -1.303187622  1.391970e-03 -1.2641254079  3.023553e-01
    ## 308  -0.112290996 -0.220814059 -1.742359e-01 -1.2670583115  3.557474e-01
    ## 309  -0.162789671 -0.211239719 -6.764532e-01  0.4661646750  1.073276e-02
    ## 310  -0.492879051 -1.119955614  1.614278e-01  0.4850091149  6.405388e-02
    ## 311   0.389686329  1.397578942 -6.422324e-02  0.7369554815 -7.666099e-01
    ## 312  -0.251696998  0.453349443  2.185577e-01 -1.2654050135 -7.536370e-01
    ## 313   0.029884304  0.130382317 -2.461188e-01  0.3911412206  4.523724e-01
    ## 314  -0.240068779 -0.709400244 -5.133733e-02 -0.0712399596 -1.739784e-01
    ## 315   0.015439565  0.345508616 -8.442942e-02  0.1041593293  2.664078e-01
    ## 316  -0.214703942 -0.475438163  2.013134e-02  0.2686386185 -3.659913e-01
    ## 317  -0.259030216 -0.661741196  2.880654e-01 -0.3408125362 -2.929629e-01
    ## 318   0.074600384  0.231431067  1.443223e-01  0.0728693829 -6.677410e-01
    ## 319  -0.028951937 -0.009499233 -4.702115e-01 -0.0722433377  4.808139e-01
    ## 320  -0.147017971 -0.354145791 -2.903117e-01 -1.2626084858 -8.331594e-02
    ## 321   0.224555588  0.069598403 -5.615456e-01 -1.2476857527  3.273636e-01
    ## 322  -0.287334132 -1.263852446  2.169731e-01 -0.8836357576 -5.176054e-01
    ## 323  -0.140096160  0.056700395  6.970068e-03  0.0056269243  3.683318e-01
    ## 324  -0.452103334  0.487513620  1.010567e+00 -0.0378132336  1.425148e-01
    ## 325  -0.016145725  0.230921652  2.871863e-01 -0.6480074654 -9.166028e-01
    ## 326   0.165234740  0.044817550 -5.102034e-01  0.1330237300  1.290823e+00
    ## 327   0.102172678  0.308672356 -7.659690e-01  0.3692527397  8.571525e-01
    ## 328   0.163523924  0.527082515 -4.537046e-01  0.6438753420  8.160958e-01
    ## 329  -0.107469198 -0.288155787  3.482394e-01  0.0223030315 -4.981094e-01
    ## 330  -0.282731767 -0.663842960  3.320412e-01 -0.0749492966 -2.942848e-01
    ## 331   0.066536720  0.104007524 -1.722316e-01 -1.3303065895  5.701391e-01
    ## 332  -0.069562744  0.257128614 -1.545403e-01  0.0697461025 -2.840594e-01
    ## 333   0.214976926  0.859950057 -1.402548e-01  0.0247400352 -2.707211e-01
    ## 334   0.209268993  0.542073169  9.648140e-02  0.1056293891 -2.572699e-01
    ## 335   0.194660466  0.536813527  5.271092e-02 -0.0988919509 -9.095232e-02
    ## 336  -0.011532507  0.174260202 -1.926556e-02 -0.4263467058  4.434490e-01
    ## 337   0.126142541  0.097176937 -1.427600e-01  0.5076871240  5.432803e-01
    ## 338  -0.095171787 -0.200136649  2.606691e-01 -1.0450733690 -1.534844e-01
    ## 339  -0.228183049  0.255473671  2.480693e-01  0.1864377476 -1.376500e+00
    ## 340   0.036295697 -0.630935644  1.892593e-01  0.9802986482  4.100731e-02
    ## 341  -0.215489555 -0.760368711  1.734524e-01  0.4748821454  9.144072e-02
    ## 342  -0.072616053 -0.454834435  1.893907e-02  0.1420086768 -1.807525e-01
    ## 343  -0.250880638 -0.396493754 -2.471061e-01 -1.1953311619  7.373630e-01
    ## 344   0.248380153  1.134632767 -2.825247e-01  0.6721078673 -2.372661e-01
    ## 345   0.299585054  0.443017390  1.333235e-02  0.2827064427 -2.146136e-01
    ## 346   0.363770048  1.051670051 -1.236213e-01  0.7058266122  1.395453e-01
    ## 347  -0.181583332 -0.732670982  6.179063e-02 -0.4253168425 -2.730028e-01
    ## 348   0.185498107  0.506816329 -5.076642e-02 -0.3162162619  2.614274e-01
    ## 349   0.231808104 -0.311996695  5.473350e-02  0.1179698799 -6.573772e-01
    ## 350  -0.202825992 -0.534633519  7.983841e-02  0.4274486973  2.024091e-01
    ## 351  -0.185123147 -0.549738533  2.133800e-01  0.7819679123 -2.386808e-01
    ## 352   0.130609323  0.910111668 -3.048159e-01 -0.3034611934 -1.812149e-01
    ## 353  -0.232115979  0.076171847  2.544114e-01 -0.9562705364 -2.639650e-01
    ## 354  -0.095754400 -0.028078608 -6.904536e-02  0.1050305389  6.744620e-01
    ## 355   0.528695953 -0.391265938 -1.033547e+00  0.0533809848  4.781680e-02
    ## 356   0.251900418  0.846136111  9.012451e-02  0.8494241012  1.082191e-01
    ## 357  -0.339160702 -0.159380082  6.388565e-02 -0.0429769395  8.971295e-02
    ## 358  -0.288769483 -0.877648185 -8.286158e-03 -0.0092948835  3.909542e-01
    ## 359   0.276806950  0.729882537 -2.056530e-01  0.0159873201  1.044437e-01
    ## 360   0.313764695  0.895573234  9.790213e-04  0.7231286067 -5.579704e-01
    ## 361  -0.082964751 -1.121081131 -4.608045e-01 -0.7476194736  3.076850e-01
    ## 362   0.313879556  0.854498738 -2.381154e-01  0.2395318744 -2.466786e-01
    ## 363  -0.099198015  0.356052373  2.270213e-01 -0.0461771072 -3.901025e-02
    ## 364  -0.460497198 -0.891390759 -2.546505e-01 -0.5849148046  4.409976e-01
    ## 365  -0.027389237  0.114761547  2.001585e-01  0.6781672302 -3.680873e-01
    ## 366   0.103639574  0.230937433  9.080899e-03  0.3093071771 -6.350467e-01
    ## 367  -0.653763344 -1.479316386 -6.304765e-02 -0.4726324901  2.988965e-01
    ## 368  -0.376301963 -0.755851479 -1.829805e-01  0.3047429714 -6.706990e-01
    ## 369   0.428278296  1.180626608 -1.578871e-01  0.0904554285 -5.392570e-01
    ## 370  -0.280540421 -0.722749112  9.782224e-02 -0.0785929502 -5.776083e-01
    ## 371   0.806984912 -0.140704184 -3.619158e-01 -1.6736351818  9.000992e-01
    ## 372  -0.276174939 -0.697708395  3.356313e-01 -0.0171963656 -3.249037e-01
    ## 373  -0.079039651  0.022980930 -1.652439e-01 -1.2357459648  2.118210e-01
    ## 374   0.048864326 -0.201702075 -6.415122e-03  0.5767337833  1.242516e-01
    ## 375  -0.176634733 -0.221759430 -1.421902e-01  0.0669557061  1.079470e-02
    ## 376  -0.012513271  0.293051483  1.533219e-01 -0.0725086811 -4.950863e-01
    ## 377   1.232808418 -1.243539546  1.539385e-01 -0.1710237487 -4.441241e-02
    ## 378   0.085141232  0.344870517 -3.767388e-01 -1.3023262498  5.763578e-02
    ## 379  -0.127988273 -0.571537037  7.711988e-02  0.4780350697 -2.709888e-01
    ## 380   0.475654419  0.953624152 -2.635200e-01 -0.2733569385  3.863913e-01
    ## 381  -0.082580566 -0.546705013 -1.865878e-02 -0.0814516176  2.102223e-01
    ## 382  -0.003294463  0.116648946  1.381225e-02  0.5268679852  3.324184e-01
    ## 383   0.374478596  0.979091984 -2.527057e-01  0.5082343200  6.828556e-01
    ## 384  -0.455865206 -0.749952532  1.246030e-02 -0.4294712283 -7.278147e-01
    ## 385  -0.114873200 -0.359716354 -3.012332e-01 -1.2026524012  3.336190e-01
    ## 386   0.125412524  0.366466507 -8.548002e-02 -0.1057598623  4.583330e-01
    ## 387   0.044586112  0.494962470  3.250795e-01  0.4208312145 -1.341755e-01
    ## 388  -0.090016070 -0.484435145 -3.209214e-01 -0.8174564604  7.265971e-01
    ## 389  -0.358459761 -0.616086834  4.076115e-01  0.0092749230 -5.927501e-01
    ## 390  -0.561890569 -1.413091479 -1.032215e-01  0.6287368588  4.340801e-01
    ## 391  -0.149599596 -0.357803687  3.981136e-01 -0.0933457093 -4.999680e-01
    ## 392   0.082470510  0.352414896 -1.639341e-01 -1.0210052767  5.388266e-01
    ## 393  -0.370188561 -0.940937264  3.155124e-01  0.3898331657 -2.238529e-01
    ## 394   0.044251991 -0.820721203  8.728426e-01  0.9454355592  5.607447e-01
    ## 395  -0.543649976 -0.824464961  3.120382e-01  0.3961197234 -1.128498e-01
    ## 396  -0.002120095  0.328462605 -6.988549e-01 -0.3765793938  1.307757e+00
    ## 397  -0.140088714 -0.439921602  7.933278e-02  0.0512625742 -6.771902e-01
    ## 398  -0.183428788 -0.196393254 -2.161082e-02 -0.2947233533  3.675131e-01
    ## 399  -0.364218087 -0.442433282  5.179696e-01 -0.3636685961 -4.090803e-01
    ## 400  -0.063862819 -0.114698217  2.076651e-01 -1.7105204282  5.050505e-01
    ## 401   0.915172069 -0.258077424  3.937856e+00  0.2001345653 -1.511010e-01
    ## 402   0.192253103  0.862400720 -4.366194e-03 -1.5717976598  1.940799e-02
    ## 403   1.216858326 -1.562823308  2.386408e-01 -0.9903197826 -6.324344e-02
    ## 404   0.425471678  0.971694921  1.808100e-01  0.2287199914 -1.325112e-01
    ## 405  -0.203553701 -0.383324855 -2.713482e-02 -0.1312234722  2.093586e-01
    ## 406   0.049529134  0.189631503 -2.182026e-01 -0.0086013948  2.956440e-01
    ## 407  -0.218516825 -0.528320470 -1.226843e-01  0.9960502818  6.976371e-01
    ## 408  -0.237195501 -0.734112309  6.341502e-02 -0.3197349748 -7.281360e-01
    ## 409  -0.224595912 -0.403483481  2.808509e-01  0.0185985524 -7.404256e-01
    ## 410   0.104022799  0.362430158 -1.455227e-01  0.4227045933  5.203054e-01
    ## 411   0.097521309  0.328867493 -7.714715e-02  0.0546623529  3.681448e-01
    ## 412   0.264301098  1.018884139 -1.020276e-01  0.6712233863 -3.122865e-01
    ## 413   0.490571794  1.361779557 -2.457923e-01  0.4222913906  7.621847e-01
    ## 414   0.029986858  0.164511462 -2.589402e-01  0.4207859529  4.776848e-01
    ## 415   0.008743748 -0.217083694  9.095867e-02  0.5314458383  1.784262e-01
    ## 416  -0.191195037 -0.501352621  2.986421e-03 -0.0460565796 -2.799758e-01
    ## 417  -0.095876066 -0.200525442 -6.937801e-02 -0.0218777928  4.048147e-01
    ## 418   0.194544957  0.471470358 -3.221265e-02  0.3903770603 -3.474012e-01
    ## 419   0.223870505  0.363600615 -1.590692e-01  0.2178451417  4.465411e-01
    ## 420   0.198971290  0.750514428 -1.108435e-01  0.3038108723  3.619821e-02
    ## 421  -0.324288058 -0.401027921  2.142890e-01 -0.0069650829  3.061140e-01
    ## 422   0.509522241 -0.206599978 -4.698808e-01  0.1432532439 -6.603090e-01
    ## 423   0.233753078  0.861085133 -1.068641e-01  0.7885436390 -1.193120e-01
    ## 424  -0.073298527  0.133256073 -1.281078e-01 -0.1136315965  3.594551e-01
    ## 425   0.254276726  0.850971648  1.522973e-02  0.7146235192  3.377725e-01
    ## 426  -0.091052449 -0.279001763 -1.915428e-02  0.5402919785 -2.680269e-01
    ## 427   0.125525054  0.509420209  4.805043e-03  0.7410839692  3.599082e-01
    ## 428  -0.429691167 -0.781796749  2.693861e-01 -1.4409296426 -4.429279e-01
    ## 429   0.211637203  0.637707905 -2.541502e-01 -0.3925695478  4.802937e-02
    ## 430  -0.198922349 -0.538313085 -6.034635e-03 -0.3650472486 -2.192554e-01
    ## 431  -0.119933584 -0.152862547  6.176712e-01  0.3867769210 -1.448586e+00
    ## 432   0.083980666  0.590252792  2.664489e-03  0.4086922630  3.329261e-01
    ## 433   0.159537064 -0.367362212 -2.880686e-01 -0.2070862534 -2.326605e-01
    ## 434   0.034044962 -0.426103854 -2.514237e-01 -0.1298321783  4.928451e-02
    ## 435  -0.582835436 -0.767617301  1.938239e+00  0.4031474068 -5.800627e-01
    ## 436  -0.131666243 -0.164801863 -2.994027e-01 -0.5135815759  5.150376e-01
    ## 437   0.058077616  0.402050939 -1.207275e-01 -0.1463622935  4.230930e-01
    ## 438  -0.327056431 -0.809098702  1.401467e-01  0.4493531695 -8.163307e-01
    ## 439  -1.285796956  0.450085035  8.354614e-01  0.6367883507  5.180148e-01
    ## 440   0.231689518 -0.096148874 -4.370129e-01  0.6292164842  4.631489e-01
    ## 441   1.489153785  0.182196078  2.932408e-01  0.6907781850  1.338774e-01
    ## 442   0.065334735  0.317229824  1.478764e-01 -0.7179778946  1.030381e-01
    ## 443   0.042142218  0.288637272  2.677140e-01  0.3655015671 -3.456132e-01
    ## 444   0.117753123  0.340351514  2.762481e-02  0.7006464279  5.106943e-01
    ## 445   0.513948048  1.267439157 -2.013722e-01  0.0886458431 -5.687130e-01
    ## 446   0.185470239  0.881300695 -4.279949e-01 -0.2671966294  4.630826e-01
    ## 447   0.231895249  0.635264980 -1.986829e-02  0.6529819108  3.513999e-01
    ## 448   0.357012155 -0.165928805  1.107220e+00  0.3702578568 -3.960206e-01
    ## 449  -0.172520630 -0.708537303  1.527718e+00 -0.0658316766  1.758098e-01
    ## 450  -0.075630337 -0.454013351  1.137472e-01 -0.0702126044 -1.994479e-01
    ## 451  -0.004022737  0.251580159 -5.108205e-01 -0.1620132290  1.254277e+00
    ## 452   0.398629958  1.631757621 -1.115443e-01  0.0785533161 -1.177482e+00
    ## 453  -0.002563673  0.114063988 -1.319448e-01  0.2081470682  1.985715e-01
    ## 454   0.050230206 -0.007553787 -2.716718e-02  0.5047859322 -5.807026e-02
    ## 455   0.180694471  0.830196849 -2.643722e-01 -0.4303854527 -2.391897e-01
    ## 456   0.517232371 -0.035049369 -4.652111e-01  0.3201981985  4.451917e-02
    ## 457   0.661695925  0.435477209  1.375966e+00 -0.2938031527  2.797980e-01
    ## 458  -0.294166318 -0.932391057  1.727263e-01 -0.0873295380 -1.561143e-01
    ## 459   0.573574068  0.176967718 -4.362069e-01 -0.0535018649  2.524053e-01
    ## 460  -0.379068307 -0.704181032 -6.568048e-01 -1.6326529569  1.488901e+00
    ## 461   0.364514210 -0.608057134 -5.395279e-01  0.1289399830  1.488481e+00
    ## 462   0.370508651 -0.576752473 -6.696054e-01 -0.7599075295  1.605056e+00
    ## 463   0.156617169 -0.652450441 -5.515722e-01 -0.7165216354  1.415717e+00
    ## 464   0.208828369 -0.511746619 -5.838132e-01 -0.2198450291  1.474753e+00
    ## 465   0.589669127  0.109541319  6.010453e-01 -0.3647002782 -1.843078e+00
    ## 466   0.551179689 -0.009802357  7.216982e-01  0.4732457514 -1.959304e+00
    ## 467   0.343282649 -0.054195664  7.096540e-01 -0.3722158669 -2.032068e+00
    ## 468   0.501543149 -0.546868812 -7.658364e-02 -0.4255503668  1.236442e-01
    ## 469   0.454031932 -0.577525824  4.596654e-02  0.4617000176  4.414616e-02
    ## 470   0.375026285  0.145399853  2.406033e-01 -0.2346492227 -1.004881e+00
    ## 471   0.615641835 -0.406427275 -7.370181e-01 -0.2796417722  1.106766e+00
    ## 472   0.536892074 -0.546126024 -6.052395e-01 -0.2637434596  1.539916e+00
    ## 473   0.743314023  0.064038356  6.778417e-01  0.0830078227 -1.911034e+00
    ## 474   0.756052550  0.140167769  6.654111e-01  0.1314637917 -1.908217e+00
    ## 475   0.645103276 -0.503529449 -5.228218e-04  0.0716957788  9.200743e-02
    ## 476   0.667926572 -0.516242362 -1.221781e-02  0.0706137031  5.850447e-02
    ## 477   0.481829697  0.146023056  1.170385e-01 -0.2175645988 -1.387760e-01
    ## 478   0.734774950 -0.435901182 -3.847659e-01 -0.2860156638  1.007934e+00
    ## 479   0.716719907 -0.448059938 -4.024069e-01 -0.2888352226  1.011752e+00
    ## 480  -1.052368256  0.204816874 -2.119007e+00  0.1702786083 -3.938441e-01
    ## 481   1.646518293 -0.278484511 -6.648413e-01 -1.1645553694  1.701796e+00
    ## 482   0.149895683 -0.601967246 -6.137243e-01 -0.4031136878  1.568445e+00
    ## 483   1.865678768  0.407809282  6.058094e-01 -0.7693481181 -1.746337e+00
    ## 484   1.757085227 -0.189709153 -5.086292e-01 -1.1893081688  1.188536e+00
    ## 485   0.316093599  0.055179227  2.106922e-01 -0.4179178814 -9.111883e-01
    ## 486   0.573898081 -0.080162775  3.184078e-01 -0.2458622027  3.382384e-01
    ## 487   0.650988236  0.254983289  6.288435e-01 -0.2381284543 -6.713323e-01
    ## 488   1.688665431 -0.078844698  1.937309e-01  0.4794959539 -5.066031e-01
    ## 489   1.887737726  0.333998185  2.876590e-01 -1.1864064091 -6.902727e-01
    ## 490   2.002882877  0.351102065  7.952551e-01 -0.7783793814 -1.646815e+00
    ## 491   1.846164793 -0.267171794 -3.108040e-01 -1.2016854580  1.352176e+00
    ## 492   1.976988396  0.256510487  4.859081e-01 -1.1988213144 -5.265674e-01
    ## 493   1.990545351  0.223785003  5.544084e-01 -1.2040415962 -4.506846e-01
    ## 494   1.963596666 -0.217413916 -5.493400e-01  0.6455452020 -3.545578e-01
    ## 495  -0.423553904 -0.800852285  7.761446e-02  0.1676084526  3.501818e-01
    ## 496  -0.502635558 -1.047397528 -5.675202e-02 -0.3406884048  5.412352e-01
    ## 497   0.159387389  0.592669569 -5.359618e-02  0.3207475021 -3.691208e-01
    ## 498   2.004109945  0.191058399  6.229277e-01 -1.2092635574 -3.747989e-01
    ## 499   2.086083001  0.760190241  7.168058e-01 -0.6467434296 -1.617043e+00
    ## 500  -1.159829951 -1.504118882 -1.925433e+01  0.5448667260 -4.781606e+00
    ## 501  -2.475962229  0.342390706 -3.564508e+00 -0.8181401900  1.534077e-01
    ## 502  -2.089609633  1.745314500  1.376816e+00 -0.5542714184 -1.610741e+00
    ## 503  -2.444883682  0.727495341 -3.450782e-01 -0.9817485515  9.952713e-01
    ## 504  -2.366836062  1.130955198  9.911532e-01 -1.0331321071 -3.271793e-01
    ## 505  -2.362344928  1.099557296  1.037199e+00 -1.0363593418 -2.547765e-01
    ## 506  -2.356896279  1.068019177  1.085617e+00 -1.0397970780 -1.820061e-01
    ## 507  -2.350633745  1.036361874  1.136051e+00 -1.0434137405 -1.089233e-01
    ## 508  -2.343673594  1.004602414  1.188212e+00 -1.0471837684 -3.557277e-02
    ## 509  -2.336110956  0.972754726  1.241866e+00 -1.0510862482  3.800908e-02
    ## 510  -2.328024416  0.940830319  1.296817e+00 -1.0551039036  1.117920e-01
    ## 511  -2.319479462  0.908838783  1.352904e+00 -1.0592223376  1.857509e-01
    ## 512  -0.377503243 -0.889596748 -7.420759e-02  0.0354455164  5.505780e-01
    ## 513   1.577548225 -1.280137123 -6.012945e-01  0.0404039215  9.955020e-01
    ## 514   1.725852827 -1.151605776 -6.800519e-01  0.1081759145  1.066878e+00
    ## 515   1.729804092 -1.208096082 -7.268392e-01  0.1125396735  1.119193e+00
    ## 516  -0.166736918 -0.521933839 -1.123763e-01 -0.5920767505  5.207913e-01
    ## 517   1.741135596 -1.251137949 -3.962191e-01  0.0957055556  1.322751e+00
    ## 518   1.746801542 -1.353148766 -7.629651e-01  0.1170278519  1.297994e+00
    ## 519   1.750729840 -1.409635709 -8.098093e-01  0.1213966599  1.350300e+00
    ## 520   1.754607609 -1.466115253 -8.567786e-01  0.1257765765  1.402587e+00
    ## 521   1.758440075 -1.522588162 -9.038602e-01  0.1301664525  1.454857e+00
    ## 522   1.762231769 -1.579055101 -9.510427e-01  0.1345652919  1.507110e+00
    ## 523   1.765986636 -1.635516647 -9.983165e-01  0.1389722274  1.559350e+00
    ## 524   1.769708129 -1.691973305 -1.045673e+00  0.1433864999  1.611577e+00
    ## 525   1.773399282 -1.748425521 -1.093104e+00  0.1478074424  1.663792e+00
    ## 526   1.777062772 -1.804873686 -1.140605e+00  0.1522344664  1.715997e+00
    ## 527   1.780700970 -1.861318147 -1.188167e+00  0.1566670507  1.768192e+00
    ## 528   1.784315984 -1.917759213 -1.235787e+00  0.1611047318  1.820378e+00
    ## 529   0.603127128  0.380690182 -6.412518e-02  0.2713801006  3.372195e-01
    ## 530   1.796826389 -1.960973717 -9.022475e-01  0.1440114422  2.024388e+00
    ## 531   0.882940442 -0.246201930  1.752227e+00  0.2199249645  1.562817e-01
    ## 532   1.802149134 -2.062934276 -1.269843e+00  0.1654091885  1.999499e+00
    ## 533   1.805769784 -2.119376168 -1.317450e+00  0.1698456305  2.051687e+00
    ## 534   1.809370933 -2.175815203 -1.365104e+00  0.1742863596  2.103868e+00
    ## 535   1.812953975 -2.232251588 -1.412803e+00  0.1787310696  2.156042e+00
    ## 536   1.816520171 -2.288685505 -1.460544e+00  0.1831794827  2.208209e+00
    ## 537  -0.106994396 -0.250050304 -5.216265e-01 -0.4489495142  1.291646e+00
    ## 538  -0.063167877 -0.207385284 -1.832608e-01 -0.1036793734  8.961784e-01
    ## 539   0.371121111 -0.322289520 -5.498559e-01 -0.5206289588  1.378210e+00
    ## 540   0.899931176  1.481271004  7.252655e-01  0.1769595829 -1.815638e+00
    ## 541   0.546589340  0.334970737  1.721057e-01  0.6235902880 -5.271143e-01
    ## 542   0.262201510 -0.633527564  9.289101e-02  0.1876126556  3.687084e-01
    ## 543   1.026421111  0.299613502  1.656800e+00  0.3284325018  1.064568e-01
    ## 544   0.371773459  0.111954592 -3.052251e-01 -1.0538352624  7.711749e-01
    ## 545  -0.159387124 -0.305154013  5.361974e-02  0.0117614442  3.751458e-01
    ## 546   0.469211945 -0.144362651 -3.179809e-01 -0.7696443095  8.078546e-01
    ## 547   1.774460092 -0.771389808  6.572670e-02  0.1039157449 -5.757793e-02
    ## 548   1.192693540  0.090355811 -3.418809e-01 -0.2159241513  1.053032e+00
    ## 549   2.839596108 -1.185442841 -1.428119e-01 -0.0861029384 -3.291133e-01
    ## 550   1.807877250 -0.890421395 -3.258144e-01  0.1230395969 -9.301375e-02
    ## 551   1.483594216  0.834310713 -1.484856e-01  0.0016688545 -3.899573e-02
    ## 552   1.080322734 -0.561384150  1.026784e-01 -0.0671947228 -4.769312e-01
    ## 553   1.582555933  0.778710245 -1.357073e-01 -0.0042778003  3.270616e-02
    ## 554   1.691042030  0.920020577 -1.511039e-01  0.0110068782  8.030327e-02
    ## 555   1.550472966  0.614572855  2.852079e-02  0.0137043245 -1.495120e-01
    ## 556   1.194888356 -0.845752532  1.906739e-01 -0.2164432525 -3.250331e-01
    ## 557   1.128641074 -0.962959729 -1.100451e-01 -0.1777332408 -8.917542e-02
    ## 558  -0.734307917 -0.599926260 -4.908301e+00  0.4101702346 -1.167660e+00
    ## 559   0.447180122  0.536203895  1.634061e+00  0.2038388491  2.187489e-01
    ## 560  -0.311999618 -0.639699711 -1.202490e-01 -0.1802176945  6.092826e-01
    ## 561  -0.176541019 -0.433469859 -5.293232e-01 -0.5970196050  1.335954e+00
    ## 562   0.247912823 -0.049586179 -2.260170e-01 -0.4012356010  8.561236e-01
    ## 563   0.148284202  0.721099541  2.661291e+00 -0.5086199335 -4.016566e-01
    ## 564   1.854266661 -0.165533726 -3.399390e-01  0.2963137625  1.364225e+00
    ## 565   2.452338505 -0.292962575 -1.893303e-01 -0.1664821043  3.803962e-02
    ## 566   2.417495412 -0.097711951  3.821545e-01 -0.1547565198 -4.039559e-01
    ## 567   2.263770356  0.620749345 -9.406867e-02  0.5367190874  3.981423e-01
    ## 568   2.248971226  0.566843572  3.374405e-02  0.5917834080  3.342288e-01
    ## 569   2.331465808  0.862996306 -6.144532e-01  0.5236478771 -7.125933e-01
    ## 570   2.014271566 -0.167416618  4.996753e-02  0.3844297190 -7.788357e-02
    ## 571   2.158143084  0.111510363  2.164138e-01  0.5846612555  7.603595e-01
    ## 572   2.153755026  0.033921618 -1.409492e-02  0.6252499553 -5.339033e-02
    ## 573   2.557943956  0.926277883  3.279494e-02  0.6380725857  3.618872e-01
    ## 574   2.427460374  0.692666548  2.030517e-02  0.4998090931  4.675936e-01
    ## 575   2.192855420 -0.282596815  8.068498e-03  0.4038583370 -1.878839e-02
    ## 576   1.775890900 -1.224757612  8.259354e-02  0.4520887274  4.638273e-01
    ## 577   2.571969593  0.206809359 -1.667801e+00  0.5584187540 -2.789788e-02
    ## 578   2.479413508  0.366932842  4.280465e-02  0.4782788918  1.577708e-01
    ## 579   2.267448261 -0.492029168 -2.393029e-01  0.4543682102 -1.016111e-01
    ## 580   2.133456284 -1.271508967 -3.530389e-02  0.6150536950  3.490238e-01
    ## 581   2.679490070 -0.047335106 -8.369823e-01  0.6253493282  1.258649e-01
    ## 582   2.549628235 -0.532227764 -2.350965e-01  0.6732091695  2.265980e-01
    ## 583   2.989552509  0.497598717 -5.092903e-01  0.7325034567  2.805280e-01
    ## 584   3.058081838  0.941180290 -2.327103e-01  0.7635080193  7.545624e-02
    ## 585   2.525115463 -0.832074337 -1.861168e-01  0.4297808337  6.971028e-01
    ## 586   2.714044802 -0.101355311 -4.396660e-01  0.5195138615  7.893280e-01
    ## 587   3.147428105  0.341677595 -1.150162e+00  0.7951897350 -1.945422e-01
    ## 588   1.190738695 -1.127670009 -2.358579e+00  0.6734613290 -1.413700e+00
    ## 589   0.549014458  0.624321411 -1.366627e-01  0.1317379187  3.092137e-02
    ## 590   0.598842606  0.615318868 -4.864989e-01  0.7392684684 -2.368448e-01
    ## 591   2.601441029  0.231910116 -3.648985e-02  0.0426396452 -4.383303e-01
    ## 592   0.144653280 -0.885681640  6.250986e-01  0.0965268919 -1.894549e-01
    ## 593   0.912700322 -0.630358133  1.908867e-01 -0.0612634320  3.797747e-01
    ## 594  -2.182691946  0.520543072 -7.605564e-01  0.6627666384 -9.484543e-01
    ## 595  -1.977196194  0.652931885 -5.197769e-01  0.5417017301 -5.386076e-02
    ## 596   2.674465879 -0.020879980 -3.024471e-01 -0.0863964000 -5.160598e-01
    ## 597   0.027934968  0.220366441  9.763479e-01 -0.2905394154  1.161002e+00
    ## 598  -1.277811583  0.719652001  4.511252e-01 -0.2580936506  6.561295e-01
    ## 599   1.030738252  0.165327776 -1.017502e+00 -0.4779833808 -3.049868e-01
    ## 600   0.698359061  0.487477721  1.228698e+00 -0.5352173332  3.882781e-01
    ## 601   0.641593528  0.841754905  1.767278e-01  0.0810041185 -2.588987e-01
    ## 602   2.070008164 -0.512625728 -2.485017e-01  0.1265496243  1.041663e-01
    ## 603   0.490182668  0.470427403 -1.262613e-01 -0.1266444588 -6.619076e-01
    ## 604   2.698174895 -0.027080891  3.667752e-01 -0.1230114417 -3.004565e-01
    ## 605   0.220670430  0.912107214 -2.863377e-01  0.4512082465  1.883153e-01
    ## 606   1.186035605 -0.040214870 -2.389301e-01  0.1101443260  4.541837e-02
    ## 607  -0.334417439 -1.014314673 -1.284273e-01 -0.9462418376  4.560903e-01
    ## 608   1.307871432  0.102825644 -1.774628e-02  0.1496959257 -9.660247e-02
    ## 609   0.533521286 -0.022179953 -2.995559e-01 -0.2264161671  3.643605e-01
    ## 610   0.609508182  0.202874285 -6.079139e-02 -0.1867333534 -1.740123e-02
    ## 611   0.070632869  0.192490772 -1.746593e-01 -0.4389080729  2.392590e-01
    ## 612  -0.062166000 -0.128168130 -4.017599e-02  0.1100403475  4.378914e-01
    ## 613  -0.083734377 -0.346930257 -5.061871e-02  0.2310444246 -4.507599e-01
    ## 614   0.402729681 -0.132128892 -3.297718e-02  0.4608608368  5.604038e-01
    ## 615   0.560475066  0.165681718 -1.375352e-02  0.4749351803 -2.187251e-01
    ## 616  -0.392667222  0.440019839  7.776592e-01  0.4185519356  2.445632e-01
    ## 617  -0.051659886 -0.084089409 -1.928464e-01 -0.9173919855  6.819528e-01
    ## 618   0.696954881  0.740003045 -1.551152e-01 -0.0506074461  2.683683e-01
    ## 619   0.493436339  0.733393182  2.023500e-01  0.4920537009 -1.837909e-01
    ## 620   0.918244481 -0.715366337  2.107473e-01 -0.0602114737  5.095346e-01
    ## 621   0.926156961 -0.817706132 -1.504344e-01 -0.0393830597  4.856398e-01
    ## 622   0.931958309 -0.874467361 -1.926395e-01 -0.0354260266  5.386651e-01
    ## 623   0.937415596 -0.931178206 -2.356967e-01 -0.0313933541  5.915583e-01
    ## 624   0.942593331 -0.987848115 -2.794463e-01 -0.0272992247  6.443442e-01
    ## 625   0.610653921  0.835795007  1.179955e+00 -0.0290908864 -3.008957e-01
    ## 626  -0.050107622  0.123760576 -1.325679e-01  0.3502305536  5.077014e-01
    ## 627   0.469199426  0.344930464 -2.037994e-01  0.3766403785  7.154855e-01
    ## 628  -0.112113510 -0.220002032 -1.210219e-01 -0.4404544484  6.715400e-01
    ## 629   0.594623132  0.372143666 -3.104558e-01 -0.6240651488  8.402161e-01
    ## 630   0.383180108 -0.213952218 -3.366398e-01  0.2370763008  2.460029e-01
    ## 631   0.679176431  0.731907051  3.330455e-01  0.3925052360 -2.741973e-01
    ## 632  -0.723326196  0.501221637 -6.968923e-01 -0.6005137264  1.275470e-01
    ## 633  -0.875146364 -0.509848661  1.313918e+00  0.3550647314  4.485516e-01
    ## 634   1.177852408  0.175331359 -1.211123e+00 -0.4468910817 -4.055204e-01
    ## 635   1.061314182  0.125737469  5.895923e-01 -0.5687306045  5.828247e-01
    ## 636  -0.175273370  0.543325052 -5.479551e-01 -0.5037216901 -3.109328e-01
    ## 637  -0.299847071  0.610478814  7.890228e-01 -0.5645119395  2.011958e-01
    ## 638   1.101671317 -0.992493707 -6.982589e-01  0.1398976735 -2.051508e-01
    ## 639   1.208054260  0.277611797  1.926580e-02  0.5085288312 -2.011829e-01
    ## 640   0.077739051  1.092436595  3.201327e-01 -0.4346425529 -3.806871e-01
    ## 641  -0.090526788  0.348589628  5.113239e-02 -0.4154298724  2.196653e-01
    ## 642   0.285831618 -0.771508089 -2.651998e-01 -0.8730771501  9.397763e-01
    ## 643   0.794372452  0.270471232 -1.436241e-01  0.0135659145  6.342031e-01
    ## 644   0.652941051  0.081930976 -2.213478e-01 -0.5235821592  2.242282e-01
    ## 645   0.785540122  0.297411984  3.085361e-01 -0.5984157464 -1.218503e-01
    ## 646  -2.457144873  1.687256642  9.771784e-01 -0.5433689766 -2.891248e-01
    ## 647  -0.527474204  0.220546095 -1.371110e+00 -0.5048989945  3.823068e-01
    ## 648  -1.394503794 -0.166029307 -1.452081e+00 -0.2518146707  1.243461e+00
    ## 649   0.299768816 -0.583282924 -1.876960e-01 -0.3292557532  7.323281e-01
    ## 650   0.119278777  0.513479450 -2.642434e-01  0.4433112880  2.951639e-02
    ## 651   0.614968997 -0.195199717  5.907113e-01 -0.2333776639 -1.642853e-01
    ## 652  -0.575924092  0.495889441  1.154128e+00 -0.0161857865 -2.079928e+00
    ## 653   2.076382721 -0.990302787 -3.303585e-01  0.1583776258  6.351446e-03
    ## 654   1.226648399 -0.695902073 -1.478490e+00 -0.0615525997  2.361550e-01
    ## 655   1.224794983 -0.656639147 -3.308107e-01 -0.0789460153  2.703059e-01
    ## 656   0.556894674  0.169775529 -1.743571e-01  0.3080613230  7.109961e-01
    ## 657   0.436390207 -0.077552723 -3.091624e+00 -0.3902008857 -2.886889e-01
    ## 658   0.401341241  0.152191259 -9.346755e-01 -0.2561483824 -4.694033e-01
    ## 659   0.578984296  1.397310820  1.045322e+00 -0.3039998693  5.294959e-03
    ## 660   0.320473876  0.611026686  1.748637e-01 -0.5021511011 -1.747128e-01
    ## 661   0.332216176  0.493980865 -8.019829e-02 -0.2533273378 -4.777997e-01
    ## 662   0.417762091 -0.648575872 -3.186174e-01 -0.6804129318  3.898687e-01
    ## 663   0.662933040  0.184086858 -8.945180e-02 -0.5059999351 -6.225889e-02
    ## 664  -0.377597095 -0.793459660 -1.323332e-01 -0.3315862690  6.648779e-01
    ## 665   0.764265625  0.473261684  5.484817e-01 -0.1568500518 -7.101869e-01
    ## 666   0.276010822  1.342044557 -1.016579e+00 -0.0713612974 -3.358693e-01
    ## 667   0.340330753  0.760169762  3.533770e-01 -0.7788927426 -7.068116e-02
    ## 668   0.868339851  0.793736498  2.173466e-01 -0.0219850652  1.458819e-01
    ## 669   1.079871408 -0.352026410 -2.183577e-01  0.1258656840 -7.418018e-02
    ## 670  -0.023254785 -0.158600834 -3.880616e-02 -0.0603272547  3.583389e-01
    ## 671  -0.008996174 -0.057036144 -5.369151e-02 -0.0263732768  4.002997e-01
    ## 672   0.102912554  0.311626102 -4.129195e+00  0.0346387322 -1.133631e+00
    ## 673  -0.049502428  0.207265101 -2.652719e-01 -0.6792937702  5.118115e-01
    ## 674  -0.511656805 -0.122724086 -4.288639e+00  0.5637966578 -9.494514e-01
    ## 675  -0.026054722 -0.295254768 -1.804585e-01 -0.4365386632  4.946491e-01
    ## 676   0.351792126  0.391248950 -2.528749e-01 -0.4980424801  1.017219e-02
    ## 677   0.352456414 -0.243677652 -1.940785e-01 -0.1722010875  7.422372e-01
    ## 678   0.838760421  0.341727096  9.475065e-01 -0.1454930025  4.932577e-02
    ## 679   0.042619457  0.397223782  7.222887e-02 -0.2422759966  5.609165e-01
    ## 680  -0.665172079 -0.632078029 -4.211758e-01 -0.4007743529 -1.640302e-03
    ## 681  27.202839157 -8.887017141  5.303607e+00 -0.6394348023  2.632031e-01
    ## 682  -0.448670517 -0.517567968  1.283334e-02  0.6992167280  5.272580e-01
    ## 683  -0.367136001 -0.891627121 -1.605781e-01 -0.1083259620  6.683743e-01
    ## 684   0.764186725 -0.275577693 -3.435715e-01  0.2330848293  6.064336e-01
    ## 685   0.345920750 -0.108001595 -1.654419e-01  0.2798945800  8.087831e-01
    ## 686   0.729827756  0.485285915  5.670054e-01  0.3235856916  4.087085e-02
    ## 687   1.096342010  0.658398655  1.711676e+00  0.3335400364  5.385914e-01
    ## 688  -1.328131789  0.189310618 -5.524184e-03 -0.8147077395  4.009243e-01
    ## 689   1.092051134 -0.041080429  9.043952e-01  0.1800156016  4.997034e-02
    ## 690   0.801312236 -0.183000576 -4.403869e-01  0.2925394119 -1.449674e-01
    ## 691   0.862912790  0.927825035 -3.430581e-01 -0.2562682304 -6.007422e-01
    ## 692   0.528421373  0.228027081 -7.492980e-01 -0.0671789627  2.157925e-01
    ## 693   0.855138263  0.774744821  5.903715e-02  0.3431998079 -4.689379e-01
    ## 694   0.256559618 -0.466245243  2.911049e-01  0.2425665528 -1.279094e+00
    ## 695  -0.016378018 -0.207609058 -1.641192e-01  0.2552801621  4.547978e-01
    ## 696  -1.004877379  1.150353588 -1.525553e-01 -1.3867446332  4.716015e-03
    ## 697   1.049731562  0.475840283  4.044801e-01  0.2820297897 -5.069011e-01
    ## 698   1.168617819  0.289530904 -3.718882e-01  0.1447612766  8.473483e-02
    ## 699   1.131130282  0.118021510 -3.327042e-01  0.1399407339  3.247579e-01
    ## 700   0.949568986 -0.428985750 -3.506755e-01  0.1975495036  1.592342e-01
    ## 701   1.215976106  0.041177744 -1.059098e+00  0.2756620623  5.742493e-02
    ## 702  -0.366507065 -0.714464901 -1.439107e-01 -0.3051776195  6.975141e-01
    ## 703  -0.387894981 -0.866811936 -1.215826e-01 -0.3561085863  6.345728e-01
    ## 704  -0.325283765 -0.734343618 -1.067253e-01 -0.2249990645  5.691673e-01
    ## 705  -0.332983054 -0.851270428 -3.707998e-01  0.2982416873  4.429299e-01
    ## 706   0.130748608  0.239389294 -9.022741e-02  0.4115723905 -2.161260e-01
    ## 707  -0.346373780 -0.663587986 -1.023258e-01  0.0179105750  6.503016e-01
    ## 708   0.155380927 -0.614880422 -1.961255e-01 -0.4643755434  1.184731e-01
    ## 709  -0.312862942 -0.687873797 -2.670025e-01 -1.1584800633  2.714600e-01
    ## 710   1.303250309 -0.016118152 -8.766699e-01  0.3822298006 -1.054624e+00
    ## 711   0.550540662 -0.067870220 -1.114692e+00  0.2690694332 -2.057246e-02
    ## 712  -1.032934652  1.196428310 -1.128567e-01  0.2547189932  6.966679e-01
    ## 713   0.746159905  0.550802012 -3.488234e-02 -0.5676084045 -5.283180e-01
    ## 714   0.174099466 -0.272505290 -3.154945e-02 -0.4061659297  1.577690e-01
    ## 715   0.713558822 -0.408953596 -3.208895e-01 -0.8042302752  9.628516e-01
    ## 716   1.066550249 -0.521657289 -3.199173e-01 -0.4058590881  9.068022e-01
    ## 717   1.376938323 -0.792016726 -7.714142e-01 -0.3795742715  7.187165e-01
    ## 718   1.269205389  0.057657252  6.293074e-01 -0.1684317735  4.437439e-01
    ## 719   0.325575470  0.014001845  8.449460e-01  0.1149628845  1.563648e-01
    ## 720   0.163966694  1.245648321 -2.692413e-01  0.5371021194 -2.207567e-01
    ## 721   1.103397745 -0.541854752  3.694322e-02 -0.3555190041  3.536344e-01
    ## 722   0.954272202 -0.451086449  1.272144e-01 -0.3394498693  3.940955e-01
    ## 723   0.911427442  0.053061281 -3.677010e-01 -0.7763377558  2.920938e-01
    ## 724   0.920899034  0.037674509  2.675398e-02 -0.7914888237  1.764928e-01
    ## 725   0.875260063 -0.102501391 -6.062827e-01 -0.7431647943  9.631894e-02
    ## 726   1.024422834  0.428756482  1.820321e-01 -0.5345978827  1.689335e-01
    ## 727  -2.504450187  1.436472378  3.515424e-01  0.6484669043  5.796814e-01
    ## 728   0.480639718  0.533516931  1.284645e+00  0.5161312704 -6.029411e-01
    ## 729  -0.936989532 -0.053811586  5.801059e-01  0.2169273025  1.516429e-01
    ## 730   1.664119015  0.785075272  6.841249e-02  0.7789609160 -8.631659e-01
    ## 731   0.015255354  0.239994342 -1.119160e-01 -0.3805758899  3.709948e-01
    ## 732  10.005998013 -2.454964095  1.684957e+00  0.1182630831 -1.531380e+00
    ## 733  -0.095308307  0.946628918 -2.974030e-01 -0.3683438912  1.987312e-01
    ## 734   1.410478565  0.279402821  5.708212e-01  0.6562710148 -2.986005e-01
    ## 735  -0.563944162 -0.902099515 -4.043824e-01 -0.0129439005  5.898362e-01
    ## 736  -0.474437305 -0.974624666 -4.815450e-02 -0.0235242679  3.621920e-01
    ## 737   8.280439326 -2.797149541  1.090707e+00 -0.1592595975  5.321564e-01
    ## 738   0.825951491  1.144169790  2.085595e-01 -0.2954974245 -6.902322e-01
    ## 739   0.450381285  0.521161530  3.083254e-01 -0.3180118407 -1.255362e+00
    ## 740   2.102343468  0.597370315 -3.280862e-01  0.4457524024  5.852806e-01
    ## 741   1.990519705  0.083353322 -6.226355e-02  0.3901878021  3.298839e-01
    ## 742   2.309880169  0.978660126 -9.613014e-02  0.4323767225 -4.356279e-01
    ## 743   2.245605934  0.546320675  3.818534e-01  0.3820245440 -8.210365e-01
    ## 744   2.502772410  0.481691290  4.809580e-01  0.3603186388 -2.933543e-01
    ## 745   2.462055911  1.054865184  5.304806e-01  0.4726698140 -2.759980e-01
    ## 746   1.909032018 -0.348739731  4.250006e-01  0.6749086698 -7.842084e-01
    ## 747   2.541636947  0.135534921 -1.023967e+00  0.4062645258  1.065928e-01
    ## 748   2.715357044  0.695602690 -1.138122e+00  0.4594422419  3.863373e-01
    ## 749   2.761157059 -0.266161810 -4.128607e-01  0.5199522157 -7.439090e-01
    ## 750   2.966842404  0.615344322 -7.664948e-01  0.4312610051 -1.049750e-01
    ## 751   2.823431432  1.153005013 -5.673428e-01  0.8430116216  5.499376e-01
    ## 752   1.508748425  1.041642099 -6.827896e-01  0.5735439463 -1.602389e+00
    ## 753   1.082235328 -0.350562500  4.830439e-01  0.6611329209 -3.965215e-01
    ## 754   0.167703335  1.503413334 -7.677554e-01  0.3719514763 -1.415639e+00
    ## 755   0.339007373  1.342922614  2.392174e-01  0.5346441664 -1.749649e-01
    ## 756  -0.907905601  1.514028276 -1.418792e-01  0.7891859660 -3.134277e-02
    ## 757  -8.755698321  3.460893345  8.965375e-01  0.2548355368 -7.380966e-01
    ## 758  -3.474096503  1.765445513  1.701257e+00  0.3815868051 -1.413417e+00
    ## 759   0.010865192  0.548257832  9.121830e-02 -1.0079594602 -8.218282e-02
    ## 760  -0.110621879 -1.257800026 -3.244184e-01 -0.4200200010 -2.195011e-01
    ## 761  -4.884982624  1.140910436  1.392953e+00  0.3489966048 -2.167510e+00
    ## 762  -4.969477722  0.976124822  1.841248e+00  0.3344177482 -7.201285e-01
    ## 763  -1.998091228  1.133706234 -4.146087e-02 -0.2153787383 -8.655986e-01
    ## 764  -5.688990272  2.510979556  9.539329e-01 -0.5425062282 -6.201520e-01
    ## 765  -5.498771680  2.941475403  9.162359e-01 -0.2555044797 -1.838348e-01
    ## 766  -4.300431866  2.865771860  1.489302e+00  0.3860392888 -2.323918e-01
    ## 767  -4.352213131  2.389040921  2.019128e+00  0.6271918696 -1.085997e+00
    ## 768  -8.228874098  3.318177770  2.585212e+00  0.1950085693 -1.194803e+00
    ## 769  -9.110422576  4.158894820  1.412928e+00  0.3828006739  4.471540e-01
    ## 770   5.453671853 -2.056177144 -2.803337e-01  0.1207714725  5.693577e-01
    ## 771 -12.615022864  5.774087269  2.750221e+00  0.5134110997 -1.608804e+00
    ## 772   1.160623288 -1.259696808 -1.598165e+01 -0.8836699352 -3.536716e+00
    ## 773 -16.922015779  5.703684008  3.510019e+00  0.0543302269 -6.719833e-01
    ## 774   0.401415741 -0.084933416 -1.976844e-01 -0.2832710790  3.875973e-01
    ## 775 -21.453736278  8.361985192  4.909111e+00  0.0983279449 -1.508739e+00
    ## 776   0.773630966  0.860618181 -3.046664e-01 -0.1554995830  4.121663e-01
    ## 777   1.137211647  0.674245235 -5.317655e-01 -0.4334093505  3.786586e-01
    ## 778   1.089084184  0.975397810 -6.255303e-01 -0.5351809281  2.474350e-01
    ## 779   0.957897452  0.145339480 -4.470390e-02 -0.5449616713 -7.577565e-01
    ## 780 -22.797603906  8.316275439  5.466230e+00  0.0238542304 -1.527145e+00
    ## 781   1.325672390  1.021226205 -2.664762e-01 -0.3708804449  3.655352e-01
    ## 782   1.272896344  1.300268449 -3.950134e-03 -0.3608477778 -5.975260e-01
    ## 783   1.620591246  1.567947009 -5.780075e-01 -0.0590448082 -1.829169e+00
    ## 784   1.242896206  0.428408392 -1.011836e-01 -0.5201991212 -1.769378e-01
    ## 785   1.189422839  0.247858476  2.944483e-01 -0.5485035022 -1.746166e-01
    ## 786   1.128472231  0.228484047  2.862577e-01 -0.5362940981 -1.049249e-01
    ## 787   1.326944222  0.102998961  5.081669e-01 -0.2780175361  1.397211e-01
    ## 788   0.284840710 -0.874383479 -8.399505e-02 -0.6514416583  4.545938e-01
    ## 789   0.439756764 -0.694099204  2.996603e-01 -0.6576006277  1.016476e-01
    ## 790  -0.173601631 -0.190973582  2.199756e-01 -0.2165965507 -1.366919e-01
    ## 791   0.632504706 -0.070837951 -4.902913e-01 -0.3599831261  5.067753e-02
    ## 792   0.800538093  0.364616671  2.336076e-01 -0.2820775782 -3.203110e-01
    ## 793  -0.412525822 -0.208823119  3.448327e-01  1.0914347316 -6.865127e-01
    ## 794   0.802316031  1.037104789  9.581542e-02 -0.3201909134 -8.073063e-02
    ## 795   0.306598170 -0.854626652  1.011770e-01 -0.2814974102  2.444394e-02
    ## 796   5.556642391 -1.501807658  1.355172e+00  0.1410932141  7.791341e-02
    ## 797   5.563300664 -1.608271961  9.653216e-01  0.1637179793  4.753087e-02
    ## 798   0.870729582  1.269472789 -2.654940e-01 -0.4805486109  1.696651e-01
    ## 799   1.244286775 -1.015232287 -1.800985e+00  0.6575856270 -4.356172e-01
    ## 800   0.535541842  0.863591564  4.507425e-01 -0.1442278444 -2.056088e-01
    ## 801   1.775029790  1.266441152 -1.994096e-01  0.0149600028 -1.908227e-05
    ## 802  -0.036122295 -0.753591203 -4.711340e-02  0.3584927675 -2.874068e-01
    ## 803   0.203727978  0.733795721 -3.655975e-02  0.3343057707  1.471710e-01
    ## 804   0.491337125 -0.984222623 -4.219794e-01 -1.0480578940  7.264115e-01
    ## 805   0.102081332 -0.531495912 -3.287410e-01  0.3930996100  5.684346e-01
    ## 806   0.586829163  0.594077545 -2.521203e-01  0.3254388279  5.627657e-01
    ## 807   1.581480357  0.261332720  6.214150e-01  0.9941099270 -6.878531e-01
    ## 808   8.664662430 -2.716382588  4.835591e-01  0.0792347976  3.110652e-01
    ## 809   0.718504288  0.893850158 -3.163172e-02  0.3229126383 -5.840641e-02
    ## 810   0.010663477  1.786681373 -1.511782e-01 -0.5820977639 -9.560615e-01
    ## 811   0.337348848  1.018190578  3.035500e-01  0.8338861198 -1.222306e+00
    ## 812   0.262325117 -0.431790203 -9.208807e-02  0.1452158980  4.577884e-01
    ## 813   0.618247762  0.800932138  1.300165e-01  0.2889459533 -3.666580e-01
    ## 814   0.109184861 -0.931071823 -6.417504e-02 -0.0070125201  3.454195e-01
    ## 815   0.382006846  0.033958045  1.876969e-01  0.3584334658 -4.889338e-01
    ## 816   0.143176822 -0.390175929  3.560295e-01 -0.7623515490  9.651034e-02
    ## 817  -0.035490726 -0.419177641  1.574360e-01 -0.7148493743  4.688593e-01
    ## 818   0.019649407 -0.211678285 -2.474519e-01 -0.2794718867  2.396461e-01
    ## 819   0.303905313 -0.647075252 -3.730136e-01  0.2608011382 -4.965664e-01
    ## 820   0.110814637  0.563860613 -4.084361e-01 -0.8800789125  1.408392e+00
    ## 821  -1.384476843 -0.348903878 -3.979948e+00 -0.8281562622 -2.419446e+00
    ## 822   1.396872063  0.092072872 -1.492882e+00 -0.2042273963  5.325109e-01
    ## 823   0.129371630 -0.803020814 -7.409836e-02 -0.0310844053  3.753662e-01
    ## 824   0.851859435  1.176926955  4.535533e-01  0.4852112071 -5.006868e-01
    ## 825  -0.664694295  1.138556446 -3.507534e-01 -0.2874673006  8.088890e-01
    ## 826  -0.843268389  0.796738518  1.314312e+00 -0.3528866531 -1.770706e+00
    ## 827   0.098341470 -0.845865920 -3.122777e-02  0.4211464102  3.883614e-01
    ## 828   0.911372759  1.042928656  9.993939e-01  0.9012604552 -4.520932e-01
    ## 829  -0.204158276 -0.511441056  7.787415e-02  0.3883351203  7.895728e-03
    ## 830   1.120532789  1.605085186 -6.186374e-01 -0.2512829969 -2.405278e-01
    ## 831   0.621621663  0.043807385  1.027110e-01 -0.6015048702  1.273714e-01
    ## 832   1.178031950  1.360988581 -2.720131e-01 -0.3259478962  2.907027e-01
    ## 833   0.857941843  0.621203387  9.648166e-01 -0.6194372756 -1.732613e+00
    ## 834  -0.664263126  1.821421635  1.135628e-01 -0.7596733472 -5.023041e-01
    ## 835   1.059737074 -0.037394796  3.487071e-01 -0.1629294829  4.105310e-01
    ## 836   0.983480729  0.899876106 -2.851034e-01 -1.9297171045  3.198685e-01
    ## 837   1.377515322  2.151787056  1.892248e-01  0.7729432781 -8.724432e-01
    ## 838   1.441622275  0.895527949  1.385511e+00 -2.0280242292  5.091310e-01
    ## 839   1.404524058 -0.760549175  3.582923e-01 -1.1859415645 -1.286177e+00
    ## 840   1.138875933  1.033664340 -8.061989e-01 -1.5110464188 -1.917314e-01
    ## 841   1.325218400  1.226744876 -1.485217e+00 -1.4707322605 -2.400534e-01
    ## 842  -0.284412899 -0.706865472  1.314049e-01  0.6007421220 -6.042643e-01
    ## 843  -0.038689615  0.204553809 -1.673130e-01  0.7915471344 -2.236755e-01
    ## 844  -0.393089990 -0.708692350  4.713087e-01 -0.0786160466 -5.446547e-01
    ## 845   0.861308008  1.249301388  1.850627e+00 -0.1174712004  1.219815e+00
    ## 846  -0.812097797 -0.295360726 -5.988806e+00  0.7143808012 -1.600024e+00
    ## 847   0.786787019  0.893064859  1.034907e+00  0.0976710804 -1.345551e+00
    ## 848   0.105593016  0.371014424  5.110472e-02  0.4015241966 -7.247658e-01
    ## 849   0.289861124 -0.172718010 -2.190952e-02 -0.3765600886  1.928174e-01
    ## 850   0.727415064 -0.301431745 -5.024333e-01 -0.4623085762  5.106827e-01
    ## 851  -0.648259077  0.511284975 -1.110045e+00  0.1789869706  2.202353e-01
    ## 852   0.713906864 -0.063867591  1.679473e-01 -0.4498639689  2.370210e-02
    ## 853   0.652683372  0.414132297  2.386907e-02 -0.2606163875  4.053165e-01
    ## 854   0.641210916 -0.256678382 -2.337233e+00 -0.1582778985  1.198797e+00
    ## 855   0.407260461 -0.397434853 -8.000585e-02 -0.1685965452  4.650585e-01
    ## 856  -0.144713036 -0.310108279 -1.015300e-01 -0.4149601723  3.765969e-01
    ## 857   0.415436545 -0.469937601  7.128040e-03 -0.3881472907 -4.933979e-01
    ## 858   0.647714388  0.126575864  2.039535e-01  0.0084949976 -1.745012e-01
    ## 859   6.215513991 -1.276908596  4.598611e-01 -1.0516854749  2.091784e-01
    ## 860   1.245582186  0.616382702  2.251439e+00 -0.0660963925  5.387103e-01
    ## 861   0.397954324 -0.945401783 -3.761375e-01 -0.2204804204  2.640030e-01
    ## 862   0.209086136 -0.425937611 -1.544395e-01 -0.0188195044  6.322340e-01
    ## 863   0.417471746 -0.817343384 -2.875240e-02  0.0257225109 -8.258353e-01
    ## 864   1.083640212  1.037323802  6.232514e-02  0.5324896038 -1.491449e-01
    ## 865   1.185579782  1.348156129 -5.368557e-02  0.2841219846 -1.174469e+00
    ## 866   1.096404666  1.064222165  6.537033e-02  0.2572094376 -6.936540e-01
    ## 867   0.124235616 -0.823864650 -7.988736e-02  0.0288277899  3.897115e-01
    ## 868  -0.048061088 -0.599350030  7.219299e-02 -0.6003514200  3.713306e-01
    ## 869  -0.140061995 -0.907720220 -6.801079e-01 -0.3491698929  5.627589e-02
    ## 870   0.397057601  0.141165218  1.719845e-01  0.3942737758 -4.446418e-01
    ## 871   0.039289490  0.181651615  7.298091e-02 -0.1552988493 -1.498914e-01
    ## 872  -0.073205306  0.561496277 -7.503426e-02 -0.4376191794  3.538408e-01
    ## 873   0.220526295  1.187013204  3.358208e-01  0.2156829641  8.031100e-01
    ## 874  -0.157868696 -0.176243752  2.743695e-02 -0.4680062028  5.806280e-02
    ## 875  -0.934126825  0.922037918 -1.802550e-01 -0.2817191182  2.992850e-01
    ## 876   0.837685440  0.761712127 -4.176943e-01 -0.4697124715 -2.259342e-01
    ## 877   1.719631260  0.343209160  1.335836e-01  0.8333395340 -8.397757e-01
    ## 878   1.636622306  0.038727306  2.782182e-01  0.7866703568  6.389456e-02
    ## 879   0.476660425  0.434278328 -1.369399e-01 -0.6200716954  6.425306e-01
    ## 880   0.540730752  0.719526342  3.792487e-01 -0.6169618600 -4.428108e-01
    ## 881   0.547096801  0.687853881  4.299387e-01 -0.6206012794 -3.696883e-01
    ## 882   0.554184996  0.656075670  4.824173e-01 -0.6243994568 -2.962886e-01
    ## 883   0.561892174  0.624206816  5.364289e-01 -0.6283337124 -2.226512e-01
    ## 884   0.229936135  0.766926567 -1.896237e-01  0.7668529580 -1.414008e-01
    ## 885   0.133814675 -0.121562328 -2.085737e-01 -0.2547517842 -9.832440e-02
    ## 886   1.167243794 -1.006617493  7.745618e-01  0.0633969166 -3.906578e-01
    ## 887   0.564449516  0.445743912 -1.411362e-01 -0.2655171164  3.622599e-01
    ## 888   0.319260754 -0.471378905 -7.589041e-02 -0.6679092649 -6.428484e-01
    ## 889   0.364130241  0.210428090 -3.660187e-01  0.0158025670  3.492115e-01
    ## 890   1.191174807 -0.967141445 -1.463421e+00 -0.6242308515 -1.764615e-01
    ## 891   0.702672040 -0.182304776 -9.210170e-01  0.1116354882 -7.162221e-02
    ## 892   1.176574848 -0.978692051 -2.783303e-01 -0.6358737938  1.235394e-01
    ## 893   0.329760146 -0.941383472 -6.074662e-03 -0.9589253522  2.392979e-01
    ## 894   0.566848532 -0.321691498 -2.813246e-01 -1.1202560383 -7.339434e-02
    ## 895   0.535619751 -0.459495920 -9.363668e-03 -1.1404357942 -6.445269e-03
    ## 896   0.547789787 -0.491959822  5.591413e-02 -1.1453685347  6.895135e-02
    ## 897   0.360500882 -0.865525510  1.399781e-01 -0.3362378707  1.284493e-01
    ## 898  -0.301001481 -0.818972281  2.068123e-01 -0.2636830318 -1.149581e-01
    ## 899  -0.152130790 -0.360735866  4.341441e-02 -0.2423795357  1.710978e-01
    ## 900   0.773960868  0.214868420 -1.842330e-01 -0.2840913709  4.934666e-01
    ## 901   0.602290585 -0.541287465 -3.546393e-01 -0.7014920583 -3.097298e-02
    ## 902   0.532320016 -0.556913022  1.924444e-01 -0.6985884442  2.500272e-02
    ## 903  -0.032643295 -0.246526436  4.841078e-01  0.3596365766 -4.359725e-01
    ## 904   0.242560469  0.841230121 -3.701566e-01 -0.0260123796  4.919542e-01
    ## 905   0.123145443 -0.713201110 -8.086791e-02 -0.9643102608  3.385676e-01
    ## 906   0.249022571 -0.480286306 -2.860802e-01 -1.1535747555 -3.557113e-02
    ## 907   1.459369058 -0.136261687  8.481772e-01 -0.2699163798 -1.095060e+00
    ## 908   0.098482322 -0.538374705 -2.179890e-01 -1.0426566175  3.143888e-01
    ## 909  -0.088519421 -0.595178282  2.581481e-01  0.0619011334 -3.541802e-01
    ## 910   0.203563420  0.293267882  1.995683e-01  0.1468676323  1.636019e-01
    ## 911   0.614220932 -0.365047253 -1.804085e-01 -0.5232714295  6.450543e-01
    ## 912   0.622200356 -0.437708334 -9.035829e-02 -0.7428021577 -3.123607e-01
    ## 913   0.163739122  0.703910440 -2.450762e-01  0.4600491738  9.202810e-01
    ## 914   0.587727520 -0.605759173  3.374555e-02 -0.7561695115 -8.171625e-03
    ## 915   1.125229259  0.805257865  1.991192e-01  0.0352062006  1.215876e-02
    ## 916   1.043587159  0.262188674 -4.792239e-01 -0.3266381997 -1.569386e-01
    ## 917  -0.326139919  1.509238687 -2.159663e-01 -0.2457269073  8.930413e-01
    ## 918  -6.389131866  2.249964393  1.670508e+00  0.1404500025  1.621474e-01
    ## 919   0.880395097 -0.130435548  2.241471e+00  0.6653455365 -1.890041e+00
    ## 920   0.473211086  0.719400298  1.224578e-01 -0.2556498521 -6.192593e-01
    ## 921  -0.315104560  0.575519700  4.908420e-01  0.7565023664 -1.426854e-01
    ## 922   0.288252720  0.831939310  1.420070e-01  0.5926149743 -1.961431e-01
    ## 923  -0.870778522  0.504849160  1.379943e-01  0.3682745690  1.031367e-01
    ## 924   0.778583979 -0.319188819  6.394190e-01 -0.2948850404  5.375025e-01
    ## 925   0.370611857  0.028234445 -1.456404e-01 -0.0810494069  5.218745e-01
    ## 926   0.751825538  0.834107690  1.909439e-01  0.0320700856 -7.396948e-01
    ## 927   0.583275999 -0.269208638 -4.561078e-01 -0.1836591295 -3.281678e-01
    ## 928  -0.164350328 -0.295135167 -7.217253e-02 -0.4502613134  3.132666e-01
    ##               V26           V27           V28        Amount Class
    ## 1   -0.2897994430 -0.2976541991 -0.0884718062 -0.3473357940     0
    ## 2    0.9382472258 -0.2477809322  0.1579251303  3.5204844022     0
    ## 3    0.2265238930  0.5298512095  0.1806447913 -0.3493326036     0
    ## 4   -0.3287412658  0.0301971677  0.0340537327 -0.1097154515     0
    ## 5    0.2066477567 -0.1094378374 -0.0570566113  0.0459957611     0
    ## 6   -0.7105380150 -0.0096815046 -0.4585440173  2.1866155891     0
    ## 7   -0.5089420838 -0.0209235749  0.0163559053 -0.1647475241     0
    ## 8   -0.2811374511  0.0239856629  0.0483496698  0.4014678062     0
    ## 9    0.1295165194  0.0298619184 -0.1381347810 -0.3174235862     0
    ## 10   0.2098868157  0.2538747507  0.0620030766 -0.3373916822     0
    ## 11  -0.3314989767 -0.0510025800 -0.0639203633  0.0140867437     0
    ## 12   0.3561852553 -0.0673659444 -0.0109807687  0.0096937626     0
    ## 13   0.3535082160  0.0058276032  0.0364232453  0.2338955445     0
    ## 14   0.1267453958 -0.0047414817  0.0149449237 -0.3234539512     0
    ## 15   0.2417467365 -0.0987182768 -0.0817087302 -0.1448992367     0
    ## 16  -0.2426934622 -0.2845605788 -0.2468178029 -0.2887094641     0
    ## 17  -0.1703455453  0.1166557339  0.0104383768 -0.1324790810     0
    ## 18  -0.3352843999  0.0132855199  0.0284350793 -0.1970159673     0
    ## 19  -0.5064166659  0.0897183756  0.0140834298 -0.3453389844     0
    ## 20  -0.6225444374 -0.0388406749 -0.1833267041 -0.3493326036     0
    ## 21  -0.2407841982  0.0357329749  0.0101551418 -0.1857938973     0
    ## 22  -0.0325299352 -0.2719523446 -0.0103931219 -0.3320002963     0
    ## 23  -0.0032954406  0.0804358104  0.0040840945 -0.3413453652     0
    ## 24  -0.5976156967  0.0184972369 -0.0042785781  0.2057804653     0
    ## 25   0.5550697298  0.2023058515  0.2010900348  0.2017868461     0
    ## 26   0.1294646561  0.2150553073  0.0680472165 -0.3353948726     0
    ## 27   0.2101568991 -0.1115165430 -0.0463190270  0.1558602253     0
    ## 28  -0.4831517993 -0.0781673233 -0.0574510232 -0.3493326036     0
    ## 29   0.0728565699 -0.0240565246  0.0360736540 -0.3054427286     0
    ## 30  -0.8407386154 -0.0986021819  0.1210401003  3.2733592461     0
    ## 31   0.6008763655 -0.4628483400 -0.0958723694 -0.3073996020     0
    ## 32  -0.2660977742  0.0403357964  0.0506742884  0.2856528494     0
    ## 33   0.7452217352 -0.0216619542 -0.0150512326 -0.2352748392     0
    ## 34  -0.2368847936  0.0348000184  0.0402220315 -0.3493326036     0
    ## 35  -0.1051756909 -0.0129957079 -0.0354348609 -0.1816804696     0
    ## 36  -0.2567807508  0.0417911783  0.0391897101 -0.0498111635     0
    ## 37  -0.0370731122  0.3368344341  0.0991455249  0.4038639777     0
    ## 38   0.5445573615 -0.0825834071 -0.0808717133 -0.2524074656     0
    ## 39  -0.5299031241  0.0845951733  0.0232509264 -0.2127907631     0
    ## 40   0.6798739607  0.4423748655  0.3301696254 -0.3453389844     0
    ## 41  -0.1452655162  0.3153859278 -0.0203226626  0.0899655085     0
    ## 42   0.9536567242 -0.0397470678  0.0068245827 -0.1137090708     0
    ## 43  -0.5186764872  0.0264526529 -0.0575485696 -0.3361935964     0
    ## 44  -0.1681442272  0.1866919977  0.0657067880 -0.2929027643     0
    ## 45   0.1775553799 -0.0551979104 -0.0296883991 -0.3493725398     0
    ## 46   0.5097074740  0.0673348990  0.0929355194 -0.1979744359     0
    ## 47   0.0429982072  0.1910923584  0.1517712032 -0.3134299670     0
    ## 48  -0.5362883157 -1.0508345570 -0.3682619698 -0.3069603039     0
    ## 49   0.0453264306 -0.5603815294 -0.1006162291 -0.2734538388     0
    ## 50  -0.4857170433 -0.6031312409 -1.3174798137  0.0459957611     0
    ## 51  -0.3549078363 -0.0548748346  0.0240035464 -0.3133900308     0
    ## 52  -0.3245578560 -0.0992742872  0.2990924288  1.4733551998     0
    ## 53  -0.0614986505  0.0172471618  0.0108752295 -0.1338768477     0
    ## 54   0.1028465314  0.2601427683  0.0825349221 -0.3314012534     0
    ## 55  -0.0011362608 -0.0373743206  0.0010696095 -0.2930625091     0
    ## 56  -0.4971633609 -0.0159181406 -0.0527589185 -0.1553625190     0
    ## 57   0.7083319721  0.0659824181 -0.0216628089 -0.2634697908     0
    ## 58  -0.1472186717 -0.0268322668  0.0974665708 -0.2535656152     0
    ## 59  -0.2760950966  0.8263352226  0.4695547002 -0.1492922178     0
    ## 60   0.9647093297 -0.1496919893 -0.4767817662 -0.1855942164     0
    ## 61  -0.0332459210  0.0464944521  0.1094532537 -0.3016088542     0
    ## 62  -0.2014304000 -0.0994063237 -0.1166030281  0.1189991201     0
    ## 63  -0.3793104062  0.3342625281  0.2586856150 -0.3433821110     0
    ## 64   0.3471635226 -0.0450077798 -0.0784115791 -0.1931022205     0
    ## 65  -0.3396647655  0.5614215354 -0.3491180363 -0.0841163525     0
    ## 66  -0.2310610516 -0.8990965517 -0.6595626568 -0.2945002120     0
    ## 67  -0.2540521111 -0.5412696063  0.3328043130 -0.3318405515     0
    ## 68   0.2019028819 -0.0768351044 -0.0727170551 -0.3134299670     0
    ## 69   0.6686919123 -0.1339383605  0.0191222446  0.8018281311     0
    ## 70   0.1503320478  0.2473924772  0.0826943874 -0.3453789206     0
    ## 71   0.1475427323  0.0701681176  0.0009766742 -0.3318405515     0
    ## 72   0.1006253462  0.2637495021  0.1177977653 -0.3134299670     0
    ## 73   0.9318201093 -0.0871401235 -0.0308727142  0.3132088218     0
    ## 74   0.3010500101 -1.0705887206  7.2787520115  0.4954376660     0
    ## 75  -0.4264762278 -0.0167195384 -0.0701300149 -0.2724953702     0
    ## 76  -0.1703211108 -0.0061334811  0.0987261382 -0.1083576210     0
    ## 77  -0.4875450008  0.2335077980  0.0471306651 -0.1736133588     0
    ## 78  -0.5889528905  0.0374341022 -0.0033502851 -0.0525268246     0
    ## 79  -0.5038718188  0.3928649621  0.2409525571 -0.2836375678     0
    ## 80  -0.6142715835  0.0422576886 -0.0298606609 -0.3493326036     0
    ## 81  -0.1592247240  0.2167347596 -0.0453378802 -0.2335575830     0
    ## 82   0.5210991644 -0.3196464365  0.2213400818  3.4744379728     0
    ## 83   0.5654776934 -0.1487450692  0.0781465345  2.4122550739     0
    ## 84   0.4543501318 -0.0275629666 -0.0417039220 -0.1622315440     0
    ## 85   0.1643068299 -0.0848177932 -0.0410215748  0.0060995053     0
    ## 86  -0.2283614298  0.0155086436  0.0079968012 -0.1468561101     0
    ## 87  -0.0528648455 -0.0363951627 -0.0695422487 -0.0817601171     0
    ## 88   0.9740914918 -0.1205547156 -0.1249923482 -0.3493326036     0
    ## 89   0.7922174981  0.2841181300  0.3163337826  1.6434833777     0
    ## 90  -0.1558227069 -0.0297775705  0.0309379969 -0.0419836699     0
    ## 91   0.1335782386 -0.0053406245 -0.0480387522 -0.3533262228     0
    ## 92  -0.3300685854  0.0441409647  0.0273534379 -0.2075591220     0
    ## 93  -0.0573491874 -0.0019656875 -0.0558701838 -0.0797633075     0
    ## 94  -0.0922308673  0.0743628119 -0.0241707462 -0.2315208372     0
    ## 95   0.0954153281 -0.0234225450  0.0184792625 -0.3481744540     0
    ## 96   0.5632564312 -0.0417905823 -0.0555643322 -0.3239731217     0
    ## 97  -0.7870621126 -0.3087562357 -0.3066661422 -0.0835173096     0
    ## 98  -0.3438147528  0.0423203172  0.0079950280 -0.3493326036     0
    ## 99  -0.0375362208 -0.0181859471  0.0611099577 -0.3481744540     0
    ## 100 -0.1934511356 -0.0660348309  0.0552149519 -0.0908655689     0
    ## 101  0.7843692730  0.0129698023  0.1167779155 -0.3373517460     0
    ## 102 -0.1114887042  0.2241433561  0.1526282821 -0.3186216720     0
    ## 103 -0.9587940070 -0.5059930369 -0.4075115510 -0.3502910722     0
    ## 104 -0.6898203265  0.0006176191 -0.0314450841  0.0560197453     0
    ## 105 -0.0455134827 -0.0887112068  0.0288702808  0.4468752565     0
    ## 106  0.3582728385 -0.0575853536  0.0755786207  1.1816213171     0
    ## 107 -0.2515488492  0.0456432209 -0.0260506125 -0.3014491094     0
    ## 108 -0.1648575294 -0.1067836003  0.1267454967  2.2864560691     0
    ## 109  0.1040977687 -0.0658228122 -0.1658506811 -0.2814810134     0
    ## 110  0.2309426847  0.3066643574  0.1509850667 -0.2017683741     0
    ## 111 -0.4823184105  0.0271561631  0.0743882733  0.8247914415     0
    ## 112  0.1408197172  0.2548496603  0.0951768359 -0.3462175806     0
    ## 113  0.1375533754  0.2484696132  0.0975117485 -0.3497719017     0
    ## 114 -0.7153095589  0.0161162419 -0.0659607533 -0.2938212967     0
    ## 115  0.4883784744 -0.2116236681  0.0974031992 -0.3502910722     0
    ## 116  0.8696380599  0.1648896744  0.1348973334 -0.3416648547     0
    ## 117  0.7882821913 -0.0901859875 -0.0338181337 -0.2026469704     0
    ## 118  0.1483039787 -0.0630156974 -0.0394611399 -0.3481744540     0
    ## 119 -0.2292977794  0.0739691478  0.0454030171 -0.0498111635     0
    ## 120  0.0373266425  0.0115032387 -0.0972106650 -0.3533262228     0
    ## 121  0.5257617148  0.2386579356  0.0371202393 -0.3502910722     0
    ## 122  0.0343656590 -0.0102630305 -0.0508477513 -0.3390290661     0
    ## 123  0.2082828699 -0.0758120650 -0.0751078150 -0.3454188568     0
    ## 124  0.1228035244  1.3156578100 -0.6489819129  1.6434833777     0
    ## 125 -0.3730409610 -0.0382261296 -0.0286168565  0.7408455659     0
    ## 126  0.0677727284 -0.0391336377  0.0088137074 -0.3454188568     0
    ## 127 -0.0411263132  0.2258855857  0.0738316811 -0.2534857428     0
    ## 128  0.7484863631  0.3500948366  0.3623031434 -0.3493326036     0
    ## 129 -0.7666157572  0.1031820762  0.0497659236 -0.3455386654     0
    ## 130 -0.2580658553  0.0711445153  0.0234728526 -0.3464971340     0
    ## 131 -0.1532972850  1.5981905650 -0.6776133193  0.5212763822     0
    ## 132 -0.7972917475 -0.0100483733 -0.0188836413 -0.2538851047     0
    ## 133  0.7146934734  0.2616984873 -0.1051956415 -0.3226552274     0
    ## 134  0.5802632858  0.1130717655  0.1497920674 -0.2934618710     0
    ## 135  0.1252320259  1.4179094058  1.1614826571 -0.1175429452     0
    ## 136  0.1276058722 -0.0134679234  0.0251964606 -0.3454188568     0
    ## 137  0.2032023404  0.0301733737  0.0721233517 -0.3373916822     0
    ## 138  1.0971535866 -0.1051653186 -0.0233502449 -0.3453389844     0
    ## 139 -0.2974911516 -0.4051701182 -0.2293322244 -0.0510092493     0
    ## 140 -0.5617783919  0.0274903402  0.0478451567  0.3263078928     0
    ## 141  0.1111918395 -0.0199810736  0.0146981834 -0.3353948726     0
    ## 142 -0.2898959206  0.0311489397  0.0425399429  0.0021058861     0
    ## 143  0.1693717143 -0.0413477040 -0.1080152772 -0.3174235862     0
    ## 144  0.1471503822  0.2431122310  0.0974460685 -0.3145082442     0
    ## 145 -0.2441850643  0.1788968650  0.0936287348 -0.2825592906     0
    ## 146 -0.2900099574  0.2865134559  0.2121832635 -0.3316408706     0
    ## 147  0.9190721556 -0.0064389391 -0.0092539852 -0.2000511179     0
    ## 148  0.0652351824  0.3881680848  0.2578352714 -0.3474955388     0
    ## 149 -0.2254200063 -0.3848157797 -0.2412389413 -0.3094363478     0
    ## 150  0.9181732350 -0.0671557943  0.0665723286  0.8439608137     0
    ## 151 -0.3212316763  0.0029506101  0.0323509517 -0.3497719017     0
    ## 152  0.3093262421 -0.0207200249  0.1289469646 -0.2339170087     0
    ## 153 -0.9218379347  0.6359920948  0.4660159962  0.0779846509     0
    ## 154  0.5560925544  0.5440427659  0.4389165963 -0.2335575830     0
    ## 155  0.2859689312 -0.1289878495 -0.0721584083  0.4617714561     0
    ## 156 -0.6427237797 -0.0232833855  0.0144061310 -0.1017282131     0
    ## 157  0.5543036612 -0.0308406492 -0.0444132376 -0.3073996020     0
    ## 158 -0.3914784796 -0.0576348063 -0.0599884660 -0.2898276775     0
    ## 159 -0.5333891058 -0.3241935583 -0.2452456396  0.0976731936     0
    ## 160 -0.0478670293  0.0173033412  0.1058629893 -0.3454188568     0
    ## 161 -0.1831081680  0.0396140857 -0.0530546028 -0.2874315060     0
    ## 162 -0.0729083673  0.2631781197  0.0531574149 -0.2335176468     0
    ## 163  0.6923426542 -0.3283317041 -0.0072395681 -0.3502910722     0
    ## 164  0.7611902462  0.0632054525 -0.0442309826 -0.3493326036     0
    ## 165  0.9756859061 -0.0477129425  0.0080934145 -0.3133900308     0
    ## 166  0.2011622682 -0.0678291190 -0.0605414501 -0.3454188568     0
    ## 167  0.2856342852 -0.0769762035 -0.0189317116 -0.2086773354     0
    ## 168  0.0455120598 -0.1384403614 -0.2102114480 -0.1736133588     0
    ## 169 -0.7027495068 -0.1409113457 -0.0295480985 -0.3493326036     0
    ## 170 -0.2050646716  0.2011561162  0.1786255925 -0.2333579020     0
    ## 171 -0.2257474238 -0.1231037736 -0.0608258337 -0.2175431700     0
    ## 172 -0.0578288128  0.0738817053  0.1686375287 -0.3497719017     0
    ## 173  0.1332044732 -0.0081432598  0.0125935636 -0.2546838285     0
    ## 174  0.1109303227  0.1923224173  0.0256488727 -0.3442207710     0
    ## 175 -0.2492258877  0.0195491813  0.0166259256 -0.2678627719     0
    ## 176 -0.2855833432  0.0561019143 -0.0426322625 -0.1929025395     0
    ## 177 -0.4846376079  0.0895195683  0.0527361829 -0.3013293008     0
    ## 178  0.2265717268  0.2626094712  0.1826893716 -0.2968963835     0
    ## 179 -0.1477832751  0.8000433540 -0.0324607291  0.0460356973     0
    ## 180  0.0706187614  0.2642788386  0.1216044717 -0.3214571416     0
    ## 181  0.3796730398  0.0089590655  0.0011658526 -0.2974155540     0
    ## 182  0.1679261519  0.2401021182  0.1983706574 -0.1989728407     0
    ## 183  0.3272423541 -0.1559875135 -0.2204493969 -0.2101150383     0
    ## 184 -0.4229124959 -0.0532071404 -0.0629574413 -0.0737728787     0
    ## 185  0.2181126193  0.3564817114  0.3095756005 -0.3230945255     0
    ## 186 -0.4130308902  0.0795728459  0.0408229578 -0.3134299670     0
    ## 187 -0.4174323281  0.0112619774  0.0074963891 -0.2936216158     0
    ## 188  0.1179729210  0.1145671852  0.0215638958 -0.3129107965     0
    ## 189 -0.2316337380  0.6494788479  0.2581688679 -0.3174235862     0
    ## 190 -0.2014986589 -0.0027074978  0.0328041243  0.3834965198     0
    ## 191  0.1742682590 -0.0771426655 -0.0702745734 -0.3497719017     0
    ## 192 -0.2238643316  0.3470184346 -0.2672327863  0.5759090929     0
    ## 193 -0.0182493420 -0.0235535464  0.0420860623 -0.1140684965     0
    ## 194  1.1009628479 -0.0395484300  0.0553584539  0.3653255524     0
    ## 195 -0.1553592299 -0.0208146328 -0.0062031229 -0.0276066408     0
    ## 196  0.4295145852 -0.0286573250  0.0402890432  0.1654449114     0
    ## 197 -0.1056465782 -0.0639315981  0.0739333173  1.4577800849     0
    ## 198  0.0214701898  0.3122991043  0.3201137237 -0.2734937750     0
    ## 199 -0.1307921435 -0.3158186423 -0.1058392428 -0.2215367892     0
    ## 200 -0.0788969382 -0.0081140096 -0.0464978015 -0.1536851990     0
    ## 201 -0.2746847984 -0.0288257901  0.0256836029  0.3822584978     0
    ## 202 -0.6014273400  0.0584527987 -0.0359788113  0.0058998243     0
    ## 203 -0.1811621002  0.1928708533  0.2210870800  1.4503918893     0
    ## 204  0.4221747370  0.0065345070 -0.2886583756  1.3299443342     0
    ## 205  0.3018464775  0.0623970261 -0.0864694865 -0.1540446247     0
    ## 206  0.2681791022  0.0308410075 -0.0658105081 -0.1378704669     0
    ## 207 -0.3136078370  0.0149365112  0.0148084877 -0.3074395382     0
    ## 208 -0.1543214329 -0.1141348659  0.0064732446  1.3124123459     0
    ## 209 -0.2536802761  0.1744468099  0.1415676854 -0.0887888869     0
    ## 210  0.1025039148  0.0790061156  0.0565524270 -0.3493326036     0
    ## 211 -0.2845384644  0.2992241140  0.0860588932 -0.1073192800     0
    ## 212 -0.5132037900 -0.0260728556 -0.0483794812 -0.3453389844     0
    ## 213  0.0377073122 -0.0767139357 -0.0794865241 -0.3322798496     0
    ## 214 -0.6448967976 -0.1558440232 -0.0370055868 -0.3483341988     0
    ## 215 -0.3529809337  0.0380629061  0.0270447404 -0.1536452628     0
    ## 216 -0.5186990472  0.3106833833  0.2830983602 -0.3134299670     0
    ## 217 -0.7213031143 -0.0832808204 -0.1485448196 -0.3448597501     0
    ## 218  0.4449091897 -0.1207849507 -0.0961350088 -0.2227348749     0
    ## 219 -0.9267693942  0.2044461204  0.2524275988 -0.3054027924     0
    ## 220  0.5326115971  0.3504570963 -0.0736183112  0.2856528494     0
    ## 221 -0.5844052527 -0.0007610864 -0.0223705544 -0.3134299670     0
    ## 222 -0.5386539153 -1.2865607025 -0.0628985089 -0.2615129174     0
    ## 223  0.1203848314  0.2251461785  0.0711708212 -0.3425833872     0
    ## 224 -0.3801615064 -0.2788487096 -0.5000289752  0.2366112056     0
    ## 225 -0.3724482276  0.1067353345 -0.0015105738 -0.2637094079     0
    ## 226 -0.7074805028 -0.0482679469  0.0266776999  1.3000321264     0
    ## 227 -0.5099541312 -0.0294326445  0.0694133790  0.9637693897     0
    ## 228  0.3570880049 -0.1184126309  0.1384771854  1.5623729717     0
    ## 229  0.0852829665 -0.0593116777 -0.0833534311 -0.1935814548     0
    ## 230  0.7019143511 -0.0427587820 -0.0065202459 -0.2135495508     0
    ## 231 -0.4419785324  0.0023152247 -0.0217850738  0.1671621677     0
    ## 232  0.6153690992  0.1910359500  0.1501632286 -0.1250908855     0
    ## 233  0.1305878105 -0.0178568383  0.0116411865 -0.3454188568     0
    ## 234 -0.5068649902  0.1132230231  0.1427241156 -0.0777664979     0
    ## 235  0.0922561401 -0.0190959748  0.0325979078 -0.3373916822     0
    ## 236 -1.0877918565  0.0442513128 -0.0342828947 -0.3375514270     0
    ## 237  0.1334262950 -0.0042563099 -0.0734683368 -0.1336771668     0
    ## 238 -0.1619498444  0.0398074706 -0.0308217260 -0.1941804977     0
    ## 239  0.1301952811 -0.0202710925  0.0127399336 -0.3134299670     0
    ## 240  0.1928999468  0.0520711303  0.0116150432 -0.2897478051     0
    ## 241  1.3019255251 -0.0189874024  0.0087416446 -0.0593958496     0
    ## 242  0.1748772142 -0.0658584227 -0.0300658416 -0.3134299670     0
    ## 243 -0.4146902235  0.0403755796  0.0109465043 -0.3270082723     0
    ## 244  0.2679535375 -0.0150275035 -0.0532584520 -0.3134699032     0
    ## 245 -0.0666369097  0.0209378819  0.2611372878  0.0089349749     0
    ## 246 -0.7569730061  0.0302854011  0.1330133755 -0.3483341988     0
    ## 247  0.9876827204 -0.1978502714 -0.0264219455  1.2521086960     0
    ## 248 -0.6284063464  0.0269765675  0.0113330101 -0.2742525626     0
    ## 249  0.0923473130  0.1377002811  0.0357027106 -0.3497719017     0
    ## 250 -0.3893841130 -1.2751892175 -0.1737095225 -0.3326792115     0
    ## 251 -0.4622203109  0.1622349493  0.1805340249 -0.2335176468     0
    ## 252 -0.2379059996 -0.0498547847 -0.0423253630 -0.0446194586     0
    ## 253 -0.3124859563 -0.1320735522 -0.2144011418 -0.2534857428     0
    ## 254 -0.4065261505 -0.0030581664 -0.0560924544 -0.3493326036     0
    ## 255  0.9589962965 -0.0921441940  0.0075400449 -0.2312013476     0
    ## 256 -0.5092359682 -0.0301106106 -0.0716105706 -0.2582381496     0
    ## 257  0.1657859286  0.2651836787  0.0302311776 -0.2814810134     0
    ## 258  0.0222848375 -0.0153305334 -0.0214887929 -0.2734538388     0
    ## 259 -0.6621161551  0.3278208164  0.2484664673  0.1418825581     0
    ## 260  0.7412294042 -0.1258892077  0.0090317248  0.0646060266     0
    ## 261 -0.0980891793  0.0244303868  0.0076559055 -0.3493326036     0
    ## 262 -0.1024421712  0.0483921952 -0.0259456576 -0.3493725398     0
    ## 263  0.6051296687 -0.0391846489 -0.1288969477 -0.1082378124     0
    ## 264  0.4491641922 -0.0135310745 -0.0398690289 -0.1756900408     0
    ## 265 -0.2788752151  0.0141761354 -0.0256759501 -0.3502511360     0
    ## 266  0.0379867812 -0.6260575078 -0.2723375235 -0.3054826648     0
    ## 267  0.0590436417 -0.1008924944 -0.5221663855  0.4046227654     0
    ## 268 -0.1201076036  0.2048658404 -0.0530128022 -0.2765688618     0
    ## 269 -0.2702100076  0.0595557331  0.0650431755 -0.3493326036     0
    ## 270  0.3384258363  2.1172041270  1.3057581095 -0.3163852452     0
    ## 271 -0.4462610905  0.4246511483  0.2799950407 -0.2095559316     0
    ## 272  0.9955861425 -0.0289121775  0.0155627742  0.4332969512     0
    ## 273 -0.2522573230  0.0210837885  0.0229346858 -0.1296835476     0
    ## 274  0.5703130603 -0.2931751506  0.0556828157 -0.3502910722     0
    ## 275 -0.1676777710 -0.2486593224 -0.4942133677 -0.3399475985     0
    ## 276 -0.2361411016  0.1282906726  0.1179859899 -0.3439412177     0
    ## 277  0.0764809987  0.3588725925  0.1566157825 -0.3353948726     0
    ## 278 -0.4424912811  0.1131949068  0.1206621590 -0.3134299670     0
    ## 279  0.5845967531 -0.0508211198 -0.0363315962 -0.0378303059     0
    ## 280  0.2255476655 -0.0705673273  0.2171573702 -0.2336374554     0
    ## 281 -0.6280793987  0.0329771972  0.0033026501  0.2457166573     0
    ## 282 -0.2886687201  0.2114378343  0.0324318869 -0.2045639076     0
    ## 283  0.0898937346 -0.0140792898  0.0336621842 -0.3481744540     0
    ## 284 -0.3273343546  0.0291898840  0.0050261636 -0.2927829557     0
    ## 285 -0.4370195864  0.0228330397  0.0122863926 -0.2839969935     0
    ## 286 -0.3576899986  0.0049675560  0.0127111973 -0.2991328103     0
    ## 287 -0.4053254527  0.0061320175 -0.0191160254 -0.3294044438     0
    ## 288  0.0703924677  0.2665102231  0.1220196905 -0.3497719017     0
    ## 289 -0.4109434668 -0.0110852397 -0.0675111362 -0.3493326036     0
    ## 290  0.0213283174  0.0502232747 -0.0065034144 -0.2814410772     0
    ## 291 -0.6050512603 -0.0011493061 -0.0363994967  0.0452369735     0
    ## 292 -0.0647138441  0.0873322333  0.0203082601 -0.2814410772     0
    ## 293  0.2278686921 -0.0551230087  0.0024258119 -0.2585177030     0
    ## 294 -0.2682206429  0.0251035147  0.0684223210  0.7209573423     0
    ## 295 -0.6249493377  0.1135918544  0.0156878088 -0.3493326036     0
    ## 296  0.5363665933  0.1515150291  0.1651616466 -0.2536854237     0
    ## 297 -1.0928621062  0.0352687748 -0.0260053376 -0.1476947702     0
    ## 298  0.1667356343 -0.0682990099 -0.0295847028 -0.3174235862     0
    ## 299 -0.0076276790  0.0161275171  0.1447723842 -0.3115929022     0
    ## 300 -0.9262838723 -0.0471048630 -0.0278412669 -0.3116328384     0
    ## 301 -0.1202895157  0.4379772086  0.1771490774 -0.3493326036     0
    ## 302 -0.2950539542 -0.0103957515  0.0099868130 -0.3497719017     0
    ## 303  0.1440872056  0.2351769516  0.0819907234 -0.3318405515     0
    ## 304  0.2673962060 -0.0501505570  0.0964232723 -0.3481744540     0
    ## 305 -0.4116515324  0.0347305672 -0.0466461020  0.0300612205     0
    ## 306 -0.2125927929 -1.2369184667  0.0100100879 -0.0542041447     0
    ## 307 -0.3636292607 -0.0096870685 -0.0047002680 -0.0750109007     0
    ## 308  0.4247424016  0.0212676894  0.0219561402 -0.0538447189     0
    ## 309  0.0392582903 -0.1031527806  0.0464342969 -0.3054027924     0
    ## 310  0.2720019092 -0.0363773970  0.0158659739 -0.1738130397     0
    ## 311 -0.2646192701  0.6183321149  0.3739921246 -0.2769282875     0
    ## 312  0.7522613209 -0.8016944767 -0.7602023337  0.5330974951     0
    ## 313 -0.3056074239  0.1859862530  0.0629283946 -0.3493326036     0
    ## 314  0.0776610306  0.2345525867  0.0838748129 -0.3461776444     0
    ## 315  0.7030719983 -0.0380602873 -0.0178959202 -0.3041647704     0
    ## 316 -0.0459557268  0.5842986629  0.3126562494 -0.3312814448     0
    ## 317  0.1986125518 -0.0766638831 -0.0732756833 -0.3353948726     0
    ## 318  0.4219039557  0.1436242662  0.1535011119 -0.0937809109     0
    ## 319 -0.0120361476 -0.1693316869  0.1222033324 -0.2929027643     0
    ## 320 -0.5816947594  0.0093962094  0.1219866728 -0.3493326036     0
    ## 321  1.1586644242 -0.0978168583  0.0482051083  1.0124915439     0
    ## 322 -0.6476865150 -0.0313198806  0.0050650098  0.4014678062     0
    ## 323 -0.6930644038  0.0750330375  0.0252322753 -0.3060018353     0
    ## 324  0.3458176913  0.4264600895 -0.2522614496  0.5077380131     0
    ## 325 -0.1084138350  0.0545064663 -0.0527860912 -0.2284058142     0
    ## 326 -0.2244614483 -0.5860030577 -0.1554461389 -0.2814410772     0
    ## 327  0.4704318377 -0.2631371938 -0.1272880441 -0.2734538388     0
    ## 328 -0.1463838020  0.2590219670  0.1133182648 -0.3357542983     0
    ## 329 -0.6495109141  0.0224088000 -0.0351067720 -0.1936213910     0
    ## 330  0.2040788554 -0.0660842051 -0.0600595302 -0.3353948726     0
    ## 331 -0.0872799780  0.0108610557 -0.0046080978 -0.3133900308     0
    ## 332 -0.3389478060  0.1431693653  0.0038914058 -0.3282462942     0
    ## 333  0.1003953499  0.5209264080  0.3572149887 -0.0341162401     0
    ## 334  0.3404590368 -0.0361096978 -0.0478027135 -0.1556420724     0
    ## 335 -0.0743063034  0.0164245512 -0.0103896700 -0.1975750740     0
    ## 336 -0.5102386695 -0.0134025350 -0.0637234654 -0.3054027924     0
    ## 337 -0.3622295112 -0.0168254617  0.0237855992  0.0091745921     0
    ## 338  0.2084419653  0.0334577757 -0.0076031868 -0.3461776444     0
    ## 339 -0.4213646341  0.2160197740  0.1742116846 -0.2335176468     0
    ## 340 -0.4702801851  0.2716182402 -0.0677217414  0.1386477266     0
    ## 341  0.0712358946 -0.0458895563  0.0065941880 -0.3454188568     0
    ## 342  0.6737785024 -0.1524961141  0.1182414916 -0.1935814548     0
    ## 343 -0.3006903665 -0.0035731089 -0.0044157530 -0.1216963092     0
    ## 344 -0.2007146500  0.0656458924 -0.0030828473 -0.3232542702     0
    ## 345  0.3723560944 -0.6352194861 -0.1223068618 -0.0830380753     0
    ## 346 -0.3482840991  0.0276630774 -0.0218089105  0.2656847533     0
    ## 347  0.1140323575  0.1156545823  0.0053068329 -0.3094762840     0
    ## 348 -0.3151384514  0.0692078985  0.0270074098 -0.2392285222     0
    ## 349 -0.4854765661 -0.0751300368  0.0066992742  1.0444404976     0
    ## 350  0.1818489678 -0.0103231351  0.0300933686 -0.0380299869     0
    ## 351 -0.6773749641  0.2892731920  0.1609574706 -0.3134299670     0
    ## 352  0.0748167858  0.5362080186  0.2267322203 -0.3468565597     0
    ## 353 -0.2596803718  0.0846079580 -0.0571415533 -0.3013293008     0
    ## 354 -0.3143616878  0.0416874525  0.0173600644 -0.3075593468     0
    ## 355  0.9118311132 -0.3397674290  0.0926359669  4.0249583797     0
    ## 356 -0.1574884886 -0.0072524155 -0.0219643029 -0.3041248343     0
    ## 357  0.9210318040 -0.1662108730 -0.3932222901  1.6446015911     0
    ## 358  0.0306548863 -0.3261055427  0.1007247531 -0.3497719017     0
    ## 359  0.0555457666 -0.2600149016  0.0084527337 -0.2974155540     0
    ## 360  0.4772812922 -0.1074916422 -0.0262163514 -0.3473357940     0
    ## 361 -0.3094355024 -0.0616080512 -0.3201884411  0.0459957611     0
    ## 362  0.0576447437  0.1222535977  0.1951686385 -0.3502511360     0
    ## 363 -0.3529671167  1.4361071915  0.8159739546 -0.3461776444     0
    ## 364 -0.3586738546  0.2875663398  0.1041758762 -0.2934219348     0
    ## 365 -0.2369121144 -0.0025577145 -0.0292630456  0.1538634157     0
    ## 366  0.2281565416  0.0336128696  0.1331649104 -0.3502910722     0
    ## 367 -0.5411088733 -0.0394146923  0.0039792816 -0.1971757121     0
    ## 368 -0.6133334798 -0.2905744486 -0.0447559285 -0.1091962811     0
    ## 369 -0.4078464543 -0.0227295011  0.1858440991 -0.3493326036     0
    ## 370  0.1506367827  0.2370207968  0.0920471220 -0.3461776444     0
    ## 371  0.1184344519  0.2240278677  0.2317207654  0.3588558893     0
    ## 372  0.2000231728 -0.0715660019 -0.0582238788 -0.2854746326     0
    ## 373 -0.1700300063  0.0848657941  0.0331452309  0.0161234895     0
    ## 374  0.6280976131  0.0169629870  0.1328078204 -0.2734538388     0
    ## 375  0.5688081351  0.0246394157  0.0593840005  0.6287047387     0
    ## 376 -0.5979267983  0.0959112744  0.0142259493  0.2816592302     0
    ## 377  0.1199816827  0.4428469874  0.1890065898 -0.3294044438     0
    ## 378 -0.5166690917  0.1460274545  0.2852534366 -0.1896677080     0
    ## 379  0.0473932425  0.1262430423  0.0399857182 -0.3139890737     0
    ## 380  0.0634625364 -0.1175973158 -0.0659329649  0.3707968107     0
    ## 381 -0.0927783914  0.0938514647 -0.1313591215 -0.1252905664     0
    ## 382 -0.4373291560  0.0636046948  0.0293337798 -0.3014491094     0
    ## 383  0.0865111343 -0.1068562588 -0.0638892250  0.1215550364     0
    ## 384 -0.8935114649 -1.0398035901 -0.4602278061 -0.2359138183     0
    ## 385 -0.2357818448  0.0335594436  0.0495212620  0.5731934318     0
    ## 386  0.2472545562 -0.1038327763 -0.0948695212 -0.3502910722     0
    ## 387 -0.1197657910 -0.1585652302 -0.0971398638 -0.3493326036     0
    ## 388 -0.1191505029 -0.0969528235 -0.1919815585 -0.1540446247     0
    ## 389 -0.4957517007  0.0355461589 -0.0229165715 -0.0851546935     0
    ## 390 -0.7628215989  0.3728414533  0.1367573590 -0.2765289256     0
    ## 391 -0.6075761288  0.0291962460 -0.0341517316 -0.3493326036     0
    ## 392  0.3098116043 -0.0898793655 -0.0899836871 -0.3071599849     0
    ## 393  0.1799894494 -0.0603393081 -0.0311436700 -0.3454188568     0
    ## 394  0.1776713675 -0.3330973806 -0.1275447883  1.5865743041     0
    ## 395 -0.3344849883  0.0230569474 -0.0490040846 -0.2894283156     0
    ## 396 -0.3744613195 -0.0274360227 -0.1353523455 -0.1525270494     0
    ## 397  0.0692429487 -0.0951656177 -0.0537826838 -0.3214172054     0
    ## 398 -0.1532580384  0.0345763217  0.0197203940 -0.1396675956     0
    ## 399  0.0965654319 -0.1394028204 -0.4237486499 -0.3174235862     0
    ## 400 -0.3442215182  0.0608272116 -0.0055552657  0.6446392793     0
    ## 401  0.2221129590 -0.4413523498  0.2741136799  6.1810335140     0
    ## 402  0.3029169811  0.0103446716 -0.0832184591 -0.1767283817     0
    ## 403 -0.6316280643 -0.1151987089  0.0280103752 -0.1264886522     0
    ## 404 -0.5329190062  0.0366219819  0.0816147239  0.2417230381     0
    ## 405 -0.2798713417  0.0417695489  0.0449408913  0.0699974125     0
    ## 406 -0.4248570337 -0.1565177738 -0.2103075716 -0.1536452628     0
    ## 407 -0.1369398796 -0.0005438784  0.0169323986 -0.0358334963     0
    ## 408 -0.8281564023  0.2380368947  0.2596604250 -0.2695001558     0
    ## 409  0.9794879641  0.0649416939  0.1348905491  0.1257084004     0
    ## 410  0.1429181961  0.0205476148  0.0383506993 -0.1143081136     0
    ## 411 -0.3968434585 -0.1017099561 -0.2415043147  1.1462378510     0
    ## 412  0.2927281897  0.2913863296  0.2699137727 -0.2275671542     0
    ## 413  0.2728151010 -0.1136911883 -0.1009807139 -0.2734538388     0
    ## 414 -0.4875977773  0.0873168653  0.0888771858 -0.3134299670     0
    ## 415 -0.5177553397  0.0015472575  0.0253819315 -0.1028064903     0
    ## 416  0.0859301522  0.2559013314  0.0835175725 -0.3481744540     0
    ## 417  0.7371696194 -0.0530015618  0.1231014539 -0.1460573863     0
    ## 418 -0.1836168520  0.0182902656  0.0192224728 -0.3093564754     0
    ## 419 -0.5183880696 -0.0666214375 -0.0480876367  0.2217549421     0
    ## 420  0.0903963804  0.6568382211  0.4372619361 -0.3502511360     0
    ## 421  0.6723179767 -0.1875480776 -0.6083724360 -0.3383102146     0
    ## 422 -0.4682356597 -0.1149574321  0.1166291802  3.1837424312     0
    ## 423  0.0410428175  0.5411383369  0.4417240326 -0.3475754112     0
    ## 424  0.5623723813 -0.0279362999  0.0171066776 -0.2055623124     0
    ## 425 -0.4492275626 -0.0004920807 -0.0486977400 -0.3493326036     0
    ## 426  0.2669337421  0.1698433897  0.1087663618 -0.3493326036     0
    ## 427 -0.3711428791  0.0306569391 -0.0544065279 -0.1736133588     0
    ## 428 -0.4213631863  0.0342086875 -0.0557043571 -0.2551630629     0
    ## 429 -0.3710992130  0.3073668757  0.1253640982 -0.2608739383     0
    ## 430  0.1046902046  0.2520907353  0.0825018122 -0.3007302579     0
    ## 431 -0.2751646058 -0.0178753506  0.1700480484 -0.3168644795     0
    ## 432 -0.4720282081 -0.0300361655 -0.0583074056 -0.3493326036     0
    ## 433  0.9032492015 -0.0963146599  0.0805882204  1.5970775226     0
    ## 434  0.6557276944 -0.1255411059  0.0336404711  0.7249509615     0
    ## 435  0.1639534517  0.1298597705 -0.1695877153 -0.2736535198     0
    ## 436 -0.2489378411  0.0341258814  0.0500569601 -0.2335176468     0
    ## 437 -0.0949485632  0.0230913372  0.0102238314 -0.3493326036     0
    ## 438  1.1198399159 -0.0734493933  0.0092257635  0.8842963676     0
    ## 439 -0.0869828832  0.4022010150 -0.1968941007  4.3325069944     0
    ## 440  0.4829725316 -0.1297334107  0.0603399004  1.2041852656     0
    ## 441 -0.0042393079 -0.4589264206  0.3006768602 -0.3496520931     0
    ## 442 -0.1930176579  0.0894359052  0.0283058280 -0.2505304646     0
    ## 443 -0.1688221549  0.1314724412  0.1477098127  0.2562997482     0
    ## 444 -0.4632641996 -0.0126277523 -0.0703164491 -0.3421440891     0
    ## 445 -0.2655405760 -0.5176625622  0.2669384959 -0.3310418277     0
    ## 446 -0.0769806799  0.3544841091  0.1107952135 -0.3493326036     0
    ## 447 -0.0237022155 -0.0715770003 -0.0639367431 -0.1159454975     0
    ## 448 -0.2668832327  0.0193982406  0.2435320166  0.3904853534     0
    ## 449 -0.4586024423  0.0114444164  0.0050575021  2.4579820138     0
    ## 450 -0.8723043155  0.0852731692  0.1137906356 -0.0875508650     0
    ## 451 -0.4513920363 -0.2346084311 -0.3032965212 -0.1536452628     0
    ## 452  0.1009396189  0.1796440098  0.0616114741 -0.3214172054     0
    ## 453 -1.2257424401  0.2630744283  0.1032552196 -0.3439412177     0
    ## 454  0.1346785344 -0.0268932420  0.0129718225 -0.3073996020     0
    ## 455 -0.1429914725  0.0608953690 -0.0327719966 -0.2785656714     0
    ## 456  0.1778397983  0.2611450026 -0.1432758747 -0.3533262228     1
    ## 457 -0.1453617148 -0.2527731225  0.0357642252  1.7592983345     1
    ## 458 -0.5426278890  0.0395659889 -0.1530287965  0.6048628321     1
    ## 459 -0.6574877548 -0.8271357146  0.8495733800 -0.1177026900     1
    ## 460  0.5667972735 -0.0100162235  0.1467927349 -0.3493326036     1
    ## 461  0.5079626778  0.7358216361  0.5135737407 -0.3493326036     1
    ## 462  0.5406753964  0.7370403817  0.4966991082 -0.3493326036     1
    ## 463  0.5552647398  0.5305073889  0.4044740545 -0.3493326036     1
    ## 464  0.4911919257  0.5188682846  0.4025280677 -0.3493326036     1
    ## 465  0.3519092984  0.5945499781  0.0993722360 -0.3493326036     1
    ## 466  0.3194755401  0.6004849165  0.1293052251 -0.3493326036     1
    ## 467  0.3667776021  0.3951706693  0.0202055389 -0.3493326036     1
    ## 468  0.3219845393  0.2640281605  0.1328167198 -0.3493326036     1
    ## 469  0.3057044339  0.5309809863  0.2437455981 -0.3493326036     1
    ## 470  0.4358319162  0.6183238052  0.1484689444 -0.3493326036     1
    ## 471  0.3238854777  0.8947666715  0.5695186003 -0.3493326036     1
    ## 472  0.5235736967  0.8910246884  0.5727406665 -0.3493326036     1
    ## 473  0.3221883984  0.6208665388  0.1850303547 -0.3493326036     1
    ## 474  0.3348075987  0.7485342848  0.1754137944 -0.3493326036     1
    ## 475  0.3084979316  0.5525909147  0.2989544787 -0.3493326036     1
    ## 476  0.3048828396  0.4180124672  0.2088582781 -0.3493326036     1
    ## 477 -0.4244528811 -1.0020414260  0.8907802880 -0.3489332417     1
    ## 478  0.4131955325  0.2802841569  0.3039365506 -0.3493326036     1
    ## 479  0.4259646214  0.4131395558  0.3082046570 -0.3493326036     1
    ## 480  0.2963671945  1.9859132180 -0.9004516384  6.8738465730     1
    ## 481  0.6908056147  2.1197487334  1.1089334033 -0.3493326036     1
    ## 482  0.5218839340  0.5279377499  0.4119104058 -0.3493326036     1
    ## 483  0.5020395167  1.9772583298  0.7116065312 -0.3493326036     1
    ## 484  0.6052418859  1.8815287524  0.8752604072 -0.3493326036     1
    ## 485  0.4665239246  0.6273932705  0.1578512825 -0.3493326036     1
    ## 486  0.0322706214 -1.5084579336  0.6080746826 -0.3533262228     1
    ## 487 -0.0335900628 -1.3317773222  0.7056975900 -0.2319601353     1
    ## 488 -0.4098627442 -3.0362713294 -0.6306051988  0.3641674028     1
    ## 489  0.6317037722  1.9342207601  0.7896871722 -0.3493326036     1
    ## 490  0.4875388233  1.4277130955  0.5831724862 -0.3493326036     1
    ## 491  0.6084245964  1.5747147838  0.8087252051 -0.3493326036     1
    ## 492  0.6348735025  1.6272090762  0.7232350685 -0.3493326036     1
    ## 493  0.6418360594  1.6059576248  0.7216436145 -0.3493326036     1
    ## 494 -0.6117638450 -3.9080804755 -0.6712482651 -0.3078389001     1
    ## 495 -0.1189409295  0.0129477659  0.0542542721 -0.3381904060     1
    ## 496 -0.0983002869 -0.0030411199  0.0498190204 -0.3381904060     1
    ## 497 -0.1366046849 -0.1008451868  0.0393471477 -0.3376312994     1
    ## 498  0.6487980368  1.5846973468  0.7200558702 -0.3493326036     1
    ## 499  0.1723468895  0.6266467810 -0.1697260311  2.7072237881     1
    ## 500 -0.0077722811  3.0523576868 -0.7750356512  4.5144562852     1
    ## 501  0.7550794608  2.7065661616 -0.9929160471 -0.3493326036     1
    ## 502  0.1537254445  1.2124772081 -1.8692904760  0.4005892100     1
    ## 503  0.8167617180  2.2629423738 -1.1780631632 -0.3493326036     1
    ## 504  0.6346929035  2.1719049446 -1.3952876839 -0.3493326036     1
    ## 505  0.6423432010  2.1611292237 -1.4012820196 -0.3493326036     1
    ## 506  0.6499208608  2.1492470779 -1.4068113321 -0.3493326036     1
    ## 507  0.6574367785  2.1364244709 -1.4119453748 -0.3493326036     1
    ## 508  0.6648997744  2.1227957542 -1.4167406149 -0.3493326036     1
    ## 509  0.6723170651  2.1084708521 -1.4212432527 -0.3493326036     1
    ## 510  0.6796946122  2.0935405711 -1.4254914538 -0.3493326036     1
    ## 511  0.6870373836  2.0780805838 -1.4295170226 -0.3493326036     1
    ## 512 -0.0271712790 -0.0249206212  0.0736053987 -0.3408661309     1
    ## 513 -0.2737427117  1.6881358964  0.5278307996  0.0459957611     1
    ## 514 -0.2337204314  1.7075205521  0.5114231691  0.0459957611     1
    ## 515 -0.2331889901  1.6840629925  0.5037397383  0.0459957611     1
    ## 516  0.0433535806  0.0151590213  0.0636117223 -0.3383102146     1
    ## 517 -0.2179545649  1.6287925315  0.4822482749  0.0459957611     1
    ## 518 -0.2248250570  1.6210523897  0.4846144027  0.0459957611     1
    ## 519 -0.2242918735  1.5976213684  0.4769198180  0.0459957611     1
    ## 520 -0.2237548568  1.5742487352  0.4692006932  0.0459957611     1
    ## 521 -0.2232144033  1.5509284500  0.4614595668  0.0459957611     1
    ## 522 -0.2226708569  1.5276552781  0.4536986390  0.0459957611     1
    ## 523 -0.2221245167  1.5044246600  0.4459198261  0.0459957611     1
    ## 524 -0.2215756447  1.4812326064  0.4381248048  0.0459957611     1
    ## 525 -0.2210244712  1.4580756113  0.4303150487  0.0459957611     1
    ## 526 -0.2204711990  1.4349505814  0.4224918578  0.0459957611     1
    ## 527 -0.2199160083  1.4118547769  0.4146563836  0.0459957611     1
    ## 528 -0.2193590587  1.3887857621  0.4068096499  0.0459957611     1
    ## 529  0.6261558694  0.5651248260  0.3059252554 -0.2600352783     1
    ## 530 -0.2042140658  1.3321530590  0.3858907280  0.0459957611     1
    ## 531 -0.2658943177  0.2206940233  0.2560768698  2.5654503065     1
    ## 532 -0.2110585224  1.3248094928  0.3880901777  0.0459957611     1
    ## 533 -0.2105020005  1.3017339650  0.3802461814  0.0459957611     1
    ## 534 -0.2099439991  1.2786809708  0.3723927143  0.0459957611     1
    ## 535 -0.2093846241  1.2556489009  0.3645304530  0.0459957611     1
    ## 536 -0.2088239712  1.2326362952  0.3566600109  0.0459957611     1
    ## 537  0.5163266738  0.0091462814  0.1533176355 -0.3506105618     1
    ## 538  0.4073868720 -0.1309183112  0.1921773821 -0.3506105618     1
    ## 539  0.5647142927  0.5532547139  0.4024000092 -0.3506105618     1
    ## 540 -0.5365171113  0.4890350204 -0.0497285788 -0.2323195610     1
    ## 541 -0.0792146544 -2.5324452980  0.3111767147  0.0652450056     1
    ## 542 -0.1324740290  0.5765607569  0.3098427055 -0.3533262228     1
    ## 543  0.6917745058  0.1967789741  0.2410851575  2.5106977872     1
    ## 544  0.2408782137  0.4184345320  0.2321697152 -0.2773675856     1
    ## 545 -0.1062987533  0.0210082654  0.0105592082 -0.3472559216     1
    ## 546  0.2281642992  0.5510018739  0.3054734200 -0.2776072028     1
    ## 547  0.2426524609 -0.2686492220 -0.7437125860  0.1470742631     1
    ## 548  0.2711390035  1.3732995770  0.6911950959 -0.2773675856     1
    ## 549  0.5236012264  0.6262827714  0.1524399381 -0.3502910722     1
    ## 550  0.2321059697 -0.3105189983 -0.7452946268 -0.1113128992     1
    ## 551  0.3895264020  1.3002364561  0.5499403742 -0.3229347807     1
    ## 552 -0.1037163561  1.1669614917  0.6636320669 -0.3493326036     1
    ## 553  0.3620140535  0.9009251375  0.5548974993 -0.3168644795     1
    ## 554  0.4121913514  0.6357888710  0.5010495505 -0.3351153193     1
    ## 555 -0.1316866561  0.4739342218  0.4737574816 -0.2955784892     1
    ## 556 -0.2703275930  0.2102140515  0.3918548863  0.0927610419     1
    ## 557 -0.0494467232  0.3034445816  0.2193804386  0.0927610419     1
    ## 558  0.5205076473  1.9374214029 -1.5525928387 -0.3041647704     1
    ## 559 -0.2218858284 -0.3085547180 -0.1645000179  2.7490369811     1
    ## 560 -0.3395244424  0.0967012906  0.1149719155 -0.3493326036     1
    ## 561  0.5470920706  0.0099789443  0.1607691500 -0.3493326036     1
    ## 562  0.6612724267  0.4925597388  0.9718343500 -0.3493326036     1
    ## 563  0.5876112163  0.5003263711  0.5517596879 -0.3493326036     1
    ## 564 -0.5189956310  2.3523334653  1.1306248353 -0.3502910722     1
    ## 565 -0.0154765350  0.7766912659  0.3975574903 -0.3502910722     1
    ## 566  0.2778949302  0.8300616386  0.2186904422  0.0952770220     1
    ## 567  0.0082765867  2.0535242533  0.8357490030 -0.3381904060     1
    ## 568  0.3868011063  2.1638977673  0.9831038340  1.0049436037     1
    ## 569  0.3246379648  2.2450914664  0.4973208490 -0.0009692007     1
    ## 570  0.5654929058  1.7920119170  0.3710070212 -0.3321600410     1
    ## 571  0.0819723427  1.4150675107  0.0351241448 -0.0203382538     1
    ## 572  0.1647087400  1.4110474942  0.3156447998 -0.3078788363     1
    ## 573  0.4445765205  1.1019231482  0.2059580125 -0.3472559216     1
    ## 574  0.4831619697  1.1956708696  0.1982943501 -0.0009692007     1
    ## 575  0.5227216361  0.7926906898  0.0677895462 -0.2324793058     1
    ## 576 -0.2969280650  0.5265055781 -0.4508904888  1.4800245438     1
    ## 577  0.3542537815  0.2733287268 -0.1529080808 -0.3533262228     1
    ## 578  0.3299009592  0.1635043410 -0.4855522127  0.1191189287     1
    ## 579  0.4469967327  0.0622926467 -0.4397698995 -0.1716564854     1
    ## 580 -0.4289227968 -0.6949353874 -0.8189704289  0.3378494523     1
    ## 581  0.1776236122 -0.8176802559 -0.5210297959 -0.2042843542     1
    ## 582 -0.0061678582 -1.1856955057 -0.7473613214 -0.1149870289     1
    ## 583  0.2800371421 -1.4066868356 -0.6636433636 -0.3502910722     1
    ## 584 -0.4538395718 -1.5089682570 -0.6868357388 -0.3134299670     1
    ## 585  0.0560310741 -1.3108876205 -0.7074029913 -0.2170639357     1
    ## 586  0.0643574266 -1.6213856864 -1.1048192086  0.7369717553     1
    ## 587  0.1459635686 -2.4586803476 -1.1898878814  0.0721939031     1
    ## 588 -0.4627623614 -2.0185752488 -1.0428041697  1.1011099540     1
    ## 589 -0.1767010824  0.5048976717  0.0698824529 -0.1957779453     1
    ## 590 -0.0460816700 -3.0114732847 -1.0221474663 -0.1378704669     1
    ## 591 -0.1258210804  0.4212998024  0.0031458761  0.3348542379     1
    ## 592  0.5329083140  0.4230447217 -0.2102663104  2.8516330584     1
    ## 593 -0.2668454652  1.1936945771  0.2574684574  0.0459957611     1
    ## 594  0.1217959258 -3.3818429294 -1.2565236214  0.2053811034     1
    ## 595  0.1126707743 -3.7653710092 -1.0712383122 -0.3493326036     1
    ## 596 -0.2951022340  0.1959851457  0.1411151027 -0.3493326036     1
    ## 597  0.6639536244  0.4560227233 -0.4056820848 -0.3493326036     1
    ## 598  0.5566763263  0.7393832125 -0.2030496738 -0.3493326036     1
    ## 599 -0.1060892751  1.8997143418  0.5114617068 -0.3493326036     1
    ## 600 -0.0094656472  2.3001642476  0.0812314897  2.2345390195     1
    ## 601  0.7076542453  0.4186486585  0.0807557537  0.4624503714     1
    ## 602 -1.0559973261 -1.2001649668 -1.0120658431 -0.0018877331     1
    ## 603 -0.3497929966  0.4548510684  0.1378434772 -0.2538851047     1
    ## 604 -0.2399962396 -0.1834630541 -0.0733597032 -0.3493326036     1
    ## 605 -0.5318457813  0.1231846878  0.0395806345 -0.3493326036     1
    ## 606 -0.5692323173  0.4810187016 -0.0475554427 -0.3493326036     1
    ## 607 -0.4532059724  0.0466270924  0.0646983544  0.0699574763     1
    ## 608 -0.3691145677 -0.0192440931 -0.2083190916 -0.3493326036     1
    ## 609 -0.4751017854  0.5714263038  0.2934258369 -0.3493326036     1
    ## 610 -0.2837510341  0.3954510491  0.2331385760 -0.3493326036     1
    ## 611 -0.2178234050 -0.0728522477  0.0104633126 -0.3493326036     1
    ## 612  0.3688088545 -0.0182873481  0.0311734033 -0.3502910722     1
    ## 613 -0.3762053809  0.0345043695  0.1577753204 -0.3230545893     1
    ## 614  0.4093661024  0.5396682853  0.2969183782 -0.3502910722     1
    ## 615  0.3028091257  0.4660309415  0.2501335388 -0.3502910722     1
    ## 616 -0.1593606891  0.0605395343  0.3569583628  0.4796628701     1
    ## 617 -0.1944185743  0.0459171450  0.0401359981 -0.3493326036     1
    ## 618 -0.4694328412 -0.4058137681 -0.1521708470 -0.2745321160     1
    ## 619 -0.1999171199  0.3952006221  0.0276930321  1.0843766896     1
    ## 620 -0.2572836929  1.1700265156  0.2293009148  0.0459957611     1
    ## 621 -0.2643246090  1.1596904601  0.2327580884  0.0459957611     1
    ## 622 -0.2639335166  1.1340950935  0.2259731640  0.0459957611     1
    ## 623 -0.2635163233  1.1088972976  0.2190211432  0.0459957611     1
    ## 624 -0.2630779231  1.0840225277  0.2119333567  0.0459957611     1
    ## 625  0.6991749526 -0.3360717090 -0.1775868279  1.7229563998     1
    ## 626  0.1896209049  0.0610156660  0.0631412802 -0.3502910722     1
    ## 627  0.2260028809  0.6285446163  0.3199179876 -0.3502910722     1
    ## 628 -0.4135184337  0.0328380408  0.0206001971 -0.3486137522     1
    ## 629 -0.1594517667  0.5994823647  0.2889155499 -0.3213772692     1
    ## 630 -0.0442276365  0.5107290749  0.2209515937 -0.3533262228     1
    ## 631  0.8023486675  0.3908085528  0.1121455315  0.0957562563     1
    ## 632 -0.7860722203  0.6060973224  0.1716965079  0.6924828374     1
    ## 633  0.1934897804  1.2145876869 -0.0139228974 -0.3461776444     1
    ## 634 -0.1657974905  1.5055159130  0.3594920504 -0.3493326036     1
    ## 635 -0.0425826451  0.9511302331  0.1589960237 -0.3500115189     1
    ## 636 -0.1639860175  1.1978950466  0.3781866882 -0.3500115189     1
    ## 637 -0.1112254306  1.1445985091  0.1022795454  0.1676014658     1
    ## 638 -0.4724122627  1.7753780565 -0.1042846259  0.8923235422     1
    ## 639 -0.2496004193  0.5622387428  0.0753085014  0.3292631710     1
    ## 640  0.2136303590  0.4236203294 -0.1051685335  0.2595345798     1
    ## 641  0.3300203649 -0.0282517555 -0.1562702730 -0.3232942064     1
    ## 642 -0.2190847435  0.8744936322  0.4704342218 -0.3493326036     1
    ## 643  0.2136931873  0.7736254667  0.3874340546 -0.3333581268     1
    ## 644  0.7563345227  0.6328004773  0.2501870928 -0.3532862866     1
    ## 645 -0.4910183102  0.7016060414  0.2069663433  1.4205196177     1
    ## 646 -0.1075864254  0.3306418978  0.1635771496 -0.3493326036     1
    ## 647  0.3955277223  0.7820359508  0.6285278640 -0.3493326036     1
    ## 648  0.4527867769  0.1322180760  0.4245994655 -0.3493326036     1
    ## 649  0.0580797143  0.5531433433  0.3188324457 -0.3463373892     1
    ## 650 -0.3351410490 -0.1888149851 -0.1233914592 -0.3329987011     1
    ## 651 -0.2774978705  0.4286100847  0.2463937462  0.7249509615     1
    ## 652 -0.5543769063  0.4551792706  0.0013206565  0.1016268766     1
    ## 653 -0.4938600133 -1.5376523784 -0.9940224908 -0.1710574425     1
    ## 654  0.5319106161  0.3023243544  0.5363753477  5.1960472742     1
    ## 655  0.4311191102  0.8213814270 -1.0560877739 -0.2775273304     1
    ## 656 -0.2310298182  0.5804952338  0.3009840379 -0.3282862304     1
    ## 657 -0.3400042174  0.0398191045 -1.0079003102  0.9214370262     1
    ## 658 -0.2824959141  0.8660769107 -0.4334659438 -0.3297239333     1
    ## 659  0.2354352436  0.9620147975 -0.6735573932  1.4630516622     1
    ## 660  1.1792420834 -1.1663153283  0.8212148031  0.0520261261     1
    ## 661  0.9917396780 -0.9525539591 -0.3903639980 -0.2772877132     1
    ## 662  0.0557502964  0.3946819554  0.2988207258 -0.3268884637     1
    ## 663 -0.0527144133  0.3228540050  0.1352676367  0.3655252334     1
    ## 664 -0.3093124341  0.0999419474  0.1229883119 -0.3493326036     1
    ## 665 -0.3664226773 -1.4867659203  0.6776641046 -0.3489332417     1
    ## 666  0.4410441760  1.5206126205 -1.1159368032  5.2463668761     1
    ## 667 -0.0333546437 -0.0619624216 -0.0626865318  0.8055421969     1
    ## 668  0.6650876606 -1.6841860487  0.3101954630  0.8243920796     1
    ## 669  0.1791155265  0.6125798192  0.2342060896 -0.3493326036     1
    ## 670  0.0769835704  0.0189358320  0.0605741495 -0.3533262228     1
    ## 671  0.0728277709  0.0270429343  0.0632375727 -0.3533262228     1
    ## 672  0.2722652462  1.8413072688 -1.7963625867  2.5235971772     1
    ## 673  1.2466044374 -0.0286711041 -0.0061120076 -0.2258898341     1
    ## 674 -0.2045321775  1.5102064874 -0.3247060126  5.0550325802     1
    ## 675 -0.2837382670 -0.0011282786  0.0350745013  0.0380883951     1
    ## 676  0.9099287216 -1.4787672460  0.7226730395  0.0520261261     1
    ## 677  0.1277897970  0.5697307904  0.2912058588 -0.3232542702     1
    ## 678  0.8310654516  0.3324213039  0.2527127286  1.6023890361     1
    ## 679 -0.5409545843  0.1506064464 -0.1171401148  1.8394103357     1
    ## 680 -0.4951623129  0.0316326197  0.0662796879  0.9246319215     1
    ## 681 -0.1088769300  1.2695663554  0.9394073628 -0.3493326036     1
    ## 682 -0.3226065030  0.0808054374  0.0354271820 -0.2750912227     1
    ## 683 -0.3523932429  0.0719927693  0.1136843701 -0.3493326036     1
    ## 684 -0.3154334622  0.7682907513  0.4596233277  0.5544234216     1
    ## 685  0.1173626330  0.5895949028  0.3090640738 -0.3381904060     1
    ## 686  0.8258135533  0.4144822719  0.2672650825  0.9170839812     1
    ## 687 -0.1935285123  0.2581941008  0.2472692140  2.9407307028     1
    ## 688  0.2862813788  0.1352149456  0.2573147691 -0.3502910722     1
    ## 689 -0.2570998378  0.8592815665  0.2259241857  1.3495530045     1
    ## 690 -0.2517439819  1.2494136743 -0.1315246435  0.6007494043     1
    ## 691 -0.1803312877  0.0267622261 -0.3583353212 -0.1734935502     1
    ## 692 -0.4536510118  0.3491910428  0.1988622682 -0.2875113784     1
    ## 693 -0.2783379869  0.6259222152  0.3955733783 -0.0460571615     1
    ## 694 -1.1235343372 -0.6309765125  0.3268392040  0.8898075621     1
    ## 695 -0.5050322159 -0.0394559688 -0.0063575822 -0.2329585401     1
    ## 696  0.2191458382 -0.0582573076  0.1580479887 -0.2337173277     1
    ## 697 -0.3717410931  0.6152569644  0.8031628419  0.1439991763     1
    ## 698 -0.1974314709  0.3286715223  0.8353949214  0.0454366544     1
    ## 699 -0.1807691304  0.1778103829  0.6615553556  0.0456363354     1
    ## 700 -0.3777908610 -0.2135617863  0.4595289181  0.5244712776     1
    ## 701 -0.2658384840 -0.5146369364  0.3885902693  0.6640882048     1
    ## 702 -0.3125447226  0.1062474714  0.1250598633 -0.3493326036     1
    ## 703 -0.3063110234  0.0940868179  0.1210647285 -0.3493326036     1
    ## 704 -0.3350326493  0.0891401409  0.1123372200 -0.3493326036     1
    ## 705 -0.5228324505  0.0001048565  0.1356983461 -0.3493326036     1
    ## 706  0.3538963292 -0.0623612632  0.0084330017  0.1618506541     1
    ## 707 -0.3323656295  0.1059486767  0.1281237286 -0.3493326036     1
    ## 708 -0.4845370350  0.3735959159  0.1876572489 -0.3493326036     1
    ## 709 -0.1553971862  0.1143284164  0.1015255944 -0.3493326036     1
    ## 710 -0.6146060369 -0.7668481119  0.4094239441  0.0735916698     1
    ## 711 -0.9634888623 -0.9188883538  0.0014535622 -0.1137090708     1
    ## 712  0.4823704167  0.1299693145  0.2239243338 -0.3525274990     1
    ## 713  0.2587824565  0.5068926949  0.1767355488 -0.3230945255     1
    ## 714 -0.1043929204  0.0737955600 -0.0415702518 -0.3493326036     1
    ## 715  0.1995580107  1.0945331821  0.5411475677 -0.3493326036     1
    ## 716  1.1657838901  1.3744953323  0.7298893419 -0.3533262228     1
    ## 717  1.1111512560  1.2777072571  0.8190811520  1.6924052129     1
    ## 718  0.2765394716  1.4412739679 -0.1279437264 -0.3041647704     1
    ## 719 -0.6194374466 -0.1203510728  0.0355941734  1.0617328687     1
    ## 720 -0.0595547733  0.4607096775 -0.0335506071 -0.3453389844     1
    ## 721  1.0424579928  1.3595156316 -0.2721881013 -0.3533262228     1
    ## 722  1.0752950989  1.6499059092 -0.3949053722  0.6567399455     1
    ## 723  0.0259997138  1.0629681997  0.5011481682 -0.3493326036     1
    ## 724 -0.1363122791  1.0875848239  0.3738335289  0.6082174722     1
    ## 725 -0.1350604974  1.2386947300  0.0998237563 -0.3493326036     1
    ## 726 -0.1498436142  0.6855172414 -0.2997283794  0.8863730496     1
    ## 727  0.0757375933  0.3467170840  0.2822086260  0.9396878659     1
    ## 728 -0.3050236899 -0.0213631449  0.1290963856  1.4488743140     1
    ## 729 -0.3321145692 -0.4698003019 -1.4950058968  2.9590214787     1
    ## 730 -0.0068096144 -1.0657340401  1.7733264456 -0.3486137522     1
    ## 731  0.1891165471 -0.0058232807 -0.0121050907 -0.3165449900     1
    ## 732 -0.6953075711 -0.1525015620 -0.1388662563 -0.3254108246     1
    ## 733 -0.0785905920  0.3667093559  0.0737667168 -0.1985335426     1
    ## 734 -0.4075713589 -1.3787025303  1.3791302915 -0.1429024271     1
    ## 735 -0.7344490522 -0.4475291832 -0.3623747751 -0.3493326036     1
    ## 736 -0.5707092669  0.0256187203  0.0818803892 -0.3493326036     1
    ## 737 -0.4971258082  0.9436216248  0.5535806918  0.6898869849     1
    ## 738 -0.3647490758  0.2293272201  0.2088298917 -0.2814410772     1
    ## 739 -0.6919634037  0.2648783812 -0.1304448044  2.0457606398     1
    ## 740 -0.3990054219  2.1160044423  1.0507438039 -0.3493326036     1
    ## 741  0.0980170826  1.8695701630  1.0008495629 -0.0503702702     1
    ## 742  0.6508927864  1.6936075080  0.8576853717 -0.3192207148     1
    ## 743  0.3943551201  1.4129609932  0.7824070538 -0.3532862866     1
    ## 744 -0.1991931580 -0.2039173282  0.3989271859 -0.1740127207     1
    ## 745  0.2824350248  0.1048860202  0.2544170968  0.9088970619     1
    ## 746 -0.2474221029  1.1595813739  0.1978178747  0.4839360427     1
    ## 747 -0.0262323892 -1.4646300236 -0.4116818199 -0.0418239251     1
    ## 748  0.5224384492 -1.4166037365 -0.4883070357  0.3995508690     1
    ## 749 -0.1678082251 -2.4983002522 -0.7110661519 -0.2322796248     1
    ## 750 -0.0100913271 -2.4008113497 -0.7205566001  0.0621299827     1
    ## 751  0.1138918756 -0.3073749996  0.0616306406 -0.3493326036     1
    ## 752 -0.3935208213 -0.4688925155  0.1059201604 -0.3493326036     1
    ## 753 -0.4133153884 -0.9975477980 -0.2350358825 -0.2018482465     1
    ## 754 -0.5170221601 -0.4346205225  0.2927210573  0.0340548397     1
    ## 755 -0.5002404266 -1.7220602673 -0.5743393633 -0.3077989639     1
    ## 756 -0.2550565816 -1.8658308891 -0.4422043115 -0.1716964216     1
    ## 757 -0.9665642338 -7.2634821463 -1.3248843066 -0.3493326036     1
    ## 758 -1.0230778939 -2.6347612923 -0.4639305463 -0.3493326036     1
    ## 759  0.1797091887  0.0077377190 -0.0688413102 -0.2185016386     1
    ## 760 -0.2688726360 -0.1445823190  0.1044640602  1.9289472782     1
    ## 761 -0.7987535380 -2.9427750731 -0.4626796190 -0.3468166235     1
    ## 762 -0.2326034231 -3.0219924729 -0.4781582349 -0.3468166235     1
    ## 763  0.2125445774  0.5328967383  0.3578916328 -0.2776072028     1
    ## 764  0.4060128667  0.0230251634  0.1647406060 -0.2191805539     1
    ## 765 -0.5845389524 -0.3154835704 -0.0972228121 -0.3493326036     1
    ## 766  0.4138267030 -3.2559809061 -0.5389632270 -0.3188213529     1
    ## 767 -0.0718032361 -3.8381979040 -0.8025644610 -0.3141088823     1
    ## 768  0.0384229399 -4.1267462520 -0.6453292190 -0.3442207710     1
    ## 769 -0.6328163364 -4.3801542591 -0.4678628692 -0.3493326036     1
    ## 770  0.1459714660  0.3001934693  1.7793638524 -0.3502910722     1
    ## 771 -0.4596243875 -4.6261266078 -0.3345607206 -0.3493326036     1
    ## 772 -0.5929648885  0.6755248332  0.4248493518 -0.3496520931     1
    ## 773 -0.2094313591 -4.9500218638 -0.4484127708 -0.3442207710     1
    ## 774 -0.5535269864  0.5162744568  0.2708281871  0.2372102484     1
    ## 775  0.0087112905 -5.4078236392 -0.1838110449 -0.3533262228     1
    ## 776 -0.2200796473  0.3923375841 -0.0200894871 -0.2653068556     1
    ## 777  0.1981072910  0.8696944119  0.2280478518 -0.3533262228     1
    ## 778  0.1603995079  0.9695816052  0.3350412937  0.0620101741     1
    ## 779 -0.0053523692  0.3181522857 -0.3235536226 -0.3442207710     1
    ## 780 -0.1452251871 -5.6823382399 -0.4391336751 -0.3532862866     1
    ## 781  0.0813720237  0.1849830409 -0.2115824555 -0.3493326036     1
    ## 782 -0.3909012285  0.5921969189 -0.2410101529  1.0322200228     1
    ## 783 -0.0724287202  0.1367343088 -0.5998475425 -0.3230146531     1
    ## 784  0.4614499826 -0.1066251807 -0.4796619478 -0.3533262228     1
    ## 785  0.4067025031 -0.4023392727 -0.8828857954 -0.3533262228     1
    ## 786  0.4209192293 -0.2975568849 -0.9461843873  0.1248697403     1
    ## 787  0.0646100247 -0.9723582893  0.3513595612 -0.3442607072     1
    ## 788  0.0503758983  0.7569525626  0.3838689386 -0.3533262228     1
    ## 789  0.4304574392  0.8246854504  0.3269522731  0.3900061191     1
    ## 790 -0.1299536500 -0.0500768353 -0.0510818961 -0.3493326036     1
    ## 791  1.0956713288  0.4717414017 -0.1066671407 -0.3502910722     1
    ## 792  0.4929199892  0.3599764495 -0.1154710095 -0.0329580905     1
    ## 793  0.0758089204  0.0338647379 -0.8328550202  2.1830213318     1
    ## 794  0.0910116872  0.0249284935 -0.0778199786 -0.2851950792     1
    ## 795 -0.1079441118  0.6252885504  0.3156773907 -0.0937409747     1
    ## 796  0.4739883404  0.2871287904  1.4686534186  0.0695581144     1
    ## 797  0.4661647830  0.2785467431  1.4719881173  0.0695581144     1
    ## 798  0.0960810570  0.0700362726  0.0637683650  0.2242309860     1
    ## 799 -0.8945089222 -0.3975573870  0.3142617141  8.1365890281     1
    ## 800 -0.5390727727  0.5034184376 -0.2378071297 -0.3093964116     1
    ## 801  0.5442098923 -1.1675658130 -1.2063543258 -0.1715766130     1
    ## 802  0.4765059570  0.2505305614  0.2509872229 -0.1935814548     1
    ## 803  0.2795564882  0.0316687807  0.0358833226 -0.3404667690     1
    ## 804  0.2686246380  0.2836894741  0.4191016364 -0.3533262228     1
    ## 805  0.7866045292 -0.1461016438  0.0762114891 -0.2534857428     1
    ## 806 -0.2562781849  0.6526738068  0.3198792280 -0.0821594791     1
    ## 807 -0.3375312528 -1.6127910464  1.2314247347  0.0194781296     1
    ## 808  0.5555442341  0.1767400040  0.3629065758 -0.3493326036     1
    ## 809 -0.4116488511  0.5738030165  0.1760672852  0.3491513947     1
    ## 810 -0.3343689919  0.7156003264  0.3704497470  2.5252744973     1
    ## 811  2.7452606722 -0.2204016003  0.1682327405 -0.3246520370     1
    ## 812  0.1677390674  0.4512425415  0.2684212097 -0.3338772973     1
    ## 813  0.0303066562  0.4311824434  0.1106982516 -0.0302424294     1
    ## 814  0.0645577482  0.4766294386  0.3237401806 -0.3533262228     1
    ## 815 -0.2588023871  0.2961454535 -0.0471736660 -0.3453389844     1
    ## 816 -0.4878613953  0.0626546228 -0.2407323492 -0.3493326036     1
    ## 817 -0.3485223182  0.4200364799 -0.3276426965  1.0945604185     1
    ## 818 -0.5083981429 -0.0155514882  0.0418807020  0.7495915919     1
    ## 819 -0.2459727623 -0.1178576687  0.1447741805  2.5348991196     1
    ## 820 -0.1374023633 -0.0012498171 -0.1827508969  1.5664863995     1
    ## 821 -0.7670696066  0.3870389109  0.3194015345 -0.3493326036     1
    ## 822 -0.2938711044  0.2126630598  0.4310947080  4.9781554106     1
    ## 823  0.0658967839  0.4882577387  0.3258716303 -0.3533262228     1
    ## 824 -0.1082841100  0.2694765489 -0.0632454644  0.1666829334     1
    ## 825  0.8239617054  0.6684965658  0.5956098287 -0.3493326036     1
    ## 826  0.0981333586  0.9567687811  0.1627773462  3.3420095601     1
    ## 827  0.0560347439  0.4918278233  0.3408466857 -0.3533262228     1
    ## 828  0.1929592138  0.1808594494 -0.0293153379  1.0244724016     1
    ## 829 -0.1209796132 -0.0195787926  0.0061547052  0.0800213967     1
    ## 830 -0.0043267562 -0.2354773111  0.0181292220 -0.0167439965     1
    ## 831 -0.1630091265  0.8537918396  0.3565031021 -0.1957779453     1
    ## 832  0.8412945885  0.6430942490  0.2011557496 -0.3532862866     1
    ## 833  0.1083611346  1.1308281638  0.4157027894  2.5536691298     1
    ## 834  0.6306388362 -0.5138802225  0.7295258501 -0.2635895994     1
    ## 835 -0.1236122653  0.8774238514  0.6675684298 -0.3201791834     1
    ## 836  0.1706364284  0.8517983057  0.3720982822  0.1280646357     1
    ## 837 -0.2006115498  0.3568559115  0.0321134825 -0.3505706256     1
    ## 838  0.1726429210  0.7267810123  0.2345139225  2.5348991196     1
    ## 839  0.0003653025  0.1696619616  0.1082756932 -0.3502511360     1
    ## 840  0.0809988304  1.2151516137 -0.9231416086  2.0144906015     1
    ## 841  0.1129720370  0.9105906438 -0.6509437128  0.4280653101     1
    ## 842  0.2629378550  0.0991446313  0.0108097183 -0.3353948726     1
    ## 843  0.4732231708 -0.1602019372  0.0650394521 -0.3502910722     1
    ## 844  0.0147767561 -0.2409299595 -0.7810551958  0.9429626337     1
    ## 845  0.0002506190  1.0360113828  0.0043667546  0.9246718577     1
    ## 846 -0.6347829343  0.8227132280  0.4943751784 -0.3453389844     1
    ## 847 -0.7883289132  1.0554419748  0.0999714026  0.2249498375     1
    ## 848 -0.2028806965  0.0921238810  0.0949558537 -0.3533262228     1
    ## 849  0.1141072242  0.5009956085  0.2595332702 -0.3493326036     1
    ## 850  0.0466646931  0.5146457799  0.1409986731 -0.3493326036     1
    ## 851 -0.6998085503  0.5804867169 -0.1736298696 -0.3177830119     1
    ## 852  0.5369052955  0.4858640266 -0.0423934809 -0.3493326036     1
    ## 853  0.0291067416  0.5198065519 -0.4695371135  2.3126142749     1
    ## 854 -0.2612578067  0.7801254840 -0.7318008162 -0.3533262228     1
    ## 855  0.2105097562  0.6487047990  0.3602243303 -0.3486137522     1
    ## 856  0.1507776987 -0.0137115779  0.0480836587 -0.3444603882     1
    ## 857  0.4664677249  0.5663696871  0.2629904662 -0.3502511360     1
    ## 858  0.5752947982  0.1528758908 -0.0981728104  0.0253487498     1
    ## 859 -0.3198594445  0.0154338686 -0.0501165002 -0.3213772692     1
    ## 860  0.5413254933 -0.1362433822 -0.0098522563  3.6253967786     1
    ## 861  0.0489347748  0.8472198403  0.5319317994 -0.3533262228     1
    ## 862  0.1929216152  0.4681808873  0.2804864709 -0.3469763683     1
    ## 863 -0.0130890305  0.4132911887 -0.1313873464 -0.3533262228     1
    ## 864  0.6395796457  0.3515676377 -0.0018165150 -0.3502910722     1
    ## 865 -0.0878322147  0.7187895780  0.6762159488 -0.3502910722     1
    ## 866 -0.3357017117  0.5770515203  0.3983483680  0.1366109808     1
    ## 867  0.0601710709  0.4851869355  0.3265515717 -0.3533262228     1
    ## 868 -0.3709509854  0.0119702494  0.1458953388 -0.3493326036     1
    ## 869 -1.1499226687 -1.8098861203  0.7230514333 -0.3489332417     1
    ## 870 -0.2631890644  0.3047029504 -0.0443622748 -0.3453389844     1
    ## 871  0.0127924878  0.0408538192  0.0229031159 -0.2838771849     1
    ## 872 -0.5213385355  0.1444648602  0.0265881155 -0.1536452628     1
    ## 873  0.0440333424 -0.0549876280  0.0823370925  0.5941998688     1
    ## 874  0.1482630600  0.0422777637  0.0405727519 -0.3493326036     1
    ## 875 -0.2638013303  0.1501559430  0.2921121665 -0.3162255004     1
    ## 876  0.5864152485 -0.3481074059  0.0877768724 -0.3105944974     1
    ## 877  0.5020096393 -1.9374728457  1.5212183728 -0.3532862866     1
    ## 878  0.1547070971 -2.0424027264  1.4051405583 -0.1227745863     1
    ## 879  0.2807173480 -2.6491067490  0.5336409398 -0.3493326036     1
    ## 880  0.3598406887 -2.6518248746  0.4221843916 -0.3493326036     1
    ## 881  0.3673487537 -2.6647670950  0.4171006216 -0.3493326036     1
    ## 882  0.3748020361 -2.6785437692  0.4123675671 -0.3493326036     1
    ## 883  0.3822083619 -2.6930356894  0.4079351259 -0.3493326036     1
    ## 884 -0.0772779620 -0.2975953954 -0.2218160342 -0.3434619834     1
    ## 885 -0.6138740167  0.0026541572  0.0723858076  1.0761897702     1
    ## 886  1.8847406747 -1.7425575146 -0.0822159347  0.6365322324     1
    ## 887 -0.4160620976  0.5073698526  0.2437441138 -0.1481740045     1
    ## 888  0.0706001067  0.4884095278  0.2923449743 -0.3533262228     1
    ## 889  0.6473022754  0.1719291954  0.1171882283 -0.3502910722     1
    ## 890  0.4003483511  0.1529468643  0.4777749246  5.6567911214     1
    ## 891 -1.1258810959 -0.1709474273  0.1262213828  4.0276341046     1
    ## 892  0.4047293085  0.7049146536 -1.2299916817 -0.2135495508     1
    ## 893 -0.0673556787  0.8210475859  0.4261752516 -0.3264092294     1
    ## 894  0.5535304299  0.7605419281  0.3867418210 -0.3502511360     1
    ## 895  0.5279703296  0.5588809961  0.1265167620 -0.3502511360     1
    ## 896  0.5350547349  0.5392630721  0.1242550558 -0.3502511360     1
    ## 897 -0.1556458150  0.7994599456  0.3921702364 -0.3377511079     1
    ## 898 -0.2406721236 -0.0066294489  0.0172577349 -0.3407862585     1
    ## 899 -0.0026009880 -0.0266665674  0.0052376674 -0.3251312713     1
    ## 900  0.7323288211  0.6750667033  0.3370762315  0.0253487498     1
    ## 901  0.0340698776  0.5733930320  0.2946863161 -0.3502511360     1
    ## 902  0.5149676032  0.3781052960 -0.0531327938 -0.3502511360     1
    ## 903 -0.2484803115  0.0215269255  0.1091918968  0.3939198659     1
    ## 904  0.2345762634 -0.2797884182 -0.3319333880 -0.3230146531     1
    ## 905  0.0686298518  0.4815879011  0.2682258544 -0.3334779354     1
    ## 906  0.5596282100  0.4094464826  0.2210481048 -0.3502511360     1
    ## 907 -0.7109052420  0.5658463023 -1.0341071866  0.8287850607     1
    ## 908  0.5432435759  0.2338510154  0.1196034609 -0.1715766130     1
    ## 909 -1.1526710661 -0.7360727554  0.7337027325 -0.3337574887     1
    ## 910 -0.6240846659 -1.3330997592  0.4286339943  0.2696783726     1
    ## 911  0.2464658376  0.9026750545  0.4735715000 -0.3345961488     1
    ## 912  0.5025749764  0.8213902874  0.3723788171 -0.3502511360     1
    ## 913 -0.2165857418 -0.0262185638 -0.0250012964 -0.3493326036     1
    ## 914  0.5327720060  0.6639695048  0.1920668354 -0.3502511360     1
    ## 915  0.6016578233  0.1374676823 -0.1713969811  0.1544225224     1
    ## 916  0.1138065714  0.3541242600  0.2875921252 -0.3518086475     1
    ## 917  0.8657580892  0.8546570167 -0.9644824223 -0.1936613272     1
    ## 918  1.2077309094  1.2689584684  0.0975382360 -0.3041647704     1
    ## 919 -0.1208027638  0.0732685586  0.5837994183 -0.3533262228     1
    ## 920 -0.4842799948  0.6835345929  0.4432994993 -0.1939808167     1
    ## 921 -0.6027766087  0.5087124589 -0.0916455257  2.1798264364     1
    ## 922 -0.1366762623  0.0201823653 -0.0154697052 -0.2736535198     1
    ## 923 -0.4142088281  0.4549815849  0.0967109642  1.0407663679     1
    ## 924  0.7883950565  0.2926799663  0.1479679287  1.2041852656     1
    ## 925  0.7394672561  0.3891518435  0.1866365475 -0.3502910722     1
    ## 926  0.4711109626  0.3851074487  0.1943614792 -0.0422632232     1
    ## 927  0.6061158103  0.8848755395 -0.2537003189  0.6251104814     1
    ## 928 -0.2896165857  0.0029875822 -0.0153088128 -0.1834775982     1

## 7. Machine Learning set-up

Under this section, we will explain the procedure of two main splitting
approach to estimate our models’ performance.

**Definition:** Often denoted as the most popular by its simplicity, the
train-test split is a sampling technique dividing the dataset between
training and testing sets. In doing so, the goal would be to have enough
(but not too much) in our training set used for the machine learning
model to predict the observations in the testing set as accurately as
possible. Most would opt for a 70/30 training-testing split,
respectively, others 80/20, 60/40, or whichever else works best for the
case scenario. Further information
[here](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/).
