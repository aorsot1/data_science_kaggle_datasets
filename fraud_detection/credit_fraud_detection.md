fraud_detection
================
Akoua Orsot
2/9/2022

-   [Fraud Detection](#fraud-detection)
    -   [1. Environment Set-up](#1-environment-set-up)
    -   [2. Initial Diagnostics](#2-initial-diagnostics)
    -   [3. Data Cleaning](#3-data-cleaning)
    -   [4. Correlation Analysis](#4-correlation-analysis)
    -   [5. Inquiry Exploration](#5-inquiry-exploration)

# Fraud Detection

This notebook will attempt to build a predictive algorithm to detect a
fraudulent transaction using a training dataset. We will explain the
thinking process at every step using LIME (Local Interpretable
Model-agnostic Explanations) principles making it accessible and
user-friendly.

## 1. Environment Set-up

``` r
## Importing libraries
library(tidyverse)
```

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --

    ## v ggplot2 3.3.5     v purrr   0.3.4
    ## v tibble  3.1.5     v dplyr   1.0.7
    ## v tidyr   1.1.4     v stringr 1.4.0
    ## v readr   2.0.2     v forcats 0.5.1

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(ggplot2)
library(e1071)
library(dplyr)
# install.packages("corrplot")
library(corrplot)
```

    ## Warning: package 'corrplot' was built under R version 4.1.2

    ## corrplot 0.92 loaded

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
