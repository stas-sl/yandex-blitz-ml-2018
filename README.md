This repo contains solutions of all 17 tasks from all qualification and final rounds of [Yandex.Blitz Machine Learning competition](https://contest.yandex.ru/contest/8470) held in the end of June, 2018.

![leaderboard](assets/leaderboard.png)

I didn't participate in the competition when it was really held so this is not real leaderboard (unfortunately). On the other hand I had more time to explore problems, learn something new and try different approaches.

Quickly jump to:

- [A. Stump](#a-stump)
- [B. Coefficients restoration](#b-coefficients-restoration)
- [C. Freshness detector](#c-freshness-detector)
- [D. Feature selection](#d-feature-selection)
- [E. Warm up](#e-warm-up)
- [F. Generalized AUC](#f-generalized-auc)
- [G. Permutations](#g-permutations)
- [H. Restaurants](#h-restaurants)
- [I. Warm up](#i-warm-up)
- [J. Linear separability problem](#j-linear-separability-problem)
- [K. Unique queries](#k-unique-queries)
- [L. Wi-Fi](#l-wi-fi)
- [M. Pairwise ranking](#m-pairwise-ranking)
- [N. Coins](#n-coins)
- [O. SVD recommender](#o-svd-recommender)
- [P. Adversarial attack (white-box)](#p-adversarial-attack-white-box)
- [Q. Adversarial attack (black-box)](#q-adversarial-attack-black-box)

# A. Stump

![a](assets/a.svg)

The are 3 important observations here:

1. All possible candidates for split (i.e. **c** value) are all points in the middle of 2 consecutive <img src="assets/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=16.715802649999993pt height=25.188841500000024pt/> coordinates (dashed lines in the image). So we have to check only <img src="assets/efcf8d472ecdd2ea56d727b5746100e3.svg" align=middle width=42.80482639999999pt height=23.755361600000015pt/> variants (or less if some points have equal <img src="assets/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=16.715802649999993pt height=25.188841500000024pt/> coordinates).
2. If we fix **c** then optimal values for **a** and **b** that minimize MSE would be just mean of <img src="assets/91aac9730317276af725abd8cef04ca9.svg" align=middle width=14.795948499999989pt height=25.188841500000024pt/> coordinates of all points on each side of the split.
3. If we will naively calculate mean for each split by iterating over all points we'll get <img src="assets/c3f65f86f2baa7f28840d7c68c00f5f2.svg" align=middle width=53.9922907pt height=30.005601399999982pt/> complexity which will not work, so instead we should sort all points by <img src="assets/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=16.715802649999993pt height=25.188841500000024pt/> and then store sums of <img src="assets/91aac9730317276af725abd8cef04ca9.svg" align=middle width=14.795948499999989pt height=25.188841500000024pt/> and <img src="assets/6e9fb305c704f8322e39f6132d0468fe.svg" align=middle width=22.142735099999992pt height=30.005601399999982pt/> for left and right sides of current split. Then going to next <img src="assets/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=16.715802649999993pt height=25.188841500000024pt/> we can update those sums only by value of points that change side. Thus we will be able to easily compute mean as well as MSE for both sides and find split that minimizes MSE in <img src="assets/e7a2f022962441f2be6dc8e70e837b4a.svg" align=middle width=45.72395804999999pt height=27.646325999999984pt/>, though overall complexity will be <img src="assets/2a614f0f77ff651ee3f7ab0f114a1a1d.svg" align=middle width=92.4921782pt height=27.646325999999984pt/> due to sorting.

Implementation: [a.py](a.py)

# B. Coefficients restoration

Let's ignore noise and define the following function (MSE):

<p align="center"><img src="assets/dc1154838a80353a1d3610a809aa9dba.svg" align=middle width=385.46306960000004pt height=50.33949715000001pt/></p>

To find coefficients *a*, *b* and *c* we minimize this function using `scipy.optimize.minimize`.

Implementation: [b.py](b.py)

# C. Freshness detector

That is pretty straightforward classification problem. Though there are 2 question to address:

1. **How to use query text**. Though we are provided with obfuscated query text, nevertheless we still can use standard `TfidfVectorizer` from sklearn to calculate tfidf features - just need to customize `token_pattern` a little bit.
2. **How to perform validation**. If we look at dates in train and test sets we'll notice that all dates in test data go the next day after training dates. Namely, training dates are 24-29 of January and test date is 30 of January. So in order to validate we can split training data into training - 24-28 of January and validation - 29 of January.

![c](assets/c.svg)

After we decided on these questions we can feed the data into a classifier. I used **lightgbm** for this purposes because it works well with sparse features and is very fast.

The last thing to note is that usage of query features is important for this task because without them we can only get <img src="assets/a4d98285c4b44ddc302e89d6858f3a77.svg" align=middle width=77.46089119999999pt height=25.188841500000024pt/>, which is not enough, but when we add query information we can get <img src="assets/4ee778f5b39cbea54f35d4284a13b806.svg" align=middle width=77.46089119999999pt height=25.188841500000024pt/>.

Implementation: [c.ipynb](c.ipynb)

# D. Feature selection

In order to extract most important features, we train catboost model with specified parameters `CatBoostRegressor(n_estimators=200, learning_rate=0.1)` and then use `get_feature_importance` method to extract top 50 most imporant features.

In the task statement it is said that cross-validation is being performed in inverse manner: we train on small amount of data (1000 samples) and validate on much larger data (9000 samples), so it is easy to overfit in this setting. The more features we have, the easier it is to overfit. Although we are constrained to select no more than 50 most important features, but we also could try to select less, and submit 40, 30, 20 top features. Indeed sending only 20 features gives us best result and we get 4 points (max for the task) on the leaderboard.

![d](assets/d.svg)

Implementation: [d.ipynb](d.ipynb)

# E. Warm up

Just feed the data into sklearn `LinearRegression` model.

Implementation: [e.ipynb](e.ipynb)

# F. Generalized AUC

![f](assets/f.svg)

This problem looks less like machine learning problem but more like traditional competive programming problem.

Let's recall the formula for generalized AUC:

<p align="center"><img src="assets/e457cd2630a616fa81d566621747dcb5.svg" align=middle width=403.5843709pt height=62.464926250000005pt/></p>

The naive solution would be to go over all pairs of <img src="assets/77a3b857d53fb44e33b53e4c8b68351a.svg" align=middle width=6.349677299999989pt height=24.311253299999994pt/> and <img src="assets/36b5afebdba34564d884d347484ac0c7.svg" align=middle width=8.645012999999988pt height=24.311253299999994pt/> and just calculate cases when <img src="assets/50a4d4d808e63c013645de93b95e08ca.svg" align=middle width=50.866177349999994pt height=22.672930299999983pt/> and <img src="assets/1d79dc63628f7df562d6d4a19ac118f9.svg" align=middle width=55.62759265pt height=19.872096900000006pt/>, though the complexity will be <img src="assets/c3f65f86f2baa7f28840d7c68c00f5f2.svg" align=middle width=53.9922907pt height=30.005601399999982pt/> and we'll get TL for <img src="assets/b5b9788de90f0b9dfb14ed2d64708bb8.svg" align=middle width=67.17020885pt height=30.005601399999982pt/>. So we need something smarter.

Let's try to put those points on a plane: <img src="assets/02ab12d0013b89c8edc7f0f2662fa7a9.svg" align=middle width=11.870269699999989pt height=22.672930299999983pt/> on <img src="assets/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=16.715802649999993pt height=25.188841500000024pt/>-axis, and <img src="assets/2b442e3e088d1b744730822d18e7aa21.svg" align=middle width=14.250977349999992pt height=15.871031600000025pt/> on <img src="assets/91aac9730317276af725abd8cef04ca9.svg" align=middle width=14.795948499999989pt height=25.188841500000024pt/>-axis. Thus going from left to right on X-axis for fixed <img src="assets/7d4058a1e5678e1dc8e8a21ba778b29c.svg" align=middle width=53.160360499999996pt height=25.188841500000024pt/> all points to the left will have <img src="assets/aea1accf0ad68965a6a7179a094f2c57.svg" align=middle width=50.86617364999999pt height=22.672930299999983pt/> and we just need to calculate number of points that also have <img src="assets/fb7c214da6059aa80d0a6685a8e8f025.svg" align=middle width=55.62758894999999pt height=19.872096900000006pt/> among them. Geometrically this means that we should calculate number of points in rectangle <img src="assets/09b1a71bfd74646f4aeca7a17736f531.svg" align=middle width=85.30475985pt height=27.646325999999984pt/>, of course we should not forget to consider cases when <img src="assets/a96c3bf71e0d8dd09b9535b7a3716ad3.svg" align=middle width=55.62758894999999pt height=15.871031600000025pt/> - points that lie on the top border of the rectangle. Naively calculating points inside this rectangle still requires <img src="assets/c3f65f86f2baa7f28840d7c68c00f5f2.svg" align=middle width=53.9922907pt height=30.005601399999982pt/>, so we should somehow organize our points to be able to quickly answer the question "how many points are there which have <img src="assets/aea1accf0ad68965a6a7179a094f2c57.svg" align=middle width=50.86617364999999pt height=22.672930299999983pt/> and <img src="assets/fb7c214da6059aa80d0a6685a8e8f025.svg" align=middle width=55.62758894999999pt height=19.872096900000006pt/>". To do so, we will store all <img src="assets/fba353e8e83ce14fc4a80553757972f7.svg" align=middle width=15.880781099999991pt height=15.871031600000025pt/> of points that lie to the left of <img src="assets/7d4058a1e5678e1dc8e8a21ba778b29c.svg" align=middle width=53.160360499999996pt height=25.188841500000024pt/> in a sorted array or a binary tree. Initially this array will be empty, but as we go from the left to the right along the <img src="assets/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=16.715802649999993pt height=25.188841500000024pt/>-axis we'll add points there. Notice, that we don't care about specific <img src="assets/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=16.715802649999993pt height=25.188841500000024pt/>-coordinates of points in this array. All we need to know is that their <img src="assets/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=16.715802649999993pt height=25.188841500000024pt/>-coordinate is less than current <img src="assets/7d4058a1e5678e1dc8e8a21ba778b29c.svg" align=middle width=53.160360499999996pt height=25.188841500000024pt/>. So, if we have such sorted array/bin tree, then we can easily answer to the required question in <img src="assets/75ff77642a7a68c66eacceb8a0740bba.svg" align=middle width=72.6022917pt height=27.646325999999984pt/> time doing binary search. The only question to cover yet is how much time do we need to insert new points into our array/bin tree. In case of sorted array if we insert some elements in the middle of it we might need to shift all other elements to the right of it, so we might need <img src="assets/e7a2f022962441f2be6dc8e70e837b4a.svg" align=middle width=45.72395804999999pt height=27.646325999999984pt/> in worst case. In case of binary tree, if the tree is balanced insertion will take <img src="assets/75ff77642a7a68c66eacceb8a0740bba.svg" align=middle width=72.6022917pt height=27.646325999999984pt/> time, though if it is not we'll end up with <img src="assets/e7a2f022962441f2be6dc8e70e837b4a.svg" align=middle width=45.72395804999999pt height=27.646325999999984pt/> still. Let's hope we will be lucky and the order of elements will be random, so the tree will be quite balanced.

I ended up with several implementations for this task using C++ and Python and trying sorted array/binary tree. Though sorted array required <img src="assets/c3f65f86f2baa7f28840d7c68c00f5f2.svg" align=middle width=53.9922907pt height=30.005601399999982pt/> time it surprisingly didn't get TL even using Python implementation. But it was very-very close to it. Binary tree version of solution was much faster using C++ implementation and took only *125ms* for the longest test case. But Python version of it was quite slow, even slower than sorted array version, so it exceeded the TL.

Implementation: [f_naive.cpp](f_naive.cpp), [f_sorted_naive.cpp](f_sorted_naive.cpp), [f_fastest.cpp](f_fastest.cpp), [f_sorted_naive.py](f_sorted_naive.py)

# G. Permutations

As it is said in the task description, stupid permutation function has a bug. As a result not all permutaitions occur evenly - some may occur more often while others less often. Let's generate 100000 stupid permutations and calculate how often each of numbers (0-7) was placed on each position (0-7).

![g](assets/g.svg)

Columns in matrix represent numbers, while rows - positions. Thus the number in each cell means how many times each number was on each position.

From the matrix we see, that the bug in stupid permutation causes some numbers to stay on initial position more often, than it should be if all permutations were evenly distributed. E.g. 0 stays on position 0, 1 on 1, 2 on 2 etc.

So all we are left to do is calculate number of such permutatations in groups of 1000 of them and sort by this number in ascending order, thus, random permutations will go first and stupid ones later.

Implementation: [g.ipynb](g.ipynb)

# H. Restaurants

We will use a simple linear model with 2 coefficients to predict restaurant score based on distance and rating:

<p align="center"><img src="assets/61fb1b29365a17ebd35401f07e228426.svg" align=middle width=168.38135195pt height=15.563833550000002pt/></p>

We may notice that the expression we are asked to minimize is exactly the negative log likelihood (divided by constant *N*) or loss function of logistic regression:

<p align="center"><img src="assets/8370200341c3c39d2d5782932bba2bec.svg" align=middle width=405.8355249pt height=54.02313945pt/></p>

Because logistic regression requires only labels 0 or 1 and target metric will be evaluated on pairs with known winner and looser, we will omit samples with ties (0.5 target) during training. To solve logistic regression we will use SGD.

After we obtain coefficients of scoring model we can use them to score all restaurants in the test data.

Another very important detail worth mentioning is rating and distance transformations before training. If we plot distance and ratings distributions we might notice that there are a lot of small distances near zero and ratings are concentrated around 7-8. Logically, users might be much more sensitive to difference between 100 meters and 1 km than 100km and 101 km. On the contrary low ratings 2 or 3 are percieved as almost equally poor, while difference between 8 and 9 rating feels more important. In order to take this into account, we might try to apply log transformation to distances and exponential transformation to ratings. Indeed, the model without any transformations scored 1.5 on the leaderboard, when using only distance transformation score raised up to 3.5, and after we applied both transformation we achieved max score of 4 points.

![h](assets/h.svg)

Implementation: [h_exploration.ipynb](h_exploration.ipynb), [h.py](h.py)

# I. Warm up

Although it is stated that we should find a linear function, I was unable to train linear model, so I used **lightgbm**, which might be an overkill for this task, but anyway. 

As it is said that only few features are relevant, we'd like to find out what are those features.

![i](assets/i.svg)

We see that features 5 and 95 are the most important ones. Let's look at them and the target:

|      | 5    | 95   | 100(target) |
| ---- | ---- | ---- | ----------- |
| 0    | 0    | 1    | 1           |
| 1    | 0    | 1    | 1           |
| 2    | 0    | 0    | 0           |
| 3    | 1    | 1    | 0           |
| 4    | 1    | 1    | 0           |
| 5    | 1    | 1    | 0           |
| 6    | 1    | 1    | 0           |
| 7    | 1    | 1    | 0           |
| 8    | 0    | 1    | 1           |
| 9    | 0    | 1    | 1           |

Looks like XOR: <img src="assets/14279deb74124feae51cac1eb8dee83f.svg" align=middle width=247.87254230000005pt height=25.59845739999999pt/>. Double-checking on the whole dataset confirms our hypothesis.

Implementation: [i.ipynb](i.ipynb)

# J. Linear separability problem

The task is given a set of points <img src="assets/9516deb02a4c8362bc9b12a69f6b87cb.svg" align=middle width=237.7464563pt height=27.646325999999984pt/> and their respective classes <img src="assets/0639f95b687db1c9499938d0e983f9f5.svg" align=middle width=277.1705668pt height=27.646325999999984pt/> to find a hyperplane separating those points according to their classes. Formally, find such a vector <img src="assets/08d68c5754b303eac3b4d98eb1598a65.svg" align=middle width=58.65881394999999pt height=25.393651300000002pt/> that:

<p align="center"><img src="assets/05d473830075ef5f0eccaaf54c7a8456.svg" align=middle width=271.40544325pt height=66.3518299pt/></p>

Although there are a few linear models that might be applicable to this task, namely linear/logistic regression, SVM, perceptron, not all of them will find a hyperplane that splits all points exactly. This can happen if, according to the respective loss function, it is 'cheaper' to misclassify a single point, but the total loss for other points will be less. Though it doesn't apply to the perceptron model - it tries to find separating hyperplane, that splits the classes exactly, if it is possible, but if not it will never converge. In the task statement it is said that the input dataset is known to be linearly separable, so we can use the perceptron here. During training only points that are misclassified contribute to the error, so if a point was already classified correctly it doesn't matter how far it is from decision boundary, so we may end up with decision boundary being very close to some of training points, but in our task that is acceptable, as we don't have any other requrements for the hyperplane.

The perceptron model has <img src="assets/0e51a2dede42189d77627c4d742822c3.svg" align=middle width=16.18256789999999pt height=15.871031600000025pt/> parameters <img src="assets/31de31d2aa8a24f524564ad3fb7914bc.svg" align=middle width=148.34946499999998pt height=27.646325999999984pt/> and maps the input vector <img src="assets/9fc20fb1d3825674c6a279cb0d5ca636.svg" align=middle width=15.74841914999999pt height=15.871031600000025pt/> to the output class <img src="assets/2b442e3e088d1b744730822d18e7aa21.svg" align=middle width=14.250977349999992pt height=15.871031600000025pt/>:

<p align="center"><img src="assets/f32523c1fbf4fe3d2a003130780dbc7f.svg" align=middle width=246.959053pt height=66.3518299pt/></p>

In order to find the parameters of vector <img src="assets/44bc9d542a92714cac84e01cbbb7fd61.svg" align=middle width=9.74238489999999pt height=15.871031600000025pt/>, an iterative update rule is used:

<p align="center"><img src="assets/2e90101bb57dc32f618881690607d249.svg" align=middle width=628.1722008500001pt height=22.06585465pt/></p>

Despite linear/logistic regression might not always find separating hyperplane correctly, my solutions using both of them were accepted - probably the tests were not so hard and the points were separated by a wide margin.

Implementation: [j.py](j.py)

# K. Unique queries

The idea for this task is to implement some kind of [Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter). *Some kind* because actually we will use *k* = 1 (number of filters). So it will be more like regular hash-table but without storing the values of elements - only their hashes. There is a formula for calculating false positive rate of *Bloom Filter* depending on number of unique elements, size of filter and number of filters, but for me it was easier to write a small test to check the error rate for various combinations of parameters.

I used 2 a bit different implementations, though in both of them I used `bytearray` which I never used before - that is just, as its name says, array of bytes, just more memory efficient than usual `list` of any objects. Using it, it was possible to allocate array of 1,000,000 elements keeping memory usage below 5MB. That is more bytes that we could probably allocate if we were using C++, because its memory limit was 10x times lower - just 500KB. For calculating hashes of queries I used built-in function `hash` which is actually used for hashing elements in regular python hash-tables aka dicts.

So, being able to use hash-table of size 1,000,000 it was possible to implement a very straightforward solution, using whole bytes to store just 1 bit of information - whether we've seen such hash before or not. Though, probably the error rate was pretty close to 5%. Another a bit more sophisticated but more robust approach was to use every bit of a single byte, thus we can achieve even lower error rates using just 200,000 bytes.

Implementation: [k_test.ipynb](k_test.ipynb), [k_bytearray.py](k_bytearray.py), [k_bitarray.py](k_bitarray.py)

# L. Wi-Fi

The most important observation in this task is similarity between **ssid** and **organization name** for those rows where *target=1*. Here are just a few samples with *target=1*:

| id   | names                                             | ssid                |
| ---- | ------------------------------------------------- | ------------------- |
| 18   | ["Аэропорт Толмачево, бухгалтерия", "Толмачево"]  | Tolmachevo-MTS-Free |
| 38   | ["Kontrolmatik"]                                  | Kontrolmatik_Staff  |
| 49   | ["ПКВ Моторс", "Pkw Motors", "Pkw Motors", "Те... | PKW Guests          |
| 77   | ["Техцентр Юста", "Tekhtsentr Yusta", "Юста", ... | YUSTA               |
| 94   | ["Респект Авто", "Автосервис"]                    | RespectAuto         |

While **ssid** contains mostly latin characters, organization names might be in other languages like russian, so we need to perform some transliteration before calculating similarity. After that we will compute similarity between 2 strings as number of n-grams ocurred in both strings. We will use **n=1..8**. Also we'll compute similarity for **ssid** and **urls** fields, because **urls** might also include some substrings from **ssid**.

Other features are more obvious: we calculate distance between user and organization and perform one-hot encoding of **rubrics**. Also we feed some fields as is, like **has_wifi**, **publishing_status**.

For validation I used 20% of data. Though it was important to perform group-aware splits, that's why I used `GroupShuffleSplit` which places all samples from one group to either train or test set, so that test and train sets have non-intersecting groups.

For training I used **lightgbm** model, which after *170* iterations scored *3.71* on the leaderboard that corresponds to 95% accuracy.

Implementation: [l.ipynb](l.ipynb)

# M. Pairwise ranking

This task might look similar to [H. Restaurants](#h-restaurants). The training dataset here is also composed of pairs of items and we are asked to maximize log likelihood of the data. Though, unlike task H, here we don't have any features of objects that can be used as input to scoring model, so instead we will consider each <img src="assets/77a3b857d53fb44e33b53e4c8b68351a.svg" align=middle width=6.349677299999989pt height=24.311253299999994pt/>-th object's score as a parameter/weight <img src="assets/73cff7830c7881710ad86fddbe27bb4c.svg" align=middle width=84.2050884pt height=27.646325999999984pt/>. 

Thus our task can rewritten as:

<p align="center"><img src="assets/7e5ccf04b2f3a6b4c5c2fc80a7a171b5.svg" align=middle width=582.27460365pt height=38.5248921pt/></p>

Or in vectorized form:

<p align="center"><img src="assets/2fcf8e9215cffe44a804684a8abac421.svg" align=middle width=102.71766840000001pt height=21.0338081pt/></p>

Where <img src="assets/6feb3bad07ddeea39e0e6487b370ddd2.svg" align=middle width=158.6880827pt height=27.646325999999984pt/> - vector of all objects' scores and <img src="assets/cbfb1b2a33b28eab8a3e59464768e810.svg" align=middle width=16.715802649999993pt height=25.188841500000024pt/> - is <img src="assets/63b142315f480db0b3ff453d62cc3e7f.svg" align=middle width=49.77191529999999pt height=21.5027202pt/> design matrix, where each row contains only 2 non-zero elements, namely <img src="assets/e33baa1a785610766d8c96a85ea0d929.svg" align=middle width=74.94207365pt height=23.755361600000015pt/> and <img src="assets/a6b6803e79a52d9c48fbe3a16bbbed28.svg" align=middle width=89.27725745pt height=23.755361600000015pt/> for all <img src="assets/f68112419890e1aaee4b3945368ad473.svg" align=middle width=71.68115264999999pt height=24.311253299999994pt/>.

There is always <img src="assets/034d0a6be0424bffe9a6e7ac9236c0f5.svg" align=middle width=9.215477149999991pt height=23.755361600000015pt/> on the right side of equation because first object <img src="assets/23a9f1f890788fca12299080b7ddeeb9.svg" align=middle width=22.30382384999999pt height=15.871031600000025pt/> is always preferred over the second <img src="assets/91cfda11becf7d409e7826d26965b2e0.svg" align=middle width=22.30382384999999pt height=15.871031600000025pt/>.

After those preparations we can just fit logistic regression to obtain scores <img src="assets/c2a29561d89e139b3c7bffe51570c3ce.svg" align=middle width=18.40963859999999pt height=15.871031600000025pt/>.

Implementation: [m.py](m.py)

# N. Coins

Well, the first straightforward idea was simply to calculate frequency of getting heads up <img src="assets/bffefda2b7e31010aac1f05438156f1a.svg" align=middle width=18.9173896pt height=26.222958199999997pt/> and sort coins by this number. But, unfortunately, it didn't work. The problem is that if we have 2 coins <img src="assets/77a3b857d53fb44e33b53e4c8b68351a.svg" align=middle width=6.349677299999989pt height=24.311253299999994pt/> and <img src="assets/36b5afebdba34564d884d347484ac0c7.svg" align=middle width=8.645012999999988pt height=24.311253299999994pt/> with <img src="assets/cf377df1cda8314d1a503e6c1b83ba72.svg" align=middle width=113.82111865pt height=25.59845739999999pt/> and <img src="assets/170dbe8c5612c56a7ce62f6dab302a13.svg" align=middle width=126.2961996pt height=25.59845739999999pt/> then we will get <img src="assets/eb6a85f7ef7ac49a13a97219e8627d42.svg" align=middle width=87.0395733pt height=31.14240049999998pt/>. Which means that we will consider both of the coins to have equally low chance of getting heads up <img src="assets/88fdf1bf393110b9b577366c6fe9bd53.svg" align=middle width=90.812356pt height=23.755361600000015pt/>. Though, intuitively, of course, it doesn't seem right: if we toss a coin only single time and didn't get heads up, we shouldn't be so confident that we will never get it at all, though if we didn't get a single heads up after 50 tosses, well, chances to get a one are really low. At this point we should probably start to realize that Bayesian statistics was invented for a reason. 

Unfortunately my understanding of Bayessian statistics not as profound as I'd like, but the idea is that instead of estimating a single number <img src="assets/0d19b0a4827a28ecffa01dfedf5f5f2c.svg" align=middle width=14.48770519999999pt height=15.871031600000025pt/> we deal with whole distibution of all possible values of <img src="assets/0d19b0a4827a28ecffa01dfedf5f5f2c.svg" align=middle width=14.48770519999999pt height=15.871031600000025pt/>. In order to do this, we choose some plausible prior distribution, which in case if we don't have any specific information, we may choose as <img src="assets/7a2a4464adb558e31e86e4e9f6b2c5e0.svg" align=middle width=55.55121169999999pt height=27.646325999999984pt/> - standard uniform distibution, i.e. all <img src="assets/0d19b0a4827a28ecffa01dfedf5f5f2c.svg" align=middle width=14.48770519999999pt height=15.871031600000025pt/> are equally likely. Then after each toss we update the resulting distribution (a posteori) according to the observed outcome, thus the initial distibution is gradually shifting towards the true distribution. In the end to be able to compare a posteori distribution of different coins, we still have to come up to a single number (point estimate), but this time we  have some options how to calculate it from whole distibution. The most common approaches are to select mode (maximum a posteori) or mean (expected value) of the distibution. Intuitively expected value contains more information about the distribution because it is integral over all possible values, while maximum value is just a single point. So we'll use mean of the distribution as its point estimate.

That was an overall justification, now let's get to our task. According to wikipedia articles [Checking whether a coin is fair](https://en.wikipedia.org/wiki/Checking_whether_a_coin_is_fair) and [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) we know that the a posterior distribution of <img src="assets/0d19b0a4827a28ecffa01dfedf5f5f2c.svg" align=middle width=14.48770519999999pt height=15.871031600000025pt/> after getting <img src="assets/a157a419f4d70215e97230c8bfe59bb8.svg" align=middle width=62.726819649999996pt height=25.59845739999999pt/> heads and <img src="assets/684b9ab190097118dc7494cc56ce8a18.svg" align=middle width=97.02116319999999pt height=25.59845739999999pt/> tails is in fact a <img src="assets/57500b0520bf2dc4f1978141f7f782fc.svg" align=middle width=39.88464764999999pt height=25.188841500000024pt/> distribution: <img src="assets/5f10e1a3673762981a9e3a8876883f25.svg" align=middle width=428.79065605pt height=27.646325999999984pt/> with probability density function:

<p align="center"><img src="assets/c247306cd4d9f509fbe04084f687c4fa.svg" align=middle width=619.1771327pt height=46.8699461pt/></p>

From the properties of <img src="assets/57500b0520bf2dc4f1978141f7f782fc.svg" align=middle width=39.88464764999999pt height=25.188841500000024pt/> distribution we know that its expected value is <img src="assets/8a0e49c8ec3a0dd45ee07822d28a29e3.svg" align=middle width=96.49313435pt height=31.7419892pt/>, while its mode is <img src="assets/1e09657fbd3f90d78db722b284d92fde.svg" align=middle width=96.6978868pt height=31.14240049999998pt/>. So if we would use mode instead of mean we would get the same results as our initial idea that didn't work.

Implementation: [n.py](n.py)

# O. SVD recommender

![o](assets/o.svg)

This task is classical recommender system. It is well known since Netflix competition in 2006-2009. One
of the competitors Simon Funk has [really nice description](http://sifter.org/simon/journal/20061211.html) of his method that uses SGD to find matrix factorization. It is good, because we don't need to deal with huge sparse matrices.

Another useful resource was [suprise](http://surpriselib.com/) library. It does exactly what is required in our task and 
it has convenient methods to get sample data for testing purposes. We will try to implement our own algorithm
to find matrix factorization and compare the results with those received using this library.

The algorithm is pretty straightforward. The [idea](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) is to represent ratings as <img src="assets/d42d1a09987f1cb87139da3ec00b2aad.svg" align=middle width=213.960418pt height=31.0090858pt/>

Where:

<img src="assets/f7115ca4804f002d341b17739b30a8a2.svg" align=middle width=16.624153649999993pt height=25.59845739999999pt/> - per user average rating minus global average, vector of size <img src="assets/6bac6ec50c01592407695ef84f457232.svg" align=middle width=14.59365284999999pt height=25.188841500000024pt/><br>
<img src="assets/f8fd3079e1d6ddd6a40ffb421d3e5ed1.svg" align=middle width=20.988693999999988pt height=25.59845739999999pt/> - per movie average rating minus global average, vector of size <img src="assets/fb97d38bcc19230b0acd442e17db879c.svg" align=middle width=19.89000859999999pt height=25.188841500000024pt/><br>
<img src="assets/13086b9ea21e25dc5977f88758215527.svg" align=middle width=17.98729099999999pt height=15.871031600000025pt/> - user embedding, vector of size <img src="assets/2103f85b8b1477f430fc407cad462224.svg" align=middle width=9.59305104999999pt height=25.59845739999999pt/> (number of factors)<br>
<img src="assets/001683f492003af3c5898ea899494eb0.svg" align=middle width=21.30657579999999pt height=15.871031600000025pt/> - movie embedding, vector of size <img src="assets/2103f85b8b1477f430fc407cad462224.svg" align=middle width=9.59305104999999pt height=25.59845739999999pt/> (number of factors)<br>

We initialize these variables with some random values and then iterate over each known user-movie-raiting tuples and compute 
error. Then we update just a little bit all parameters to minimize the error:

<p align="center"><img src="assets/acf0e13f23be91d60bb822910780b719.svg" align=middle width=239.57638750000004pt height=101.36986385000002pt/></p>

Where <img src="assets/9be29c69a1e76559ef3b12be5c5f459b.svg" align=middle width=139.53533055pt height=25.59845739999999pt/>, <img src="assets/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg" align=middle width=10.751395249999991pt height=25.59845739999999pt/> - regularization parameter, <img src="assets/11c596de17c342edeed29f489aa4b274.svg" align=middle width=10.566169549999989pt height=15.871031600000025pt/> - learning rate.

Implementation: [o_test.ipynb](o_test.ipynb), [o.py](o.py)

# P. Adversarial attack (white-box)

![p](assets/p.svg)

The idea is to start with the source image and incrementally update it a little bit in opposite gradient (w.r.t. input image) direction, so that with each iteration the model predicts higher and higher probability that the image belongs to the target class. As long as we know the underlying model is plain linear classifier with softmax activation, we can analytically compute the gradient of the loss function w.r.t. the input.

The loss function that we are going to minimize is:

<p align="center"><img src="assets/2a1465ceb2e667bfb0fadb45da84ae23.svg" align=middle width=376.86357215pt height=130.39180915000003pt/></p>

Where <img src="assets/c97fba47d1059b5c2d3fa5f52768a2a7.svg" align=middle width=10.533774199999991pt height=20.929253500000016pt/> - is original the image, <img src="assets/4f4f4e395762a3af4575de74c019ebb5.svg" align=middle width=6.655624749999991pt height=22.672930299999983pt/> - target class, <img src="assets/227f4d8d12b0de49c4ca84f74fa98023.svg" align=middle width=15.483926499999992pt height=15.871031600000025pt/> - <img src="assets/36b5afebdba34564d884d347484ac0c7.svg" align=middle width=8.645012999999988pt height=24.311253299999994pt/>-th output of softmax, <img src="assets/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg" align=middle width=10.751395249999991pt height=25.59845739999999pt/> - is regularization parameter. Then partial derivatives will be:

<p align="center"><img src="assets/30fdddbcf71e200a82cd233381cd6f45.svg" align=middle width=454.0076675pt height=755.9377426000001pt/></p>

The whole gradient in vector form can be written as:

<p align="center"><img src="assets/b0fc023df9bb92300cc08cb93082221a.svg" align=middle width=297.21883365pt height=21.0338081pt/></p>

Where <img src="assets/f70e5a55c48604acfb3be95ff3209f30.svg" align=middle width=169.18174150000002pt height=31.0090858pt/> - column vector of outputs of softmax, <img src="assets/71c0437a67c94e48f18cc11d0c17a38c.svg" align=middle width=14.14961769999999pt height=15.871031600000025pt/> - unit column vector <img src="assets/c0f929543ff8b52491172f6b25974c14.svg" align=middle width=50.17291944999999pt height=23.755361600000015pt/> where all elements are zeros except <img src="assets/4f4f4e395762a3af4575de74c019ebb5.svg" align=middle width=6.655624749999991pt height=22.672930299999983pt/>-th which is <img src="assets/034d0a6be0424bffe9a6e7ac9236c0f5.svg" align=middle width=9.215477149999991pt height=23.755361600000015pt/>.

Then incremental update rule will be:

<p align="center"><img src="assets/8fde4192a2fad7ec82882fe6d488b15d.svg" align=middle width=228.0022473pt height=21.8939027pt/></p>

Implementation: [p.ipynb](p.ipynb)

# Q. Adversarial attack (black-box)

This task is similar to the previous white box adversarial attack, but here we don't have direct access to gradient, so we need to somehow estimate it using only output of the classifier. 

There are a few different approaches to do it:

- Use *substitute network* - new model that is trained to give the same answers as black-box model. And then hope that it's gradients are similar to the gradients of original model.
- Use *finite difference method* to estimate gradient, while being precise it requires as much model evaluations as the dimensions of input image. We have *32x32x3* input image, so there will be *3072* evaluations per iteration. It actually may be acceptable but we'll use another method.
- *Natural Evolution Strategies (NES)*. This method is well described in the [Black-box Adversarial Attacks with Limited Queries and Information](https://arxiv.org/abs/1804.08598) paper. In short we choose <img src="assets/55a049b8f161ae7cfeb0197d75aff967.svg" align=middle width=11.06286124999999pt height=15.871031600000025pt/> points in the neighborhood of <img src="assets/332cc365a4987aacce0ead01b8bdcc0b.svg" align=middle width=10.533774199999991pt height=15.871031600000025pt/> and estimate gradient using function values at those points using formula:

<p align="center"><img src="assets/378250a2c12c428256e808ccff8adc06.svg" align=middle width=268.92129100000005pt height=50.33949715000001pt/></p>

Where <img src="assets/5852701715a58770b77e393ca9c28a7f.svg" align=middle width=97.01044244999999pt height=27.646325999999984pt/> and <img src="assets/8cda31ed38c6d59d14ebefa440099572.svg" align=middle width=11.192949549999991pt height=15.871031600000025pt/> is standard deviation.

Here we don't depend on size of the input image and we can choose <img src="assets/55a049b8f161ae7cfeb0197d75aff967.svg" align=middle width=11.06286124999999pt height=15.871031600000025pt/> (number of model evaluations) to be *50* or *100*. Then using the estimated gradient we iteratively update image using SGD as in the white-box setting.

<p align="center"><img src="assets/6e511b373a0b32d9f3dc651389067770.svg" align=middle width=237.32007015pt height=21.8939027pt/></p>

![q](assets/q.svg)

After roughly *5000* iterations we've got desired target class probability > 0.5. It look like the *MSE=687* is quite large (much larger than *235* which is the threshold to get max points for this task) and the image is distorted quite a lot. After spending some time figuring out what's wrong and how it is possible to approach much lower threshold, I submitted this solution and got good results. As it turned out there was only one test for this task and the input image in it required less distortions to fool the classifier, so it worked well. After tuning learning rate, regularization parameter and number of points for NES the desired threshold was reached.

Implementation: [q_visualization.ipynb](q_visualization.ipynb), [q.py](q.py)