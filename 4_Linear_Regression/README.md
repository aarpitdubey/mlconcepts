# Linear Regression

- Linear Regression is also known as **Ordinary Least Squares.**
- Regression predicts a **countinuous value**. Now we will build an algorithm that solves the task of predicting marks given a labelled dataset.
- The goal of the algorithm is to learn a **linear model** that predicts a ***y** for an unseen **x** with minimum error*.

  **Input (X) :** It is the independent feature or **input feature** which is **Time Spent**.

  **Output (Y) :** It is the dependent feature (or) the **target** variable which is **Marks**.

![img](./output/time_vs_marks.png "Arpit Dubey")

#### Equation of Line: Y = m * x + c

where, m is the slope, c is the intercept and x is the coefficient.

![img](./output/linear_regression.png "Arpit Dubey")

## Hypothesis

**Step 1: Assume a Hypothesis function**

- Linear Regression is a **parametric method**, which means it makes an assumption about the form of the function relating X and Y. So, we decide to approximate **y** as a linear function of **x**

  ![img](./output/time_vs_marks_plot.png "Arpit Dubey")

**Step 2: Decide Loss/Error Function**

- Given a training set. how do we learn, the best values for the parameters ***θ***, so we define an error function that measures for each value of ***θ*** how close the predictions are to the actual **y** values.

**Loss Function:** The loss function is a measure of how close we are to the true/target value or in general, it is a measure of how good the algorithm is doing. The lower the loss the better our predictions will be.

**J(***θ***) = $\frac{1}{2m} \Sigma_{i=1}^m ( {y}-\hat{y} )^2$**

(OR)

J($\Theta_{0}, \Theta_{1})$ = $\frac{1}{2m} \Sigma_{i=1}^m ( {y}^{i}-h_{\Theta}^{i}(x) )^2$

![img](./output/abs_error.png "Arpit Dubey")

Here in above diagram we can see the Line L1 is better as compare to Line L2 because the distance/error/residual from actual datapoint from the prediction lines L1 & L2 is lesser in comparison.

**Absolute Error :** The absolute error, denoted as **|𝑦 − 𝑦̂|**, measures the absolute difference between the actual value **(𝑦)** and the predicted or estimated value **(𝑦̂)**. It quantifies the magnitude of the error without considering its direction. The formula for absolute error is: **∣𝑦−𝑦̂∣** and Total absolute error will be:

**Total Absolute error = $\Sigma_{i=1}^m  | {y}-\hat{y} |$**

**Sum of Squared Error (SSE) :** The Sum of Squared Error (SSE) is a metric used to quantify the overall discrepancy between the actual data points and the values predicted by a model. It is calculated by taking the sum of the squared differences between each actual data point and its corresponding predicted value. The formula for SSE can be expressed as:

**𝑆𝑆𝐸 = $\Sigma_{i=1}^m ( {y}-\hat{y} )^2$**

**Mean Square Error (MSE) :** The mean square error, often denoted as MSE, is a metric that calculates the average of the squares of the errors between the actual and predicted values. It is represented by the formula:

**𝑀𝑆𝐸 = $\frac{1}{m} \Sigma_{i=1}^m ( {y}-\hat{y} )^2$**

Here, m represents the number of data points, $ y_{i}$ denotes the actual values, and $\hat y_{i}$ represents the predicted values.

**We Prefer MSE Over Absolute Error:**

* **MSE penalizes larger errors more significantly** due to the squaring operation, providing a more accurate representation of the model's performance.

  ![img](./output/differentiability_of_mse.png "Arpit Dubey")
* It is "**differentiable"** and exhibits some desirable mathematical properties, making it convenient for optimization algorithms.

**Step 3: Training - an algorithm that reduces error on training data**

- Starting with a random $\Theta$, we need an algorithm that iteratively improves $\Theta$ by reducing $J(\Theta)$ in each step and converges eventually to minimum error.
- Task: find the parameters that minimize loss, i.e. make our model as accurate as possible.

### Gradient Descent

- The **gradient descent algorithm is an optimization technique used to minimize the cost function by iteratively adjusting the model parameters**.
- To efficiently perform this optimization, it requires the **cost function to be differentiable**.
- The differential function allows the algorithm to calculate the gradient, indicating the direction and magnitude of the steepest ascent or descent. Without a differentiable cost function, the algorithm cannot determine the appropriate direction for parameter updates, hindering its convergence towards the optimal solution.

  ![img](./output/gradient_descent.jpg "Arpit Dubey")
- MSE is preferred over absolute error due to **its enhanced sensitivity to larger errors and mathematical properties**, while a differential function is essential for the gradient descent algorithm to **iteratively optimize the model parameters** in a differentiable cost function.

Let's code Gradient Descent in python using a quadratic function

$y = f(x) = (x - 5)^2 + 3$

**Step 1: Import the necessary libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
```

**Step 2: Get data for X and Y**

```python
# Generate data using a dummy function
X = np.arange(10)
Y = (X - 5)**2 + 3

print(X, Y)
```

![img](./output/gd1.png)

**Step 3: Plot a Graph for these value**

```Python
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

![img](./output/gd2.png)

**Step 4: Perform Gradient Descent Optimization**

```python
# Gradient Descent when slope/gradient is negative
# Initialize x with any random value
x = 0
# Learning rate (lr)
lr = 0.1

# Plot X and Y
plt.plot(X, Y)

for i in range(50):
	# let's define the dy/dx or slope or we said it as gradient which is differenciation of y = f(x)
	gradient = 2 * (x - 5) 

	# Update in value of x
	x = x - lr * gradient
	# Value of y w.r.t x (small x)
	y = (x - 5) ** 2 + 3

	# plotting data points in scatter points
	plt.scatter(x, y)

	print(x)

plt.show()

```

![img](./output/gd3.png)

![img](./output/gd4.png)

 **Step 5:** **Performing Gradient Descent when slope is positive slope**

```python
# Gradient Descent when slope/gradient is positive
# Initialize x with any random value
x = 9
# Learning rate (lr)
lr = 0.1

# Plot X and Y
plt.plot(X, Y)

for i in range(50):
	# let's define the dy/dx or slope or we said it as gradient which is differenciation of y = f(x)
	gradient = 2 * (x - 5) 

	# Update in value of x
	x = x - lr * gradient
	# Value of y w.r.t x (small x)
	y = (x - 5) ** 2 + 3

	# plotting data points in scatter points
	plt.scatter(x, y)

	print(x)

plt.show()
```

![img](./output/gd5.png)

![img](./output/gd6.png)

- If there is a function $f(x)$ and we want the minimum value of $f(x)$ we using this optimization algorithm
- Starting with any random value of $x$, then updation of that value of $x$ and then, It will converge upto the local minima.
- The Cost function $J(\Theta)$ in linear regression, it have only one minima so, that lacal minima is considered as the global minima since, the goal of this optimization is to reach the global minima.

$J(\Theta)$ = J($\Theta_{0}, \Theta_{1})$ = $\frac{1}{2m} \Sigma_{i=1}^m ( h_{\Theta}^{i}(x) - {y}^{i} )^2$

![img](./output/gradient_descent_1.png)

parameters $(\Theta)$ = [$\Theta_{0}, \Theta_{1}]$

![img](./output/gradient_descent_2.png)

 **Mathematics behind Gradient Descent**

Gradient Descent Convergence formula:

$\Theta_{j+1} $ := $\Theta_{j}$ - $\alpha * \frac{\partial}{\partial \theta} J(\theta) $

* $\theta_{j+1}$  represents the updated value of the parameter at iteration ( j +1) of new value
* $\theta_j $ represents the current value of the parameter at iteration ( j )
* $\alpha $ represents the learning rate
* $\frac{\partial}{\partial \theta} J(\theta) $ represents the partial derivative of the cost function ( $J(\theta)$ ) with respect to the parameter ( $\theta$ )

In simple terms: parameter = parameter - LR * partial derivative of Loss w.r.t parameter

![img](./output/maths.jpg "Arpit Dubey")

A **partial derivative** $\frac{\partial}{\partial \theta} J(\theta)$  indicates how much total loss is increased (or) decreased if you increase $\Theta$ parameter by small amount.

### Implementation of Linear Regression

#### 1. Data Generation

**Step1: Import Required Libraries**

```python
# Importing required Libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
```

Let's assume a linear function $Y = f(x)$, where $f(x$) = 3 * X + 1 + (noise) is a linear function.

**Step 2: Data Generation**

```python
def generateDataset(m):  
	# array of X  
	X     = np.random.randn(m) * 10  
	noise = np.random.randn(m) * 5  
	Y     = 3 * X + 1 + noise  
        return X, Y
```

`def generateDataset(m):` - This line defines a function named `generateDataset` that takes one parameter `m`.

`X = np.random.randn(m) * 10` - This line generates an array of random numbers following a standard normal distribution with mean 0 and standard deviation 1, using `np.random.randn(m)`, and then scales these numbers by 10 to create an array `X`. This array represents the input variable X in a dataset.

`noise = np.random.randn(m) * 5` - This line generates another array of random numbers following a standard normal distribution, this time scaled by 5 to create a noise array that represents random noise in the dataset.

`Y = 3 * X + 1 + noise` - This line calculates the output variable `Y` using a linear relationship with `X`, a constant term `1`, and the noise. Basically, it's a linear relationship with

some added noise.

`return X, Y` - This line returns the arrays `X` and `Y` as the output of the function.

The `np.random.randn()` function is used here to generate random numbers that follow a standard normal distribution. In statistics, the standard normal distribution has a mean ($\mu$) of 0 and a standard deviation($\sigma$) of 1, and generating random numbers from a standard normal distribution is used to simulate real-world scenarios where many natural phenomena tend to follow a normal distribution. Additionally, using a standard normal distribution simplifies the modeling process and allows for straightforward easier interpretation of the generated dataset.

![img](./output/std_normal_dist.png "Arpit Dubey")

```python
# Calling generateDataset() function and distribution of data along X and Y printing shape of X and Y.
X, Y = generateDataset(100)
print(X.shape, Y.shape)

# Mean and Standard Deviation
print(f"Mean : {X.mean()} and Standard Deviation : {X.std()}")
```

![img](./output/lr1.png)

**Step 3: Visualization of Generated data**

```python
def plotData(X, Y, color="orange", title="Data"):
    plt.xlabel('Marks')
    plt.ylabel('Work Efficiency')
    plt.scatter(X, Y, color=color)
    plt.title(title)
    plt.show()
  
plotData(X, Y)
```

Let Y = Work Efficiency and X = Marks just assume then the plot looks like below Plot:

![img](./output/generated_data.png)

#### 2. Data Normalization

- Data normalization is a preprocessing technique **used to rescale the values of numeric features into a specific range**, typically to improve the performance and convergence of machine learning algorithms.
- It aims to **bring** **all feature values to a similar scale** so that no single feature dominates the others, which can help in achieving better performance and stability in machine learning models.
- **Data normalization involves transforming the numeric features of a dataset to a common scale, often between 0 and 1 or with a mean of 0 and a standard deviation of 1**. This is typically achieved using mathematical transformations such as **Min-Max scaling or standardization**.

  let's took a data about Height (cm), Age (years) and Salary (US dollars $)

| Feature                        | Min Value | Max Value |
| ------------------------------ | --------- | --------- |
| Height (cm)$ \Theta_{1}$     | 140 cm    | 200 cm    |
| Age (years)$ \Theta_{2}$     | 20 years  | 60 years  |
| Salary (dollars$)\Theta_{3}$ | $ 30,000  | 150,000   |

**Impact of different scale in features of same dataset:** The salary feature has a much larger scale compared to height and age, which can impact the learning process, causing the model to be biased towards salary.

$\Theta_j(new) = \Theta_j(old) - \alpha * \frac{\partial}{\partial \theta} J(\theta)$

(OR)

$\Theta_j(new) = \Theta_j(old) - \alpha$ * $\Sigma_{i=1}^m ( h_{\Theta}^{i}(x) - {y}^{i} ) * {x}^{(i)}_{j}$

**Hence, updating the $\Theta_j(new)$ is directly proportional to the feature (${x}^{(i)}_{j}$) value so, we need to scale.**

- The presence of feature ($x$) in the formula will affect the step size of the gradient descent
- The different ranges of features will cause different step sizes for each features. for example - features like age, salary will have very different scales.
- To ensure that the gradient descent moves smoothly towards the minima and that the steps for gradient descent are updated at the same rate for all the features, we have to scaled the data before feeding it to the model.

Min-Max scaling use to bring all features to a common scale between 0 and 1 using the formula:

$x_{\text{norm}} = \dfrac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)} $

**Starndardization (**Z-score normalization**):**

- Standardization is a data preprocessing technique used to rescale the features of a dataset to have a mean of 0 and a standard deviation of 1.
- It is also known as z-score normalization.
- Standardization transforms the features to a common scale, making it easier to compare and interpret them, and it is particularly useful in scenarios where the data follows a normal distribution.

  $x_{\text{std}} = \dfrac{x - \bar{x}_{mean}}{\sigma}$

  * $x_{\text{std}}$ is the standardized value of the feature
  * $x$ is the original value of the feature
  * $\bar{x}_{mean}$ is the mean of the feature
  * $\sigma$ is the standard deviation of the feature
- The mean of the standardized feature becomes 0, and the standard deviation becomes 1, putting all features on a common scale.
- Standardization is less sensitive to outliers compared to some other scaling techniques, making it a robust choice when dealing with datasets containing outliers.

```python
def normalizeData(X):
    X = (X - X.mean())//X.std()
    return X

X = normalizeData(X)
plotData(X, Y)

print(f"Mean : {X.mean()} and Standard Deviation : {X.std()}")
```

Plot the datset after normalization (or standardization)

![img](./output/normalized_standardized.png)

Mean and Standard Deviation after z-test normalization or standardization ($\mu_{mean}$ is close to 0 and, $\sigma_{std_dev}$ is close to 1 )

![img](./output/mean_std_dev_after_normatization.png)

Difference between Standardization and Normalization:

* **Standardization** : Scales the features to have a mean of 0 and a standard deviation of 1. It assumes that the data follows a normal distribution or at least is not heavily affected by outliers.
* **Normalization** : Scales the features to a range typically between 0 and 1. It is less influenced by outliers and is suitable for data that doesn't necessarily follow a normal distribution.
