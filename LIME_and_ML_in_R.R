# required packages
# install vip from github repo: devtools::install_github("koalaverse/vip")

# https://cran.r-project.org/web/packages/lime/vignettes/Understanding_lime.html


library(lime)       # ML local interpretation
library(vip)        # ML global interpretation
library(pdp)        # ML global interpretation
library(ggplot2)    # visualization pkg leveraged by above packages
library(caret)      # ML model building
library(h2o)        # ML model building

# initialize h2o
h2o.init()
h2o.no_progress()

# create data sets
df <- rsample::attrition %>% 
  dplyr::mutate_if(is.ordered, factor, ordered = FALSE) %>%
  dplyr::mutate(Attrition = factor(Attrition, levels = c("Yes", "No")))

# Pull back relevant features for examination

index <- 1:5
train_obs <- df[-index,]
local_obs <- df[index,]

any(is.na(train_obs))
any(is.na(local_obs))
# Local observations are going to be the 5 new instances we want to understand why the 
# predicted response was made

# Create H2o objects for modelling

y <- "Attrition"
x <- setdiff(names(train_obs), y)
# USe the setdiff function to select everything else in the data frame

train.obs.h2o <- as.h2o(train_obs)
local.obs.h2o <- as.h2o(local_obs)


#------------------------------- Random Forest model ----------------------------------- #


fit.caret <- caret::train(as.formula(paste0(y, " ~ .")),
                          data = train_obs,
                          method = 'ranger',
                          trControl = trainControl(method = "cv", number = 5, classProbs = TRUE),
                          tuneLength = 1,
                          importance = 'impurity')

# Create H2o models

h2o_rf <- h2o.randomForest(x, y, training_frame = train.obs.h2o)
h2o_glm <- h2o.glm(x, y, training_frame = train.obs.h2o, family = "binomial")
h2o_gbm <- h2o.gbm(x,y, training_frame = train.obs.h2o)

# Ranger model - model type not built into LIME

fit.ranger <- ranger::ranger(
  as.formula(paste0(y, " ~ .")),
  data = train_obs,
  importance = 'impurity',
  probability = TRUE
)

# Global interpretation 
# The two most common methods for global interpretation are variable importance measures
# and partial dependence plots

varImp(fit.caret) # Variable importance for caret


vip(fit.ranger) + ggtitle("Ranger: Random Forest")

#After the most globally relevant variables have been identified, the next step is to attempt to understand how the response variable changes based on these variables. For this we can use partial dependence plots (PDPs) and individual conditional expectation (ICE) curves. These techniques plot the change in the predicted value as specified feature(s) vary over their marginal distribution. Consequently, we can gain some local understanding how the reponse variable changes across the distribution of a particular variable but this still only provides a global understanding of this relationships across all observed data.

# built-in PDP support in H2O
h2o.partialPlot(h2o_rf, data = train.obs.h2o, cols = "MonthlyIncome")

#We can gain further insight by using centered ICE curves which can help draw out further details. For example, the following ICE curves show a similar trend line as the PDP above but by centering we identify the decrease as monthly income approaches $5,000 followed by an increase in probability of attriting once an employeeâs monthly income approaches $20,000. Futhermore, we see some turbulence in the flatlined region between $5-$20K) which means there appears to be certain salary regions where the probability of attriting changes.

fit.ranger %>% 
  partial(pred.var = "MonthlyIncome", grid.resolution = 25, ice = TRUE) %>% 
  autoplot(rug = TRUE, train = train_obs, alpha = 0.1, center = TRUE)


# These visualizations help us to understand our model from a global perspective: identifying the variables with the largest overall impact and the typical influence of a feature on the response variable across all observations. However, what these do not help us understand is given a new observation, what were the most influential variables that determined the predicted outcome. Say we obtain information on an employee that makes about $10,000 per month and we need to assess their probabilty of leaving the firm. Although monthly income is the most important variable in our model, it may not be the most influential variable driving this employee to leave. To retain the employee, leadership needs to understand what variables are most influential for that specific employee. This is where lime can help.

#####################################Local Interpretation############################################

#Local Interpretable Model-agnostic Explanations (LIME) is a visualization technique that helps explain individual predictions. As the name implies, it is model agnostic so it can be applied to any supervised regression or classification model. Behind the workings of LIME lies the assumption that every complex model is linear on a local scale and asserting that it is possible to fit a simple model around a single observation that will mimic how the global model behaves at that locality. The simple model can then be used to explain the predictions of the more complex model locally
#The generalized algorithm LIME applies is:

#1 Given an observation, permute it to create replicated feature data with slight value modifications.
#2 Compute similarity distance measure between original observation and permuted observations.
#3 Apply selected machine learning model to predict outcomes of permuted data.
#4 Select m number of features to best describe predicted outcomes.
#5 Fit a simple model to the permuted data, explaining the complex model outcome with m features from the permuted data weighted by its similarity to the original observation .
#6 Use the resulting feature weights to explain local behavior.

#lime::lime
#The application of the LIME algorithm via the lime package is split into two operations: lime::lime and lime::explain. The lime::lime function creates an âexplainerâ object, which is just a list that contains the machine learning model and the feature distributions for the training data. The feature distributions that it contains includes distribution statistics for each categorical variable level and each continuous variable split into n bins (default is 4 bins). These feature attributes will be used to permute data.

#The following creates our lime::lime object and I change the number to bin our continuous variables into to 5.

lime_explain_caret <- lime::lime(train_obs, fit.caret, n_bins = 5)
class(lime_explain_caret)
summary(lime_explain_caret)

#------------------------------------LIME EXPLAINER -------------------------------#

#lime::explain
#Once we created our lime objects, we can now perform the generalized LIME algorithm using the lime::explain function. This function has several options, each providing flexibility in how we perform the generalized algorithm mentioned above.

#x: Contains the one or more single observations you want to create local explanations for. In our case, this includes the 5 observations that I included in the local_obs data frame. Relates to algorithm step 1.
#explainer: takes the explainer object created by lime::lime, which will be used to create permuted data. Permutations are sampled from the variable distributions created by the lime::lime explainer object. Relates to algorithm step 1.
#n_permutations: The number of permutations to create for each observation in x (default is 5,000 for tabular data). Relates to algorithm step 1.
#dist_fun: The distance function to use. The default is Gowerâs distance but can also use euclidean, manhattan, or any other distance function allowed by ?dist(). To compute similarity distance of permuted observations, categorical features will be recoded based on whether or not they are equal to the actual observation. If continuous features are binned (the default) these features will be recoded based on whether they are in the same bin as the observation. Using the recoded data the distance to the original observation is then calculated based on a user-chosen distance measure. Relates to algorithm step 2.
#kernel_width: To convert the distance measure to a similarity value, an exponential kernel of a user defined width (defaults to 0.75 times the square root of the number of features) is used. Smaller values restrict the size of the local region. Relates to algorithm step 2.
#n_features: The number of features to best describe predicted outcomes. Relates to algorithm step 4.
#feature_select: To select the best n features, lime can use forward selection, ridge regression, lasso, or a tree to select the features. In this example I apply a ridge regression model and select the m features with highest absolute weights. Relates to algorithm step 4.
#For classification models we also have two additional features we care about and one of these two arguments must be given:

#labels: Which label do we want to explain? In this example, I want to explain the probability of an observation to attrit (âYesâ).
#n_labels: The number of labels to explain. With this data I could select n_labels = 2 to explain the probability of âYesâ and âNoâ responses.


explainer_caret <- lime::explain(
  x = local_obs,
  explainer = lime_explain_caret,
  n_permutations = 5000,
  dist_fun = "gower",
  kernel_width = .75,
  n_features = 10, 
  feature_select = "highest_weights",
  labels = "Yes" # Links to the attrition label
)

# The explain function above first creates permutations, then calculates similarities, followed by selecting the m features. Lastly, explain will then fit a model (algorithm steps 5 & 6). lime applies a ridge regression model with the weighted permuted observations as the simple model.3 If the model is a regressor, the simple model will predict the output of the complex model directly. If the complex model is a classifier, the simple model will predict the probability of the chosen class(es).
# The explain output is a data frame containing different information on the simple model predictions. Most importantly, for each observation in local_obs it contains the simple model fit (model_r2) and the weighted importance (feature_weight) for each important feature (feature_desc) that best describes the local relationship.

lime_outputs <- tibble::glimpse(explainer_caret)


# Visualising results

# However the simplest approach to interpret the results is to visualize them. There are several plotting functions provided by lime but for tabular data we are only concerned with two. The most important of which is plot_features. This will create a visualization containing an individual plot for each observation (case 1, 2, â¦, n) in our local_obs data frame. Since we specified labels = "Yes" in the explain() function, it will provide the probability of each observation attriting. And since we specified n_features = 10 it will plot the 10 most influential variables that best explain the linear model in that observations local region and whether the variable is causes an increase in the probability (supports) or a decrease in the probability (contradicts). It also provides us with the model fit for each model (âExplanation Fit: XXâ), which allows us to see how well that model explains the local region.
# Consequently, we can infer that case 3 has the highest liklihood of attriting out of the 5 observations and the 3 variables that appear to be influencing this high probability include working overtime, being single, and working as a lab tech.


plot_features(explainer_caret, ncol = 1)
# Creates a feature plot for the 5 cases used in the training data

plot_explanations(explainer_caret)


# Tuning

# As you saw in the above plot_features plot, the output provides the model fit. In this case the best simple model fit for the given local regions was 
# R2=  0.59 for case 3. Considering there are several knobs we can turn when performing the LIME algorithm, we can treat these as tuning parameters to try find the best fit model for the local region. This helps to maximize the amount of trust we can have in the local region explanation.

# As an example, the following changes the distance function to use the manhattan distance algorithm, we increase the kernel width substantially to create a larger local region, and we change our feature selection approach to a LARS lasso model. The result is a fairly substantial increase in our explanation fits.

tune_explainer_caret <- explain(
  x = local_obs,
  explainer = lime_explain_caret,
  n_permutations = 5000, 
  dist_fun = "manhattan",
  kernel_width = 3, 
  n_features = 10, 
  feature_select = "lasso_path",
  labels = "Yes"
)

plot_features(tune_explainer_caret)

?lime::explain
?lime::plot_features


# Supported vs non supported models

explainer_h2o_rf  <- lime(train_obs, h2o_rf, n_bins = 5)
explainer_h2o_glm <- lime(train_obs, h2o_glm, n_bins = 5)
explainer_h2o_gbm <- lime(train_obs, h2o_gbm, n_bins = 5)

explanation_rf <- explain(local_obs, explainer_h2o_rf, n_features = 5, labels = "Yes", kernel_width = .1, feature_select = "highest_weights")
explanation_glm <- explain(local_obs, explainer_h2o_glm, n_features = 5, labels = "Yes", kernel_width = .1, feature_select = "highest_weights")
explanation_gbm <- explain(local_obs, explainer_h2o_gbm, n_features = 5, labels = "Yes", kernel_width = .1, feature_select = "highest_weights")

p1 <- plot_features(explanation_rf, ncol = 1) + ggtitle("rf")
p2 <- plot_features(explanation_glm, ncol = 1) + ggtitle("glm")
p3 <- plot_features(explanation_gbm, ncol = 1) + ggtitle("gbm")
gridExtra::grid.arrange(p1, p2, p3, nrow = 1)

explainer_ranger <- lime(train, fit.ranger, n_bins = 5)
## Error in UseMethod("lime", x): no applicable method for

# We can work with this pretty easily by building two functions that make lime compatible with an unsupported package. First, we need to create a model_type function that specifies what type of model this unsupported package is using. model_type is a lime specific function, we just need to create a ranger specific method. We do this by taking the class name for our ranger object and creating the model_type.ranger method and simply return the type of model (“classification” for this example).

# get the model class
class(fit.ranger)
## [1] "ranger"

# need to create custom model_type function
model_type.ranger <- function(x, ...) {
  # Function tells lime() what model type we are dealing with
  # 'classification', 'regression', 'survival', 'clustering', 'multilabel', etc
  
  return("classification")
}

model_type(fit.ranger)
## [1] "classification"

# We then need to create a predict_model method for ranger as well. The output for this function should be a data frame. For a regression problem it should produce a single column data frame with the predicted response and for a classification problem it should create a column containing the probabilities for each categorical class (binary “Yes” “No” in this example).

# need to create custom predict_model function
predict_model.ranger <- function(x, newdata, ...) {
  # Function performs prediction and returns data frame with Response
  pred <- predict(x, newdata)
  return(as.data.frame(pred$predictions))
}

predict_model(fit.ranger, newdata = local_obs)

# Now that we have those methods developed and in our global environment we can run our lime functions and produce our outputs.

explainer_ranger <- lime(train_obs, fit.ranger, n_bins = 5)
explanation_ranger <- explain(local_obs, explainer_ranger, n_features = 5,
                              n_labels = 2, kernel_width = .1)
plot_features(explainer_ranger, ncol = 2) + ggtitle("ranger")
