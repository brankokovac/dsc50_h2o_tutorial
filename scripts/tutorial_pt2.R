### "End-to-End Machine Learning with H2O DSC5.0 Tutorial
### https://bit.ly/dsc50_h2o_tutorial
###
### Lecturer: Branko Kovac, Logikka
### Startit Center Belgrade, November 2019 

### Part 2 - Classification

## Classification Part One: H2O AutoML

# Let's go
library(h2o) # for H2O Machine Learning
library(mlbench) # for Datasets

# Enter your lucky number :)
n_seed <- 12345

# Data - Pima Indians Diabetes from `mlbench`
data("PimaIndiansDiabetes")

# Data Prep
# Convert pos and neg to 1 and 0
d_new <- PimaIndiansDiabetes[, -ncol(PimaIndiansDiabetes)]
d_new$diabetes <- 0
d_new[which(PimaIndiansDiabetes$diabetes == "pos"), ]$diabetes <- 1
PimaIndiansDiabetes <- d_new
rm(d_new)

# Define Target and Features
target <- "diabetes" 
features <- setdiff(colnames(PimaIndiansDiabetes), target)
print(features)

# Start a local H2O Cluster (JVM)
h2o.init()
h2o.removeAll()   # Optional: remove anything from previous session 

# Convert R dataframe into H2O dataframe
h_diabetes <- as.h2o(PimaIndiansDiabetes)

# Make sure the target is a factor (for classification)
h_diabetes$diabetes <- as.factor(h_diabetes$diabetes)

# Split Data into Train/Test
h_split <- h2o.splitFrame(h_diabetes, ratios = 0.8, seed = n_seed)
h_train <- h_split[[1]] # 80% for modelling
h_test <- h_split[[2]] # 20% for evaluation

dim(h_train)
dim(h_test)

# H2O AutoML
# Run AutoML (try n different models)
# Check out all options using ?h2o.automl
automl = h2o.automl(x = features,
                    y = target,
                    training_frame = h_train,
                    nfolds = 5,                        # 5-fold Cross-Validation
                    max_models = 20,                   # Max number of models
                    stopping_metric = "logloss",       # Metric to optimize
                    project_name = "automl_diabetes",  # Specify a name so you can add more models later
                    sort_metric = "logloss",
                    seed = n_seed)

# Leaderboard
as.data.frame(automl@leaderboard)

## Classification Part Two: XAI

# Package `DALEX`
# Descriptive mAchine Learning EXplanations (DALEX)
library(DALEX)

# Custom Predict Function
custom_predict <- function(model, newdata) {
  newdata_h2o <- as.h2o(newdata)
  res <- as.data.frame(h2o.predict(model, newdata_h2o))
  return(round(res$p1)) # round the probabil
  }

# Explainer for H2O Models
explainer_automl <- DALEX::explain(model = automl@leader, 
                                data = as.data.frame(h_test)[, features],
                                y = as.numeric(as.character(as.data.frame(h_test)[, target])),
                                predict_function = custom_predict,
                                label = "H2O AutoML")

# Variable importance
library(ingredients)
vi_automl <- feature_importance(explainer_automl, type="difference")
plot(vi_automl)

# Partial Dependence Plots
# Let's look at feature `age` 
pdp_automl_rm <- partial_dependency(explainer_automl, variables = "age")
plot(pdp_automl_rm)

# Prediction Understanding
library(iBreakDown)

# Prediction: Diabetes = Negative (0)
pb_automl <- break_down(explainer_automl, new_observation = as.data.frame(h_test)[1, ])
plot(pb_automl)

# Prediction: Diabetes = Positive (1)
pb_automl <- break_down(explainer_automl, new_observation = as.data.frame(h_test)[6, ])
plot(pb_automl)

## Bring Your Own Data + Q&A
## Get your hands dirty!
