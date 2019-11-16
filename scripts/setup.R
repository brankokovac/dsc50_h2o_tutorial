# Instalacija svih paketa koji su neophodni

pkgs <- c("h2o", "DALEX", "ingredients", "iBreakDown", "breakDown", "pdp",
          "knitr", "rmdformats", "DT", "xgboost", "mlbench") # lista paketa

for (pkg in pkgs) {
  
  if (!(pkg %in% rownames(installed.packages()))) { install.packages(pkg)}
  
}

DALEX::install_dependencies()

rm(list = ls())
