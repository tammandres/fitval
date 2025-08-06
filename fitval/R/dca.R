# Decision curve analysis
# Partly based on DCA vignette at https://cran.r-project.org/web/packages/dcurves/vignettes/dca.html
#install.packages("dcurves", version="0.4.0")
library(dcurves)
root <- "C:\\Users\\1cgQ\\Desktop\\fitval\\results\\colofit\\"
ex_name <- "impute_none_nottingham"

out_path <- file.path(root, ex_name)
data_path <- file.path(root, ex_name, 'boot')

# Load data -- avg of imputed data sets
df <- read.csv(file.path(data_path, "predictions_orig.csv"))
#df <- read.csv(file.path(data_path, "predictions_orig_avg.csv"))
df <- df[df$m == 0]
head(df, 10)
colnames(df)

# Transform data to wide format
df2 <- reshape(df, idvar=c("idx", "y_true"), timevar="model_name", direction="wide", drop=c("m", "b"), sep='_')
colnames(df2) <- gsub("-", "_", colnames(df2))  # Replace - in model names, e.g. "exeter-lr" -> "exeter_lr"
colnames(df2) <- gsub("y_pred_", "", colnames(df2))
nrow(df2)
head(df2, 10)
colnames(df2)

# Replace fit value with fit > 10 indicator
df2$fit10 = ifelse(df2$fit >= 10, 1, 0)

# Formula for DCA
models <- unique(df$model_name)
models <- models[models != 'fit']
model_str <- paste(models, collapse=" + ")
model_str <- gsub("-", "_", model_str)
model_str <- paste(model_str, 'fit10', sep=' + ')

f_str <- paste("y_true ~", model_str)
f <- formula(f_str)
d <- dca(f, data=df2, thresholds=seq(0, 0.20, by = 0.01))

png(filename=file.path(out_path, "R_dca.png"), width=700, 500)
plot(d, smooth=TRUE)
dev.off()

df_dca <- d$dca
df_dca$variable <- gsub("_", "-", df_dca$variable)
df_dca$label <- gsub("_", "-", df_dca$label)
fout <- file.path(out_path, "R_dca.csv")
write.csv(df_dca, fout)
