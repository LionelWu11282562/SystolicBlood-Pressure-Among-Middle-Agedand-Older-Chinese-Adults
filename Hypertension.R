library(dplyr)
library(tidyr)
library(glmnet)
library(ggplot2)
library(ggcorrplot)
library(gridExtra)
library(patchwork)
library(car)
library(pROC)
library(splines)
library(MASS)

# 数据读取
hypertension <- read.csv("/Users/huanqiwu/Desktop/467FinalData.csv")

# 类型转换
hypertension$Xa <- as.numeric(hypertension$Xa)

hypertension$Xb <- factor(hypertension$Xb,
                          levels=c(0,1),
                          labels=c("NonHan","Han"))

# 二分类变量
binary_vars <- c("Xd","Xe","Xf","Xg","Xh","Xi","Xj","Xk","Xl",
                 "Xn","Xo","Xp","Xr","Xv","Xw","Xx","Xy")

hypertension[binary_vars] <- lapply(hypertension[binary_vars], as.numeric)

# 饮食变量
diet_vars <- paste0("Xq",1:20)
hypertension[diet_vars] <- lapply(hypertension[diet_vars], as.numeric)

# 有序变量
hypertension$Xc <- ordered(hypertension$Xc, levels=1:6)
hypertension$Xm <- ordered(hypertension$Xm, levels=1:5)
hypertension$Xs <- ordered(hypertension$Xs, levels=1:3)
hypertension$Xt <- ordered(hypertension$Xt, levels=1:3)
hypertension$Xu <- ordered(hypertension$Xu, levels=1:3)

# 数值变量
num_vars <- c("X1","X2","X4","X5","Y1")
hypertension[num_vars] <- lapply(hypertension[num_vars], as.numeric)

hypertension$Y2 <- as.numeric(hypertension$Y2)

# Box-Cox 变换
tmp <- na.omit(hypertension)
bc <- boxcox(Y1 ~ 1, data=tmp)
lambda <- bc$x[which.max(bc$y)]

hypertension$Y1_bc <- (hypertension$Y1^lambda - 1) / lambda

hypertension_clean <- na.omit(hypertension)
hypertension_clean$Y1 <- hypertension_clean$Y1_bc

# ======================
# LASSO - Y1
# ======================
y_y1 <- hypertension_clean$Y1_bc

predictors_y1 <- setdiff(names(hypertension_clean),
                         c("Y1","Y1_bc","Y2","X3"))

x_y1 <- model.matrix(~ ., data=hypertension_clean[predictors_y1])[,-1]

set.seed(123)
lasso_y1 <- cv.glmnet(x_y1, y_y1, alpha=1, family="gaussian")

coef_df_y1 <- data.frame(
  variable = rownames(coef(lasso_y1, s="lambda.min")),
  coefficient = as.numeric(coef(lasso_y1, s="lambda.min"))
) %>%
  filter(variable != "(Intercept)" & coefficient != 0)

ggplot(coef_df_y1, aes(reorder(variable, coefficient), coefficient)) +
  geom_col(fill="steelblue") +
  coord_flip()

# ======================
# LASSO - Y2
# ======================
y_y2 <- hypertension_clean$Y2

predictors_y2 <- setdiff(names(hypertension_clean),
                         c("Y1","Y1_bc","Y2","X3"))

x_y2 <- model.matrix(~ ., data=hypertension_clean[predictors_y2])[,-1]

set.seed(123)
lasso_y2 <- cv.glmnet(x_y2, y_y2, alpha=1, family="binomial")

coef_df_y2 <- data.frame(
  variable = rownames(coef(lasso_y2, s="lambda.min")),
  coefficient = as.numeric(coef(lasso_y2, s="lambda.min"))
) %>%
  filter(variable != "(Intercept)" & coefficient != 0)

ggplot(coef_df_y2, aes(reorder(variable, coefficient), coefficient)) +
  geom_col(fill="tomato") +
  coord_flip()

# ======================
# 相关性分析
# ======================
numeric_only <- hypertension_clean %>%
  select(where(is.numeric)) %>%
  select(-Y1)

cor_mat <- cor(numeric_only, use="pairwise.complete.obs")

ggcorrplot(cor_mat, type="lower", lab=FALSE,
           colors=c("blue","white","red"))

df_num <- hypertension_clean %>%
  select(where(is.numeric))

# Y1相关性
cor_y1 <- sapply(df_num[setdiff(names(df_num), c("Y1","Y1_bc","Y2"))],
                 function(x) cor(df_num$Y1_bc, x))

top10_y1 <- names(sort(abs(cor_y1), decreasing=TRUE))[1:10]

ggplot(data.frame(v=top10_y1, c=cor_y1[top10_y1]),
       aes(reorder(v,c), c, fill=c)) +
  geom_col() +
  coord_flip()

# Y2相关性
cor_y2 <- sapply(df_num[setdiff(names(df_num), c("Y1","Y1_bc","Y2"))],
                 function(x) cor(df_num$Y2, x))

top10_y2 <- names(sort(abs(cor_y2), decreasing=TRUE))[1:10]

ggplot(data.frame(v=top10_y2, c=cor_y2[top10_y2]),
       aes(reorder(v,c), c, fill=c)) +
  geom_col() +
  coord_flip()

# ======================
# 自动画图
# ======================
plot_one <- function(v){
  x <- hypertension_clean[[v]]
  if(is.numeric(x) & length(unique(x)) > 10){
    ggplot(hypertension_clean, aes_string(x=v)) +
      geom_histogram(bins=25)
  } else {
    ggplot(hypertension_clean, aes_string(x=v)) +
      geom_bar()
  }
}

plot_list <- lapply(names(hypertension_clean), plot_one)

for(i in seq(1, length(plot_list), 4)){
  grid.arrange(grobs=plot_list[i:min(i+3,length(plot_list))],
               ncol=2, nrow=2)
}

# ======================
# 回归模型
# ======================
model1 <- lm(Y1 ~ X1 + I(X1^2) + ns(X4,3) + X5 +
               Xo + Xp + Xi + Xd,
             data=hypertension_clean)

model2 <- lm(Y1 ~ X1 + I(X1^2) + ns(X4,3) + X5 +
               Xo + Xp + Xi + Xd +
               Xr + Xq14 + Xq5 + Xq15,
             data=hypertension_clean)

model3 <- glm(Y2 ~ X1 + I(X1^2) + ns(X4,3) +
                X5 + Xo + Xi + Xp + Xd +
                Xq16 + Xq5 + Xq15,
              data=hypertension_clean,
              family=binomial)

# ======================
# 模型诊断
# ======================
ggplot(model1, aes(.fitted, .resid)) +
  stat_binhex() +
  geom_hline(yintercept=0)

ggplot(model1, aes(sample=.stdresid)) +
  stat_qq() + stat_qq_line()

ggplot(model2, aes(.fitted, .resid)) +
  stat_binhex() +
  geom_hline(yintercept=0)

ggplot(model2, aes(sample=.stdresid)) +
  stat_qq() + stat_qq_line()

# ======================
# ROC
# ======================
prob3 <- predict(model3, type="response")
roc3 <- roc(hypertension_clean$Y2, prob3)
plot(roc3, col="blue")
auc(roc3)

# ======================
# 非线性预测（logistic）
# ======================
newdata <- data.frame(
  X1 = seq(20,80,length=200),
  X4 = median(hypertension_clean$X4),
  X5 = median(hypertension_clean$X5),
  Xo=0, Xi=0, Xp=0, Xd=0,
  Xq16 = median(hypertension_clean$Xq16),
  Xq5 = median(hypertension_clean$Xq5),
  Xq15 = median(hypertension_clean$Xq15)
)

new_ns <- as.data.frame(predict(ns(hypertension_clean$X4,3), newdata$X4))
nd <- cbind(newdata, new_ns)

pred <- predict(model3, nd, type="link", se.fit=TRUE)

invlogit <- function(z) exp(z)/(1+exp(z))

nd$prob <- invlogit(pred$fit)
nd$lower <- invlogit(pred$fit - 1.96*pred$se.fit)
nd$upper <- invlogit(pred$fit + 1.96*pred$se.fit)

ggplot(nd, aes(X1, prob)) +
  geom_line(color="blue") +
  geom_ribbon(aes(ymin=lower, ymax=upper), alpha=0.3)

# ======================
# 线性模型预测
# ======================
new_lm <- data.frame(
  X1 = seq(20,80,length=200),
  X4 = median(hypertension_clean$X4),
  X5 = median(hypertension_clean$X5),
  Xo=0, Xp=0, Xi=0, Xd=0
)

new_lm_ns <- cbind(new_lm,
                   as.data.frame(predict(ns(hypertension_clean$X4,3),
                                         new_lm$X4)))

pred_lm <- predict(model1, new_lm_ns, interval="confidence")

new_lm$fit <- pred_lm[,1]
new_lm$lower <- pred_lm[,2]
new_lm$upper <- pred_lm[,3]

ggplot(new_lm, aes(X1, fit)) +
  geom_line(color="red") +
  geom_ribbon(aes(ymin=lower, ymax=upper),
              fill="pink", alpha=0.3)