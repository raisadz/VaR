# main program to reproduce results of Weak Aggregating Algorithm for VaR
#data was downloaded from Yahoo Finance from January 2011 to December 2018
file = "WMT"  #Walmart    ("AAPL" for Apple, "WPP" for WPP inc.)
q = 0.05       #significance level
df = read.csv(paste(file, ".csv", sep=""), header = T)   #load data
df$r = c(0, diff(df$Adj.Close)) / c(1, df$Adj.Close[1:dim(df)[1]-1])  #returns
df = df[-1, ]  #remove first observation
N = dim(df)[1] 
l = 500        #length of the moving window
N_test = N - l #length of backtesting  
h = 50         #length of window to calculate sigma 
df$sigma_h = 0
for (i in 1:(N-h)) {
  df$sigma_h[i+h] = sqrt(mean(df$r[i:(i+h-1)]^2))
}
df$r_1 = c(0, df$r[1:(N-1)])
df$sigma_1 = c(0, df$sigma_h[1:(N-1)])
df_test = df[(l+1):N, ]  #backtesting data

sigma_grid = seq(0, 0.03, 0.0025)  #create grid of normal distribution experts
r_grid = -qnorm(1-q, 0, 1) * sigma_grid 
X = matrix(rep(r_grid, N_test, each = T), nrow = N_test, byrow = T)
outcomes = df_test$r
df_test$VaR_WAA = WAA(outcomes, X, q, C = 200)$gamma   #apply WAA to test set
weights_WAA = WAA(outcomes, X, q, C = 200)$weights_norm

#create model experts
lambda = 0.96  #parameter of EWMA
library(quantreg)
for (i in 1:N_test) {
  df_test$VaR1[i] = quantile(df$r[i:(i+l-1)], probs = q)
  df_test$VaR2[i] = -qnorm(1-q, 0, 1) * sqrt(mean(df$r[i:(i+l-1)]^2))
  if (i==1) {
    df_test$VaR3[i] = -qnorm(1-q, 0, 1) *sqrt(lambda*df$sigma_h[N]^2 + (1-lambda)*df$r[N]^2)
  } else {
    df_test$VaR3[i] = -qnorm(1-q, 0, 1) *sqrt(lambda*df_test$sigma_h[i-1]^2 + (1-lambda)*df_test$r[i-1]^2)
  }
  qr1 <- rq(r ~ r_1+sigma_1, data=df[(i+h+1):(i+l-1), ], tau = q)
  df_test$VaR_qr[i] = predict.rq(qr1, df_test[i,])
}

library(rugarch)
#GARCH
spec = ugarchspec(distribution.model = "std")
mod = ugarchroll(spec, data = df[, "r"], n.ahead = 1, 
                 n.start = l,  refit.every = 50, refit.window = "recursive", 
                 solver = "hybrid", fit.control = list(),
                 calculate.VaR = TRUE, VaR.alpha = q,
                 keep.coef = TRUE)
df_test$VaR4 = mod@forecast$VaR[,1]
experts = c( "VaR1", "VaR2", "VaR3", "VaR4")
predictions = as.matrix(df_test[, experts])
df_test$VaR_WAA2 = WAA(outcomes, predictions, q, C = 200)$gamma   #apply WAA to test set
weights_WAA2 = WAA(outcomes, predictions, q, C = 200)$weights_norm

#backtesting
vector_VaRtest = c("actual.exceed", "uc.LRp", "cc.LRp", "uc.Decision", "cc.Decision")
VaR_models = as.matrix(cbind(df_test$VaR1, df_test$VaR2, df_test$VaR3, df_test$VaR4, df_test$VaR_qr, df_test$VaR_WAA, df_test$VaR_WAA2, df_test$VaR_mix))
VaR_names = c("Historical", "Variance-Covariance", "EWMA", "GARCH", "Quantile Regression", "WAA", "WAA2")
table_VaR = matrix(0, nrow = dim(VaR_models)[2], ncol = length(vector_VaRtest)+1)
table_VaR = data.frame(table_VaR)
for (i in 1:dim(VaR_models)[2]) {
  table_VaR[i, 1] = VaR_names[i]
  for (j in 2:(length(vector_VaRtest)+1)) {
    table_VaR[i, j] = VaRTest(q, df_test$r, VaR_models[,i])[[vector_VaRtest[j-1]]]
  }
}
xtable(table_VaR, digits=c(0,4,0,4,4, 0,0))

#calculate pinball loss of experts
calc_loss = function(outcomes, predictions, q) {
  #function calculates pinball loss for vectors outcomes
  #and predictions for quantile q
  losses = outcomes - predictions
  for (j in 1:length(losses)) {
    losses[j] = ifelse(losses[j] > 0, q*losses[j], (q-1)*losses[j])
  }
  return(losses)
}

loss_models = matrix(0, nrow = N_test, ncol = length(VaR_names))
Loss_models = matrix(0, nrow = N_test, ncol = length(VaR_names))
for (i in 1:length(VaR_names)) {
  loss_models[, i] = calc_loss(df_test$r, VaR_models[, i], q)
  Loss_models[, i] = cumsum(loss_models[, i])
}

#plot weights of experts
library(reshape2)
melted_cormat <- melt(weights_WAA)
head(melted_cormat)
melted_cormat$sigma = sigma_grid[unlist(melted_cormat["Var2"])]
melted_cormat$Time = melted_cormat$Var1
library(ggplot2)
pdf(paste(file, "_1_q", 100*q, ".pdf", sep=""), height = 8.5, width = 8.5, paper = "special")
ggplot(data = melted_cormat, aes(x=Time, y=sigma, fill=value)) +   geom_tile() +
  scale_fill_distiller(direction = 1, limit = c(0, max(weights_WAA)))+
  theme(text = element_text(size=30), plot.title = element_text(hjust = 0.5))
dev.off() 

model_names = c("Historical", "Var-Cov", "EWMA", "GARCH")
melted_cormat2 <- melt(weights_WAA2)
head(melted_cormat2)
melted_cormat2$expert = model_names[unlist(melted_cormat2["Var2"])]
melted_cormat2$Time = melted_cormat2$Var1

pdf(paste(file, "_2_q", 100*q, ".pdf", sep=""), height = 8.5, width = 8.5, paper = "special")
ggplot(data = melted_cormat2, aes(x=Time, y=expert, fill=value)) +   geom_tile() +
  theme(text = element_text(size=30))+
  scale_fill_distiller(direction = 1, limit = c(min(weights_WAA2), max(weights_WAA2)))+
  theme(text = element_text(size=30), plot.title = element_text(hjust = 0.5))
dev.off() 

#plot returns and VaR for WAA
pdf(paste(file, "_returns_q", 100*q, ".pdf", sep=""), height = 8.5, width = 8.5, paper = "special")
period = seq(1, 1510)
plot(df_test$r[period], type="l", col = "blue", xlim = c(0, length(period)), ylim = c(-0.1, 0.12), lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, xlab = "Time", ylab = "")
lines(df_test$VaR_WAA[period], col = "red", lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
lines(df_test$VaR_WAA2[period], col = "green", lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
legend("topleft", legend=c("WAAn", "WAAm"),col=c("red", "green"), lty=1, lwd=3, cex=2, bty="n")
dev.off()

