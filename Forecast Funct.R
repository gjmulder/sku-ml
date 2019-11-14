library(foreach)
library(doSNOW)
library(forecast)
library(lubridate)
library(plyr)
# library(foreach)
library(ggplot2)
library(tsintermittent)
library(EnvStats)
library(robustbase)
library(RSNNS)
library(rpart)
library(caret)
library(e1071)
library(kernlab)
library(brnn)
library(grnn)
library(ddpcr)
library(robustbase)
library(zoo)
library(randomForest)
library(gbm)


# setwd("C:/Users/FSU_Team/Desktop/Paper Interm METRO")

robustness <- function(ind){
  lambda <- BoxCox.lambda(ind, method = "loglik", lower = 0, upper = 1)
  transformed <- BoxCox(ind, lambda) 
  RM <- InvBoxCox(mean(transformed)+sd(transformed), lambda)
  return(RM)
}

#ML Methods
CreateSamples<-function(datasample,xi){
  xo<-1
  sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
  for (cid in (xi+xo):length(datasample)){
    sample[cid,]<-datasample[(cid-xi-xo+1):cid]
  }
  sample<-as.matrix(data.frame(na.omit(sample)))
  return(sample)
}
MLP_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  train <- as.matrix(dftest[,1:(ncol(dftest)-1)])
  test <- dftest[,ncol(dftest)]
  
  #Train model
  frc_f <- NULL
  for (ssn in c(1:10)){
    modelMLP <- mlp(train, test, 
                    size = (2*ni), maxit = 500,initFunc = "Randomize_Weights", 
                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic", 
                    shufflePatterns = FALSE, linOut = TRUE)
    
    #Extrapolate
    testsample <- as.numeric(input) 
    tempin <- (testsample-MIN)/(MAX-MIN)
    tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
    colnames(tempin) <- head(paste0("X",c(1:100)),ni)
    tempin <- tail(tempin,1)
    MLf <- as.numeric(predict(modelMLP,tempin))*(MAX-MIN)+MIN
    frc <- rep(MLf, fh)
    frc_f <- rbind(frc_f, frc)
  }
  frc <- colMedians(frc_f)
  return(frc)
}
BNN_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  train <- as.matrix(dftest[,1:(ncol(dftest)-1)])
  test <- dftest[,ncol(dftest)]
  
  #Train model
  frc_f <- NULL
  for (ssn in c(1:10)){
    modelBNN <-10101
    while (length(modelBNN)==1){
      modelBNN<-tryCatch(model<-brnn(train, as.numeric(test), 
                                     neurons=(2*ni),normalize=FALSE,
                                     epochs=500,verbose=F), error=function(e) 100)
    }
    
    #Extrapolate
    testsample <- as.numeric(input) 
    tempin <- (testsample-MIN)/(MAX-MIN)
    tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
    colnames(tempin) <- head(paste0("X",c(1:100)),ni)
    tempin <- tail(tempin,1)
    MLf <- as.numeric(predict(modelBNN,tempin))*(MAX-MIN)+MIN
    frc <- rep(MLf, fh)
    frc_f <- rbind(frc_f, frc)
  }
  frc <- colMedians(frc_f)
  return(frc)
}
RBF_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  train <- as.matrix(dftest[,1:(ncol(dftest)-1)])
  test <- dftest[,ncol(dftest)]
  
  #Train model
  frc_f <- NULL
  for (ssn in c(1:10)){
    modelRBF <- rbf(train, test,
                    size = (2*ni), maxit = 500,
                    initFunc = "RBF_Weights_Kohonen", initFuncParams = c(0, 1, 0, 0.02, 0.04),
                    learnFunc = "RadialBasisLearning", updateFunc = "Topological_Order", 
                    shufflePatterns = FALSE, linOut = TRUE)
    #Extrapolate
    testsample <- as.numeric(input) 
    tempin <- (testsample-MIN)/(MAX-MIN)
    tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
    colnames(tempin) <- head(paste0("X",c(1:100)),ni)
    tempin <- tail(tempin,1)
    MLf <- as.numeric(predict(modelRBF,tempin))*(MAX-MIN)+MIN
    frc <- rep(MLf, fh)
    frc_f <- rbind(frc_f, frc)
  }
  frc <- colMedians(frc_f)
  
  return(frc)
}
GRNN_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  train <- as.matrix(dftest[,1:(ncol(dftest)-1)])
  test <- as.matrix(dftest[,ncol(dftest)])
  
  #Train model
  MSE <- c()
  ssize <- c(1:length(test)) 
  train_ho <- sample(ssize, round(0.8*length(test),0))
  test_ho <- ssize[ssize %!in% train_ho]
  train_tr <- train[train_ho,] ; test_tr <- test[train_ho]
  train_va <- train[test_ho,] ; test_va <- test[test_ho]
  sigmalist <- c(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00)
  for (sid in sigmalist){
    modelGRNN <- smooth(learn(cbind(test_tr,train_tr)), sigma=sid)
    fitted.m <- unlist(lapply(c(1:nrow(train_va)), function(x) guess(modelGRNN, t(train_va[x,]))))
    MSE <- c(MSE, mean((fitted.m - as.numeric(test_va))^2))
  }
  modelGRNN <- smooth(learn(cbind(test,train)), sigma=sigmalist[which.min(MSE)])
  
  #Extrapolate
  testsample <- as.numeric(input) 
  tempin <- (testsample-MIN)/(MAX-MIN)
  tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
  colnames(tempin) <- head(paste0("X",c(1:100)),ni)
  tempin <- tail(tempin,1)
  MLf <- as.numeric(guess(modelGRNN,as.matrix(tempin)))*(MAX-MIN)+MIN
  frc <- rep(MLf, fh)
  return(frc)
}
KNN_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  train <- as.matrix(dftest[,1:(ncol(dftest)-1)])
  test <- dftest[,ncol(dftest)]
  
  #Train model
  MSE <- c()
  ssize <- c(1:length(test)) 
  train_ho <- sample(ssize, round(0.8*length(test),0))
  test_ho <- ssize[ssize %!in% train_ho]
  train_tr <- train[train_ho,] ; test_tr <- test[train_ho]
  train_va <- train[test_ho,] ; test_va <- test[test_ho]
  Klist <- seq(3,31,2)
  for (nK in Klist){
    modelKNN <- knnreg(train_tr, test_tr, k = nK)   
    fitted.m <- unlist(lapply(c(1:nrow(train_va)), function(x) predict(modelKNN, t(train_va[x,])))) 
    MSE <- c(MSE, mean((fitted.m - as.numeric(test_va))^2))
  }
  modelKNN <- knnreg(train, test, k = Klist[which.min(MSE)])
  
  #Extrapolate
  testsample <- as.numeric(input) 
  tempin <- (testsample-MIN)/(MAX-MIN)
  tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
  colnames(tempin) <- head(paste0("X",c(1:100)),ni)
  tempin <- tail(tempin,1)
  MLf <- as.numeric(predict(modelKNN,tempin))*(MAX-MIN)+MIN
  frc <- rep(MLf, fh)
  return(frc)
}
CART_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  
  #Train model
  modelCART <- rpart(Y~., method="anova", data= dftest)
  modelCART <- prune(modelCART, cp = modelCART$cptable[which.min(modelCART$cptable[,"xerror"]),"CP"])
  
  #Extrapolate
  testsample <- as.numeric(input) 
  tempin <- (testsample-MIN)/(MAX-MIN)
  tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
  colnames(tempin) <- head(paste0("X",c(1:100)),ni)
  tempin <- tail(tempin,1)
  MLf <- as.numeric(predict(modelCART,tempin))*(MAX-MIN)+MIN
  frc <- rep(MLf, fh)
  return(frc)
}
SVR_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  
  #Train model
  
  modelSVR <- svm(Y~., scale=F, data= dftest, type="nu-regression")
  
  #Extrapolate
  testsample <- as.numeric(input) 
  tempin <- (testsample-MIN)/(MAX-MIN)
  tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
  colnames(tempin) <- head(paste0("X",c(1:100)),ni)
  tempin <- tail(tempin,1)
  MLf <- as.numeric(predict(modelSVR,tempin))*(MAX-MIN)+MIN
  frc <- rep(MLf, fh)
  return(frc)
}
GP_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  train <- as.matrix(dftest[,1:(ncol(dftest)-1)])
  test <- dftest[,ncol(dftest)]
  
  #Train model
  modelGP <- gausspr(x=train, y=test, scaled = FALSE, kernel="besseldot")

  #Extrapolate
  testsample <- as.numeric(input)
  tempin <- (testsample-MIN)/(MAX-MIN)
  tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
  colnames(tempin) <- head(paste0("X",c(1:100)),ni)
  tempin <- tail(tempin,1)
  MLf <- as.numeric(predict(modelGP,tempin))*(MAX-MIN)+MIN
  frc <- rep(MLf, fh)
  return(frc)
}
RF_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  
  #Train model
  modelRF <- randomForest(formula = Y ~ .,  data= dftest, ntree=500)
  
  #Extrapolate
  testsample <- as.numeric(input) 
  tempin <- (testsample-MIN)/(MAX-MIN)
  tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
  colnames(tempin) <- head(paste0("X",c(1:100)),ni)
  tempin <- tail(tempin,1)
  MLf <- as.numeric(predict(modelRF,tempin))*(MAX-MIN)+MIN
  frc <- rep(MLf, fh)
  return(frc)
}
GBT_frc <- function(input, fh, ni){
  
  #Scale data
  MAX<-max(input) ; MIN<-min(input)
  Sinsample<-(input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate<-CreateSamples(datasample=Sinsample,xi=ni)
  dftest<-data.frame(samplegenerate)
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y")
  
  #Train model
  modelGBT <- 100100
  returnML<-tryCatch(modelGBT <- gbm(Y ~ . ,data = dftest ,distribution = "gaussian", 
                                     n.trees = 500, shrinkage = 0.01, interaction.depth = 4, cv.folds = 3)
                     , error=function(e) 100)
  if (length(returnML)==1){
    modelGBT <- rpart(Y~., method="anova", data= dftest)
    modelGBT <- prune(modelGBT, cp = modelGBT$cptable[which.min(modelGBT$cptable[,"xerror"]),"CP"])
    opt_tr <- 1
  }else{
    opt_tr <- gbm.perf(modelGBT, method = "cv", plot.it = F)
  }
  
  #Extrapolate
  testsample <- as.numeric(input) 
  tempin <- (testsample-MIN)/(MAX-MIN)
  tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
  colnames(tempin) <- head(paste0("X",c(1:100)),ni)
  tempin <- tail(tempin,1)
  if (opt_tr==1){
    MLf <- as.numeric(predict(modelGBT,tempin))*(MAX-MIN)+MIN 
  }else{
    MLf <- as.numeric(predict(modelGBT,tempin, n.trees = opt_tr))*(MAX-MIN)+MIN 
  }
  frc <- rep(MLf, fh)
  return(frc)
}

#Statistical Methods
intervals <- function(x){
  y<-c()
  k<-1
  counter<-0
  for (tmp in (1:length(x))){
    if(x[tmp]==0){
      counter<-counter+1
    }else{
      k<-k+1
      y[k]<-counter
      counter<-1
    }
  }
  y<-y[y>0]
  y[is.na(y)]<-1
  y
}
demand <- function(x){
  y<-x[x!=0]
  y
}
recompose <- function(x,y1,y2,k){
  z1=z2<-c()
  
  tmp<-1
  for (t in (1):(length(x)-k)){
    if (x[t]==0){
      tmp<-tmp
    }else{
      tmp<-tmp+1
    }
    z1[t+1]<-y1[tmp]
    z2[t+1]<-y2[tmp]
  }
  z<-z1/z2
  head(z, length(x))
}
SES <- function(a, x, h, job, init){
  y<-c()  
  if (init=="naive"){
    y[1] <- x[1]
  }else{
    y[1] <- mean(x)
  }
  for (t in 1:(length(x))){
    y[t+1] <- a*x[t]+(1-a)*y[t]
  }
  fitted <- head(y,(length(y)-1))
  forecast <- rep(tail(y,1),h)
  if (job=="train"){
    return(mean((fitted - x)^2))
  }else if (job=="fit"){
    return(fitted)
  }else{
    return(list(fitted=fitted,mean=forecast))
  }
}
MA <- function(x, h, type){
  if (type=="sbj"){
    mse <- c() ; xd <- demand(x) ; xz <- intervals(x)
    for (k in 2:5){
      yd = yz <- rep(NA, k)
      for (i in (k+1):length(x)){
        yd <- c(yd, mean(xd[(i-k):i]))
        yz <- c(yz, mean(xz[(i-k):i]))
      }
      y <- yd/yz
      mse <- c(mse, mean((y-x)^2, na.rm = T))
    }
    k <- which.min(mse)+1
    forecast <- rep(mean(as.numeric(tail(xd, k)))/mean(as.numeric(tail(xz, k))), h)*((k-1)/k)
  }else{
    mse <- c()
    for (k in 2:5){
      y <- rep(NA, k)
      for (i in (k+1):length(x)){
        y <- c(y, mean(x[(i-k):i]))
      }
      mse <- c(mse, mean((y-x)^2, na.rm = T))
    }
    k <- which.min(mse)+1
    forecast <- rep(mean(as.numeric(tail(x, k))), h)
  }
  return(forecast)
}
Croston <- function(x, h, type){
  
  if (type=="classic"){
    a1 = a2 <- 0.1 ; mult <- 1 ; init <- "naive"
  }else if (type=="optimized"){
    mult <- 1 ; init <- "naive"
    a1 <- optim(c(0), SES, x=demand(x), h=1, job="train", init=init, lower = 0.1, upper = 0.3, method = "L-BFGS-B")$par
    a2 <- optim(c(0), SES, x=intervals(x), h=1, job="train", init=init, lower = 0.1, upper = 0.3, method = "L-BFGS-B")$par
  }else if (type=="sba"){
    a1 = a2 <- 0.1 ; mult <- (1-0.1/2) ; init <- "naive"
  }else if (type=="sba-opt"){
    init <- "naive" ; msel <- c()
    for (asel in c(0.1,0.2,0.3)){
      y1 <- SES(a=asel, x=demand(x), h=1, job="fit", init=init)
      y2 <- SES(a=asel, x=intervals(x), h=1, job="fit", init=init)
      msel <- c(msel, sum(na.omit((x-recompose(x,y1,y2,0))^2)))
    }
    a1 = a2 <- which.min(msel)/10
    mult <- (1-a1/2) 
  }
  
  yd <- SES(a=a1, x=demand(x), h=1, job="forecast", init=init)$mean
  yi <- SES(a=a2, x=intervals(x), h=1, job="forecast", init=init)$mean
  forecast <- rep(as.numeric(yd/yi), h)*mult
  return(forecast)
}
Naive <- function(x, h, type){
  frcst <- rep(tail(x,1), h)
  if (type=="seasonal"){
    frcst <- head(rep(as.numeric(tail(x,6)), h), h) 
  }
  return(frcst)
}
SexpS <- function(x, h, init){
  a <- optim(c(0), SES, x=x, h=1, job="train", init=init, lower = 0.1, upper = 0.3, method = "L-BFGS-B")$par
  y <- SES(a=a, x=x, h=1, job="forecast", init=init)$mean
  forecast <- rep(as.numeric(y), h)
  return(forecast)
}
TSB <- function(x, h){
  n <- length(x)
  p <- as.numeric(x != 0)
  z <- x[x != 0]
  
  a <- c(0.1, 0.2, 0.3) 
  b <- c(0.01,0.02,0.03,0.05,0.1,0.2,0.3)
  MSE <- c() ; forecast <- NULL
  for (atemp in a){
    for (btemp in b){
      zfit <- vector("numeric", length(x))
      pfit <- vector("numeric", length(x))
      zfit[1] <- z[1] ; pfit[1] <- p[1]
      
      for (i in 2:n) {
        pfit[i] <- pfit[i-1] + atemp*(p[i]-pfit[i-1])
        if (p[i] == 0) {
          zfit[i] <- zfit[i-1]
        }else {
          zfit[i] <- zfit[i-1] + btemp*(x[i]-zfit[i-1])
        }
      }
      yfit <- pfit * zfit
      forecast[length(forecast)+1] <- list(rep(yfit[n], h))
      yfit <- c(NA, head(yfit, n-1))
      MSE <- c(MSE, mean((yfit-x)^2, na.rm = T) )
    }
  }
  return(forecast[[which.min(MSE)]])
}
ADIDA <- function(x, h){
  al <- max( c(round(mean(intervals(x)),0), 2)) #mean inter-demand interval
  #Aggregated series (AS)
  AS <- as.numeric(na.omit(as.numeric(rollapply(tail(x, (length(x) %/% al)*al), al, FUN=sum, by = al))))
  forecast <- rep(SexpS(AS, 1, "naive")/al, h)
  return(forecast)
}
iADIDA <- function(x, h){
  #Inverted series (IS)
  al <- max( c(round(mean(x[x>0]),0), 2)) #mean demand size
  cdata <- data.frame(cumsum(demand(x)), intervals(x)) ; colnames(cdata) <- c("cx", "cy")
  ccomplete <- data.frame(c(1:max(cdata$cx)), NA) ; colnames(ccomplete) <- c("cx", "cn")
  cdata <- merge(ccomplete, cdata, by=c("cx"), all.x=TRUE) ; cdata$cn <- NULL
  cdata[is.na(cdata$cy),]$cy <- 0
  IS <- cdata$cy
  #Aggregated inverted series (AIS)
  AIS <- as.numeric(na.omit(as.numeric(rollapply(tail(IS, (length(IS) %/% al)*al), al, sum, by = al))))
  forecast <- SexpS(AIS, 1, "naive")
  if (forecast<1){
    forecast <- rep(al,h)
  }else{
    forecast <- rep(al/forecast,h)
  }
  return(forecast)
}
iMAPA <- function(x, h){
  mal <- max( c(round(mean(intervals(x)),0), 2))
  frc <- NULL
  for (al in 1:mal){
    frc <- rbind(frc, rep(SexpS(as.numeric(na.omit(as.numeric(rollapply(tail(x, (length(x) %/% al)*al), al, FUN=sum, by = al)))), 1, "naive")/al, h))
  }
  forecast <- colMeans(frc)
}

#Forecast Modules
ML_forc_methods <- function(tsid){
  
  wsample <- ts(na.omit(as.numeric(input[tsid,])), frequency = 6)
  insample <- head(wsample, length(wsample)-fh)
  outsample <- tail(wsample, fh)
  
  Methods <- NULL
  ni <- 12
  
  #ML methods
  Methods <- cbind(Methods, MLP_frc(insample, fh, ni)) #MLP

  invisible(capture.output(y <- BNN_frc(insample, fh, ni))) #BNN
  Methods <- cbind(Methods, y)

  Methods <- cbind(Methods, GRNN_frc(insample, fh, ni)) #GRNN

  Methods <- cbind(Methods, RBF_frc(insample, fh, ni)) #RBF

  Methods <- cbind(Methods, CART_frc(insample, fh, ni)) #CART

  Methods <- cbind(Methods, RF_frc(insample, fh, ni)) #RF

  Methods <- cbind(Methods, GBT_frc(insample, fh, ni)) #GBT

  Methods <- cbind(Methods, KNN_frc(insample, fh, ni)) #KNN

  Methods <- cbind(Methods, SVR_frc(insample, fh, ni)) #SVR

  # invisible(capture.output(ydot <- GP_frc(insample, fh, ni))) #GP
  # Methods <- cbind(Methods, ydot)
  
  #Set negatives to zero
  for (i in 1:nrow(Methods)){
    for (j in 1:ncol(Methods)){
      if (Methods[i,j]<0){ Methods[i,j]<-0  } 
    }
  }
  
  #Error estimation
  errors_mae = errors_mse = errors_me <- c()
  for (j in 1:ncol(Methods)){
    errors_mae <- c(errors_mae, mean(abs(as.numeric(Methods[,j])-as.numeric(outsample))))
    errors_mse <- c(errors_mse, mean((as.numeric(Methods[,j])-as.numeric(outsample))^2))
    errors_me <- c(errors_me, abs(mean((as.numeric(Methods[,j])-as.numeric(outsample)))))
  }
  sME <- errors_me/mean(insample)
  sMAE <- errors_mae/mean(insample)
  sMSE <- errors_mse/(mean(insample)^2)
  errors <- data.frame(rbind(sME, sMAE, sMSE)) ; colnames(errors) <- ML_names[1:ncol(Methods)]
  errors$Error <- nerrors
  errors$id <- tsid ; errors$dataset <- sn
  row.names(errors) <- NULL
  return(errors)
}
ML_forc_methods2 <- function(tsid){
  
  wsample <- ts(na.omit(as.numeric(input[tsid,])), frequency = 6)
  insample <- head(wsample, length(wsample)-fh)
  outsample <- tail(wsample, fh)
  
  Methods <- NULL
  ni <- 12
  
  #ML methods
  invisible(capture.output(ydot <- GP_frc(insample, fh, ni))) #GP
  Methods <- cbind(Methods, ydot)
  
  #Set negatives to zero
  for (i in 1:nrow(Methods)){
    for (j in 1:ncol(Methods)){
      if (Methods[i,j]<0){ Methods[i,j]<-0  } 
    }
  }
  
  #Error estimation
  errors_mae = errors_mse = errors_me <- c()
  for (j in 1:ncol(Methods)){
    errors_mae <- c(errors_mae, mean(abs(as.numeric(Methods[,j])-as.numeric(outsample))))
    errors_mse <- c(errors_mse, mean((as.numeric(Methods[,j])-as.numeric(outsample))^2))
    errors_me <- c(errors_me, abs(mean((as.numeric(Methods[,j])-as.numeric(outsample)))))
  }
  sME <- errors_me/mean(insample)
  sMAE <- errors_mae/mean(insample)
  sMSE <- errors_mse/(mean(insample)^2)
  errors <- data.frame(rbind(sME, sMAE, sMSE)) ; colnames(errors) <- ML_names[length(ML_names)]
  errors$Error <- nerrors
  errors$id <- tsid ; errors$dataset <- sn
  row.names(errors) <- NULL
  return(errors)
}
S_forc_methods <- function(tsid){
  
  wsample <- ts(na.omit(as.numeric(input[tsid,])), frequency = 6)
  insample <- head(wsample, length(wsample)-fh)
  outsample <- tail(wsample, fh)
  
  Methods <- NULL
  #Statistical methods
  Methods <- cbind(Methods, Naive(insample, fh, type="simple"))
  Methods <- cbind(Methods, Naive(insample, fh, type="seasonal"))
  Methods <- cbind(Methods, SexpS(insample, fh, "naive"))
  Methods <- cbind(Methods, MA(insample, fh, "simple"))
  Methods <- cbind(Methods, Croston(insample, fh, "classic"))
  Methods <- cbind(Methods, Croston(insample, fh, "optimized"))
  Methods <- cbind(Methods, Croston(insample, fh, "sba"))
  Methods <- cbind(Methods, Croston(insample, fh, "sba-opt"))
  Methods <- cbind(Methods, MA(insample, fh, "sbj"))
  Methods <- cbind(Methods, TSB(insample, fh))
  Methods <- cbind(Methods, ADIDA(insample, fh))
  Methods <- cbind(Methods, iADIDA(insample, fh))
  Methods <- cbind(Methods, iMAPA(insample, fh))
  
  #Set negatives to zero
  for (i in 1:nrow(Methods)){
    for (j in 1:ncol(Methods)){
      if (Methods[i,j]<0){ Methods[i,j]<-0  } 
    }
  }
  
  #Error estimation
  errors_mae = errors_mse = errors_me <- c()
  for (j in 1:ncol(Methods)){
    errors_mae <- c(errors_mae, mean(abs(as.numeric(Methods[,j])-as.numeric(outsample))))
    errors_mse <- c(errors_mse, mean((as.numeric(Methods[,j])-as.numeric(outsample))^2))
    errors_me <- c(errors_me, abs(mean((as.numeric(Methods[,j])-as.numeric(outsample)))))
  }
  sME <- errors_me/mean(insample)
  sMAE <- errors_mae/mean(insample)
  sMSE <- errors_mse/(mean(insample)^2)
  errors <- data.frame(rbind(sME, sMAE, sMSE)) ; colnames(errors) <- S_names
  errors$Error <- nerrors
  errors$id <- tsid ; errors$dataset <- sn
  row.names(errors) <- NULL
  return(errors)
}
ML_forc_methods_Global <- function(input){
  
  ni <- 12 ; nrep <- 3
  
  ######################### Scale data & create samples ##################################
  input_train <- input[,1:(ncol(input)-fh)]
  dftest = Maxes = Mins = Type = TypeInd <- NULL
  ADI = CV2 <- c()
  for (i in 1:nrow(input_train)){
    temp_sample <- na.omit(as.numeric(input_train[i,]))
    MAX<-max(temp_sample) ; MIN<-min(temp_sample)
    Sinsample<-(temp_sample-MIN)/(MAX-MIN)
    Maxes[length(Maxes)+1] <- list(MAX)
    Mins[length(Mins)+1] <- list(MIN)
    Avs <- CreateSamples(datasample=Sinsample,xi=ni)
    nrepts <- min(c(nrep,nrow(Avs)))
    dftest <- rbind(dftest, Avs[sample(c(1:nrow(Avs)), nrepts, replace = F),])
    
    ADI <- c(ADI, mean(intervals(temp_sample)))
    CV2 <- c(CV2, (sd(demand(temp_sample))/mean(demand(temp_sample)))^2)
    TypeInd <- rbind(TypeInd, c(ADI[i], CV2[i]))
    for (j in 1:nrepts){ Type <- rbind(Type, c(ADI[i], CV2[i])) }
    
  }
  Type <- data.frame(Type) ; colnames(Type) <- c("ADI","CV2")
  TypeInd <- data.frame(TypeInd) ; colnames(TypeInd) <- c("ADI","CV2")
  #Create training sample
  row.names(dftest) <- NULL
  dftest <- data.frame(dftest)
  dftest <- cbind(dftest, Type) 
  colnames(dftest)<-c(head(paste0("X",c(1:100)),ni),"Y", "ADI","CV2")
  train <- as.matrix(dftest[,colnames(dftest)%!in%c("Y")])
  test <- dftest[,c("Y")]
  ###################################################################################
  
  ########################### Train models ####################################
  print(paste("Start", sn))
  Models <- NULL ; ModelNames <- c()
  #MLP
  frc_f <- NULL
  for (ssn in c(1:10)){
    modelMLP <- mlp(train, test,
                    size = (2*ni), maxit = 500,initFunc = "Randomize_Weights",
                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
                    shufflePatterns = FALSE, linOut = TRUE)
    frc_f[length(frc_f)+1] <- list(modelMLP)
  }
  Models[[length(Models)+1]] <- list(frc_f) ; ModelNames <- c(ModelNames, "MLP")
  print("Done MLP")
  
  #BNN
  frc_f <- NULL
  for (ssn in c(1:10)){
    modelBNN <-10101
    while (length(modelBNN)==1){
      invisible(capture.output( modelBNN<-tryCatch(model<-brnn(train, as.numeric(test), 
                                                               neurons=(2*ni),normalize=FALSE,
                                                               epochs=500,verbose=F), error=function(e) 100)))
    }
    frc_f[length(frc_f)+1] <- list(modelBNN)
  }
  Models[[length(Models)+1]] <- list(frc_f) ; ModelNames <- c(ModelNames, "BNN")
  print("Done BNN")
  
  #GRNN
  MSE <- c()
  ssize <- c(1:length(test)) 
  train_ho <- sample(ssize, round(0.8*length(test),0))
  test_ho <- ssize[ssize %!in% train_ho]
  train_tr <- train[train_ho,] ; test_tr <- test[train_ho]
  train_va <- train[test_ho,] ; test_va <- test[test_ho]
  sigmalist <- c(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00)
  for (sid in sigmalist){
    modelGRNN <- smooth(learn(cbind(test_tr,train_tr)), sigma=sid)
    fitted.m <- unlist(lapply(c(1:nrow(train_va)), function(x) guess(modelGRNN, t(train_va[x,]))))
    MSE <- c(MSE, mean((fitted.m - as.numeric(test_va))^2))
  }
  modelGRNN <- smooth(learn(cbind(test,train)), sigma=sigmalist[which.min(MSE)])
  Models[length(Models)+1] <- list(modelGRNN) ; ModelNames <- c(ModelNames, "GRNN")
  print("Done GRNN")
  
  #RBF
  frc_f <- NULL
  for (ssn in c(1:10)){
    modelRBF <- rbf(train, test,
                    size = (2*ni), maxit = 500,
                    initFunc = "RBF_Weights_Kohonen", initFuncParams = c(0, 1, 0, 0.02, 0.04),
                    learnFunc = "RadialBasisLearning", updateFunc = "Topological_Order", 
                    shufflePatterns = FALSE, linOut = TRUE)
    frc_f[length(frc_f)+1] <- list(modelRBF)
  }
  Models[[length(Models)+1]] <- list(frc_f) ; ModelNames <- c(ModelNames, "RBF")
  print("Done RBF")
  
  #CART
  modelCART <- rpart(Y~., method="anova", data= dftest)
  modelCART <- prune(modelCART, cp = modelCART$cptable[which.min(modelCART$cptable[,"xerror"]),"CP"])
  Models[length(Models)+1] <- list(modelCART) ; ModelNames <- c(ModelNames, "CART")
  print("Done CART")
  
  #RF
  modelRF <- randomForest(formula = Y ~ .,  data= dftest, ntree=500)
  Models[length(Models)+1] <- list(modelRF) ; ModelNames <- c(ModelNames, "RF")
  print("Done RF")
  
  #GBT
  modelGBT <- 100100
  returnML<-tryCatch(modelGBT <- gbm(Y ~ . ,data = dftest ,distribution = "gaussian", 
                                     n.trees = 500, shrinkage = 0.01, interaction.depth = 4, cv.folds = 3)
                     , error=function(e) 100)
  if (length(returnML)==1){
    modelGBT <- rpart(Y~., method="anova", data= dftest)
    opt_tr <- 1
  }else{
    opt_tr <- gbm.perf(modelGBT, method = "cv", plot.it = F)
  }
  Models[length(Models)+1] <- list(modelGBT) ; ModelNames <- c(ModelNames, "GBT")
  print("Done GBT")
  
  #KNN
  MSE <- c()
  ssize <- c(1:length(test)) 
  train_ho <- sample(ssize, round(0.8*length(test),0))
  test_ho <- ssize[ssize %!in% train_ho]
  train_tr <- train[train_ho,] ; test_tr <- test[train_ho]
  train_va <- train[test_ho,] ; test_va <- test[test_ho]
  Klist <- seq(3,31,2)
  for (nK in Klist){
    modelKNN <- knnreg(train_tr, test_tr, k = nK)   
    fitted.m <- unlist(lapply(c(1:nrow(train_va)), function(x) predict(modelKNN, t(train_va[x,])))) 
    MSE <- c(MSE, mean((fitted.m - as.numeric(test_va))^2))
  }
  modelKNN <- knnreg(train, test, k = Klist[which.min(MSE)])
  Models[length(Models)+1] <- list(modelKNN) ; ModelNames <- c(ModelNames, "KNN")
  print("Done KNN")
  
  #SVR
  modelSVR <- svm(Y~., scale=F, data= dftest, type="nu-regression")
  Models[length(Models)+1] <- list(modelSVR) ; ModelNames <- c(ModelNames, "SVR")
  print("Done SVR")
  
  #GP
  invisible(capture.output( modelGP <- gausspr(x=train, y=test, scaled = FALSE, kernel="besseldot") ))
  Models[length(Models)+1] <- list(modelGP) ; ModelNames <- c(ModelNames, "GP")
  print("Done GP")
  ######################################################################################
  
  ################################# Generate forecasts ##################################
  frc <- data.frame(matrix(NA, nrow=nrow(input_train), ncol=length(ModelNames))) ; colnames(frc) <- ModelNames
  for (i in 1:nrow(input_train)){
    testsample <- as.numeric(na.omit(as.numeric(input_train[i,]))) 
    tempin <- (testsample-Mins[[i]])/(Maxes[[i]]-Mins[[i]])
    tempin <- data.frame(CreateSamples(datasample=tempin,xi=(ni-1)))
    colnames(tempin) <- head(paste0("X",c(1:100)),ni)
    tempin <- tail(tempin,1)
    tempin <- cbind(tempin, TypeInd[i,]) #Add type
    colnames(tempin)<-c(head(paste0("X",c(1:100)),ni), "ADI","CV2")
    
    for (j in 1:length(ModelNames)){
      
      if ((ModelNames[j]=="MLP")|(ModelNames[j]=="BNN")|(ModelNames[j]=="RBF")){
        
        frc_temp <- median(c(as.numeric(predict(Models[[j]][[1]][[1]], tempin)), as.numeric(predict(Models[[j]][[1]][[2]], tempin)),
                             as.numeric(predict(Models[[j]][[1]][[3]], tempin)), as.numeric(predict(Models[[j]][[1]][[4]], tempin)),
                             as.numeric(predict(Models[[j]][[1]][[5]], tempin))))*(Maxes[[i]]-Mins[[i]])+Mins[[i]]
        frc[i,j] <- max(c(0,frc_temp))
        
      }else if (ModelNames[j]=="GBT"){
        if (opt_tr==1){
          frc_temp <- as.numeric(predict(Models[[j]], tempin))*(Maxes[[i]]-Mins[[i]])+Mins[[i]]
        }else{
          frc_temp <- as.numeric(predict(Models[[j]], tempin, n.trees = opt_tr))*(Maxes[[i]]-Mins[[i]])+Mins[[i]]
        }
        frc[i,j] <- max(c(0,frc_temp))
      }else if (ModelNames[j]=="GRNN"){
        frc_temp <- as.numeric(guess(Models[[j]], as.matrix(tempin)))*(Maxes[[i]]-Mins[[i]])+Mins[[i]]
        frc[i,j] <- max(c(0,frc_temp))
      }else{
        frc_temp <- as.numeric(predict(Models[[j]], tempin))*(Maxes[[i]]-Mins[[i]])+Mins[[i]]
        frc[i,j] <- max(c(0,frc_temp))
      }
    }
    
  }
  ######################################################################################
  
  ################################## Compute errors ##################################
  for (j in 1:length(ModelNames)){
    errors_mae = errors_mse = errors_me <- c()
    for (i in 1:nrow(input)){
      wsample <- ts(na.omit(as.numeric(input[i,])), frequency = 6)
      insample <- head(wsample, length(wsample)-fh)
      outsample <- tail(wsample, fh)
      
      errors_mae <- c(errors_mae, mean(abs(as.numeric(rep(frc[i,j], fh))-as.numeric(outsample)))/mean(insample))
      errors_mse <- c(errors_mse, mean((as.numeric(rep(frc[i,j], fh))-as.numeric(outsample))^2)/(mean(insample)^2))
      errors_me <- c(errors_me, abs(mean((as.numeric(rep(frc[i,j], fh))-as.numeric(outsample))))/mean(insample))
    }
    namreps <- c(rep("sME",length(errors_me)),rep("sMAE",length(errors_me)),rep("sMSE",length(errors_me)))
    errors <- data.frame(c(errors_me, errors_mae, errors_mse), namreps) ; colnames(errors) <- c(ModelNames[j],"Error")
    errors$id <- c(1:i) ; errors$dataset <- sn
    row.names(errors) <- NULL
    if (j==1){
      Methods <- errors
    }else{
      Methods <- merge(Methods, errors, by=c("Error", "id", "dataset"))
    }
  }
  ###################################################################################
  
  return(Methods)
}

ML_names <- c("MLP","BNN","GRNN","RBF","CART","RF","GBT","KNN","SVR","GP")
# ML_names <- c("MLP","BNN","GRNN","RBF","CART","RF","GBT","KNN","SVR","GP")
S_names <-c("Naive", "sNaive", "SES", "MA", "Croston", "optCroston","SBA", "optSBA", "SBJ", "TSB", "ADIDA", "iADIDA", "iMAPA")
nerrors <- c("sME", "sMAE", "sMSE")
'%!in%' <- function(x,y)!('%in%'(x,y))
#Experiment settings
fh = 12
model_set <- "ML-G"

cl = registerDoSNOW(makeCluster(10, type = "SOCK"))

#Forecasting and estimation of errors

Error_matrix <- NULL
for (sn in 1:5){
  if (sn==1){
    input <- head(read.csv("sample1.csv", stringsAsFactors =  F), -fh)
  }else if (sn==2){
    input <- head(read.csv("sample2.csv", stringsAsFactors =  F), -fh) 
  }else if (sn==3){
    input <- head(read.csv("sample3.csv", stringsAsFactors =  F), -fh) 
  }else if (sn==4){
    input <- head(read.csv("sample4.csv", stringsAsFactors =  F), -fh) 
  }else if (sn==5){
    input <- head(read.csv("sample5.csv", stringsAsFactors =  F), -fh) 
  }
  input$X <- NULL
  if (model_set=="ML-G"){
    Error_matrix <- rbind(Error_matrix, ML_forc_methods_Global(input))
  }else{
    if (model_set=="S"){
      Error_matrix <- rbind(Error_matrix, foreach(tsi=1:nrow(input), .combine='rbind', .packages=c('zoo')) %dopar% S_forc_methods(tsi))
    }else if (model_set=="ML"){
      Error_matrix1 <- foreach(tsi=1:nrow(input), .combine='rbind', .packages=c('zoo','RSNNS','robustbase','brnn','grnn','caret','rpart',
                                                                                'e1071','kernlab','ddpcr','gbm','randomForest','EnvStats','plyr')) %dopar% ML_forc_methods(tsi)
      Error_matrix2 <- foreach(tsi=1:nrow(input), .combine='rbind', .packages=c('zoo','RSNNS','robustbase','brnn','grnn','caret','rpart',
                                                                                'e1071','kernlab','ddpcr','gbm','randomForest','EnvStats','plyr')) %do% ML_forc_methods2(tsi)
      # Error_matrix1 <- foreach(tsi=1:nrow(input), .combine='rbind', .packages=c('zoo','robustbase','brnn','grnn','caret','rpart',
      #                                                                           'e1071','kernlab','ddpcr','gbm','randomForest','EnvStats','plyr')) %dopar% ML_forc_methods(tsi)
      # Error_matrix2 <- foreach(tsi=1:nrow(input), .combine='rbind', .packages=c('zoo','robustbase','brnn','grnn','caret','rpart',
      #                                                                           'e1071','kernlab','ddpcr','gbm','randomForest','EnvStats','plyr')) %do% ML_forc_methods2(tsi)
      Error_matrix <- rbind(Error_matrix, merge(Error_matrix1, Error_matrix2, by=c("Error","id","dataset"), all = T))
    }
  }
}

#Summarize results
Examine_set <- Error_matrix
Examine_set$dataset = Examine_set$id <- NULL

temp <- Examine_set[Examine_set$Error=="sME",] ; temp$Error <- NULL
sME <- as.numeric(round(colMeans(temp),4))
temp <- Examine_set[Examine_set$Error=="sMSE",] ; temp$Error <- NULL
sMSE <- as.numeric(round(colMeans(temp),4))
temp <- Examine_set[Examine_set$Error=="sMAE",] ; temp$Error <- NULL
sMAE <- as.numeric(round(colMeans(temp),4))
RM <- round(unlist(lapply(c(1:ncol(temp)), function(x) robustness(temp[,x]))),4)

toprint <- rbind(sME, sMAE, RM, sMSE)
if (model_set=="S"){
  colnames(toprint) <- S_names
  save.image("StatResults.Rdata")
}else if (model_set=="ML"){
  colnames(toprint) <- ML_names
  save.image("MLResults.Rdata")
}else if (model_set=="ML-G"){
  colnames(toprint) <- ML_names
  save.image("MLglobalResults.Rdata")
}
toprint
