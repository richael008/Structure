rmse <- function(x,y) sqrt(mean((x-y)^2))


trank_X <- function(x) 
{
  x_unique <- unique(x)
  x_ranks <- rank(x_unique, ties.method = "max")
  tx <- x_ranks[match(x,x_unique)] - 1
  
  tx <- tx / length(unique(tx))
  tx <- tx / max(tx)
  
  return(tx)
}


quantile_normalize_bart_X <- function(X) 
{
  apply(X = X, MARGIN = 2, trank_X)
}


preprocess_df_X <- function(X) 
{
  stopifnot(is.data.frame(X))
}



GetSigma_X <- function(X,Y) 
{ 
  require("glmnet")
  stopifnot(is.matrix(X) | is.data.frame(X))
  
  if(is.data.frame(X)) 
  {
    X <- model.matrix(~.-1, data = X)
  }
  if (nrow(X)>1000)
  {
    X0=X[1:1000,]
    Y0=Y[1:1000]   
  }  else
  {
    X0=X
    Y0=Y
  }  

  
  fit <- cv.glmnet(x = X0, y = Y0)
  fitted <- predict(fit, X0)
  sigma_hat <- sqrt(mean((fitted - Y0)^2))
  return(sigma_hat)
}


unnormalize_bart_X <- function(z, a, b) {
  y <- (b - a) * (z + 0.5) + a
  return(y)
}




writeBartMpiFiles=function(numnodes,x,y,xp=NULL,xroot="x",yroot="y",xproot="xp",verbose=FALSE) 
{
  
  filelist = NULL 
  p=ncol(x)
  n=nrow(x)
  
  pv=rep(0,numnodes)
  pc=rep(0,numnodes)
  for(i in 1:numnodes) 
  {
    pv[i]=round((i*n*1.0)/numnodes, digits = 0)
    if(i==1)
    {
      pc[i]=pv[i]
    } else {
      pc[i]=pv[i]-pv[i-1]
    }
  }
  stopifnot(pv[numnodes]==n) 
  
  
  if (verbose) cat("Generating ",numnodes," train datasets of sizes ",pc,"\n")
  for(i in 1:(numnodes)) 
  {
    if(i==1) {
      ii=1:pv[1]
    } else {
      ii=(pv[i-1]+1):pv[i]
    }
    
    xx=x[ii,]
    yy=y[ii]
    tempfile = paste(c(yroot,xroot),i-1,'.csv',sep="") 
    
    
    write.table(yy,tempfile[1],row.names=FALSE,col.names=FALSE,sep=",")
    write.table(xx,tempfile[2],row.names=FALSE,col.names=FALSE,sep=",")    
    filelist = c(filelist,tempfile)
  }
  

  if(!is.null(xp)) 
  {
    if (nrow(xp)==0)
    {
      xp=NULL
    }
  }
  
    
  if(!is.null(xp)) 
  {
    for(i in 1:numnodes) 
    {
      pv[i]=round((i*nrow(xp)*1.0)/numnodes, digits = 0)
      if(i==1)
      {
        pc[i]=pv[i]
      } else {
        pc[i]=pv[i]-pv[i-1]
      }
    }
    stopifnot(pv[numnodes]==nrow(xp))     
   
    if (verbose) cat("Generating ",numnodes," predictive X datasets of sizes ",pc,"\n")
    
    for(i in 1:numnodes) #c indices of slaves
    {
      if(i==1) {
        ii=1:pv[1]
      } else {
        ii=(pv[i-1]+1):pv[i]
      }
      
      xx=xp[ii,]
      tempfile = paste(xproot,i-1,'.csv',sep="") 
      
      write.table(xx,tempfile,row.names=FALSE,col.names=FALSE,sep=",")            
      filelist = c(filelist,tempfile)
    }    
  }
  return(filelist)
}

writeBartMpiFilesALL=function(numnodes,x,y,xp=NULL,xroot="x",yroot="y",xproot="xp",verbose=FALSE) 
{
  
  filelist = NULL 
  p=ncol(x)
  n=nrow(x)
  
  pv=rep(0,numnodes)
  pc=rep(0,numnodes)
  # for(i in 1:numnodes) 
  # {
  #   pv[i]=round((i*n*1.0)/numnodes, digits = 0)
  #   if(i==1)
  #   {
  #     pc[i]=pv[i]
  #   } else {
  #     pc[i]=pv[i]-pv[i-1]
  #   }
  # }
  # stopifnot(pv[numnodes]==n) 
  
  
  # if (verbose) cat("Generating ",numnodes," train datasets of sizes ",pc,"\n")
  for(i in 1:(numnodes)) 
  {
    # if(i==1) {
    #   ii=1:pv[1]
    # } else {
    #   ii=(pv[i-1]+1):pv[i]
    # }
    
    # xx=x[ii,]
    # yy=y[ii]

    xx=x
    yy=y
    tempfile = paste(c(yroot,xroot),i-1,'.csv',sep="") 
    
    
    write.table(yy,tempfile[1],row.names=FALSE,col.names=FALSE,sep=",")
    write.table(xx,tempfile[2],row.names=FALSE,col.names=FALSE,sep=",")    
    filelist = c(filelist,tempfile)
  }
  

  if(!is.null(xp)) 
  {
    if (nrow(xp)==0)
    {
      xp=NULL
    }
  }
  
    
  if(!is.null(xp)) 
  {
    for(i in 1:numnodes) 
    {
      pv[i]=round((i*nrow(xp)*1.0)/numnodes, digits = 0)
      if(i==1)
      {
        pc[i]=pv[i]
      } else {
        pc[i]=pv[i]-pv[i-1]
      }
    }
    stopifnot(pv[numnodes]==nrow(xp))     
   
    if (verbose) cat("Generating ",numnodes," predictive X datasets of sizes ",pc,"\n")
    
    for(i in 1:numnodes) #c indices of slaves
    {
      if(i==1) {
        ii=1:pv[1]
      } else {
        ii=(pv[i-1]+1):pv[i]
      }
      
      xx=xp[ii,]
      tempfile = paste(xproot,i-1,'.csv',sep="") 
      
      write.table(xx,tempfile,row.names=FALSE,col.names=FALSE,sep=",")            
      filelist = c(filelist,tempfile)
    }    
  }
  return(filelist)
}





STRUBART<-function(
  numnodes,
  CPath,
  Prefix,
  X,
  Y,
  X_TEST=NULL,
  alpha=1,
  beta=2,
  gamma=0.95,
  k=2,
  sigma_hat=NULL,
  shape=1,
  no_split_penality=NULL,
  num_tree=20,
  alpha_scale=NULL,
  alpha_shape_1=0.5,
  alpha_shape_2=1,
  tau_rate=10,
  num_burn = 15, 
  num_save = 25, 
  update_sigma_mu = TRUE, 
  update_s = TRUE,
  update_alpha = TRUE,
  update_beta = FALSE,
  update_gamma = FALSE,
  update_tau = TRUE,
  update_tau_mean = FALSE,
  verbose=TRUE,
  binary=FALSE,
  binaryOffset=NULL,
  n_min=2,
  p_catagory=0,
  p_continuous=0,
  max_depth=10,
  n_cutpoints=100,
  RepeatT=10,
  Max_catagory=10,
  Return_Result=TRUE,
  mtry=NULL,
  Selected=FALSE,
  depthconstrain=TRUE,
  delhigest=TRUE,
  ver=0,
  widetype=1,
  try=3,
  MixRestart=TRUE,
  selectType=4,
  winnercontrol=1,
  Horder=2,
  NT2=5,
  NT3=0,
  resetP=0.0)
{ 
  
  
  if (verbose) cat("***** Running Distributed SOFT Bart\n")




  p=ncol(X)
  if(p_catagory+p_continuous!=p)
  {
    stop("p_catagory+p_continuous must be p")
  }
  
  for(i in 1:p) 
  {
    if ( (length(unique(X[,i]))> Max_catagory) != (i<=p_continuous)  )
    {
      cat("Data rules must be obeyed\n")
      cat("The first p_continuous must be continuous,unique count greater than Max_catagory \n")
      cat("The later p_catagory must be catagory,unique count less than or equal to Max_catagory \n")
      stop("break the Data rules\n")
    }
    
  }  
  
  
  
  
  if(!is.null(X_TEST) && nrow(X_TEST)<numnodes) 
  {
    X_TEST=NULL
    cat("'@@@@@@@@@@@@@ prediction x has less than numnodes, set to NULL\n'") 
  }
  

 
  if (binary)
  {
    
    if(is.factor(Y)) {
      if(length(levels(Y)) != 2) stop("Y is a factor with number of levels != 2")
      Y = as.numeric(Y)-1
    } 
    else if((length(unique(Y)) == 2) & (max(Y) == 1) & (min(Y) == 0)) 
    {
        cat('NOTE: assumming numeric response is binary\n')
    }
    else
    {
      stop("Y is not suitable for binary anlaysis")
      
    }
    if(length(binaryOffset)==0) binaryOffset=qnorm(mean(Y))
  }
  else
  {
    if((!is.vector(Y)) || (typeof(Y)!="double")) stop("argument Y must be a double vector")
    if(length(unique(Y)) == 2 )  stop("argument Y must be a Non binary vector")
  }  
  
  if(is.null(alpha_scale))    alpha_scale    <-  ncol(X)
  if (binary)
  {
    sigma_mu       <-  3 / (k * sqrt(num_tree+NT2+NT3))    
  }
  else
  {
    sigma_mu       <-  0.5 / (k * sqrt(num_tree+NT2+NT3))
  }  
  

  if (!binary)
  {  
    YMin           <- min(Y)
    YMax           <- max(Y)
    YS             <- (Y - YMin) / (YMax - YMin) - 0.5    
    
    
    if(is.null(sigma_hat))      sigma_hat      <-  GetSigma_X(X,YS)
    sigma          <-  sigma_hat
    if (verbose) cat('sighat: ',sigma_hat,'\n') 
  }
  else
  {
    YS<-Y
    YMin           <- 0
    YMax           <- 1   
    sigma_hat=1
    sigma =1
  }  
 

  
  n <- nrow(X)
  idx_train <- 1:n
  if(p_continuous>0)
  {
  
    if (p_catagory>0)
    {
      X_TRANS <- rbind(X, X_TEST)
      X_TRANS_CONT<-X_TRANS[,1:p_continuous]
      X_TRANS_CATA<-X_TRANS[,(p_continuous+1):p]
      
      
      X_TRANS_CONT <- quantile_normalize_bart_X(X_TRANS_CONT)
      X_TRANS <-cbind(X_TRANS_CONT,X_TRANS_CATA)
      X <- X_TRANS[idx_train,,drop=FALSE]
      X_TEST <- X_TRANS[-idx_train,,drop=FALSE]     
    } else{
     
      X_TRANS <- rbind(X, X_TEST)
      X_TRANS <- quantile_normalize_bart_X(X_TRANS)
      X <- X_TRANS[idx_train,,drop=FALSE]
      X_TEST <- X_TRANS[-idx_train,,drop=FALSE] 
    }

  }
  
  
  
  
  

  
  xroot=paste(Prefix,"x",sep="")
  yroot=paste(Prefix,"y",sep="")
  xproot=paste(Prefix,"xp",sep="")
  
  if (verbose)
  {
    cat('xroot,yroot,xproot:\n')
    print(xroot);print(yroot);print(xproot)
  }
  if (Selected)
  {
    tempfiles = writeBartMpiFilesALL(numnodes,X,YS,X_TEST,xroot,yroot,xproot,verbose=verbose)  
  }
  else
  {
    tempfiles = writeBartMpiFiles(numnodes,X,YS,X_TEST,xroot,yroot,xproot,verbose=verbose)
  }
                              

  if (binary)
  {
    update_sigma_mu<-0
  }
  else
  {
    update_sigma_mu<-ifelse(update_sigma_mu,1,0)
  }
 
  if(!binary) {
    binaryOffset = -1000.0
  } 

  if (is.null(mtry)) 
  {
    mtry = dim(X)[2]
    cat("mtry = p, use all variables. \n")
  }
  
  if (mtry > dim(X)[2])
  {
    mtry = dim(X)[2]
    cat("mtry cannot exceed p, set to mtry = p. \n")
  }
  
  
  
  if(is.null(no_split_penality))    no_split_penality    <-  log(n_cutpoints)
  
    
  
  binary         <-ifelse(binary,1,0)    
  update_s       <-ifelse(update_s,1,0)
  update_alpha   <-ifelse(update_alpha,1,0)
  update_beta    <-ifelse(update_beta,1,0)
  update_gamma   <-ifelse(update_gamma,1,0)
  update_tau     <-ifelse(update_tau,1,0)
  update_tau_mean<-ifelse(update_tau_mean,1,0)
  verbose        <-ifelse(verbose,1,0)
  Selected       <-ifelse(Selected,1,0)  
  depthconstrain <-ifelse(depthconstrain,1,0)  
  delhigest      <-ifelse(delhigest,1,0)  
  MixRestart     <-ifelse(MixRestart,1,0)





  
  if (verbose) cat('*****running mcmc code in parallel\n')
  cmd = paste('mpiexec -np ',numnodes,' ',CPath,'Structure_',ver,sep='')
  cmd = paste(cmd,Prefix)
  cmd = paste(cmd,alpha)
  cmd = paste(cmd,beta)
  cmd = paste(cmd,gamma)
  cmd = paste(cmd,k)
  cmd = paste(cmd,sigma_hat)
  cmd = paste(cmd,shape)
  cmd = paste(cmd,no_split_penality)
  cmd = paste(cmd,num_tree)
  cmd = paste(cmd,alpha_scale)
  cmd = paste(cmd,alpha_shape_1)
  cmd = paste(cmd,alpha_shape_2)
  cmd = paste(cmd,tau_rate)
  cmd = paste(cmd,num_burn)
  cmd = paste(cmd,num_save)
  cmd = paste(cmd,update_sigma_mu)
  cmd = paste(cmd,update_s)
  cmd = paste(cmd,update_alpha)
  cmd = paste(cmd,update_beta)
  cmd = paste(cmd,update_gamma)
  cmd = paste(cmd,update_tau)
  cmd = paste(cmd,update_tau_mean)
  cmd = paste(cmd,verbose)
  cmd = paste(cmd,binary)
  cmd = paste(cmd,binaryOffset)
  cmd = paste(cmd,n_min)
  cmd = paste(cmd,p_catagory)
  cmd = paste(cmd,p_continuous)
  cmd = paste(cmd,max_depth)
  cmd = paste(cmd,n_cutpoints)
  cmd = paste(cmd,RepeatT)
  cmd = paste(cmd,YMax)
  cmd = paste(cmd,YMin)
  cmd = paste(cmd,sigma_mu)
  cmd = paste(cmd,mtry)
  cmd = paste(cmd,Selected)
  cmd = paste(cmd,depthconstrain)
  cmd = paste(cmd,delhigest) 
  cmd = paste(cmd,widetype)
  cmd = paste(cmd,try)  
  cmd = paste(cmd,MixRestart)
  cmd = paste(cmd,selectType)
  cmd = paste(cmd,winnercontrol)
  cmd = paste(cmd,Horder)
  cmd = paste(cmd,NT2)
  cmd = paste(cmd,NT3)    
  cmd = paste(cmd,resetP) 

  if (verbose) 
  {
    cat('cmd:\n')
    cat(cmd)
    cat('\n')
  }
  
  
  
  ptime = system.time({system(cmd)})   
  


  #system(paste("rm ",xroot,"*",sep=""))  
  #system(paste("rm ",yroot,"*",sep=""))  
  #system(paste("rm ",xproot,"*",sep=""))  
  
  
  

    
    

  
  if (Return_Result)
  {

    if (Selected)
    {

      ytrain<-read.csv(paste(Pre,'_R_ytrain',0,'.csv',sep="") ,sep=',',header=FALSE) 
      for(i in 0:(numnodes-1)) 
      { 
        if (i==0)
        {
  
          ytest <-read.csv(paste(Pre,'_R_ytest',i,'.csv',sep="") ,sep=',',header=FALSE)   
        }
        else 
        {

          ytest<-rbind(ytest,read.csv(paste(Pre,'_R_ytest',i,'.csv',sep="") ,sep=',',header=FALSE))   
        }
      }


    }
    else
    {

      for(i in 0:(numnodes-1)) 
      { 
        if (i==0)
        {
          ytrain<-read.csv(paste(Pre,'_R_ytrain',i,'.csv',sep="") ,sep=',',header=FALSE)   
          ytest <-read.csv(paste(Pre,'_R_ytest',i,'.csv',sep="") ,sep=',',header=FALSE)   
        }
        else 
        {
          ytrain<-rbind(ytrain,read.csv(paste(Pre,'_R_ytrain',i,'.csv',sep="") ,sep=',',header=FALSE))
          ytest<-rbind(ytest,read.csv(paste(Pre,'_R_ytest',i,'.csv',sep="") ,sep=',',header=FALSE))   
        }
      }
    }


    fit<-list()
    
    fit$ytrain <-as.vector(as.numeric(unlist(ytrain)))
    fit$ytest <-as.vector(as.numeric(unlist(ytest)))
    fit$sigma <-read.csv(paste(Pre,'_R_sigma.csv',sep="") ,sep=',',header=FALSE)
    fit$sigma_mu <-read.csv(paste(Pre,'_R_sigma_mu.csv',sep="") ,sep=',',header=FALSE)
    fit$Var_counts <-read.csv(paste(Pre,'_R_VC.csv',sep="") ,sep=',',header=FALSE)
    fit$tau_history <-read.csv(paste(Pre,'_R_TH.csv',sep="") ,sep=',',header=FALSE)
    fit$change_histroy <-read.csv(paste(Pre,'_R_CH.csv',sep="") ,sep=',',header=FALSE)
    fit$lasttree <-read.csv(paste(Pre,'_R_TREE.csv',sep="") ,sep=',',header=FALSE)

    fit$SCAT<-read.csv(paste(Pre,'_split_count_all_tree.csv',sep="") ,sep=',',header=FALSE)
    
    
    fit$binaryOffset <- binaryOffset
    cat("Nodes counts:","\n")
    for(i in 1:(nrow(fit$Var_counts))) 
    {
      for(j in 1:(ncol(fit$Var_counts))) 
      {
        cat(fit$Var_counts[i,j]," ")
      }
      cat("\n")
    }  

    # cat("tau history:","\n")
    # for(i in 1:(nrow(fit$tau_history)))
    # {
    #   for(j in 1:(ncol(fit$tau_history)))
    #   {
    #     cat(round(fit$tau_history[i,j], digits = 2) ," ")
    #   }
    #   cat("\n")
    # }

    # cat("change history:","\n")
    # for(i in 1:(nrow(fit$change_histroy)))
    # {
    #   for(j in 1:(ncol(fit$change_histroy)))
    #   {
    #     cat(fit$change_histroy[i,j]," ")
    #   }
    #   cat("\n")
    # }

    # cat("last tree:","\n")
    # for(i in 1:(nrow(fit$lasttree)))
    # {
    #   for(j in 1:(ncol(fit$lasttree)))
    #   {
    #     cat(fit$lasttree[i,j]," ")
    #   }
    #   cat("\n")
    # }    
    
            
    
    class(fit) <- "dsxbart"
    
    return(fit)                              
  }
    
}