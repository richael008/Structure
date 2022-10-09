rm(list = ls())

Split_Count=0







mpath="/Your Working Directory/"
setwd(mpath)

Version=7

source(paste0("Stru_",Version,"_Functions.R"))

Pre      <- paste0("_Stru",rawToChar(as.raw(65+Version)),"R_")
nt       <- 2000
p        <- 10
iter     <- 40

SNB=5000
SNS=5000





##Sample size Part
##Sample List to produce different Result files
##if want ordinal result,just set SSIZE a vector of one element

SSIZE=c(2000)



SD=1
##SD=0.14


for (iSample in SSIZE)
{ 
  result=as.data.frame(matrix(data = NA,nrow=2*(iter+16),ncol=CVP$TC+2))
  FNAME=paste("S",iSample,"SD",SD,sep = "-")
  n=iSample
  
  for(istep in (1:iter))
  {
    
    
    set.seed(istep+60)        
    x = matrix(runif(p * n, -2, 2), n, p) 
    #x = matrix(rnorm(p * n), n, p) 
    
    xtest = matrix(runif(p * nt, -2, 2), nt, p)        
    ##xtest = matrix(rnorm(p * nt), nt, p)        
    
    f = function(x) { 10*sin(3.14*x[,2]*x[,3]) + 20*(x[,4]-0.5)^2+10*x[,5]+5*x[,6]}
    # f = function(x) { 
    # x[,2]+x[,3]^2+x[,4]+x[,5]^2+x[,6]+x[,2]*x[,3]+x[,3]*x[,4]+x[,4]*x[,5]
    # }
    
    # f = function(x) { 
    # x[,2]+x[,3]^2+x[,4]+x[,5]^2+x[,6]
    # }
    
    
    ftrue = f(x)
    ftest = f(xtest)
    
    if (SD==0)
    {
      sigma =sd(ftrue)
    } else  {
      sigma =SD
    }
    
    
    
    
    
    
    y = ftrue + sigma * rnorm(n)
    y_test = ftest + sigma * rnorm(nt)
    
    
    cat("Iteration ",istep, " Sigma ",sigma,"\n")
    

      
      

        
        t1=proc.time()
        
        fit1=STRUBART(numnodes=10,
                      CPath=mpath,
                      Prefix=Pre,
                      X=x,
                      Y=y,
                      X_TEST=xtest,
                      beta= 1.1,
                      num_burn = 100, 
                      num_save = 60, 
                      verbose=FALSE,
                      binary=FALSE,
                      binaryOffset=NULL,
                      n_min=5,
                      p_catagory=0,
                      p_continuous=p,
                      max_depth=10,
                      n_cutpoints=100,
                      RepeatT=5,
                      Max_catagory=10,
                      num_tree=7,
                      mtry=10,
                      Return_Result=TRUE,
                      Selected=TRUE,
                      depthconstrain=TRUE,
                      delhigest=TRUE,
                      ver=Version,
                      widetype=2,
                      try=15,
                      MixRestart=TRUE,
                      selectType=4,
                      winnercontrol=0,
                      Horder=2,
                      NT2=30,
                      resetP=0.3)
        t2=proc.time()
        t=t2-t1
        
        result[2*CVP$LLCount+istep,cvv+1]=rmse(ftest,fit1$ytest)
        result[2*CVP$LLCount+istep+iter,cvv+1] = round(t[3], 3)
        
        if (Split_Count==0)
        {
          Split_Count=1 
          Split_count_Summary= fit1$SCAT
        } else {
          
          Split_count_Summary <- cbind(Split_count_Summary,fit1$SCAT)        
        }
        
        
        cat("Iteration ",istep," Type ",cvv,  " SDBart Cost ",result[2*CVP$LLCount+istep+iter,cvv+1]," Seconds","\n")
        cat("TEST RMSE:",result[2*CVP$LLCount+istep,cvv+1],"\n") 
        
        system(paste("rm ",Pre,"*",sep=""))
        
      }    

    

  FNAME=paste(FNAME,"-V",Version,"-T-",Sys.time(),".csv",sep="")
  write.table(result,FNAME,row.names=FALSE,col.names=TRUE,sep=",")
  
}

FNAME=paste("SplitCount-V",Version,"-T-",Sys.time(),".csv",sep="")
write.table(Split_count_Summary,FNAME,row.names=FALSE,col.names=FALSE,sep=",")





